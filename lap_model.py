#!/usr/bin/python3
# -- coding: UTF-8
"""LAP cascade-VLA client wrapper (RoboTwin Stage 2 deployment).

Talks to the WebSocket inference server launched on the pod via
``policy/lap/scripts/serve_policy.py --env LAP_ROBOTWIN``. Mirrors pi05's
``PI0`` client surface so ``script/eval_policy.py`` can drive it through
the generic ``policy.<name>.eval()`` interface.

Differences vs pi05 client:
  * **Image schema**: 2 cameras (``base_0_rgb`` = head, ``left_wrist_0_rgb``
    = active arm wrist), HWC uint8, matching ``RoboTwinTaskDataset`` output.
    Stickiness for the active wrist (A1.b decision) is approximated here by
    always sending ``left_camera`` as the wrist; refinement to follow a
    parsed ``arm_tag`` from cascade output is a future enhancement.
  * **State**: 14-DoF bimanual (left endpose 7 + right endpose 7).
  * **No CFG / negative prompts** — Stage 2 didn't train with negative
    distillation, so those code paths are stubbed out.
"""

from __future__ import annotations

import os
import time

import numpy as np
import websockets.exceptions as _ws_exc
from openpi_client import websocket_client_policy


# Image keys the LAP RoboTwin server expects (must match LAPConfig.image_keys
# in ``lap.training.config.lap_robotwin_finetune``).
BASE_KEY = "base_0_rgb"
WRIST_KEY = "left_wrist_0_rgb"


class LAP:
    """Client for the LAP RoboTwin cascade-VLA policy server."""

    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: int,
        pi0_step: int,
        server_host: str = "localhost",
        server_port: int = 8000,
        infer_max_retries: int = 3,
        infer_retry_backoff_s: float = 1.0,
        # Action smoothing (D from stage2_sim_eval_diagnosis.md): low-pass
        # filter applied per-executed action to dampen the chunk-boundary
        # jumps that compound into off-trajectory drift in IL eval.
        # ``smoothed_t = α · raw_t + (1−α) · smoothed_{t−1}``.
        # α=1.0 → disabled (raw actions); 0.5 ≈ moderate smoothing; <0.3
        # is very heavy and will lag the policy noticeably.
        action_smooth_alpha: float = 1.0,
        # Plan-cache (stage2_design_discussions_zh.md §2.5.5). When True,
        # the first inference of each episode lets the server AR-generate
        # the full ``[plan][stage][action]`` cascade. The client extracts
        # the ``[plan]`` segment from ``reasoning_text`` and on every
        # subsequent inference sends ``plan`` + ``plan_position="prompt"``
        # in the obs dict so the tokenizer routes through the
        # plan-in-prompt path (training-distribution Context 2). AR then
        # only regenerates ``[stage]`` + ``[action]``. Reduces per-infer
        # cascade variance and matches the training observation that plan
        # is episode-static.
        cache_plan: bool = True,
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pi0_step = pi0_step
        self.exec_horizon = pi0_step
        self.img_size = (224, 224)

        self.observation_window: dict | None = None
        self.instruction: str | None = None
        self._infer_count = 0
        self.verbose = False
        self.verbose_dir: str | None = None

        # Populated by ``get_action`` if the server returns a ``reasoning_text``
        # field alongside ``actions``. Used by the eval driver to set the
        # rollout-video overlay text per inference.
        self.last_reasoning_text: str = ""

        # A1.b last_arm stickiness state. ``last_arm`` says which physical
        # wrist camera (``left`` or ``right``) is packed into the model's
        # single wrist slot for *this* inference. Updated by
        # ``_parse_arm_from_cascade`` after each successful infer. Default
        # "right" because pick_place / arrange tasks usually start by acting
        # with the right gripper (matches the bias in training data).
        # ``last_wrist`` mirrors it for the video overlay.
        self.last_arm: str = "right"
        self.last_wrist: str = self.last_arm

        # Plan-cache state. Cleared on ``reset_obsrvationwindows``.
        self.cache_plan = bool(cache_plan)
        self.cached_plan: str = ""

        # Action-smoothing state. Cleared on ``reset_obsrvationwindows``.
        self.action_smooth_alpha = float(action_smooth_alpha)
        self._smoothed_action: np.ndarray | None = None

        self.server_host = server_host
        self.server_port = server_port
        self.infer_max_retries = max(1, int(infer_max_retries))
        self.infer_retry_backoff_s = float(infer_retry_backoff_s)

        print(f"[LAP] Connecting to inference server at {server_host}:{server_port}...")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=server_host, port=server_port
        )
        meta = self.policy.get_server_metadata()
        print(f"[LAP] Connected. Server metadata: {meta}")

    def smooth_action(self, raw_action) -> np.ndarray:
        """EMA-blend ``raw_action`` with the previously-smoothed one.

        Returns the smoothed action to be sent to the env. Updates internal
        state so subsequent calls chain correctly. When
        ``action_smooth_alpha >= 1.0`` this is a pass-through (raw action).
        """
        raw = np.asarray(raw_action, dtype=np.float32)
        a = self.action_smooth_alpha
        if a >= 1.0 or self._smoothed_action is None:
            self._smoothed_action = raw
        else:
            self._smoothed_action = (a * raw + (1.0 - a) * self._smoothed_action).astype(np.float32)
        return self._smoothed_action

    # ----- generic eval-policy contract -----

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"[LAP] set instruction: {instruction!r}")

    def set_neg_prompts(self, neg_prompts, guidance_scale=1.5):
        """Stub: Stage 2 LAP doesn't use CFG / negative prompts."""
        if neg_prompts:
            print(f"[LAP] (warn) set_neg_prompts ignored — Stage 2 has no CFG distillation")

    def update_observation_window(self, img_arr, state):
        """Pack the latest observation into a server-bound dict.

        Args:
            img_arr: list of HWC uint8 RGB images in the order
                [head_camera, right_camera, left_camera] (RoboTwin convention
                from ``encode_obs`` — note pi05 mirrored this order). LAP only
                uses head + left for now (single wrist slot per the Stage 2
                training schema).
            state: 14-d float vector (left_endpose 7 + right_endpose 7).
        """
        head, right, left = img_arr[0], img_arr[1], img_arr[2]

        # Resize each cam to the model's expected (224, 224) resolution if
        # needed; sim usually outputs the right size already.
        def _ensure_224(img):
            if img.shape[:2] == self.img_size:
                return img
            try:
                import cv2
                return cv2.resize(img, (self.img_size[1], self.img_size[0]),
                                  interpolation=cv2.INTER_AREA)
            except ImportError:
                from PIL import Image
                arr = np.array(Image.fromarray(img).resize(
                    (self.img_size[1], self.img_size[0])))
                return arr

        head = _ensure_224(head)
        # A1.b "last_arm stickiness" inference protocol: the wrist slot must
        # match what the training dataset packed (which was
        # ``left_camera if phase.arm_tag=="left" else right_camera``). The
        # client maintains ``self.last_arm`` and parses each cascade response
        # for the arm mention (see ``get_action``). This is the short-term
        # fix for the train/test wrist mismatch documented in
        # ``stage2_design_discussions_zh.md §1``.
        wrist_src = right if self.last_arm == "right" else left
        wrist = _ensure_224(wrist_src)
        self.last_wrist = self.last_arm

        # State is 14-d for bimanual RoboTwin; pad/truncate defensively.
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != 14:
            if state_arr.shape[0] < 14:
                state_arr = np.concatenate(
                    [state_arr, np.zeros(14 - state_arr.shape[0], dtype=np.float32)])
            else:
                state_arr = state_arr[:14]

        # Match RoboTwinTaskDataset emit format (HWC uint8 + dict-of-cams) and
        # carry the routing flags ``TokenizePromptAndReasoning`` reads
        # unconditionally (``is_vqa_sample`` / ``is_prediction_sample``).
        # When ``cache_plan`` is on AND we've extracted a plan from a prior
        # cascade response (this episode), include ``plan`` + ``plan_position
        # ="prompt"`` so the tokenizer routes through the plan-in-prompt
        # branch (training Context 2). AR will then only need to generate
        # ``[stage]`` + ``[action]``. On the first infer of each episode
        # ``cached_plan`` is empty → fall through to the legacy "prompt
        # only" path (Context 1) and let AR produce the full cascade,
        # which ``get_action`` then mines for the plan to cache.
        self.observation_window = {
            "image": {
                BASE_KEY: head.astype(np.uint8),
                WRIST_KEY: wrist.astype(np.uint8),
            },
            "image_mask": {
                BASE_KEY: True,
                WRIST_KEY: True,
            },
            "state": state_arr,
            "prompt": self.instruction or "",
            "is_vqa_sample": False,
            "is_prediction_sample": False,
        }
        if self.cache_plan and self.cached_plan:
            # Server's TokenizePromptAndReasoning will encode
            # ``[plan] <cached_plan>`` into the prompt span (NOT the AR
            # target). AR then only needs to emit ``[stage]`` + ``[action]``.
            self.observation_window["plan"] = self.cached_plan
            self.observation_window["plan_position"] = "prompt"

    def get_action(self):
        assert self.observation_window is not None, (
            "[LAP] observation_window not set; call update_observation_window first")
        last_exc = None
        for attempt in range(1, self.infer_max_retries + 1):
            try:
                t0 = time.time()
                result = self.policy.infer(self.observation_window)
                elapsed = time.time() - t0
                actions = result["actions"]
                # Optional cascade text — present only if the server-side
                # policy wrapper additionally runs ``infer_reasoning``. Falls
                # back to empty so the eval-side overlay just hides the field.
                self.last_reasoning_text = str(
                    result.get("reasoning_text", "") or ""
                ).strip()
                # Update which wrist camera goes into the next inference's
                # wrist slot based on the cascade text the model just emitted
                # (A1.b stickiness — see ``stage2_design_discussions_zh.md §1``).
                self.last_arm = self._parse_arm_from_cascade(
                    self.last_reasoning_text, self.last_arm
                )
                # Plan-cache: on the first infer of each episode the cascade
                # response contains a freshly-generated ``[plan]<text>``
                # segment. Extract it, store it, and subsequent inferences
                # will route through the plan-in-prompt path (avoiding the
                # plan-drifts-each-tick problem we observed in early eval
                # videos). The plan span is everything between ``[plan]``
                # and ``[stage]`` (or end-of-text if no stage marker yet).
                if self.cache_plan and not self.cached_plan and self.last_reasoning_text:
                    plan_text = self._extract_plan_segment(self.last_reasoning_text)
                    if plan_text:
                        self.cached_plan = plan_text
                        print(f"[LAP] plan cached ({len(plan_text)} chars): "
                              f"{plan_text[:100]!r}...")
                # Per-inference timing breakdown — written by the server-side
                # CascadePipelinePolicy. Lets us see how much of each infer
                # is LLM (AR) vs flow-matching vs round-trip overhead.
                pt = result.get("policy_timing", {}) or {}
                st = result.get("server_timing", {}) or {}
                ar_ms = pt.get("ar_ms")
                flow_ms = pt.get("flow_ms")
                cascade_tokens = pt.get("cascade_tokens")
                server_ms = st.get("infer_ms")
                wire_ms = max(0.0, elapsed * 1000.0 - (server_ms or 0.0))
                parts = [f"client_total={elapsed*1000:.0f}ms"]
                if server_ms is not None:
                    parts.append(f"server={server_ms:.0f}ms")
                if ar_ms is not None and flow_ms is not None:
                    total_model = ar_ms + flow_ms
                    ar_share = (ar_ms / total_model * 100.0) if total_model > 0 else 0.0
                    parts.append(
                        f"AR={ar_ms:.0f}ms({ar_share:.0f}%) flow={flow_ms:.0f}ms({100-ar_share:.0f}%)"
                    )
                if cascade_tokens is not None:
                    parts.append(f"tokens={cascade_tokens}")
                parts.append(f"net_overhead={wire_ms:.0f}ms")
                print(f"[LAP-perf #{self._infer_count + 1:04d}] " + "  ".join(parts))
                self._infer_count += 1
                if self.verbose:
                    self._log_inference(elapsed, actions)
                if attempt > 1:
                    print(f"[LAP] infer succeeded on retry {attempt}/{self.infer_max_retries}")
                return actions
            except _ws_exc.ConnectionClosed as e:
                last_exc = e
                print(f"[LAP] infer attempt {attempt}/{self.infer_max_retries} "
                      f"failed: {type(e).__name__}: {e}")
                if attempt >= self.infer_max_retries:
                    break
                time.sleep(self.infer_retry_backoff_s * attempt)
                try:
                    self._reconnect()
                except Exception as re:
                    print(f"[LAP] reconnect failed: {type(re).__name__}: {re}")
                    last_exc = re
        raise last_exc

    def reset_obsrvationwindows(self):     # name kept for parity with pi05 / eval_policy.py
        self.instruction = None
        self.observation_window = None
        self._infer_count = 0
        self._smoothed_action = None
        # Reset A1.b stickiness state on episode boundary. Default to "right"
        # since most RoboTwin task families (pick_place, arrange_blocks)
        # initiate motion with the right gripper.
        self.last_arm = "right"
        self.last_wrist = self.last_arm
        # Drop the cached plan so the first infer of the next episode
        # regenerates one (the new episode's plan can differ — e.g. the
        # scene snapshot rolled a different cube placement, so the
        # task-prompt's `Arrange the blocks into an L shape.` resolves to
        # a different multi-step plan).
        self.cached_plan = ""
        print("[LAP] reset observation window + instruction")

    @staticmethod
    def _extract_plan_segment(text: str) -> str:
        """Return the text between ``[plan]`` and the next segment marker.

        The server-side ``CascadePipelinePolicy`` produces a single decoded
        string like ``"[plan] <plan_text> [stage] <reasoning> [action]
        <langact>"``. Plan-cache (see ``stage2_design_discussions_zh.md
        §2.5.5``) wants the ``<plan_text>`` substring without the
        ``[plan]`` marker so we can pass it back as the ``plan`` field of
        the next obs dict (where ``TokenizePromptAndReasoning`` will
        re-prepend the marker itself).

        Returns the empty string if no ``[plan]`` marker is found.
        """
        if not text:
            return ""
        marker = "[plan]"
        i = text.find(marker)
        if i == -1:
            return ""
        start = i + len(marker)
        # End at the next segment marker (whichever comes first).
        end = len(text)
        for nxt in ("[stage]", "[action]"):
            j = text.find(nxt, start)
            if j != -1 and j < end:
                end = j
        return text[start:end].strip()

    @staticmethod
    def _parse_arm_from_cascade(text: str, prev: str) -> str:
        """Heuristic: extract ``left`` / ``right`` from the cascade text.

        Searches in priority order ``[action]`` → ``[stage]`` → ``[plan]``.
        Looks for the phrases ``left gripper`` / ``right gripper`` / ``left
        arm`` / ``right arm``. If both appear in the same segment, takes
        whichever appears LAST (most recent action). Falls through to the
        previous value if no signal is found.

        Cascade training templates include explicit ``{arm}`` substitution
        (see ``robotwin_dataset.py::_PICKPLACE_REASONING_TEMPLATES``) so the
        signal should be present at training-distribution frames.
        """
        if not text:
            return prev
        lower = text.lower()
        # Slice from the most specific section back to the broadest.
        for marker in ("[action]", "[stage]", "[plan]"):
            idx = lower.find(marker)
            if idx == -1:
                continue
            seg = lower[idx + len(marker):]
            # Cut at the next marker if there is one (rare but safe).
            for nxt in ("[plan]", "[stage]", "[action]"):
                ni = seg.find(nxt)
                if ni != -1:
                    seg = seg[:ni]
            left_idx = max(
                seg.rfind("left gripper"),
                seg.rfind("left arm"),
                seg.rfind("left hand"),
            )
            right_idx = max(
                seg.rfind("right gripper"),
                seg.rfind("right arm"),
                seg.rfind("right hand"),
            )
            if left_idx == -1 and right_idx == -1:
                continue
            return "right" if right_idx > left_idx else "left"
        # No segment markers — try whole text.
        if "right gripper" in lower or "right arm" in lower:
            return "right"
        if "left gripper" in lower or "left arm" in lower:
            return "left"
        return prev

    # ----- internals -----

    def _reconnect(self):
        print(f"[LAP] reconnecting to {self.server_host}:{self.server_port}...")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=self.server_host, port=self.server_port,
        )
        print(f"[LAP] reconnected. Server metadata: {self.policy.get_server_metadata()}")

    def _log_inference(self, elapsed: float, actions):
        obs = self.observation_window
        prompt = obs.get("prompt", "")
        state = obs.get("state")
        print(f"\n[LAP-Verbose] Inference #{self._infer_count}  |  {elapsed:.3f}s")
        print(f"  Prompt : {prompt}")
        if state is not None:
            arr = np.asarray(state)
            print(f"  State  : shape={arr.shape}  "
                  f"range=[{arr.min():.4f}, {arr.max():.4f}]")
        actions_arr = np.asarray(actions)
        print(f"  Actions: shape={actions_arr.shape}  "
              f"range=[{actions_arr.min():.4f}, {actions_arr.max():.4f}]")
        if self.verbose_dir:
            self._save_obs_images()

    def _save_obs_images(self):
        from PIL import Image
        obs = self.observation_window
        images = obs.get("image", {})
        step_dir = os.path.join(self.verbose_dir, f"infer_{self._infer_count:04d}")
        os.makedirs(step_dir, exist_ok=True)
        for k, img in images.items():
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(step_dir, f"{k}.png"))
        print(f"  Images saved to {step_dir}")
