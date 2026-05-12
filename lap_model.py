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
        # Tag of which wrist image the client packed into ``left_wrist_0_rgb``.
        # Stays "left" until A1.b stickiness is implemented.
        self.last_wrist: str = "left"

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
        wrist = _ensure_224(left)   # always feed left wrist for now (A1.b stickiness deferred)

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
        # unconditionally (``is_vqa_sample`` / ``is_prediction_sample``). For
        # inference we set both to False (= regular task sample, no auxiliary
        # heads). language_actions / langact / plan are omitted so the
        # tokenizer goes through the legacy "prompt only" Context 1 path.
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
        print("[LAP] reset observation window + instruction")

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
