"""RoboTwin task-suite dataset adapter for LAP Stage 2 (action-expert) training.

Mirrors ``bridge_ecot_dataset.py`` shape so the two paths compose cleanly inside
``create_data_loader``. The interesting differences vs Bridge V2:

* **Source**: HDF5 (image as JPEG-encoded byte vectors per frame) + per-episode
  metadata JSON, both produced by RoboTwin's data collection pipeline.
* **Bimanual**: 14-DoF actions (left + right endpose 7d each), with both arms
  visible per frame. Only one arm is *active* in any phase (``phase.arm_tag``).
* **Cameras**: head_camera (third-person) + left_camera + right_camera (wrists)
  + front_camera (overhead, not used). Per the A1.b decision we feed
  ``head_camera → base_0_rgb`` and the wrist matching the active arm
  (``left_camera`` if arm_tag="left" else ``right_camera``) into the
  ``left_wrist_0_rgb`` slot.
* **Cascade-VLA layout**: identical to Bridge V2 (3-segment plan/stage/action),
  with a per-dataset cascade extractor that pulls task_prompt, synthesizes a
  multi-step plan from unique subgoal_prompts, and threads
  ``subgoal_reasoning`` (Bridge-style causal reasoning) into ``[stage]`` and
  ``phase_prompts`` into ``[action]``.
* **C2 mixed-reasoning frequency**: not every frame emits text. Phase-boundary
  frames always emit full cascade; mid-phase frames emit full cascade with
  probability ``p_full_reasoning`` (default 0.20) and otherwise emit the
  prompt-only "action-vector-only" Context 1 layout.

The adapter assumes the metadata JSON has been augmented with
``subgoal_reasoning: list[str]`` per phase (matching ``stack_blocks_*_K3``).
For datasets where this field is missing today, we fall back to a templated
reasoning so the dataloader still works during early development.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random
from collections import OrderedDict
from typing import Any, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ROBOTWIN_DATA_ROOT = pathlib.Path(
    "/data/zhaoqc/RoboTwin/data"  # path on the pod; override locally via constructor
)

# Datasets in the Stage 2 mix and their nominal sampling weights. Override via
# constructor for ablations / smoke tests.
DEFAULT_DATASET_WEIGHTS: dict[str, float] = {
    "pick_place_primitive": 0.5,
    "arrange_blocks_line": 1.0,
    "arrange_blocks_l_shape": 1.0,
    "arrange_blocks_u_shape": 1.0,
    "stack_blocks_n_stack_n_v1_open_K3": 1.0,
}

# Slot in the LAP image dict that the active wrist camera lands in. Even though
# the slot is named "left_wrist_0_rgb" we route the *active* arm's wrist into
# it (per A1.b stickiness inference protocol — the slot represents "the wrist
# the model should attend to for this frame").
WRIST_SLOT_KEY = "left_wrist_0_rgb"
BASE_SLOT_KEY = "base_0_rgb"


# ----------------------------------------------------------------------------
# Cascade extraction (per-dataset)
# ----------------------------------------------------------------------------

def _synth_pickplace_plan(task_spec: dict) -> str:
    """Pick-and-place is a single-subgoal task; the plan is one sentence."""
    pick_obj = task_spec["pick"]["descriptor"]
    place_loc = task_spec["place"]["loc_at_descriptor"]
    return f"Pick up {pick_obj} and place it {place_loc}."


_PICKPLACE_REASONING_TEMPLATES: dict[str, list[str]] = {
    "approach": [
        "The {color} cube is the pick target named in the task, so the {arm} arm must position above it before grasping.",
        "Because the {color} cube is the only valid pick, the {arm} arm hovers over it to set up the grasp.",
        "The task specifies the {color} cube as the pick target, and a clean approach from above is needed before closing the gripper.",
        "Approaching the {color} cube from above gives the {arm} gripper room to descend without bumping the cube sideways.",
    ],
    "grasp": [
        "The gripper must close on the {color} cube without slipping the cube, so it lowers and grips firmly.",
        "Now that the {arm} gripper is positioned over the {color} cube, lowering and closing the fingers secures the grasp.",
        "Grasping the {color} cube requires controlled descent followed by finger closure to avoid kicking the cube.",
        "The {color} cube is centered under the {arm} gripper, so closing the fingers now picks it up cleanly.",
    ],
    "transport": [
        "The {color} cube is now in hand and needs to reach {dest}, so the {arm} arm carries it through the air.",
        "Because the {color} cube is grasped, the {arm} arm transports it horizontally toward {dest}.",
        "The cube has to clear the table before placing, so the {arm} arm lifts and travels above the workspace toward {dest}.",
        "Transporting the {color} cube to {dest} requires a smooth airborne trajectory to avoid disturbing other blocks.",
    ],
    "lift_down": [
        "The {arm} arm has arrived above {dest}, so lowering and releasing places the {color} cube in position.",
        "Now that the {color} cube is over {dest}, opening the {arm} gripper drops it precisely.",
        "Placing the {color} cube at {dest} requires lowering carefully and then releasing it to set it down.",
        "The transport phase ended above {dest}, so lowering and releasing finishes the placement of the {color} cube.",
    ],
}


def _pickplace_reasoning(task_spec: dict, phase: dict) -> list[str]:
    """Synthesize a 4-variant subgoal_reasoning list for a pick_place phase."""
    color = task_spec["pick"]["color"]
    arm = phase.get("arm_tag", "left")
    dest = task_spec["place"]["loc_at_descriptor"]
    templates = _PICKPLACE_REASONING_TEMPLATES.get(phase["kind"]) or _PICKPLACE_REASONING_TEMPLATES["approach"]
    return [t.format(color=color, arm=arm, dest=dest) for t in templates]


def cascade_pick_place(meta: dict, phase_idx: int, phase: dict, rng: random.Random) -> dict:
    plan = _synth_pickplace_plan(meta["task_spec"])
    sr_list = phase.get("subgoal_reasoning") or _pickplace_reasoning(meta["task_spec"], phase)
    pp_list = phase.get("phase_prompts") or [
        f"Move the {phase.get('arm_tag','left')} gripper to manipulate the cube."
    ]
    return {
        "task_prompt": plan,           # task prompt = plan (single-step)
        "plan": plan,
        "stage": rng.choice(sr_list) if sr_list else "",
        "action_lang": rng.choice(pp_list),
    }


def _arrange_plan_from_phases(meta: dict) -> str:
    """Build a multi-step plan by concatenating unique subgoal_prompts in order."""
    phases = meta.get("phases", [])
    plan_parts: list[str] = []
    seen: set[str] = set()
    for ph in phases:
        sp = (ph.get("subgoal_prompt") or "").strip()
        if sp and sp not in seen:
            seen.add(sp)
            if not sp.endswith("."):
                sp = sp + "."
            plan_parts.append(sp)
    return " ".join(plan_parts) or (meta.get("task_prompt") or "")


def cascade_arrange(meta: dict, phase_idx: int, phase: dict, rng: random.Random) -> dict:
    sr_list = phase.get("subgoal_reasoning")
    if not sr_list:
        # Fall back to subgoal_prompt + cot.per_phase_text [Step] suffix
        cot = meta.get("cot", {})
        per_phase = cot.get("per_phase_text", []) if isinstance(cot, dict) else []
        text = per_phase[phase_idx] if phase_idx < len(per_phase) else ""
        if "[Subgoal]" in text:
            sr_list = [text.split("[Step]")[0].split("[Subgoal]", 1)[-1].strip(". ").strip()]
        else:
            sr_list = [phase.get("subgoal_prompt", "")]
    pp_list = phase.get("phase_prompts") or [meta.get("task_prompt", "")]
    return {
        "task_prompt": meta.get("task_prompt", ""),
        "plan": _arrange_plan_from_phases(meta),
        "stage": rng.choice(sr_list) if sr_list else "",
        "action_lang": rng.choice(pp_list),
    }


def cascade_stack(meta: dict, phase_idx: int, phase: dict, rng: random.Random) -> dict:
    sr_list = phase.get("subgoal_reasoning")
    if not sr_list:
        sr_list = [phase.get("subgoal_prompt", "")]
    pp_list = phase.get("phase_prompts") or [meta.get("task_prompt", "")]
    return {
        "task_prompt": meta.get("task_prompt", ""),
        "plan": _arrange_plan_from_phases(meta),
        "stage": rng.choice(sr_list) if sr_list else "",
        "action_lang": rng.choice(pp_list),
    }


_CASCADE_BY_FAMILY = {
    "pick_place_primitive": cascade_pick_place,
    "arrange_blocks_line": cascade_arrange,
    "arrange_blocks_l_shape": cascade_arrange,
    "arrange_blocks_u_shape": cascade_arrange,
    "arrange_blocks_i_shape": cascade_arrange,
    "stack_blocks_n_stack_n_v1_open_K3": cascade_stack,
    "stack_blocks_n_stack_n_v1_open": cascade_stack,
    "stack_blocks_n_stack_n_v1": cascade_stack,
}


def cascade_extract(dataset_name: str, meta: dict, phase_idx: int, phase: dict, rng: random.Random) -> dict:
    fn = _CASCADE_BY_FAMILY.get(dataset_name)
    if fn is None:
        # Closest-match by family prefix.
        for key, fn_ in _CASCADE_BY_FAMILY.items():
            if dataset_name.startswith(key):
                fn = fn_
                break
    if fn is None:
        raise ValueError(f"No cascade extractor registered for dataset {dataset_name!r}")
    return fn(meta, phase_idx, phase, rng)


# ----------------------------------------------------------------------------
# HDF5 image + action loader (with bounded LRU cache)
# ----------------------------------------------------------------------------

class _RoboTwinHDF5Reader:
    """Bounded LRU cache of open HDF5 file handles + JPEG decode pipeline.

    RoboTwin episodes are 86-840 frames each at 4 cameras × ~16KB/jpeg, so
    keeping ~32 episodes warm fits comfortably in <1GB. ``max_handles=32``
    chosen to balance reader overhead vs memory.
    """

    def __init__(self, max_handles: int = 32, image_size: tuple[int, int] = (224, 224)):
        self._cache: OrderedDict[pathlib.Path, Any] = OrderedDict()
        self._max_handles = max_handles
        self._image_size = image_size

    def _open(self, hdf5_path: pathlib.Path):
        import h5py
        if hdf5_path in self._cache:
            self._cache.move_to_end(hdf5_path)
            return self._cache[hdf5_path]
        if len(self._cache) >= self._max_handles:
            old_path, old_h = self._cache.popitem(last=False)
            try:
                old_h.close()
            except Exception:
                pass
        h = h5py.File(str(hdf5_path), "r")
        self._cache[hdf5_path] = h
        return h

    def decode_frame(self, hdf5_path: pathlib.Path, camera_key: str, frame_idx: int) -> np.ndarray:
        """Return RGB HWC uint8 image at the requested resolution."""
        import cv2
        h = self._open(hdf5_path)
        ds = h[f"observation/{camera_key}/rgb"]
        raw = ds[frame_idx]
        if hasattr(raw, "tobytes"):
            buf = raw.tobytes()
        else:
            buf = bytes(raw)
        bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imdecode failed for {hdf5_path}::{camera_key}[{frame_idx}]")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != self._image_size:
            rgb = cv2.resize(rgb, (self._image_size[1], self._image_size[0]), interpolation=cv2.INTER_AREA)
        return rgb

    def read_actions(self, hdf5_path: pathlib.Path, frame_idx: int, action_horizon: int) -> np.ndarray:
        """Read joint_action/vector slice of length ``action_horizon`` starting at ``frame_idx``.

        Pads at the right with the last valid action when ``frame_idx + action_horizon``
        exceeds episode length (so the dataloader can work on any frame).
        """
        h = self._open(hdf5_path)
        vec = h["joint_action/vector"]
        n_frames = vec.shape[0]
        end = min(frame_idx + action_horizon, n_frames)
        out = np.asarray(vec[frame_idx:end], dtype=np.float32)
        if out.shape[0] < action_horizon:
            pad_count = action_horizon - out.shape[0]
            pad = np.repeat(out[-1:], pad_count, axis=0) if out.shape[0] > 0 else np.zeros(
                (action_horizon, vec.shape[1]), dtype=np.float32
            )
            out = np.concatenate([out, pad], axis=0)
        return out

    def read_state(self, hdf5_path: pathlib.Path, frame_idx: int) -> np.ndarray:
        """Read 14-d state = [left_endpose 7d || right_endpose 7d] at ``frame_idx``."""
        h = self._open(hdf5_path)
        left = np.asarray(h["endpose/left_endpose"][frame_idx], dtype=np.float32)
        right = np.asarray(h["endpose/right_endpose"][frame_idx], dtype=np.float32)
        return np.concatenate([left, right], axis=0)


# ----------------------------------------------------------------------------
# RoboTwin dataset (per-task) and unified mixer
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class RoboTwinTaskDataset:
    """Streaming dataset over one RoboTwin task directory (e.g. arrange_blocks_l_shape).

    Each iteration yields a per-frame sample dict in the same shape Bridge V2
    emits (see ``bridge_ecot_dataset.BridgeECoTSampleBuilder.build``), so it
    plugs into ``BridgeDataLoader``-style consumers with no further changes.
    """

    task_dir: pathlib.Path                       # e.g. /data/.../arrange_blocks_l_shape
    action_horizon: int = 8
    p_plan: float = 0.15                         # Bridge-style plan-as-target probability
    p_full_reasoning: float = 0.20               # mid-phase frames emitting full cascade
    image_size: tuple[int, int] = (224, 224)
    max_episodes: Optional[int] = None
    seed: int = 0

    def __post_init__(self):
        self.task_dir = pathlib.Path(self.task_dir)
        self.dataset_name = self.task_dir.name
        self._meta_dir = self.task_dir / "demo_clean" / "metadata"
        self._data_dir = self.task_dir / "demo_clean" / "data"
        if not self._meta_dir.is_dir():
            raise FileNotFoundError(f"metadata dir missing: {self._meta_dir}")
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"data dir missing: {self._data_dir}")
        # Discover episodes by scanning the metadata dir; episode_idx encoded in filename.
        self._episode_ids = sorted(
            int(p.stem.removeprefix("episode"))
            for p in self._meta_dir.glob("episode*.json")
            if p.stem.removeprefix("episode").isdigit()
        )
        if self.max_episodes is not None:
            self._episode_ids = self._episode_ids[: self.max_episodes]
        self._reader = _RoboTwinHDF5Reader(image_size=self.image_size)
        self._rng = random.Random(self.seed)

    # ----- helpers -----

    def _meta_path(self, ep: int) -> pathlib.Path:
        return self._meta_dir / f"episode{ep}.json"

    def _hdf5_path(self, ep: int) -> pathlib.Path:
        return self._data_dir / f"episode{ep}.hdf5"

    def _load_meta(self, ep: int) -> dict:
        with open(self._meta_path(ep)) as f:
            return json.load(f)

    @staticmethod
    def _phase_for_frame(phases: list[dict], frame_idx: int) -> tuple[Optional[int], Optional[dict]]:
        for i, ph in enumerate(phases):
            if ph["start_frame"] <= frame_idx < ph["end_frame"]:
                return i, ph
        return None, None

    @staticmethod
    def _is_phase_boundary(phases: list[dict], frame_idx: int, window: int = 2) -> bool:
        for ph in phases:
            if abs(frame_idx - ph["start_frame"]) <= window:
                return True
            if abs(frame_idx - ph["end_frame"]) <= window:
                return True
        return False

    # ----- main iteration -----

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[dict]:
        """Yield per-frame sample dicts in random episode + frame order."""
        emitted = 0
        eps = list(self._episode_ids)
        self._rng.shuffle(eps)
        for ep in eps:
            try:
                meta = self._load_meta(ep)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning("Skipping episode %d: %s", ep, e)
                continue
            phases = meta.get("phases", [])
            if not phases:
                continue
            n_frames = phases[-1]["end_frame"]
            hdf5 = self._hdf5_path(ep)
            if not hdf5.exists():
                continue
            # Iterate frames in random order (sampling within episode).
            frame_indices = list(range(n_frames))
            self._rng.shuffle(frame_indices)
            for frame_idx in frame_indices:
                phase_idx, phase = self._phase_for_frame(phases, frame_idx)
                if phase is None:
                    continue
                sample = self._build_sample(meta, phase_idx, phase, ep, frame_idx)
                if sample is None:
                    continue
                yield sample
                emitted += 1
                if max_samples is not None and emitted >= max_samples:
                    return

    def _build_sample(
        self, meta: dict, phase_idx: int, phase: dict, episode: int, frame_idx: int
    ) -> Optional[dict]:
        # 1. Cascade text fields.
        try:
            cas = cascade_extract(self.dataset_name, meta, phase_idx, phase, self._rng)
        except Exception as e:
            logger.warning("cascade_extract failed for %s ep%d frame%d: %s",
                           self.dataset_name, episode, frame_idx, e)
            return None

        # 2. Decide whether this frame emits full cascade (3-segment) or
        #    action-vector-only (Context 1 with no reasoning).
        is_boundary = self._is_phase_boundary(meta["phases"], frame_idx)
        emit_full = is_boundary or (self._rng.random() < self.p_full_reasoning)

        # 3. Decide plan_position (only relevant if emit_full).
        if emit_full:
            plan_position = "target" if self._rng.random() < self.p_plan else "prompt"
            plan_text = cas["plan"]
            stage_text = cas["stage"]
            action_text = cas["action_lang"]
        else:
            plan_position = "none"
            plan_text = None
            stage_text = None
            action_text = None

        # 4. Image: head_camera → base; active arm's wrist → wrist slot.
        arm_tag = phase.get("arm_tag", "left")
        wrist_camera = "left_camera" if arm_tag == "left" else "right_camera"
        try:
            head_img = self._reader.decode_frame(self._hdf5_path(episode), "head_camera", frame_idx)
            wrist_img = self._reader.decode_frame(self._hdf5_path(episode), wrist_camera, frame_idx)
        except Exception as e:
            logger.warning("image decode failed for %s ep%d frame%d: %s",
                           self.dataset_name, episode, frame_idx, e)
            return None

        # 5. Action chunk + state.
        actions = self._reader.read_actions(self._hdf5_path(episode), frame_idx, self.action_horizon)
        state = self._reader.read_state(self._hdf5_path(episode), frame_idx)

        # 6. Pack into the bridge-compatible per-step sample dict.
        return {
            # Cascade text (None when emit_full=False; tokenizer treats those as Context 1 no-reason).
            "prompt": cas["task_prompt"],
            "language_actions": stage_text,
            "langact": action_text,
            "plan": plan_text,
            "plan_position": plan_position,
            # Visual.
            "image": {
                BASE_SLOT_KEY: head_img,
                WRIST_SLOT_KEY: wrist_img,
            },
            "image_mask": {
                BASE_SLOT_KEY: np.bool_(True),
                WRIST_SLOT_KEY: np.bool_(True),
            },
            # State + actions (Stage 2 specific — Bridge had zeros here).
            "state": state.astype(np.float32),
            "actions": actions.astype(np.float32),
            # Routing flags.
            "is_vqa_sample": np.bool_(False),
            "is_prediction_sample": np.bool_(False),
            "sample_mask": np.bool_(True),
            # Bookkeeping (not used by training; useful for debugging dumps).
            "_dataset": self.dataset_name,
            "_episode": episode,
            "_frame_idx": frame_idx,
            "_phase_idx": phase_idx,
            "_arm_tag": arm_tag,
            "_emit_full_reasoning": emit_full,
        }


@dataclasses.dataclass
class RoboTwinMixedDataset:
    """Weighted union of multiple RoboTwinTaskDatasets.

    Sampling protocol per epoch tick:
      1. Choose one dataset from the mix according to ``weights``.
      2. Fetch one sample from that dataset's per-task iterator.

    Weights are renormalized to sum=1.0; smaller weights → less-frequent picks.
    """

    data_root: pathlib.Path = DEFAULT_ROBOTWIN_DATA_ROOT
    weights: dict[str, float] = dataclasses.field(default_factory=lambda: dict(DEFAULT_DATASET_WEIGHTS))
    action_horizon: int = 8
    p_plan: float = 0.15
    p_full_reasoning: float = 0.20
    image_size: tuple[int, int] = (224, 224)
    max_episodes_per_dataset: Optional[int] = None
    seed: int = 0

    def __post_init__(self):
        self.data_root = pathlib.Path(self.data_root)
        # Build per-task datasets only for those that exist on disk.
        self._tasks: list[tuple[str, float, RoboTwinTaskDataset]] = []
        for name, w in self.weights.items():
            task_dir = self.data_root / name
            if not task_dir.is_dir():
                logger.warning("RoboTwin task dir missing, skipping: %s", task_dir)
                continue
            try:
                ds = RoboTwinTaskDataset(
                    task_dir=task_dir,
                    action_horizon=self.action_horizon,
                    p_plan=self.p_plan,
                    p_full_reasoning=self.p_full_reasoning,
                    image_size=self.image_size,
                    max_episodes=self.max_episodes_per_dataset,
                    seed=self.seed + abs(hash(name)) % 1000,
                )
            except FileNotFoundError as e:
                logger.warning("RoboTwin task %s init failed, skipping: %s", name, e)
                continue
            self._tasks.append((name, float(w), ds))

        if not self._tasks:
            raise RuntimeError(
                f"No RoboTwin tasks found under {self.data_root}. "
                f"Expected at least one of: {list(self.weights.keys())}"
            )

        # Renormalize weights over present tasks.
        total = sum(w for _, w, _ in self._tasks)
        self._renorm = [(name, w / total, ds) for (name, w, ds) in self._tasks]
        self._rng = random.Random(self.seed)

        # Per-task iterators (lazy; rebuild on exhaust).
        self._iters: dict[str, Iterator[dict]] = {}

    def _next_from_task(self, name: str, ds: RoboTwinTaskDataset) -> Optional[dict]:
        it = self._iters.get(name)
        if it is None:
            it = ds.iter_samples()
            self._iters[name] = it
        try:
            return next(it)
        except StopIteration:
            it = ds.iter_samples()
            self._iters[name] = it
            try:
                return next(it)
            except StopIteration:
                return None

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[dict]:
        names = [n for n, _, _ in self._renorm]
        weights = [w for _, w, _ in self._renorm]
        ds_lookup = {n: ds for n, _, ds in self._renorm}
        emitted = 0
        while True:
            if max_samples is not None and emitted >= max_samples:
                return
            chosen = self._rng.choices(names, weights=weights, k=1)[0]
            sample = self._next_from_task(chosen, ds_lookup[chosen])
            if sample is None:
                continue
            yield sample
            emitted += 1

    @property
    def dataset_names(self) -> list[str]:
        return [n for n, _, _ in self._renorm]
