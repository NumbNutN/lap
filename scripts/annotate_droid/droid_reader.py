"""DROID episode → annotator-ready dict.

Two paths are supported:

1. **RLDS / TFDS** path (the canonical DROID release). Yields one episode
   at a time, lazily, via :func:`iter_droid_rlds`. Mirrors the fields used
   by ``policy/pi05/src/openpi/training/droid_rlds_dataset.py``.

2. **Pre-decoded JSONL** path (offline dev / pilot). When the user has
   already dumped a handful of episodes to a custom JSONL+images layout
   (no TF / dlimp dependency on the annotator host), :func:`iter_jsonl`
   reads them. This is what we use for the 100-episode pilot if we don't
   want to ship the 1.7 TB RLDS to the annotation worker.

Both return an :class:`EpisodeBundle`:

    EpisodeBundle:
        episode_id: str          — stable identifier
        task_instruction: str
        fps: float
        gripper_width: np.ndarray (T,) — open ~0.08, closed 0.0
        ee_pos:        np.ndarray (T, 3) — world-frame translation (or None)
        frame_loader:  Callable[[int], np.ndarray] — returns RGB uint8 image
                       at frame_idx, lazily decoded from RLDS bytes
        n_frames:      int
        wrist_loader:  Optional[Callable[[int], np.ndarray]] — wrist camera

Callers (keyframe detector, prompt builder) only need ``gripper_width``,
``ee_pos``, and the lazy ``frame_loader``. We load images only for the
selected keyframes — never the full episode into memory.

TODO marked sections require user to plug in the actual data root once
they pick a path; rest is provider-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeBundle:
    episode_id: str
    task_instruction: str
    fps: float
    n_frames: int
    gripper_width: np.ndarray
    ee_pos: np.ndarray | None
    frame_loader: Callable[[int], np.ndarray]
    wrist_loader: Callable[[int], np.ndarray] | None = None
    # Full 6-DoF end-effector pose, shape (T, 6) = [x, y, z, rx, ry, rz].
    # rx/ry/rz is a Rodrigues rotation vector (DROID convention). Used by
    # the prompt builder to compute per-keyframe pose deltas (cm + axis
    # name) so the VLM can emit finer-grained axis-aware actions.
    ee_pose: np.ndarray | None = None
    # Hand-eye calibration: 6D pose (xyz + Rodrigues rotvec) of the wrist
    # camera in the EE frame. Constant across an episode. Loaded from
    # the raw HDF5 (`camera_extrinsics/{wrist_serial}_left_gripper_offset`)
    # when reading via `iter_droid_raw`. Used by `pose_utils.pose_delta`
    # to project Δp into the actual wrist camera frame instead of
    # falling back to the empirical EE-axis approximation.
    T_ee_wrist: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Path 1 — RLDS / TFDS reader
# ---------------------------------------------------------------------------


def iter_droid_rlds(
    data_dir: str,
    *,
    dataset_name: str = "droid",
    dataset_version: str | None = None,
    max_episodes: int | None = None,
    success_only: bool = True,
    skip: int = 0,
) -> Iterator[EpisodeBundle]:
    """Iterate DROID RLDS episodes, yielding one EpisodeBundle at a time.

    Args:
        data_dir: path to the TFDS-formatted DROID root. Expected layout
            is ``<data_dir>/<dataset_name>/<version>/...``.
        dataset_name: TFDS builder name. Use ``"droid"`` for the full
            76k-episode release, ``"droid_100"`` for the 2 GB pilot subset.
        dataset_version: optional version pin (e.g. ``"1.0.1"``). When
            None, TFDS picks the latest version present.
        max_episodes: stop after this many bundles (None = full set).
        success_only: filter the RLDS stream by the standard "success"
            file_path regex used by the openpi DROID loader. The
            ``droid_100`` subset is already all-success, so this filter
            is a no-op on it but harmless.
        skip: skip the first N episodes (for resume after crash).

    Memory: streams via tf.data; only one episode's tensors materialise at
    a time on the Python side.
    """
    # Lazy import — tensorflow / dlimp are heavy and not needed for the
    # JSONL path.
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import dlimp as dl

    tf.config.set_visible_devices([], "GPU")
    builder_kwargs: dict = {"data_dir": data_dir}
    if dataset_version is not None:
        builder_kwargs["version"] = dataset_version
    builder = tfds.builder(dataset_name, **builder_kwargs)
    ds = dl.DLataset.from_rlds(builder, split="train", shuffle=False)

    if success_only:
        ds = ds.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0],
                ".*success.*",
            )
        )

    n_emitted = 0
    n_seen = 0
    for traj in ds.as_numpy_iterator():
        n_seen += 1
        if n_seen <= skip:
            continue

        # ---- decode metadata --------------------------------------------
        meta = traj["traj_metadata"]["episode_metadata"]
        # file_path is a (T,) array of identical bytes — take the first.
        file_path = meta["file_path"][0]
        if isinstance(file_path, bytes):
            file_path = file_path.decode("utf-8")
        recording = meta.get("recording_folderpath", [b""])[0]
        if isinstance(recording, bytes):
            recording = recording.decode("utf-8")
        episode_id = f"{recording}::{file_path}" if recording else file_path

        # Sample a language instruction (DROID has up to 3 per episode).
        instr_keys = [
            "language_instruction",
            "language_instruction_2",
            "language_instruction_3",
        ]
        task_instruction = ""
        for k in instr_keys:
            v = traj.get(k)
            if v is None:
                continue
            s = v[0]
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            if s.strip():
                task_instruction = s.strip()
                break

        # ---- state curves ----------------------------------------------
        obs = traj["observation"]
        gripper_width = np.asarray(obs["gripper_position"]).astype(np.float32)
        # Franka EE pose: shape (T, 6) translation + axis-angle. Take first 3.
        ee_pose_arr = np.asarray(obs.get("cartesian_position", obs.get("joint_position"))).astype(np.float32)
        ee_pos = ee_pose_arr[:, :3] if ee_pose_arr.ndim == 2 else None
        # Keep full 6-DoF (xyz + Rodrigues rotvec) when present — used by
        # the prompt builder to compute axis-aware deltas. cartesian_position
        # is (T, 6); joint_position is (T, 7) and not directly usable as
        # an EE pose, so only attach when shape matches.
        full_ee_pose = ee_pose_arr if (ee_pose_arr.ndim == 2 and ee_pose_arr.shape[1] == 6) else None
        T = int(gripper_width.shape[0])

        # ---- lazy image loaders ----------------------------------------
        # The exterior images come as encoded JPEG bytes per frame; decode
        # only when requested for a keyframe.
        ext1_bytes = obs.get("exterior_image_1_left")
        ext2_bytes = obs.get("exterior_image_2_left")
        wrist_bytes = obs.get("wrist_image_left")

        def _decode_jpeg(b: bytes) -> np.ndarray:
            return tf.io.decode_image(b, expand_animations=False, dtype=tf.uint8).numpy()

        # Use exterior_1 by default; some episodes only have one of the two.
        primary_bytes = ext1_bytes if ext1_bytes is not None else ext2_bytes

        def frame_loader(idx: int, _bytes=primary_bytes) -> np.ndarray:
            return _decode_jpeg(_bytes[idx])

        wrist_loader = None
        if wrist_bytes is not None:
            def wrist_loader(idx: int, _bytes=wrist_bytes) -> np.ndarray:  # noqa: F811
                return _decode_jpeg(_bytes[idx])

        yield EpisodeBundle(
            episode_id=episode_id,
            task_instruction=task_instruction,
            fps=15.0,  # DROID standard
            n_frames=T,
            gripper_width=gripper_width,
            ee_pos=ee_pos,
            frame_loader=frame_loader,
            wrist_loader=wrist_loader,
            ee_pose=full_ee_pose,
        )

        n_emitted += 1
        if max_episodes is not None and n_emitted >= max_episodes:
            return


# ---------------------------------------------------------------------------
# Path 1b — Raw HDF5 + MP4 reader (DROID 1.0.1 raw release)
# ---------------------------------------------------------------------------


def iter_droid_raw(
    data_dir: str,
    *,
    success_only: bool = True,
    labs: list[str] | None = None,
    max_episodes: int | None = None,
    skip: int = 0,
    skipped_log_path: str | None = None,
) -> "Iterator[EpisodeBundle]":
    """Iterate DROID raw episodes (HDF5 trajectory + MP4 streams).

    Raw layout (e.g. ``/data/datasets/droid_data_raw/1.0.1/``):

        <root>/<lab>/<success|failure>/<date>/<timestamp>/
            metadata_<uuid>.json     # camera serials + extrinsics + task hint
            trajectory.h5            # robot state + per-frame camera extrinsics
            recordings/MP4/
                <wrist_serial>.mp4    # left eye of Zed-Mini wrist cam
                <ext1_serial>.mp4
                <ext2_serial>.mp4

    Per-episode work:
      - Parse `metadata_*.json` for camera serials + task instruction
      - Read `trajectory.h5`:
          * `observation/robot_state/cartesian_position` → ee_pose (T, 6)
          * `observation/robot_state/gripper_position`   → gripper_width (T,)
          * `observation/camera_extrinsics/<wrist>_left_gripper_offset[0]`
                                                         → T_ee_wrist (constant)
      - Open MP4s with cv2.VideoCapture and expose lazy frame_loader /
        wrist_loader (single-eye 1280x720 frames). h5 trajectory has one
        more frame than MP4 (verified ≥30 eps in droid_100); we clamp to
        `min(T_h5, T_mp4)` so indexes line up 1-to-1.

    Episodes that lack the wrist gripper_offset or wrist MP4 are SKIPPED
    (no empirical fallback) and recorded to ``skipped_log_path`` (when
    provided) so we can audit which subset got included.

    Args:
        data_dir: root containing ``<lab>/`` subdirs.
        success_only: skip ``failure/`` subtree.
        labs: optional whitelist of lab names; None = all labs.
        max_episodes: yield at most this many bundles.
        skip: skip first N episodes (for resume after crash).
        skipped_log_path: append-mode file path to log skipped episodes.
    """
    import h5py
    import cv2

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"raw data dir not found: {root}")

    # Stream-friendly: glob top-level once; recurse-glob inside loop.
    lab_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if labs is not None:
        lab_set = set(labs)
        lab_dirs = [p for p in lab_dirs if p.name in lab_set]

    skipped_log = None
    if skipped_log_path is not None:
        skipped_log = open(skipped_log_path, "a")

    def _log_skip(ep_path: Path, reason: str) -> None:
        line = f"{ep_path}\t{reason}\n"
        logger.info("[skip] %s: %s", ep_path, reason)
        if skipped_log is not None:
            skipped_log.write(line)
            skipped_log.flush()

    n_seen = 0
    n_yielded = 0
    try:
        for lab in lab_dirs:
            outcomes = ["success"] if success_only else ["success", "failure"]
            for outcome in outcomes:
                outcome_dir = lab / outcome
                if not outcome_dir.exists():
                    continue
                # date dirs
                for date_dir in sorted(outcome_dir.iterdir()):
                    if not date_dir.is_dir():
                        continue
                    for ep_dir in sorted(date_dir.iterdir()):
                        if not ep_dir.is_dir():
                            continue
                        n_seen += 1
                        if n_seen <= skip:
                            continue
                        if max_episodes is not None and n_yielded >= max_episodes:
                            return

                        bundle = _build_raw_bundle(
                            ep_dir, root, _log_skip, cv2_mod=cv2, h5py_mod=h5py,
                        )
                        if bundle is None:
                            continue
                        n_yielded += 1
                        yield bundle
    finally:
        if skipped_log is not None:
            skipped_log.close()


def _build_raw_bundle(
    ep_dir: Path,
    root: Path,
    log_skip: Callable[[Path, str], None],
    *,
    cv2_mod,
    h5py_mod,
) -> "EpisodeBundle | None":
    """Build one EpisodeBundle from a raw episode directory.

    Returns None and logs the reason if required files / keys are missing.
    """
    h5_path = ep_dir / "trajectory.h5"
    if not h5_path.exists():
        log_skip(ep_dir, "missing trajectory.h5")
        return None

    meta_paths = list(ep_dir.glob("metadata_*.json"))
    if not meta_paths:
        log_skip(ep_dir, "missing metadata_*.json")
        return None
    with open(meta_paths[0]) as f:
        md = json.load(f)

    wrist_serial = md.get("wrist_cam_serial")
    ext1_serial = md.get("ext1_cam_serial")
    if not wrist_serial:
        log_skip(ep_dir, "metadata missing wrist_cam_serial")
        return None

    task_instruction = md.get("current_task", "") or ""

    with h5py_mod.File(h5_path, "r") as f:
        if "observation/robot_state/cartesian_position" not in f:
            log_skip(ep_dir, "h5 missing cartesian_position")
            return None
        ee_pose = f["observation/robot_state/cartesian_position"][:].astype(np.float32)
        gripper_width = (
            f["observation/robot_state/gripper_position"][:].astype(np.float32)
        )
        T_h5 = int(ee_pose.shape[0])
        # Hand-eye calibration (must exist; no empirical fallback).
        offset_key = (
            f"observation/camera_extrinsics/{wrist_serial}_left_gripper_offset"
        )
        if offset_key not in f:
            log_skip(ep_dir, f"h5 missing {offset_key}")
            return None
        T_ee_wrist = f[offset_key][0].astype(np.float32)  # constant; take t=0

    # Pick primary external MP4 (ext1 first, else ext2).
    primary_serial = ext1_serial or md.get("ext2_cam_serial")
    if not primary_serial:
        log_skip(ep_dir, "metadata missing both ext1/ext2 serials")
        return None
    primary_mp4 = ep_dir / "recordings" / "MP4" / f"{primary_serial}.mp4"
    wrist_mp4 = ep_dir / "recordings" / "MP4" / f"{wrist_serial}.mp4"
    if not primary_mp4.exists() or not wrist_mp4.exists():
        log_skip(ep_dir, f"missing MP4 (primary={primary_mp4.exists()}, wrist={wrist_mp4.exists()})")
        return None

    # Probe MP4 frame count to clamp h5 to a 1-to-1 alignment (we observed
    # T_mp4 = T_h5 - 1 systematically across 30 sampled eps).
    cap_probe = cv2_mod.VideoCapture(str(primary_mp4))
    T_primary = int(cap_probe.get(cv2_mod.CAP_PROP_FRAME_COUNT))
    fps_probe = float(cap_probe.get(cv2_mod.CAP_PROP_FPS)) or 15.0
    cap_probe.release()
    cap_probe_w = cv2_mod.VideoCapture(str(wrist_mp4))
    T_wrist = int(cap_probe_w.get(cv2_mod.CAP_PROP_FRAME_COUNT))
    cap_probe_w.release()
    T = min(T_h5, T_primary, T_wrist)
    if T < 5:
        log_skip(ep_dir, f"too few frames T={T}")
        return None
    # Sanity: MP4 fps label is often wrong (Zed encodes as 60); the actual
    # robot control rate is 15 Hz per spec. Hard-code 15.0 for downstream
    # detector and pose-delta math that depend on dt.
    fps = 15.0

    ee_pose = ee_pose[:T]
    gripper_width = gripper_width[:T]

    # Lazy frame loaders. cv2.VideoCapture is not thread-safe; each
    # loader keeps its own capture. For sequential reads this is fast;
    # for random access we seek with CAP_PROP_POS_FRAMES (acceptable for
    # keyframe rates).
    def _make_loader(path: str):
        cap = [None]
        def load(idx: int) -> np.ndarray:
            if cap[0] is None:
                cap[0] = cv2_mod.VideoCapture(path)
            cap[0].set(cv2_mod.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap[0].read()
            if not ok or img is None:
                raise RuntimeError(f"failed to read frame {idx} from {path}")
            return cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2RGB)
        return load

    frame_loader = _make_loader(str(primary_mp4))
    wrist_loader = _make_loader(str(wrist_mp4))

    # episode_id = h5 relative path (matches RLDS episode_id tail so
    # annotation JSONLs can cross-reference).
    rel_h5 = h5_path.relative_to(root)
    episode_id = str(rel_h5)

    return EpisodeBundle(
        episode_id=episode_id,
        task_instruction=task_instruction,
        fps=fps,
        n_frames=T,
        gripper_width=gripper_width,
        ee_pos=ee_pose[:, :3],
        ee_pose=ee_pose,
        frame_loader=frame_loader,
        wrist_loader=wrist_loader,
        T_ee_wrist=T_ee_wrist,
    )


# ---------------------------------------------------------------------------
# Path 2 — pre-decoded JSONL reader (for offline pilot / dev)
# ---------------------------------------------------------------------------


def iter_jsonl(
    jsonl_path: str,
    images_root: str,
    *,
    max_episodes: int | None = None,
    skip: int = 0,
) -> Iterator[EpisodeBundle]:
    """Iterate episodes from a pre-decoded JSONL + image folder.

    JSONL line schema (one episode per line)::

        {
          "episode_id": "...",
          "task_instruction": "...",
          "fps": 15.0,
          "gripper_width": [...],          # length T
          "ee_pos":        [[x,y,z], ...], # length T, optional
          "image_dir":     "<rel path under images_root>",
                                           # contains primary_{0000..T-1}.jpg
                                           # optional wrist_{0000..T-1}.jpg
        }

    This is the format you should dump 100 pilot episodes into so the
    annotation worker does not need TFDS / dlimp / the full 1.7TB.
    """
    from PIL import Image

    p = Path(jsonl_path)
    images_root_p = Path(images_root)
    if not p.exists():
        raise FileNotFoundError(f"jsonl manifest not found: {p}")

    n_emitted = 0
    n_seen = 0
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            if n_seen <= skip:
                continue
            d = json.loads(line)

            gripper = np.asarray(d["gripper_width"], dtype=np.float32)
            ee_pos = None
            if "ee_pos" in d and d["ee_pos"] is not None:
                ee_pos = np.asarray(d["ee_pos"], dtype=np.float32)
            T = int(gripper.shape[0])
            image_dir = images_root_p / d["image_dir"]

            def frame_loader(idx: int, _dir=image_dir) -> np.ndarray:
                pth = _dir / f"primary_{idx:04d}.jpg"
                return np.asarray(Image.open(pth).convert("RGB"))

            wrist_loader = None
            wrist_probe = image_dir / "wrist_0000.jpg"
            if wrist_probe.exists():
                def wrist_loader(idx: int, _dir=image_dir) -> np.ndarray:  # noqa: F811
                    return np.asarray(Image.open(_dir / f"wrist_{idx:04d}.jpg").convert("RGB"))

            yield EpisodeBundle(
                episode_id=str(d["episode_id"]),
                task_instruction=str(d.get("task_instruction", "")).strip(),
                fps=float(d.get("fps", 15.0)),
                n_frames=T,
                gripper_width=gripper,
                ee_pos=ee_pos,
                frame_loader=frame_loader,
                wrist_loader=wrist_loader,
            )
            n_emitted += 1
            if max_episodes is not None and n_emitted >= max_episodes:
                return


# ---------------------------------------------------------------------------
# Synthetic reader for unit tests / dry-run without DROID data
# ---------------------------------------------------------------------------


def make_synthetic_bundle(*, T: int = 200, fps: float = 15.0) -> EpisodeBundle:
    """Make a synthetic episode mirroring the keyframe.py demo. For tests only."""
    width = np.full(T, 0.08, dtype=np.float32)
    width[40:50] = 0.0
    width[50:60] = 0.08
    width[60:150] = 0.0
    width[150:] = 0.08
    ee = np.zeros((T, 3), dtype=np.float32)
    for t in range(70, 110):
        ee[t] = ((t - 70) * 0.01, 0.0, 0.0)
    for t in range(110, 150):
        ee[t] = (0.4, (t - 110) * 0.01, 0.0)

    def frame_loader(idx: int) -> np.ndarray:
        # Deterministic colour gradient per index — useful for visual sanity
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[:, :, 0] = (idx * 7) % 256
        img[:, :, 1] = (idx * 13) % 256
        img[:, :, 2] = (idx * 23) % 256
        return img

    return EpisodeBundle(
        episode_id="synthetic_001",
        task_instruction="Pick up the red block and place it on the table.",
        fps=fps,
        n_frames=T,
        gripper_width=width,
        ee_pos=ee,
        frame_loader=frame_loader,
        wrist_loader=None,
    )


if __name__ == "__main__":
    bundle = make_synthetic_bundle()
    print(f"synthetic bundle: id={bundle.episode_id}  T={bundle.n_frames}  "
          f"fps={bundle.fps}  task={bundle.task_instruction!r}")
    img = bundle.frame_loader(50)
    print(f"  frame 50 shape={img.shape} dtype={img.dtype}")
