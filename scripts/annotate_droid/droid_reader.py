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
        ee_pose = np.asarray(obs.get("cartesian_position", obs.get("joint_position"))).astype(np.float32)
        ee_pos = ee_pose[:, :3] if ee_pose.ndim == 2 else None
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
        )

        n_emitted += 1
        if max_episodes is not None and n_emitted >= max_episodes:
            return


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
