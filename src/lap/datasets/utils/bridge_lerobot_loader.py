"""Image loader that pulls Bridge V2 RGB frames from the LeRobot HF datasets
``jnogga/bridge_data_v2_teleop`` and ``jnogga/bridge_data_v2_scripted``, and
maps them to the per-step (file_path, episode_id, step_idx) addressing used by
``Embodied-CoT/embodied_features_bridge``.

Strategy
========

1. **Build a static mapping** from (ECoT file_path, ECoT episode_id) to a
   LeRobot (dataset_kind, episode_index, frame_index_offset). This mapping is
   built once and cached as a JSON file (~30 MB) so we do not repeat the
   matching cost every run.

2. **Match algorithm**
   - LeRobot uuid example::

         raw/bridge_data_v2/<workspace>/<task>/<run_id>/<datetime>/raw/traj_group<G>/traj<N>

   - ECoT JSON file_path example::

         /nfs/.../numpy_256/bridge_data_v2/<workspace>/<task>/<run_id>/train/out.npy

   - We strip everything before ``numpy_256/`` (or ``scripted_numpy_256/``) and
     the trailing ``/(train|val)/out.npy`` from the ECoT path, then prepend
     ``raw/`` to get a candidate ``task_root``.
   - We strip ``/<datetime>/raw/traj_group\d+/traj\d+`` from the LeRobot uuid to
     get its task_root, and group LeRobot episodes by task_root.
   - For each ECoT episode within ``out.npy``, we match it to a candidate
     LeRobot episode under the same task_root by ``n_steps == adapter.length``.
     When multiple LeRobot episodes share the same length, we use the
     ECoT episode's integer ``episode_id`` (as it appears in the JSON) as a
     tiebreaker by sorting matches and taking the (id mod N)-th.

3. **Image fetching** — given a LeRobot (episode_index, frame_index), we read
   the corresponding frame from the ``observation.images.camera_0`` mp4
   (Bridge "fixed" external camera). We use ``decord`` for random-access
   frame reading.

Limitations
-----------
- The ``episode_id`` -> LeRobot ``traj_index`` ordering inside an ``out.npy``
  is heuristic. This loader treats it as best-effort: we may misalign images
  with annotations for ~10–20% of episodes when multiple trajs in the same
  task_root have identical length. This is acceptable for *language*
  pretraining (model still learns task / plan / subtask correlations from
  approximately-matched images) but should be re-validated before fine-tuning
  with action heads.
- Episodes without LeRobot match (~14% per LeRobot's filtering) are dropped.

Public API
----------
- :class:`LeRobotBridgeImageLoader` — concrete ``BridgeV2ImageLoader``.
- :func:`build_ecot_to_lerobot_mapping` — one-shot mapping builder; cached.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import re
from collections import defaultdict
from typing import Any

import numpy as np

from lap.datasets.bridge_ecot_dataset import BridgeV2ImageLoader

logger = logging.getLogger(__name__)


# Default snapshot directories (HF cache layout).
DEFAULT_TELEOP_SNAP = pathlib.Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--jnogga--bridge_data_v2_teleop/snapshots/"
        "38b9d67fa978ed3cc59f28e607a061457d20a865"
    )
)
DEFAULT_SCRIPTED_SNAP_PARENT = pathlib.Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--jnogga--bridge_data_v2_scripted/snapshots"
    )
)
DEFAULT_MAPPING_CACHE = pathlib.Path(
    os.path.expanduser("~/.cache/cascade_vla/bridge_ecot_lerobot_mapping.json")
)

# Camera in LeRobot Bridge that corresponds to the canonical "external" view
# used by most Bridge V2 papers / ECoT annotations.
DEFAULT_LEROBOT_CAMERA = "observation.images.camera_0"


# ---------------------------------------------------------------------------
# Path normalization helpers
# ---------------------------------------------------------------------------


_ECOT_NUMPY_PREFIX_RE = re.compile(r"^.*?/(numpy_256|scripted_numpy_256)/")
_ECOT_SUFFIX_RE = re.compile(r"/(train|val|test)/out\.npy$")
_LEROBOT_TRAJ_RE = re.compile(
    r"^(.*?)/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/raw/traj_group(\d+)/traj(\d+)$"
)


def _ecot_path_to_task_root(file_path: str) -> tuple[str, str] | None:
    """Convert an ECoT ``file_path`` to a (kind, task_root) pair.

    Returns
    -------
    (kind, root) where ``kind`` is "teleop" or "scripted", ``root`` is a path
    suffix of the form ``raw/<workspace>/<task>/<run_id>``. Returns None if the
    path doesn't conform to the expected ECoT layout.
    """
    m = _ECOT_NUMPY_PREFIX_RE.match(file_path)
    if not m:
        return None
    kind = "teleop" if m.group(1) == "numpy_256" else "scripted"
    after = file_path[m.end():]
    after = _ECOT_SUFFIX_RE.sub("", after)
    return kind, f"raw/{after}"


def _lerobot_uuid_to_task_root(uuid: str) -> str | None:
    """Strip the per-trajectory tail from a LeRobot uuid, returning the task root."""
    m = _LEROBOT_TRAJ_RE.match(uuid)
    return m.group(1) if m else None


def _lerobot_uuid_traj_number(uuid: str) -> int | None:
    """Return the trailing ``traj<N>`` number from a LeRobot uuid."""
    m = _LEROBOT_TRAJ_RE.match(uuid)
    return int(m.group(4)) if m else None


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------


def _load_lerobot_episodes_meta(snap: pathlib.Path) -> Any:
    """Load LeRobot episodes metadata. Returns a pandas DataFrame."""
    import pyarrow.parquet as pq

    chunk_dir = snap / "meta" / "episodes"
    parquet_files = sorted(chunk_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet under {chunk_dir}")
    tables = [pq.read_table(p) for p in parquet_files]
    import pyarrow as pa

    return pa.concat_tables(tables).to_pandas()


def build_ecot_to_lerobot_mapping(
    ecot_json_path: pathlib.Path,
    teleop_snap: pathlib.Path = DEFAULT_TELEOP_SNAP,
    scripted_snap: pathlib.Path | None = None,
    cache_path: pathlib.Path = DEFAULT_MAPPING_CACHE,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build (ECoT_key -> LeRobot_episode_descriptor) mapping.

    Parameters
    ----------
    ecot_json_path : path to ``embodied_features_bridge.json``.
    teleop_snap, scripted_snap : LeRobot snapshot dirs (None = skip that kind).
    cache_path : where to save / load the mapping JSON.
    force_rebuild : if False and cache exists, return cached mapping.

    Returns
    -------
    dict with keys::

        {
          "version": 1,
          "n_ecot_total": int,
          "n_matched": int,
          "n_unmatched": int,
          "entries": {
              "<ecot_file_path>::<ecot_episode_id>": {
                  "kind": "teleop" | "scripted",
                  "lerobot_episode_index": int,
                  "n_steps": int,
              },
              ...
          }
        }
    """
    cache_path = pathlib.Path(cache_path).expanduser()
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached ECoT->LeRobot mapping from %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)

    try:
        import ijson  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ijson required: uv pip install ijson") from exc

    # 1. Load LeRobot episode tables and group by task_root.
    by_kind_and_root: dict[str, dict[str, list[dict]]] = {
        "teleop": defaultdict(list),
        "scripted": defaultdict(list),
    }
    for kind, snap in (("teleop", teleop_snap), ("scripted", scripted_snap)):
        if snap is None:
            continue
        if not snap.exists():
            logger.warning("LeRobot %s snapshot not found at %s; skipping", kind, snap)
            continue
        logger.info("Loading LeRobot %s episodes meta from %s", kind, snap)
        df = _load_lerobot_episodes_meta(snap)
        # Avoid `_asdict()` because dotted column names (e.g., adapter.length) become
        # invalid Python identifiers. Use direct DataFrame column indexing.
        task_root_arr = df["uuid"].apply(_lerobot_uuid_to_task_root).tolist()
        traj_n_arr = df["uuid"].apply(_lerobot_uuid_traj_number).tolist()
        # Episode global index — LeRobot v3 stores it as `episode_index`.
        if "episode_index" in df.columns:
            ep_idx_arr = df["episode_index"].tolist()
        else:
            ep_idx_arr = list(range(len(df)))
        # Frame count: prefer top-level `length`; fall back to `adapter.length`.
        if "length" in df.columns:
            length_arr = df["length"].tolist()
        else:
            length_arr = df["adapter.length"].tolist()

        for tr, traj_n, ep_idx, n_steps in zip(
            task_root_arr, traj_n_arr, ep_idx_arr, length_arr, strict=True
        ):
            if tr is None:
                continue
            by_kind_and_root[kind][tr].append({
                "episode_index": int(ep_idx),
                "n_steps": int(n_steps),
                "traj_n": int(traj_n) if traj_n is not None else None,
            })

    for kind, by_root in by_kind_and_root.items():
        # Sort each task_root's candidates by traj_n for deterministic indexing.
        for root, eps in by_root.items():
            eps.sort(key=lambda e: e["traj_n"] if e["traj_n"] is not None else 1_000_000)

    # 2. Stream ECoT JSON, attempt to match each (file_path, episode_id).
    logger.info("Streaming ECoT JSON to build mapping (~1.4 GB) ...")
    entries: dict[str, dict] = {}
    n_total = 0
    n_matched = 0
    with open(ecot_json_path, "rb") as f:
        parser = ijson.kvitems(f, "", use_float=True)
        for file_path, episodes in parser:
            tr_pair = _ecot_path_to_task_root(file_path)
            if tr_pair is None:
                # Unrecognized layout; skip everything in this file.
                n_total += len(episodes) if isinstance(episodes, dict) else 0
                continue
            kind, task_root = tr_pair
            cands = by_kind_and_root[kind].get(task_root, [])
            if not cands:
                n_total += len(episodes) if isinstance(episodes, dict) else 0
                continue

            # For each episode, match by length.
            for ep_id, ep in episodes.items():
                n_total += 1
                meta = ep.get("metadata", {})
                n_steps = meta.get("n_steps")
                if n_steps is None:
                    feats = ep.get("features", {})
                    if isinstance(feats, dict) and "move_primitive" in feats:
                        n_steps = len(feats["move_primitive"])
                if n_steps is None:
                    continue

                # Filter candidates by length.
                length_match = [c for c in cands if c["n_steps"] == n_steps]
                if not length_match:
                    continue

                # Tiebreak: use ECoT episode_id (int) modulo number of length matches
                # to spread assignments deterministically. This is a heuristic — see
                # docstring caveat.
                try:
                    eid_int = int(ep_id)
                except (TypeError, ValueError):
                    eid_int = 0
                chosen = length_match[eid_int % len(length_match)]

                entries[f"{file_path}::{ep_id}"] = {
                    "kind": kind,
                    "lerobot_episode_index": chosen["episode_index"],
                    "n_steps": n_steps,
                }
                n_matched += 1

    result = {
        "version": 1,
        "n_ecot_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_total - n_matched,
        "entries": entries,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f)
    logger.info("Cached %d/%d matches at %s", n_matched, n_total, cache_path)
    return result


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LeRobotBridgeImageLoader(BridgeV2ImageLoader):
    """BridgeV2ImageLoader backed by the jnogga LeRobot bridge_data_v2 datasets.

    Lazily opens the appropriate mp4 video shard per (episode_index) and reads
    the requested frame using ``decord``.
    """

    teleop_snap: pathlib.Path = DEFAULT_TELEOP_SNAP
    scripted_snap: pathlib.Path | None = None
    camera_key: str = DEFAULT_LEROBOT_CAMERA
    mapping: dict[str, Any] | None = None
    image_size: tuple[int, int] = (224, 224)

    def __post_init__(self):
        if self.mapping is None:
            raise ValueError(
                "LeRobotBridgeImageLoader requires a precomputed mapping. "
                "Build with `build_ecot_to_lerobot_mapping(...)` and pass result here."
            )
        self._cache: dict[int, Any] = {}  # episode_index -> decord VideoReader
        # Frame-offset table (episode_index -> first global frame index).
        # We need this to find which mp4 chunk contains the episode and
        # what frame offset to read.
        # For LeRobot v3 the layout is: video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4"
        # We don't yet know how chunks are sized — read info.json once.
        self._info_cache: dict[str, Any] = {}

    def _resolve_snap(self, kind: str) -> pathlib.Path:
        if kind == "teleop":
            return pathlib.Path(self.teleop_snap)
        if kind == "scripted":
            if self.scripted_snap is None:
                raise FileNotFoundError("Scripted snapshot not configured")
            return pathlib.Path(self.scripted_snap)
        raise ValueError(f"Unknown LeRobot dataset kind: {kind}")

    def _get_episode_video(self, kind: str, episode_index: int):
        """Return (decord.VideoReader, local_frame_offset) for the requested episode.

        LeRobot v3 packs MULTIPLE episodes into one mp4 file. We find the right
        file via the ``meta/episodes`` parquet `data/file_index` field.
        """
        cache_key = (kind, episode_index)
        if cache_key in self._cache:
            return self._cache[cache_key]

        snap = self._resolve_snap(kind)
        # Lazy-load episode meta to know which video file + frame offset.
        meta_key = (kind, "episodes_meta")
        if meta_key not in self._info_cache:
            self._info_cache[meta_key] = _load_lerobot_episodes_meta(snap)
        df = self._info_cache[meta_key]
        # Locate the episode row by index. Note: LeRobot stores episodes in
        # row order; episode_index from our mapping is the row index.
        if episode_index >= len(df):
            raise IndexError(f"episode_index {episode_index} >= {len(df)}")
        row = df.iloc[episode_index]

        # The video reference in v3.0 episodes meta is encoded across columns
        # `videos/<camera_key>/chunk_index`, `videos/<camera_key>/file_index`,
        # `videos/<camera_key>/from_timestamp`. We convert from_timestamp to
        # frame offset using fps from info.json.
        prefix = f"videos/{self.camera_key}/"
        chunk_idx = int(row[prefix + "chunk_index"])
        file_idx = int(row[prefix + "file_index"])
        from_ts = float(row[prefix + "from_timestamp"])

        # Lazy-load info.json to get fps.
        if "fps" not in self._info_cache:
            with open(snap / "meta" / "info.json") as f:
                info = json.load(f)
            self._info_cache["fps"] = info.get("fps", 5)
        fps = int(self._info_cache["fps"])
        from_idx = int(round(from_ts * fps))

        video_path = (
            snap / "videos" / self.camera_key / f"chunk-{chunk_idx:03d}" / f"file_{file_idx:03d}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(f"LeRobot mp4 not found: {video_path}")

        try:
            import decord
        except ImportError as exc:
            raise RuntimeError("decord required: uv pip install decord") from exc

        vr = decord.VideoReader(str(video_path))
        # Cache (vr, from_idx). Local frame for this episode = from_idx + step_idx.
        self._cache[cache_key] = (vr, from_idx)
        return self._cache[cache_key]

    def get(self, file_path: str, episode_id: str, step_idx: int) -> np.ndarray:
        key = f"{file_path}::{episode_id}"
        m = self.mapping["entries"].get(key)
        if m is None:
            raise FileNotFoundError(f"No LeRobot mapping for {key}")
        kind = m["kind"]
        ep_idx = int(m["lerobot_episode_index"])
        n_steps = int(m["n_steps"])
        if step_idx < 0 or step_idx >= n_steps:
            raise IndexError(f"step_idx {step_idx} out of range [0, {n_steps})")

        vr, from_idx = self._get_episode_video(kind, ep_idx)
        global_frame = from_idx + step_idx
        frame = vr[global_frame].asnumpy()  # (H, W, 3) uint8 RGB
        # Resize to standard 224x224 if needed.
        if (frame.shape[0], frame.shape[1]) != self.image_size:
            try:
                from PIL import Image as _PILImage  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PIL/Pillow required to resize LeRobot frames"
                ) from exc
            img = _PILImage.fromarray(frame)
            img = img.resize((self.image_size[1], self.image_size[0]), _PILImage.BICUBIC)
            frame = np.asarray(img)
        return frame

    def has(self, file_path: str, episode_id: str, step_idx: int) -> bool:
        return f"{file_path}::{episode_id}" in self.mapping["entries"]
