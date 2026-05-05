"""
Viewer for the two Embodied-CoT datasets:

  1. embodied_features_bridge   — single 1.4 GB JSON of per-frame ECoT labels
                                  (no demos; refers to original Bridge V2 npy files).
  2. embodied_features_and_demos_libero
                                — TFDS / TFRecord shards bundling demos + ECoT labels.

Usage:
    cd /home/numbnut/worksapce/RoboTwin
    source .venv/bin/activate

    # Bridge: list the first 2 file_paths, first 2 episodes each, first 5 steps each
    uv run python policy/lap/scripts/view_ecot_datasets.py bridge \
        --num-files 2 --num-episodes 2 --num-steps 5

    # Libero: open shard 0, dump first 2 episodes, first 5 steps each
    uv run python policy/lap/scripts/view_ecot_datasets.py libero \
        --shard 0 --num-episodes 2 --num-steps 5

Optional deps:
    bridge streaming : uv pip install ijson    (otherwise the script falls back
                                                to json.load which needs ~10 GB RAM)
    libero parsing   : uv pip install tfrecord (TF-free TFRecord reader; avoids
                                                pulling in tensorflow + new numpy)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

BRIDGE_ROOT = Path(
    "/home/numbnut/.cache/huggingface/hub/"
    "datasets--Embodied-CoT--embodied_features_bridge/snapshots"
)
LIBERO_ROOT = Path(
    "/home/numbnut/.cache/huggingface/hub/"
    "datasets--Embodied-CoT--embodied_features_and_demos_libero/snapshots"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_snapshot(root: Path) -> Path:
    """Return the single snapshot subdirectory under a HF cache root."""
    if not root.exists():
        sys.exit(f"[error] dataset cache not found: {root}")
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    if not snaps:
        sys.exit(f"[error] no snapshot under {root}")
    return snaps[0]


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _kv(k: str, v, indent: int = 2) -> None:
    pad = " " * indent
    text = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    if len(text) > 200:
        text = text[:200] + f"... <truncated, total len={len(text)}>"
    print(f"{pad}{k}: {text}")


# ---------------------------------------------------------------------------
# bridge viewer
# ---------------------------------------------------------------------------

def _bridge_path() -> Path:
    snap = _resolve_snapshot(BRIDGE_ROOT)
    p = snap / "embodied_features_bridge.json"
    if not p.exists():
        sys.exit(f"[error] bridge json not found at {p}")
    return p


def _stream_bridge_with_ijson(path: Path, num_files: int):
    """Yield (file_path, episodes_dict) using ijson — constant memory."""
    import ijson  # type: ignore

    with open(path, "rb") as f:
        # Top level is a dict whose keys are file paths.
        # use_float=True so we get floats instead of Decimal (json-serializable).
        parser = ijson.kvitems(f, "", use_float=True)
        for i, (file_path, episodes) in enumerate(parser):
            if i >= num_files:
                break
            yield file_path, episodes


def _stream_bridge_fallback(path: Path, num_files: int):
    """Fallback: json.load the whole 1.4 GB file (high RAM)."""
    print("[warn] ijson not installed — loading full 1.4 GB JSON into RAM.")
    print("       Install with: pip install ijson  (recommended)")
    with open(path, "r") as f:
        data = json.load(f)
    for i, (file_path, episodes) in enumerate(data.items()):
        if i >= num_files:
            break
        yield file_path, episodes


def view_bridge(num_files: int, num_episodes: int, num_steps: int) -> None:
    path = _bridge_path()
    size_mb = path.stat().st_size / 1024 / 1024
    _hr(f"Embodied-CoT  bridge  ({path})")
    print(f"  size: {size_mb:.1f} MB")
    print("  layout: { <original_npy_file_path>: { <episode_id>: {features, metadata, reasoning} } }")

    try:
        import ijson  # noqa: F401
        stream = _stream_bridge_with_ijson(path, num_files)
    except ImportError:
        stream = _stream_bridge_fallback(path, num_files)

    for fi, (file_path, episodes) in enumerate(stream):
        _hr(f"[file {fi}] {file_path}")
        ep_ids = list(episodes.keys())
        print(f"  num_episodes_in_file: {len(ep_ids)}    sample_ids: {ep_ids[:5]}")

        for ej, ep_id in enumerate(ep_ids[:num_episodes]):
            ep = episodes[ep_id]
            _hr(f"  [file {fi} / episode {ep_id}]")

            meta = ep.get("metadata", {})
            print("  >>> metadata")
            for k, v in meta.items():
                _kv(k, v, indent=6)

            feats = ep.get("features", {})
            n_steps = len(feats.get("move_primitive", []))
            print(f"\n  >>> features  (n_steps={n_steps})")
            print("      keys: " + ", ".join(feats.keys()))
            print("      sample (first {} steps):".format(min(num_steps, n_steps)))
            for s in range(min(num_steps, n_steps)):
                mp = feats.get("move_primitive", [None] * n_steps)[s]
                gp = feats.get("gripper_position", [None] * n_steps)[s]
                bb = feats.get("bboxes", [None] * n_steps)[s]
                bb_text = json.dumps(bb, ensure_ascii=False)
                if len(bb_text) > 140:
                    bb_text = bb_text[:140] + "..."
                print(f"        step {s}:  move='{mp}'  gripper={gp}  bboxes={bb_text}")

            reasoning = ep.get("reasoning", {})
            r_keys = sorted(reasoning.keys(), key=lambda x: int(x))
            print(f"\n  >>> reasoning  (n_steps={len(r_keys)})")
            for s in r_keys[: min(num_steps, len(r_keys))]:
                r = reasoning[s]
                print(f"      step {s}:")
                for k in ("task", "plan", "subtask", "subtask_reason", "move", "move_reason"):
                    if k in r:
                        _kv(k, r[k], indent=10)


# ---------------------------------------------------------------------------
# libero viewer
# ---------------------------------------------------------------------------

def _libero_root() -> Path:
    snap = _resolve_snapshot(LIBERO_ROOT)
    inner = snap / "libero_lm_90_openpi" / "1.0.1"
    if not inner.exists():
        sys.exit(f"[error] libero data not found at {inner}")
    return inner


def view_libero(shard: int, num_episodes: int, num_steps: int) -> None:
    root = _libero_root()
    info = json.loads((root / "dataset_info.json").read_text())
    feats = json.loads((root / "features.json").read_text())

    _hr(f"Embodied-CoT  libero  ({root})")
    n_shards = len(info["splits"][0]["shardLengths"])
    n_eps = sum(int(x) for x in info["splits"][0]["shardLengths"])
    print(f"  module        : {info['moduleName']}")
    print(f"  version       : {info['version']}")
    print(f"  format        : {info['fileFormat']}  ({n_shards} shards)")
    print(f"  total_episodes: {n_eps}")
    print(f"  shard {shard} length: {info['splits'][0]['shardLengths'][shard]} episodes")

    print("\n  >>> per-step features (from features.json)")
    step_feats = feats["featuresDict"]["features"]["steps"]["sequence"]["feature"][
        "featuresDict"
    ]["features"]
    for k, v in step_feats.items():
        if k == "observation":
            obs = v["featuresDict"]["features"]
            for ok, ov in obs.items():
                shape = ov.get("tensor", ov.get("image", {})).get("shape", {}).get("dimensions", "?")
                dtype = ov.get("tensor", ov.get("image", {})).get("dtype", "?")
                print(f"      observation.{ok:<14} shape={shape}  dtype={dtype}")
        else:
            t = v.get("tensor", {})
            shape = t.get("shape", {}).get("dimensions", "scalar/text")
            dtype = t.get("dtype", v.get("text", {}) and "string")
            print(f"      {k:<28} shape={shape}  dtype={dtype}")

    try:
        from tfrecord.reader import tfrecord_loader  # type: ignore
    except ImportError:
        print("\n[warn] `tfrecord` package not installed — cannot decode TFRecord shards.")
        print("       Install with: uv pip install tfrecord")
        print("       (schema printed above is sufficient to understand the layout)")
        return

    shard_path = root / f"libero_lm_90_openpi-train.tfrecord-{shard:05d}-of-00128"
    if not shard_path.exists():
        sys.exit(f"[error] shard not found: {shard_path}")
    _hr(f"  decoding shard: {shard_path.name}")

    # tfrecord_loader yields one dict per Example; here, one Example == one episode.
    # Per-step fields are flat numpy arrays:
    #   text/bytes  → ndarray of dtype |S<N>, shape (n_steps,)
    #   int/float   → ndarray of dtype int64/float32, shape (n_steps * dim,)
    iterator = tfrecord_loader(str(shard_path), None, description=None)

    # Per-step vector dims (from features.json shape).
    vec_dims = {
        "steps/action": 7,
        "steps/observation/joint_state": 7,
        "steps/observation/state": 8,
    }
    text_keys = [
        "steps/language_instruction",
        "steps/language_motions",
        "steps/language_motions_future",
    ]
    scalar_keys = [
        "steps/is_first",
        "steps/is_last",
        "steps/is_terminal",
        "steps/reward",
        "steps/discount",
    ]

    for ep_idx, ex in enumerate(iterator):
        if ep_idx >= num_episodes:
            break

        _hr(f"  [shard {shard} / episode {ep_idx}]")
        meta_keys = sorted(k for k in ex if k.startswith("episode_metadata/"))
        step_keys = sorted(k for k in ex if k.startswith("steps/"))
        print(f"  episode_metadata keys: {meta_keys}")
        print(f"  steps/* keys         : {len(step_keys)} keys")
        for mk in meta_keys:
            v = ex[mk]
            # tfrecord returns single-entry bytes as raw `bytes`, multi-entry as ndarray.
            if isinstance(v, bytes):
                _kv(mk, v.decode("utf-8", errors="replace"), indent=6)
            elif hasattr(v, "dtype") and v.dtype.kind == "S":
                _kv(mk, v[0].decode("utf-8", errors="replace"), indent=6)
            else:
                _kv(mk, list(v.tolist()) if hasattr(v, "tolist") else v, indent=6)

        # n_steps = length of any per-step text array.
        n_total = None
        for k in text_keys:
            if k in ex and hasattr(ex[k], "shape"):
                n_total = int(ex[k].shape[0])
                break
        print(f"\n  >>> per-step preview  (episode_n_steps={n_total})")
        for s in range(min(num_steps, n_total or 0)):
            print(f"      step {s}:")
            for k in text_keys:
                if k in ex:
                    raw = ex[k][s]
                    if isinstance(raw, (bytes, bytearray)):
                        text = raw.decode("utf-8", errors="replace")
                    else:
                        text = str(raw)
                    if len(text) > 160:
                        text = text[:160] + "..."
                    _kv(k.split("/", 1)[1], text, indent=10)
            for k in scalar_keys:
                if k in ex:
                    v = ex[k]
                    val = v[s].item() if s < len(v) else None
                    _kv(k.split("/", 1)[1], val, indent=10)

        # Per-step vectors: action / joint_state / state.
        for k, dim in vec_dims.items():
            if k in ex:
                flat = ex[k]
                n_steps_inferred = flat.shape[0] // dim if dim else 0
                print(f"\n      {k}: total_floats={flat.shape[0]} -> ~{n_steps_inferred} steps × dim={dim}")
                print(f"        step 0 vector: {flat[:dim].tolist()}")

        # Image fields: report shape + first JPEG byte length only (don't dump bytes).
        for k in ("steps/observation/image", "steps/observation/wrist_image"):
            if k in ex:
                arr = ex[k]
                first_len = len(arr[0]) if hasattr(arr[0], "__len__") else "?"
                print(f"      {k}: n_steps={arr.shape[0]}  first_jpeg_bytes={first_len}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bridge", help="view embodied_features_bridge.json")
    pb.add_argument("--num-files", type=int, default=2)
    pb.add_argument("--num-episodes", type=int, default=2)
    pb.add_argument("--num-steps", type=int, default=5)

    pl = sub.add_parser("libero", help="view embodied_features_and_demos_libero TFRecords")
    pl.add_argument("--shard", type=int, default=0)
    pl.add_argument("--num-episodes", type=int, default=2)
    pl.add_argument("--num-steps", type=int, default=5)

    args = p.parse_args()
    if args.cmd == "bridge":
        view_bridge(args.num_files, args.num_episodes, args.num_steps)
    else:
        view_libero(args.shard, args.num_episodes, args.num_steps)


if __name__ == "__main__":
    main()
