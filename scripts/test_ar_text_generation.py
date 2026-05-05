"""Inspect what LAP's VLM autoregressively generates for a given (image, prompt) pair.

LAP differs from pi05's FAST head in that the VLM directly produces *language*
tokens (the language-action / chain-of-thought trace) — i.e. each generated
token is an actual subword piece, not a quantised action code. This script
loads a checkpoint, picks one HDF5 frame (head + left + (optional right) cam),
runs ``model.sample_tokens`` with the user's prompt, and prints the decoded
text plus a per-token breakdown so you can see what the VLM is "describing".

Usage from ``policy/lap``:

    JAX_PLATFORMS=cuda uv run --group cuda scripts/test_ar_text_generation.py \\
        --config_name lap \\
        --checkpoint_dir checkpoints/<exp>/<step> \\
        --hdf5 ../../data/stack_blocks_two/.../episode0.hdf5 \\
        --frame 0 \\
        --prompt "Use the left arm to pick up the red block..." \\
        --gpu 0

Notes
-----
* ``sample_tokens`` runs only the VLM expert (expert 0); the action expert
  (if present in the config) is never invoked, so this is the *pure* language
  read-out of the model.
* ``state`` is passed to the tokenizer so the discretised state bins land in
  the prompt exactly as during training (LAPConfig.discrete_state_input=True
  by default). For ``Observation.state`` itself we use zeros — that field is
  only consumed by the action expert during action diffusion.
* For bimanual checkpoints (``use_bimanual=True``) we also load the right
  wrist cam; otherwise we stop at base + left wrist to match the model's
  expected image set.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys

# Parse --gpu before importing JAX (JAX reads CUDA_VISIBLE_DEVICES at import).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

# Make `lap` and `openpi` importable when running from the lap project root.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_LAP_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_LAP_ROOT / "src"))
sys.path.insert(0, str(_LAP_ROOT / "third_party" / "openpi" / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from openpi.models import model as _model  # noqa: E402

from lap.models.model_adapter import CoTObservation  # noqa: E402
from lap.models.tokenizer import Gemma3Tokenizer, PaligemmaTokenizer  # noqa: E402
from lap.training import config as _config  # noqa: E402


# RoboTwin HDF5 cam → openpi image key.
_CAM_MAP = {
    "base_0_rgb": "head_camera",
    "left_wrist_0_rgb": "left_camera",
    "right_wrist_0_rgb": "right_camera",
}


def _decode_jpeg_or_raw(arr: np.ndarray):
    """Robotwin HDF5 stores RGB as either JPEG bytes (S<N>) or raw HWC."""
    import cv2

    if isinstance(arr, (bytes, bytearray, np.bytes_)) or arr.dtype.kind == "S":
        raw = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if arr.dtype == np.uint8 and arr.ndim == 1:
        img = cv2.imdecode(np.frombuffer(arr.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(arr)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise RuntimeError(f"Unexpected RGB shape: {img.shape}")
    return img


def load_images_from_hdf5(hdf5_path: str, frame_idx: int, image_keys):
    """Return {image_key: float32 [H, W, 3] in [-1, 1]} for the requested cams."""
    import cv2
    import h5py

    out = {}
    with h5py.File(hdf5_path, "r") as f:
        T = None
        for key in image_keys:
            cam = _CAM_MAP[key]
            ds_path = f"observation/{cam}/rgb"
            if ds_path not in f:
                raise KeyError(f"{ds_path} missing from {hdf5_path}")
            ds = f[ds_path]
            if T is None:
                T = ds.shape[0]
                if not (0 <= frame_idx < T):
                    raise IndexError(f"--frame {frame_idx} out of range [0, {T})")
            img = _decode_jpeg_or_raw(ds[frame_idx])
            img = cv2.resize(img, (224, 224))
            out[key] = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
    return out


def build_state(hdf5_path: str, frame_idx: int, state_from: str, action_dim: int) -> np.ndarray:
    """Construct a state vector of shape (action_dim,).

    The LAP "lap" prompt format discretises this into the textual prompt
    (LAPConfig.discrete_state_input=True), so the value matters for what the
    VLM sees. ``state_from``:

      * ``zeros``   — all-zero (cleanest baseline).
      * ``left``    — endpose/left_endpose[frame] padded/truncated to action_dim.
      * ``right``   — endpose/right_endpose[frame] padded/truncated to action_dim.
      * ``bimanual``— concat(left_endpose, right_endpose) padded/truncated.
    """
    if state_from == "zeros":
        return np.zeros((action_dim,), dtype=np.float32)
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        if state_from == "left":
            v = np.asarray(f["endpose/left_endpose"][frame_idx], dtype=np.float32)
        elif state_from == "right":
            v = np.asarray(f["endpose/right_endpose"][frame_idx], dtype=np.float32)
        elif state_from == "bimanual":
            v = np.concatenate(
                [
                    np.asarray(f["endpose/left_endpose"][frame_idx], dtype=np.float32),
                    np.asarray(f["endpose/right_endpose"][frame_idx], dtype=np.float32),
                ]
            )
        else:
            raise ValueError(f"Unknown --state-from: {state_from}")

    if v.shape[0] < action_dim:
        v = np.concatenate([v, np.zeros(action_dim - v.shape[0], dtype=np.float32)])
    elif v.shape[0] > action_dim:
        v = v[:action_dim]
    return v


def build_observation(images, tokens, mask, state):
    """Assemble a batch-1 ``CoTObservation`` ready for ``model.sample_tokens``."""
    images_b = {k: jnp.asarray(v)[None, ...].astype(jnp.float32) for k, v in images.items()}
    image_masks_b = {k: jnp.ones((1,), dtype=jnp.bool_) for k in images}
    state_b = jnp.asarray(state, dtype=jnp.float32)[None, ...]
    tokens_b = jnp.asarray(tokens, dtype=jnp.int32)[None, ...]
    mask_b = jnp.asarray(mask, dtype=jnp.bool_)[None, ...]
    return CoTObservation(
        images=images_b,
        image_masks=image_masks_b,
        state=state_b,
        tokenized_prompt=tokens_b,
        tokenized_prompt_mask=mask_b,
        token_ar_mask=None,
        token_loss_mask=None,
        tokenized_langact_mask=None,
    )


def build_tokenizer(model_config):
    """Pick the right tokenizer based on the LAP variant."""
    paligemma_variant = getattr(model_config, "paligemma_variant", "")
    common = {
        "max_len": model_config.max_token_len,
        "prompt_format": getattr(model_config, "prompt_format", "lap"),
        "prediction_format": getattr(model_config, "prediction_format", "default"),
        "reasoning_mask_prob": 0.0,
    }
    if "gemma3" in paligemma_variant:
        # Gemma3 needs a manually-downloaded SentencePiece model. Pull from the
        # default config if the user has set it; otherwise raise an actionable error.
        gemma3_path = os.environ.get("GEMMA3_TOKENIZER_PATH")
        if gemma3_path is None:
            raise RuntimeError(
                "Gemma3 variant detected but GEMMA3_TOKENIZER_PATH env var is unset. "
                "Set it to the local path of the gemma3 sentencepiece model."
            )
        return Gemma3Tokenizer(tokenizer_model_path=gemma3_path, **common)
    return PaligemmaTokenizer(**common)


def first_eos_index(tokens: np.ndarray, eos_id: int) -> int | None:
    hits = np.where(tokens == eos_id)[0]
    return int(hits[0]) if len(hits) else None


def main():
    p = argparse.ArgumentParser(
        description="Read an HDF5 frame + prompt, run LAP autoregressive text generation, "
        "and inspect what the VLM produces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config_name", default="lap",
                   help="LAP TrainConfig name registered in lap.training.config (default: lap).")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to checkpoint dir (must contain params/, or be the params dir itself).")
    p.add_argument("--hdf5", required=True, help="Path to a RoboTwin episode HDF5.")
    p.add_argument("--frame", type=int, default=0, help="Frame index in the HDF5 (default 0).")
    p.add_argument("--prompt", required=True, help="Task prompt fed to the VLM.")
    p.add_argument("--max-decoding-steps", dest="max_decoding_steps", type=int, default=256,
                   help="Cap on autoregressive decoding length (default 256).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 → argmax (greedy). >0 → categorical sampling.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling (only used when temperature>0).")
    p.add_argument("--state-from", dest="state_from",
                   choices=["zeros", "left", "right", "bimanual"], default="zeros",
                   help="Source for the discretised state in the prompt (default: zeros).")
    p.add_argument("--show-tokens", dest="show_tokens", type=int, default=80,
                   help="How many decoded tokens to dump per-token (default 80; 0 disables).")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (parsed before JAX import).")
    args = p.parse_args()

    print(f"Loading config '{args.config_name}'...")
    config = _config.get_config(args.config_name)
    # Inference-time tweak: stop_action_to_vlm_grad has no meaning at inference and
    # would only complicate model construction.
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, stop_action_to_vlm_grad=False),
    )

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    params_path = ckpt_dir / "params"
    if not params_path.exists() and ckpt_dir.name != "params":
        if (ckpt_dir / "default").exists() or (ckpt_dir / "_METADATA").exists():
            params_path = ckpt_dir
    print(f"Restoring params from {params_path} ...")
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    print(f"Model: {type(model).__name__}  "
          f"(action_dim={config.model.action_dim}, max_token_len={config.model.max_token_len}, "
          f"image_keys={tuple(config.model.image_keys)})")

    tokenizer = build_tokenizer(config.model)
    image_keys = list(config.model.image_keys)

    print(f"\nLoading frame {args.frame} from {args.hdf5} (cams: {image_keys}) ...")
    images = load_images_from_hdf5(args.hdf5, args.frame, image_keys)
    state_vec = build_state(args.hdf5, args.frame, args.state_from, config.model.action_dim)
    print(f"  state ({args.state_from}, dim={state_vec.shape[0]}): "
          f"{np.array2string(state_vec, precision=3, max_line_width=120)}")

    # Tokenize: ``reasoning=None`` puts us in inference mode (no language-action targets).
    state_for_prompt = state_vec if config.model.discrete_state_input else None
    tok_out = tokenizer.tokenize(
        args.prompt,
        reasoning=None,
        state=state_for_prompt,
    )
    tokens, mask = tok_out[0], tok_out[1]

    # Show what the prompt actually looks like once formatted (state bins included).
    formatted_prompt = tokenizer._prompt_format.format_prompt(  # noqa: SLF001
        args.prompt, state_for_prompt, state_type=None,
    )
    print("\nFormatted prompt fed to the VLM:")
    print(f"  {formatted_prompt!r}")
    n_valid = int(mask.sum())
    print(f"  → tokenized: {n_valid}/{len(tokens)} non-pad tokens")

    obs = build_observation(images, tokens, mask, state_vec)

    print(f"\nGenerating tokens (max_steps={args.max_decoding_steps}, "
          f"temperature={args.temperature}) ...")
    rng = jax.random.PRNGKey(args.seed)
    output_tokens = model.sample_tokens(
        rng, obs,
        max_decoding_steps=args.max_decoding_steps,
        temperature=args.temperature,
    )
    output_np = np.asarray(output_tokens[0]).astype(np.int32)

    eos_idx = first_eos_index(output_np, int(model.EOS_TOKEN))
    valid = output_np[:eos_idx] if eos_idx is not None else output_np
    print(f"  → produced {len(valid)} tokens before EOS"
          f"{'' if eos_idx is not None else ' (no EOS within max_decoding_steps)'}")

    text = tokenizer.decode(valid)
    print("\n" + "=" * 70)
    print("Decoded text from the VLM:")
    print("=" * 70)
    print(text)
    print("=" * 70)

    if args.show_tokens > 0:
        print(f"\nPer-token breakdown (first {min(args.show_tokens, len(valid))}):")
        sp = getattr(tokenizer, "_tokenizer", None)
        vocab = sp.vocab_size() if sp is not None else 257_152
        for i, t in enumerate(valid[: args.show_tokens]):
            tid = int(t)
            if 0 <= tid < vocab and sp is not None:
                piece = sp.id_to_piece(tid)
            else:
                piece = "<oob>"
            print(f"  [{i:3d}] id={tid:>6d}  piece={piece!r}")
        if len(valid) > args.show_tokens:
            print(f"  ... ({len(valid) - args.show_tokens} more tokens hidden)")


if __name__ == "__main__":
    main()
