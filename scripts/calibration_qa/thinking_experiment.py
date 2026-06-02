"""Compare Opus annotation quality with/without extended thinking.

Calls the Anthropic-protocol endpoint of MiMo gateway twice with
identical prompts on ep000 — once with `thinking: enabled`, once
without — and saves both outputs side-by-side for inspection.

Setup:
    export MIMO_API_KEY=...
    pip install anthropic   # or: uv pip install anthropic
    python3 thinking_experiment.py

Output:
    /home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps/ep000_*/annotation_opus_thinking_on.json
    /home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps/ep000_*/annotation_opus_thinking_off.json
"""
import base64, glob, json, os, sys, time

# Lazy-install anthropic if missing
try:
    import anthropic
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "anthropic"])
    import anthropic  # noqa

API_KEY = os.environ.get("MIMO_API_KEY")
if not API_KEY:
    sys.exit("Set MIMO_API_KEY env var first (the mimo gateway forwards to Anthropic).")

# Anthropic-protocol endpoint on MiMo gateway.
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"
MODEL = "claude-opus-4-7"  # adjust if gateway uses different alias

ED_DIR = "/home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps/ep000__AUTOLab_failure_2023-07-07_Fri_Jul__7_09-45-39_2023"
PROMPT_PATH = "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts/annotate_droid/prompt_ssaa.md"


def load_episode(ep_dir):
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    keyframes = meta["keyframes"]
    images = []  # list of (label, base64) for ext + wrist of each kf
    for kf in keyframes:
        ext_p = os.path.join(ep_dir, kf["image_file"])
        wrist_p = os.path.join(ep_dir, kf.get("wrist_image_file") or "")
        if os.path.exists(ext_p):
            images.append((f"kf{kf['idx']:02d} ext", open(ext_p, "rb").read()))
        if wrist_p and os.path.exists(wrist_p):
            images.append((f"kf{kf['idx']:02d} wrist", open(wrist_p, "rb").read()))
    return meta, images


def build_messages(meta, images, system_prompt):
    """Build a multimodal messages list. Images as base64 image blocks."""
    blocks = []
    blocks.append({
        "type": "text",
        "text": (
            f"Task: {meta['task_instruction']!r}\n"
            f"Episode: {meta['episode_id']}\n"
            f"FPS={meta['fps']}, T={meta['n_frames']}, "
            f"n_keyframes={len(meta['keyframes'])}\n\n"
            "Keyframe metadata (frame_idx + type + gripper + pose_delta_str):\n"
        ),
    })
    for kf in meta["keyframes"]:
        blocks.append({
            "type": "text",
            "text": (
                f"[kf{kf['idx']:02d}] frame={kf['frame_idx']} "
                f"type={kf['type']} gripper={kf['gripper_state']} "
                f"ctx={kf.get('interaction_context') or '-'}\n"
                f"  {kf['pose_delta_str']}\n"
            ),
        })
    # Append images, labelled
    for label, raw in images:
        blocks.append({"type": "text", "text": label})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(raw).decode("ascii"),
            },
        })
    blocks.append({"type": "text", "text":
        "Produce the SSAA-v2 annotation per the system prompt. JSON only."})
    return [{"role": "user", "content": blocks}]


def call(client, messages, system, with_thinking, max_tokens=12000, budget=8000):
    kwargs = dict(model=MODEL, max_tokens=max_tokens,
                  system=system, messages=messages)
    if with_thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return client.messages.create(**kwargs)


def main():
    meta, images = load_episode(ED_DIR)
    system_prompt = open(PROMPT_PATH).read()
    messages = build_messages(meta, images, system_prompt)
    client = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)

    print(f"Loaded ep000: {len(meta['keyframes'])} keyframes, "
          f"{len(images)} images, prompt {len(system_prompt)} chars")
    print(f"Model: {MODEL}  base_url: {BASE_URL}\n")

    for label, with_thinking in [("thinking_off", False), ("thinking_on", True)]:
        print(f"=== Calling Opus  thinking={with_thinking} ===")
        t0 = time.time()
        try:
            resp = call(client, messages, system_prompt, with_thinking)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        elapsed = time.time() - t0
        # Find the text block (ignore thinking blocks)
        text = ""
        n_think_tokens = 0
        for blk in resp.content:
            if blk.type == "text":
                text += blk.text
            elif blk.type == "thinking":
                n_think_tokens += len(blk.thinking)
        out_path = os.path.join(ED_DIR, f"annotation_opus_{label}.json")
        # Try to extract just the JSON (in case model wraps in fence)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        try:
            parsed = json.loads(cleaned)
            json.dump(parsed, open(out_path, "w"), indent=2, ensure_ascii=False)
            valid = True
        except Exception as e:
            with open(out_path + ".raw.txt", "w") as f:
                f.write(text)
            valid = False
            print(f"  JSON parse failed: {e}")
        usage = getattr(resp, "usage", None)
        print(f"  elapsed: {elapsed:.1f}s")
        print(f"  thinking chars: {n_think_tokens}")
        print(f"  usage: {usage}")
        print(f"  valid_json: {valid}  → {out_path}")
        print()


if __name__ == "__main__":
    main()
