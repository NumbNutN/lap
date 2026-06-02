"""SSAA annotation with per-episode human hints, dual backends.

Reads /home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps/hints.md (markdown with `## <ep_dir_name>` headers),
loads each episode's meta + keyframe images, and calls one of:

  --backend mimo     Xiaomi MiMo (OpenAI-compat)
  --backend sonnet   Anthropic Claude Sonnet (Anthropic API)

Both backends receive:
  - The SSAA prompt as system message
  - The hint (when present) as the FIRST user-message text block,
    explicitly tagged "[HUMAN HINT FOR THIS EPISODE]"
  - Per-keyframe metadata as text blocks
  - External + wrist image for each keyframe, each preceded by a label
    that includes BOTH the kf index AND the frame number, e.g.
       "kf05 frame=89 type=grasp gripper=open  view=external"

Writes per-episode annotations:
  /home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps/<ep_dir>/annotation_<backend>_hinted.json

Usage:
  export MIMO_API_KEY=...
  python3 annotate_with_hints.py --backend mimo --eps ep000,ep001,...

  export ANTHROPIC_API_KEY=...
  python3 annotate_with_hints.py --backend sonnet --eps ep000,...
"""
from __future__ import annotations
import argparse, base64, glob, json, os, re, sys, time
from pathlib import Path

PROMPT_PATH = "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts/annotate_droid/prompt_ssaa.md"
RAW_ROOT = "/home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps"


# ───────────────────────────────────────────────────────────────────────────
# Hint loading
# ───────────────────────────────────────────────────────────────────────────

def parse_hints(md_path: str) -> dict[str, str]:
    """Parse `## <ep_dir_name>` sections out of hints.md."""
    if not os.path.exists(md_path):
        return {}
    txt = open(md_path).read()
    out: dict[str, str] = {}
    cur_key: str | None = None
    cur_lines: list[str] = []
    for line in txt.splitlines():
        m = re.match(r"^##\s+(\S+)", line)
        if m:
            if cur_key is not None:
                out[cur_key] = "\n".join(cur_lines).strip()
            cur_key = m.group(1)
            cur_lines = []
        elif cur_key is not None:
            cur_lines.append(line)
    if cur_key is not None:
        out[cur_key] = "\n".join(cur_lines).strip()
    # Drop placeholder/empty hints
    return {k: v for k, v in out.items()
            if v and not v.startswith("<replace with")}


# ───────────────────────────────────────────────────────────────────────────
# Per-episode loading
# ───────────────────────────────────────────────────────────────────────────

def find_ep_dirs(eps_filter: list[str] | None) -> list[str]:
    all_dirs = sorted(d for d in glob.glob(f"{RAW_ROOT}/ep*") if os.path.isdir(d))
    if not eps_filter:
        return all_dirs
    keep = []
    for d in all_dirs:
        name = os.path.basename(d)
        for f in eps_filter:
            if name.startswith(f):
                keep.append(d); break
    return keep


def load_keyframe_blocks(ep_dir: str) -> tuple[dict, list[dict]]:
    """Return (meta, content_blocks). content_blocks is a flat list of text/image
    items for the user message, with each image labelled by kf+frame."""
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    blocks: list[dict] = []
    blocks.append({
        "type": "text",
        "text": (
            f"Episode: {meta['episode_id']}\n"
            f"Task instruction: {meta['task_instruction']!r}\n"
            f"FPS={meta['fps']}, T={meta['n_frames']}, "
            f"n_keyframes={len(meta['keyframes'])}\n\n"
            "Per-keyframe metadata follows. Each keyframe is then represented "
            "by its external camera image and wrist camera image, each labelled "
            "with its kf index and frame number.\n"
        ),
    })
    # Metadata table (compact)
    md_lines = []
    for kf in meta["keyframes"]:
        md_lines.append(
            f"  [kf{kf['idx']:02d}] frame={kf['frame_idx']:>4} "
            f"type={kf['type']:<8} gripper={kf['gripper_state']:<8} "
            f"ctx={kf.get('interaction_context') or '-'}\n"
            f"    {kf['pose_delta_str']}"
        )
    blocks.append({"type": "text", "text": "\n".join(md_lines)})
    # Append images, kf-by-kf with labels including frame number
    for kf in meta["keyframes"]:
        ext_p = os.path.join(ep_dir, kf["image_file"])
        wrist_p = os.path.join(ep_dir, kf.get("wrist_image_file") or "")
        label = (f"--- kf{kf['idx']:02d}  frame={kf['frame_idx']}  "
                 f"type={kf['type']}  gripper={kf['gripper_state']} ---")
        blocks.append({"type": "text", "text": label})
        if os.path.exists(ext_p):
            blocks.append({"type": "text",
                           "text": f"  view=external  (kf{kf['idx']:02d} frame={kf['frame_idx']})"})
            blocks.append(_image_block(ext_p))
        if wrist_p and os.path.exists(wrist_p):
            blocks.append({"type": "text",
                           "text": f"  view=wrist  (kf{kf['idx']:02d} frame={kf['frame_idx']})"})
            blocks.append(_image_block(wrist_p))
    blocks.append({
        "type": "text",
        "text": ("Produce the SSAA-v2 annotation per the system prompt. "
                 "JSON only, no commentary. The keyframes array MUST be in "
                 "frame_idx order with one entry per keyframe shown above."),
    })
    return meta, blocks


def _image_block(path: str) -> dict:
    raw = open(path, "rb").read()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }


def prepend_hint_block(blocks: list[dict], hint: str | None) -> list[dict]:
    if not hint:
        return blocks
    return [{
        "type": "text",
        "text": f"[HUMAN HINT FOR THIS EPISODE]\n{hint}\n[END HUMAN HINT]\n",
    }] + blocks


# ───────────────────────────────────────────────────────────────────────────
# Backends
# ───────────────────────────────────────────────────────────────────────────

def call_anthropic(system_prompt: str, blocks: list[dict],
                   *, model: str, with_thinking: bool, max_tokens: int = 12000):
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MIMO_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY (or MIMO_API_KEY for the proxied endpoint).")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs = dict(api_key=api_key)
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    call_kwargs = dict(model=model, max_tokens=max_tokens,
                       system=system_prompt,
                       messages=[{"role": "user", "content": blocks}])
    if with_thinking:
        call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
    resp = client.messages.create(**call_kwargs)
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = getattr(resp, "usage", None)
    return text, usage


def call_mimo_openai(system_prompt: str, blocks: list[dict],
                     *, model: str = "mimo-v2.5", max_tokens: int = 24000):
    """MiMo speaks the OpenAI Chat Completions API.

    MiMo is a thinking model — output splits into:
      - `reasoning_content`: chain-of-thought (NOT JSON)
      - `content`: the final JSON answer
    We use only `content` for JSON parsing; reasoning is discarded.
    `max_completion_tokens` must cover BOTH reasoning + output tokens
    — for SSAA on 14-21 kf, reasoning often burns 8-15k tokens before
    MiMo emits the JSON, so default bumped to 24k.
    """
    from openai import OpenAI
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        sys.exit("Set MIMO_API_KEY.")
    base_url = os.environ.get(
        "MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    oai_content = []
    for b in blocks:
        if b["type"] == "text":
            oai_content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            src = b["source"]
            url = f"data:{src['media_type']};base64,{src['data']}"
            oai_content.append({"type": "image_url",
                                "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": oai_content},
        ],
    )
    msg = resp.choices[0].message
    # Use only content for JSON parsing; reasoning_content is CoT prose.
    text = msg.content or ""
    if not text:  # rare: model put everything in reasoning
        text = getattr(msg, "reasoning_content", "") or ""
    usage = resp.usage
    return text, usage


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> str:
    """Tolerate fence wrappers and prose preamble; isolate the JSON object."""
    t = text.strip()
    # Strip ``` fence(s)
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    # Fall back: take the substring from first '{' to last '}' if json.loads
    # fails on the bare strip (handles e.g. "Here is the annotation: {...}")
    try:
        import json as _j
        _j.loads(t)
        return t.strip()
    except Exception:
        i = t.find("{"); j = t.rfind("}")
        if 0 <= i < j:
            return t[i:j+1]
        return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sonnet", "opus", "mimo"], required=True)
    ap.add_argument("--eps", default="",
                    help="comma-separated ep prefixes to process; empty = all")
    ap.add_argument("--hints", default=os.path.join(RAW_ROOT, "hints.md"))
    ap.add_argument("--no-thinking", action="store_true",
                    help="for opus only: disable extended thinking")
    ap.add_argument("--out-suffix", default=None,
                    help="output file suffix (default = backend name)")
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()

    system_prompt = open(PROMPT_PATH).read()
    hints = parse_hints(args.hints)
    print(f"Loaded {len(hints)} hints from {args.hints}")
    for k in hints: print(f"  hint for {k}: {hints[k][:80]!r}")

    eps_filter = [e.strip() for e in args.eps.split(",") if e.strip()]
    ep_dirs = find_ep_dirs(eps_filter)
    print(f"\nProcessing {len(ep_dirs)} episodes\n")

    suffix = args.out_suffix or args.backend
    if args.backend == "opus" and args.no_thinking:
        suffix += "_nothink"
    elif args.backend == "opus":
        suffix += "_think"

    for ep_dir in ep_dirs:
        name = os.path.basename(ep_dir)
        hint = hints.get(name)
        meta, blocks = load_keyframe_blocks(ep_dir)
        if hint:
            blocks = prepend_hint_block(blocks, hint)
        out_path = os.path.join(ep_dir, f"annotation_{suffix}.json")
        print(f"[{args.backend}{'+thinking' if args.backend=='opus' and not args.no_thinking else ''}] "
              f"{name}: {len(meta['keyframes'])} kf, hint={'yes' if hint else 'no'}")
        t0 = time.time()
        try:
            if args.backend == "mimo":
                text, usage = call_mimo_openai(
                    system_prompt, blocks, max_tokens=args.max_tokens)
            else:
                model = ("claude-opus-4-20250514" if args.backend == "opus"
                         else "claude-sonnet-4-20250514")
                text, usage = call_anthropic(
                    system_prompt, blocks,
                    model=model, with_thinking=(args.backend == "opus" and not args.no_thinking),
                    max_tokens=args.max_tokens)
        except Exception as e:
            print(f"   FAILED: {e}")
            continue
        elapsed = time.time() - t0
        cleaned = extract_json(text)
        try:
            parsed = json.loads(cleaned)
            json.dump(parsed, open(out_path, "w"), indent=2, ensure_ascii=False)
            valid = True
        except Exception as e:
            with open(out_path + ".raw.txt", "w") as f:
                f.write(text)
            valid = False
            print(f"   JSON parse FAILED: {e}  raw saved to {out_path}.raw.txt")
        print(f"   elapsed={elapsed:.1f}s  usage={usage}  valid={valid}\n")


if __name__ == "__main__":
    main()
