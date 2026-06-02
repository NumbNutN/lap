"""SSAA v3 — tool-augmented annotation runner.

Differences from `annotate_with_hints.py` (v2):
  - System prompt = prompt_ssaa_v3.md (tool-use paradigm).
  - VLM is given the keyframe images + minimal text context, but NO
    pre-computed pose deltas.
  - VLM calls `get_pose_delta(idx1, idx2)` / `get_image(frame_idx, view)` /
    `get_keyframe_list()` to fetch motion data on demand.
  - Tools dispatched against `data_pipeline/tools.py`; ep_path is bound
    by the runner so the model never sees it.

Only Anthropic backends (sonnet, opus) supported in v3 — MiMo tool-use
behavior unverified.

Usage:
  export ANTHROPIC_API_KEY=...
  python3 annotate_with_tools.py --backend sonnet --eps ep000,ep007
  python3 annotate_with_tools.py --backend opus --no-thinking --eps ep007
"""
from __future__ import annotations
import argparse, base64, glob, json, os, re, sys, time
from pathlib import Path

# Make sibling annotate_droid / data_pipeline modules importable
for _p in (
    "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts",
    "/data/zhaoqc/RoboTwin/policy/lap/scripts",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from data_pipeline import tools as droid_tools  # noqa: E402

PROMPT_PATH = ("/home/numbnut/worksapce/RoboTwin/policy/lap/scripts/"
               "annotate_droid/prompt_ssaa_v3.md")
RAW_ROOT = "/home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps"

MAX_TOOL_ITERS = 32   # hard cap on tool-loop iterations per episode

# ───────────────────────────────────────────────────────────────────────────
# Tool schemas (Anthropic format). ep_path is bound by the runner — hidden
# from the model.
# ───────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "get_keyframe_list",
        "description": (
            "Return the structured list of all rule-detector keyframes "
            "for this episode (kf_idx, frame_idx, type, gripper_state, "
            "near_interaction, interaction_context). Read-only reference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_pose_delta",
        "description": (
            "Return pose delta from frame idx1 to idx2 (idx2 > idx1) "
            "in both robot base frame and idx1-wrist camera frame. "
            "Also returns rotation decompositions, n_frames, any "
            "interaction events inside [idx1, idx2], and gaps to the "
            "first future grasp/release after idx2. "
            "Use this to fetch motion data over the phase [frame_idx, "
            "chunk_end_frame] you have decided semantically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "idx1": {"type": "integer",
                         "description": "Start frame index."},
                "idx2": {"type": "integer",
                         "description": "End frame index (must be > idx1)."},
            },
            "required": ["idx1", "idx2"],
        },
    },
    {
        "name": "get_image",
        "description": (
            "Return the JPEG image of `view` ('ext' or 'wrist') at the "
            "given frame_idx (any frame in the trajectory, need not be a "
            "keyframe). Use sparingly to inspect mid-segment evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_idx": {"type": "integer"},
                "view": {"type": "string", "enum": ["ext", "wrist"]},
            },
            "required": ["frame_idx", "view"],
        },
    },
]


def dispatch_tool(name: str, args: dict, ep_path: str) -> dict:
    """Execute a tool call. Returns a dict with EITHER 'text' or 'image_b64'.
    Image returns are wrapped so the caller knows to build an image block."""
    try:
        if name == "get_keyframe_list":
            data = droid_tools.get_keyframe_list(ep_path)
            return {"text": json.dumps(data, ensure_ascii=False)}
        if name == "get_pose_delta":
            data = droid_tools.get_pose_delta(
                ep_path, int(args["idx1"]), int(args["idx2"]))
            return {"text": json.dumps(data, ensure_ascii=False)}
        if name == "get_image":
            raw = droid_tools.get_image(
                ep_path, int(args["frame_idx"]), args["view"])
            return {"image_b64": base64.b64encode(raw).decode("ascii")}
        return {"text": json.dumps({"error": f"unknown tool {name}"})}
    except Exception as e:
        return {"text": json.dumps({"error": f"{type(e).__name__}: {e}"})}


# ───────────────────────────────────────────────────────────────────────────
# Initial-context blocks
# ───────────────────────────────────────────────────────────────────────────

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


def load_initial_blocks(ep_dir: str) -> tuple[dict, list[dict]]:
    """Initial user message: instruction, kf table, kf images. No deltas."""
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    blocks: list[dict] = [{
        "type": "text",
        "text": (
            f"Episode: {meta['episode_id']}\n"
            f"Task instruction: {meta['task_instruction']!r}\n"
            f"FPS={meta['fps']}, T={meta['n_frames']}, "
            f"n_keyframes={len(meta['keyframes'])}\n\n"
            "Below is the keyframe table (no motion data — call "
            "`get_pose_delta(idx1, idx2)` for the [frame_idx, "
            "chunk_end_frame] span you decide for each keyframe). "
            "Then per-keyframe images follow.\n"
        ),
    }]
    md_lines = []
    for kf in meta["keyframes"]:
        md_lines.append(
            f"  [kf{kf['idx']:02d}] frame={kf['frame_idx']:>4} "
            f"type={kf['type']:<8} gripper={kf['gripper_state']:<8} "
            f"near_interaction={kf['near_interaction']} "
            f"ctx={kf.get('interaction_context') or '-'}"
        )
    blocks.append({"type": "text", "text": "\n".join(md_lines)})

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
        "text": ("Produce the SSAA-v3 annotation per the system prompt. "
                 "Call the tools as needed. End with JSON only — no "
                 "commentary outside the JSON object."),
    })
    return meta, blocks


def prepend_hint_block(blocks: list[dict], hint: str | None) -> list[dict]:
    if not hint:
        return blocks
    return [{
        "type": "text",
        "text": f"[HUMAN HINT FOR THIS EPISODE]\n{hint}\n[END HUMAN HINT]\n",
    }] + blocks


# ───────────────────────────────────────────────────────────────────────────
# Hint parsing (reuse v2 format)
# ───────────────────────────────────────────────────────────────────────────

def parse_hints(md_path: str) -> dict[str, str]:
    if not os.path.exists(md_path):
        return {}
    txt = open(md_path).read()
    out: dict[str, str] = {}
    cur_key, cur_lines = None, []
    for line in txt.splitlines():
        m = re.match(r"^##\s+(\S+)", line)
        if m:
            if cur_key is not None:
                out[cur_key] = "\n".join(cur_lines).strip()
            cur_key, cur_lines = m.group(1), []
        elif cur_key is not None:
            cur_lines.append(line)
    if cur_key is not None:
        out[cur_key] = "\n".join(cur_lines).strip()
    return {k: v for k, v in out.items()
            if v and not v.startswith("<replace with")}


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


# ───────────────────────────────────────────────────────────────────────────
# Anthropic tool-use loop
# ───────────────────────────────────────────────────────────────────────────

def run_episode_anthropic(
    *, system_prompt: str, initial_blocks: list[dict], ep_path: str,
    model: str, with_thinking: bool, max_tokens: int,
    tool_log: list[dict] | None = None,
) -> tuple[str, dict, list[dict]]:
    """Run the tool-use loop. Returns (final_text, usage_summary, messages)."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY.")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs = dict(api_key=api_key)
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)

    messages = [{"role": "user", "content": initial_blocks}]
    usage_in = usage_out = 0
    last_text = ""

    for it in range(MAX_TOOL_ITERS):
        call_kwargs = dict(
            model=model, max_tokens=max_tokens,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        if with_thinking:
            call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
        resp = client.messages.create(**call_kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            usage_in += getattr(usage, "input_tokens", 0) or 0
            usage_out += getattr(usage, "output_tokens", 0) or 0

        # Append the assistant turn (raw content, including any tool_use blocks)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            last_text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text")
            return last_text, {"in": usage_in, "out": usage_out, "iters": it + 1}, messages

        # Execute all tool calls in this assistant turn, build tool_result blocks
        tool_result_blocks: list[dict] = []
        for blk in resp.content:
            if getattr(blk, "type", None) != "tool_use":
                continue
            name = blk.name
            args = blk.input or {}
            result = dispatch_tool(name, args, ep_path)
            if tool_log is not None:
                tool_log.append({
                    "iter": it, "name": name, "args": args,
                    "kind": "image" if "image_b64" in result else "text",
                    "preview": ((result.get("text") or "")[:200])
                               if "text" in result else "<jpeg>",
                })
            if "image_b64" in result:
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": blk.id,
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": result["image_b64"],
                        },
                    }],
                })
            else:
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": blk.id,
                    "content": result["text"],
                })
        messages.append({"role": "user", "content": tool_result_blocks})

    return last_text, {"in": usage_in, "out": usage_out,
                       "iters": MAX_TOOL_ITERS, "hit_cap": True}, messages


# ───────────────────────────────────────────────────────────────────────────
# JSON extraction (reused from v2)
# ───────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    try:
        import json as _j
        _j.loads(t)
        return t.strip()
    except Exception:
        i = t.find("{"); j = t.rfind("}")
        if 0 <= i < j:
            return t[i:j+1]
        return t.strip()


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sonnet", "opus"], required=True)
    ap.add_argument("--eps", default="",
                    help="comma-separated ep prefixes; empty = all")
    ap.add_argument("--hints", default=os.path.join(RAW_ROOT, "hints.md"))
    ap.add_argument("--no-thinking", action="store_true",
                    help="for opus only: disable extended thinking")
    ap.add_argument("--out-suffix", default=None,
                    help="output file suffix (default = backend name + _v3)")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--save-tool-log", action="store_true",
                    help="dump per-iteration tool calls beside the annotation")
    args = ap.parse_args()

    system_prompt = open(PROMPT_PATH).read()
    hints = parse_hints(args.hints)
    print(f"Loaded {len(hints)} hints from {args.hints}")
    for k in hints:
        print(f"  hint for {k}: {hints[k][:80]!r}")

    eps_filter = [e.strip() for e in args.eps.split(",") if e.strip()]
    ep_dirs = find_ep_dirs(eps_filter)
    print(f"\nProcessing {len(ep_dirs)} episodes\n")

    suffix = args.out_suffix or f"{args.backend}_v3"
    if args.backend == "opus" and args.no_thinking:
        suffix += "_nothink"
    elif args.backend == "opus":
        suffix += "_think"

    for ep_dir in ep_dirs:
        name = os.path.basename(ep_dir)
        hint = hints.get(name)
        meta, blocks = load_initial_blocks(ep_dir)
        if hint:
            blocks = prepend_hint_block(blocks, hint)

        out_path = os.path.join(ep_dir, f"annotation_{suffix}.json")
        tool_log: list[dict] = []
        print(f"[{args.backend}{'+thinking' if args.backend=='opus' and not args.no_thinking else ''}] "
              f"{name}: {len(meta['keyframes'])} kf, hint={'yes' if hint else 'no'}")
        t0 = time.time()
        try:
            model = ("claude-opus-4-20250514" if args.backend == "opus"
                     else "claude-sonnet-4-20250514")
            text, usage, _msgs = run_episode_anthropic(
                system_prompt=system_prompt,
                initial_blocks=blocks,
                ep_path=ep_dir,
                model=model,
                with_thinking=(args.backend == "opus" and not args.no_thinking),
                max_tokens=args.max_tokens,
                tool_log=tool_log if args.save_tool_log else None,
            )
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")
            continue
        elapsed = time.time() - t0
        cleaned = extract_json(text)
        valid = False
        try:
            parsed = json.loads(cleaned)
            json.dump(parsed, open(out_path, "w"), indent=2, ensure_ascii=False)
            valid = True
        except Exception as e:
            with open(out_path + ".raw.txt", "w") as f:
                f.write(text)
            print(f"   JSON parse FAILED: {e}  raw saved to {out_path}.raw.txt")
        if args.save_tool_log and tool_log:
            with open(out_path + ".tools.json", "w") as f:
                json.dump(tool_log, f, indent=2, ensure_ascii=False)
        print(f"   elapsed={elapsed:.1f}s  usage={usage}  valid={valid}  "
              f"tool_calls={len(tool_log) if args.save_tool_log else '(off)'}\n")


if __name__ == "__main__":
    main()
