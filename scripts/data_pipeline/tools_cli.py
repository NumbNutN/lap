"""CLI wrapper around tools.py — for the v3 annotation subagent.

Usage:
  python3 tools_cli.py keyframes <ep_path>
  python3 tools_cli.py pose_delta <ep_path> <idx1> <idx2>
  python3 tools_cli.py image      <ep_path> <frame_idx> <view>   # writes to /tmp/_tool_img.jpg

Prints JSON to stdout. Errors go to stderr with non-zero exit.

Audit logging: every successful invocation appends one JSONL row to
`<ep_path>/.tool_audit.jsonl` with timestamp, cmd, args, and a brief
result summary. This is the ground-truth record we cross-check against
the subagent's self-reported audit file.
"""
from __future__ import annotations
import json, os, sys, time

# Make sibling modules importable
for _p in (
    "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts",
    "/data/zhaoqc/RoboTwin/policy/lap/scripts",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from data_pipeline import tools as T  # noqa: E402


def _summarize(cmd: str, data) -> dict:
    """Compact preview of the tool output for the audit log."""
    if cmd == "keyframes":
        return {"n": len(data) if isinstance(data, list) else None}
    if cmd == "pose_delta":
        if not isinstance(data, dict):
            return {}
        return {
            "n_frames": data.get("n_frames"),
            "delta_robot": data.get("delta_robot"),
            "delta_ee": data.get("delta_ee"),
            "delta_rot_world": data.get("delta_rot_world"),
            "events": data.get("interaction_events_in_range"),
            "gap_to_grasp": (data.get("gap_to_grasp") or {}).get("target_frame"),
            "gap_to_release": (data.get("gap_to_release") or {}).get("target_frame"),
        }
    if cmd == "image":
        return {"bytes": data.get("bytes") if isinstance(data, dict) else None,
                "saved": data.get("saved") if isinstance(data, dict) else None}
    return {}


def _append_audit(ep_path: str, row: dict) -> None:
    try:
        audit_path = os.path.join(ep_path, ".tool_audit.jsonl")
        os.makedirs(ep_path, exist_ok=True)
        with open(audit_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Audit logging is best-effort; never fail the tool call.
        pass


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    ep = sys.argv[2]
    args_extra = sys.argv[3:]
    ok = False
    summary: dict = {}
    err: str | None = None
    try:
        if cmd == "keyframes":
            data = T.get_keyframe_list(ep)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            summary = _summarize(cmd, data); ok = True
        elif cmd == "pose_delta":
            idx1, idx2 = int(args_extra[0]), int(args_extra[1])
            data = T.get_pose_delta(ep, idx1, idx2)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            summary = _summarize(cmd, data); ok = True
        elif cmd == "image":
            frame_idx = int(args_extra[0])
            view = args_extra[1]
            raw = T.get_image(ep, frame_idx, view)
            out = args_extra[2] if len(args_extra) > 2 else "/tmp/_tool_img.jpg"
            with open(out, "wb") as f:
                f.write(raw)
            data = {"saved": out, "bytes": len(raw)}
            print(json.dumps(data, ensure_ascii=False))
            summary = _summarize(cmd, data); ok = True
        else:
            print(f"unknown command: {cmd}", file=sys.stderr)
            sys.exit(2)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(json.dumps({"error": err}), file=sys.stderr)
    finally:
        _append_audit(ep, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cmd": cmd,
            "args": args_extra,
            "ok": ok,
            "summary": summary,
            "error": err,
        })
    if not ok and err is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
