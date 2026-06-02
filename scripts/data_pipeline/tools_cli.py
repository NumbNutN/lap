"""CLI wrapper around tools.py — for the v3 annotation subagent.

Usage:
  python3 tools_cli.py keyframes <ep_path>
  python3 tools_cli.py pose_delta <ep_path> <idx1> <idx2>
  python3 tools_cli.py image      <ep_path> <frame_idx> <view>   # writes to /tmp/_tool_img.jpg

Prints JSON to stdout. Errors go to stderr with non-zero exit.
"""
from __future__ import annotations
import json, os, sys

# Make sibling modules importable
for _p in (
    "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts",
    "/data/zhaoqc/RoboTwin/policy/lap/scripts",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from data_pipeline import tools as T  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    ep = sys.argv[2]
    try:
        if cmd == "keyframes":
            data = T.get_keyframe_list(ep)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif cmd == "pose_delta":
            idx1, idx2 = int(sys.argv[3]), int(sys.argv[4])
            data = T.get_pose_delta(ep, idx1, idx2)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif cmd == "image":
            frame_idx = int(sys.argv[3])
            view = sys.argv[4]
            raw = T.get_image(ep, frame_idx, view)
            out = sys.argv[5] if len(sys.argv) > 5 else "/tmp/_tool_img.jpg"
            with open(out, "wb") as f:
                f.write(raw)
            print(json.dumps({"saved": out, "bytes": len(raw)},
                             ensure_ascii=False))
        else:
            print(f"unknown command: {cmd}", file=sys.stderr)
            sys.exit(2)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}),
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
