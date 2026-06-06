"""SSAA-v3 annotation viewer (read-only).

Loads per-episode v3 JSONs and visualizes them through anchor-centric
panels + a phase-ribbon timeline:

  • Anchor: the latest keyframe whose frame_idx ≤ slider. Right panel
    shows that anchor's S / A / A_correct / S_pred (agent-side prediction) and
    the demo's A / imitation_supervised flag (reference).
  • Timeline (matplotlib via gr.Plot): full demo bar; per-kf action
    chunks for imitation_supervised=true; recovery branch for false.
  • Slider scrub re-resolves anchor and redraws cameras.

Layout:
  images_dir/
  ├── ep00/
  │   ├── meta.json
  │   ├── kf00_f0000.jpg + _wrist.jpg
  │   ├── annotation_<suffix>.json
  │   └── annotation_<suffix>.json.audit.json (optional)
  └── ...

Usage:
  python3 scripts/view_droid_v3.py \\
      --images-dir /home/.../local_data/raw_eps \\
      --suffix subagent_v3 \\
      --port 7862
"""
from __future__ import annotations
import argparse, glob, json, logging, os, sys
from dataclasses import dataclass, field
from pathlib import Path

# Pre-import patch: gradio 4.x sometimes crashes on bool-shaped JSON schemas.
# Patch the helper before import.
try:
    import gradio_client.utils as _gcu
    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "any"
        try:
            return _orig_get_type(schema)
        except Exception:
            return "any"
    _gcu.get_type = _safe_get_type
    _orig_j2p = _gcu._json_schema_to_python_type
    def _safe_j2p(schema, defs=None):
        if isinstance(schema, bool):
            return "any"
        try:
            return _orig_j2p(schema, defs)
        except Exception:
            return "any"
    _gcu._json_schema_to_python_type = _safe_j2p
except Exception:
    pass

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("view_droid_v3")


_INTRO_MD = """\
### Concepts

**keyframe** — a moment auto-detected by rules over the trajectory
(`begin` / `motion` / `grasp` / `release` / `retry` / `filler` / `end`).
Keyframes sample the demo; they are NOT phase boundaries.

**phase** (aka **chunk**) — a *semantic* interval `[frame_idx, chunk_end_frame]`
chosen by the annotator. One phase = one sub-intent. Phases may span
multiple keyframes (a single approach can cover 4-5 kfs) and may
overlap (the ribbon stacks them in lanes). Each phase carries one
`phase_type` tag (open-vocabulary, e.g. `approach`, `fine_align`,
`pick`, `transport`, `pour_hold`, `failure`, `recovery`).

**anchor** — the current keyframe whose phase the slider is sitting in.
Picked as the latest kf with `frame_idx ≤ slider`. The right-hand panel
always describes the anchor.

### Fields trained as supervised targets

| field | role at train time |
|-------|-------------------|
| `S` | scene observation prefix; the deployed model writes this from camera input |
| `S_pred` | **world-model CE target** — forecast the key state change at `chunk_end_frame` |
| `A` | the action over the span (cm/° from pose-delta). **Always present**: world-model input, and the policy/BC target when supervised |
| `A_correct` | a corrective action — present **only** when it overrides the demo (`imitation_supervised=false`); the policy CE target there |
| `phase_type` | structural tag for analysis & ribbon coloring |
| `imitation_supervised` | `true` → policy follows the demo (`A` is target); `false` → policy overrides it (`A_correct` is target). `A` still feeds the world model either way |
| `chunk_end_frame` | structural boundary defining the phase |

**Thinking is inline `<think>…</think>`** (the 🧠 callouts), prefixing the
policy-target field — `A` when supervised, `A_correct` when not — wherever
the step warranted deliberation. The 🧠/⚡ chip flags whether a keyframe
contains a `<think>` block. `A` is on every keyframe (a failure is still
valid world-model data); only the *policy target* flips at a failure.

The deployed model is conditioned on observations alone — it never sees
frame indices, keyframes, or any hint that a demo exists. The companion
`audit.json` is for human review only and may use any vocabulary.

### Ribbon legend

- **demo + A_correct** (top tier): chunks where `imitation_supervised=true`.
  Lanes stack upward when chunks overlap.
- **recovery** (bottom tier): chunks where `imitation_supervised=false`
  (pre-failure intervention or post-failure recovery). Branches downward
  from the first divergence with a dotted red elbow.
- **▼** red triangle marks the current anchor.
- Vertical red line marks the slider position.
- Faint top bar = full episode duration.
"""


# ───────────────────────────────────────────────────────────────────────────
# Data model
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class V3Keyframe:
    kf_idx: int
    frame_idx: int
    chunk_end_frame: int
    phase_type: str
    imitation_supervised: bool
    S: str
    S_pred: str
    A: str
    A_correct: str       # "" unless imitation_supervised=false (override action)
    mode_marker: str = ""  # legacy/optional; think is now inline <think> in A/A_correct
    # Meta-side (from meta.json):
    kf_type: str = ""
    gripper_state: str = ""
    ext_img_path: str = ""
    wrist_img_path: str = ""
    # Optional motion summary (filled when --pose-delta is on)
    pose_delta: dict | None = None


@dataclass
class V3Episode:
    short_id: str
    episode_id: str
    ep_dir: str
    task_instruction: str
    description: str
    n_frames: int
    keyframes: list[V3Keyframe]
    audit_self: dict = field(default_factory=dict)  # the .audit.json companion
    audit_log_count: int = 0                        # rows in .tool_audit.jsonl
    hint: str = ""                                  # text from hints.md (or "")
    outcome: str = "unknown"                        # "success" | "failure" | "unknown"
    axis_overlay: dict | None = None                # per-frame wrist-triad (pose_overlay.json)


def _short_id(episode_id: str, ep_dir: str) -> str:
    """Best-effort short label for the dropdown."""
    base = os.path.basename(ep_dir.rstrip("/"))
    # The ep_dir basename is more recognisable than the long episode_id
    return base[-80:]


def _detect_outcome(ep_dir: str) -> str:
    """Parse 'success' / 'failure' from the ep dir basename (DROID convention)."""
    base = os.path.basename(ep_dir.rstrip("/")).lower()
    if "_success_" in base or base.endswith("_success"):
        return "success"
    if "_failure_" in base or base.endswith("_failure"):
        return "failure"
    return "unknown"


def parse_hints_md(path: str) -> dict[str, str]:
    """Parse `## <ep_dir_basename>` sections out of hints.md.
    Returns {ep_dir_basename: hint_text}. Empty / placeholder hints are dropped."""
    import re as _re
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    cur_key: str | None = None
    cur_lines: list[str] = []
    for line in Path(path).read_text().splitlines():
        m = _re.match(r"^##\s+(\S+)", line)
        if m:
            if cur_key is not None:
                out[cur_key] = "\n".join(cur_lines).strip()
            cur_key, cur_lines = m.group(1), []
        elif cur_key is not None:
            cur_lines.append(line)
    if cur_key is not None:
        out[cur_key] = "\n".join(cur_lines).strip()
    return {k: v for k, v in out.items() if v and not v.startswith("<replace with")}


def save_hint_to_md(path: str, key: str, text: str) -> None:
    """Upsert the `## <key>` section of hints.md with `text` (in-place).
    Creates the file/section if absent. Other sections are left untouched."""
    import re as _re
    text = (text or "").strip()
    header = f"## {key}"
    body = f"{header}\n{text}\n"
    existing = Path(path).read_text() if os.path.exists(path) else \
        "# Per-episode hints for SSAA distributed annotation\n"
    # replace the section (from its header up to the next `## ` or EOF)
    pat = _re.compile(rf"^##\s+{_re.escape(key)}\s*\n.*?(?=^##\s|\Z)", _re.M | _re.S)
    if pat.search(existing):
        new = pat.sub(body, existing)
    else:
        new = existing.rstrip("\n") + "\n\n" + body
    Path(path).write_text(new)


def _fill_pose_deltas(ep: V3Episode) -> None:
    """Populate kf.pose_delta for each keyframe.

    Resolution order:
      1. `<ep_dir>/pose_deltas.json` (pre-computed; HF Space path) — keyed by str(kf_idx)
      2. Live call to data_pipeline.tools.get_pose_delta (needs h5 + DROID_RAW_ROOT)

    Silently leaves kf.pose_delta=None when neither source is available.
    """
    # Path 1: pre-computed
    cache_path = os.path.join(ep.ep_dir, "pose_deltas.json")
    if os.path.exists(cache_path):
        try:
            cache = json.loads(Path(cache_path).read_text())
            n_filled = 0
            for kf in ep.keyframes:
                v = cache.get(str(kf.kf_idx))
                if v:
                    kf.pose_delta = v
                    n_filled += 1
            logger.info("pose_delta: loaded %d/%d from %s",
                        n_filled, len(ep.keyframes), os.path.basename(cache_path))
            return
        except Exception as e:
            logger.warning("pose_delta cache read failed for %s: %s",
                           ep.short_id, e)

    # Path 2: live compute
    try:
        for p in (
            "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts",
            "/data/zhaoqc/RoboTwin/policy/lap/scripts",
        ):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        from data_pipeline import tools as droid_tools  # noqa: E402
    except Exception as e:
        logger.warning("pose_delta: tools import failed: %s", e)
        return
    for kf in ep.keyframes:
        try:
            if kf.chunk_end_frame <= kf.frame_idx:
                continue
            idx2 = min(kf.chunk_end_frame, ep.n_frames - 1)
            if idx2 <= kf.frame_idx:
                continue
            d = droid_tools.get_pose_delta(ep.ep_dir, kf.frame_idx, idx2)
            kf.pose_delta = d
        except Exception as e:
            logger.debug("pose_delta failed for %s kf%d: %s",
                         ep.short_id, kf.kf_idx, e)


def _fill_axis_overlay(ep: V3Episode) -> None:
    """Populate ep.axis_overlay with the per-frame wrist-triad projection.

    Resolution order (mirrors _fill_pose_deltas):
      1. `<ep_dir>/axis_overlay.json` (pre-computed; HF Space path — h5-free)
      2. Live compute via data_pipeline.pose_overlay (needs h5 + DROID_RAW_ROOT)
    Silently leaves ep.axis_overlay=None when neither is available.
    """
    cache_path = os.path.join(ep.ep_dir, "axis_overlay.json")
    if os.path.exists(cache_path):
        try:
            ep.axis_overlay = json.loads(Path(cache_path).read_text())
            logger.info("axis_overlay: loaded %d frames from %s",
                        ep.axis_overlay.get("n_frames", 0),
                        os.path.basename(cache_path))
            return
        except Exception as e:
            logger.warning("axis_overlay cache read failed for %s: %s",
                           ep.short_id, e)
    try:
        for p in ("/home/numbnut/worksapce/RoboTwin/policy/lap/scripts",
                  "/data/zhaoqc/RoboTwin/policy/lap/scripts"):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        from data_pipeline import pose_overlay as _po  # noqa: E402
        ep.axis_overlay = _po.compute_overlay(ep.ep_dir)
        logger.info("axis_overlay: live-computed %d frames for %s",
                    ep.axis_overlay.get("n_frames", 0), ep.short_id)
    except Exception as e:
        logger.warning("axis_overlay: live compute failed for %s: %s",
                        ep.short_id, e)


# Axis colors in RGB (the viewer's frames are RGB). X=red, Y=green, Z=blue.
_AXIS_RGB = {"x": (235, 64, 52), "y": (52, 199, 89), "z": (52, 120, 235)}


def _draw_pose_gizmo(img: "np.ndarray", ep: V3Episode, frame_idx: int,
                     anchor_kf: "V3Keyframe | None") -> "np.ndarray":
    """Overlay SCENE-ANCHORED wrist-frame triads on the (RGB) ext image.

    Two coordinate frames are drawn at their real projected gripper locations:
    the current frame (bright) and the anchor's chunk_end_frame (faint). Faint
    connectors link corresponding axis tips to show the motion *sweep* of the
    frame over the chunk. Projection is the standard pinhole with the verified
    euler 'xyz' extrinsics (fx≈700 guess — placement indicative, not metrology).
    """
    ov = ep.axis_overlay
    if not ov:
        return img
    frames = ov.get("frames", [])
    if not (0 <= frame_idx < len(frames)):
        return img
    cur = frames[frame_idx]
    if not cur.get("valid") or not cur.get("o"):
        return img
    img = img.copy()
    # Overlay coords are stored at the JSON's reference resolution; the actual
    # displayed frame may be resized (HF Space decodes at --resize-w 640). Scale
    # to the real image size so the triad stays anchored on the gripper.
    h_img, w_img = img.shape[:2]
    sx = w_img / float(ov.get("image_w") or w_img)
    sy = h_img / float(ov.get("image_h") or h_img)

    end = None
    if anchor_kf is not None:
        ei = min(anchor_kf.chunk_end_frame, len(frames) - 1)
        if ei != frame_idx and frames[ei].get("valid") and frames[ei].get("o"):
            end = frames[ei]

    def _pt(p):
        return (int(round(p[0] * sx)), int(round(p[1] * sy)))

    def _triad(rec, bright):
        o = _pt(rec["o"])
        for ax in ("x", "y", "z"):
            if not rec.get(ax):
                continue
            base = _AXIS_RGB[ax]
            col = base if bright else tuple(int(c * 0.4 + 110) for c in base)
            cv2.line(img, o, _pt(rec[ax]), col, 3 if bright else 2, cv2.LINE_AA)
            if bright:
                cv2.putText(img, ax.upper(), (_pt(rec[ax])[0] + 3, _pt(rec[ax])[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        cv2.circle(img, o, 4, (255, 255, 255), -1 if bright else 1)

    # connectors first (underneath): faint line per corresponding axis tip + origins
    if end is not None:
        for ax in ("x", "y", "z"):
            if cur.get(ax) and end.get(ax):
                cv2.line(img, _pt(cur[ax]), _pt(end[ax]),
                         tuple(int(c * 0.3 + 150) for c in _AXIS_RGB[ax]), 1, cv2.LINE_AA)
        cv2.line(img, _pt(cur["o"]), _pt(end["o"]), (200, 200, 200), 1, cv2.LINE_AA)
        _triad(end, bright=False)
    _triad(cur, bright=True)

    # caption (top-left), rotation over the chunk
    rot = ""
    if anchor_kf is not None and anchor_kf.pose_delta:
        rot = anchor_kf.pose_delta.get("delta_rot_world") or ""
    cap = "wrist frame on ext view  •  bright=now faint=chunk_end"
    if rot:
        cap += f"  •  rot {rot}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(img, cap, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _build_unannotated_episode(ep_dir: str, meta_p: str) -> V3Episode | None:
    """Build a V3Episode from meta.json only (no annotation).

    Each kf gets stub S/A/etc; chunk_end_frame defaults to next-kf.frame_idx
    so the ribbon shows the rule-detector partition as a baseline. Marked
    in short_id with a [no-ann] suffix.
    """
    try:
        meta = json.loads(Path(meta_p).read_text())
    except Exception as e:
        logger.warning("skip unannotated %s: %s", ep_dir, e)
        return None
    meta_kfs = meta.get("keyframes", [])
    if not meta_kfs:
        return None
    kfs: list[V3Keyframe] = []
    for i, mkf in enumerate(meta_kfs):
        fi = int(mkf.get("frame_idx", 0))
        if i + 1 < len(meta_kfs):
            ce = int(meta_kfs[i + 1].get("frame_idx", fi + 1))
        else:
            ce = min(int(meta.get("n_frames", fi + 1)) - 1, fi + 30)
        if ce <= fi:
            ce = fi + 1
        ext_p = os.path.join(ep_dir, mkf.get("image_file", "") or "")
        wrist_p = os.path.join(ep_dir, mkf.get("wrist_image_file", "") or "")
        kfs.append(V3Keyframe(
            kf_idx=i, frame_idx=fi, chunk_end_frame=ce,
            phase_type=mkf.get("type", "") or "unannotated",
            imitation_supervised=True,
            S="(no annotation — preview only)",
            S_pred="", A="", A_correct="",
            kf_type=mkf.get("type", "") or "",
            gripper_state=mkf.get("gripper_state", "") or "",
            ext_img_path=ext_p if os.path.exists(ext_p) else "",
            wrist_img_path=wrist_p if os.path.exists(wrist_p) else "",
        ))
    episode_id = str(meta.get("episode_id", os.path.basename(ep_dir)))
    return V3Episode(
        short_id=_short_id(episode_id, ep_dir) + "  [no-ann]",
        episode_id=episode_id,
        ep_dir=ep_dir,
        task_instruction=str(meta.get("task_instruction", "")),
        description="(no annotation yet — preview for hint-writing)",
        n_frames=int(meta.get("n_frames", 0)),
        keyframes=kfs,
        audit_self={}, audit_log_count=0,
    )


def load_v3_episodes(images_dir: str, suffix: str,
                     max_episodes: int | None = None,
                     include_unannotated: bool = False,
                     hints_path: str | None = None) -> list[V3Episode]:
    """Load v3 episodes from per-ep dirs.

    When `include_unannotated=True`, dirs that have meta.json but no
    annotation_<suffix>.json are still loaded as "preview" episodes
    (keyframes from meta, but no S/S_pred/A/A_correct). Useful for
    inspecting raw video while writing hints.
    """
    hints = parse_hints_md(hints_path) if hints_path else {}
    if hints:
        logger.info("hints: %d entries from %s", len(hints), hints_path)
    ep_dirs = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                     if os.path.isdir(p)
                     and os.path.exists(os.path.join(p, "meta.json")))
    out: list[V3Episode] = []
    for ep_dir in ep_dirs:
        if max_episodes is not None and len(out) >= max_episodes:
            break
        meta_p = os.path.join(ep_dir, "meta.json")
        ann_p = os.path.join(ep_dir, f"annotation_{suffix}.json")
        if not os.path.exists(meta_p):
            continue
        ep_basename = os.path.basename(ep_dir.rstrip("/"))
        ep_hint = hints.get(ep_basename, "")
        ep_outcome = _detect_outcome(ep_dir)
        if ep_outcome == "unknown":          # uuid-named dirs lack the token;
            try:                             # the real outcome is in meta.json
                mo = json.load(open(meta_p)).get("outcome")
                if mo in ("success", "failure"):
                    ep_outcome = mo
            except Exception:
                pass
        if not os.path.exists(ann_p):
            if include_unannotated:
                ep = _build_unannotated_episode(ep_dir, meta_p)
                if ep is not None:
                    ep.hint = ep_hint
                    ep.outcome = ep_outcome
                    out.append(ep)
            continue
        try:
            meta = json.load(open(meta_p))
            ann = json.load(open(ann_p))
        except Exception as e:
            logger.warning("skip %s: %s", ep_dir, e)
            continue
        if not isinstance(ann, dict) or "keyframes" not in ann:
            continue

        meta_by_frame: dict[int, dict] = {
            int(kf["frame_idx"]): kf for kf in meta.get("keyframes", [])
        }
        kfs: list[V3Keyframe] = []
        for i, raw in enumerate(ann["keyframes"]):
            fi = int(raw.get("frame_idx", 0))
            ce = raw.get("chunk_end_frame")
            ce = int(ce) if isinstance(ce, int) else fi + 1
            meta_kf = meta_by_frame.get(fi, {})
            ext_p = os.path.join(ep_dir, meta_kf.get("image_file", "") or "")
            wrist_p = os.path.join(ep_dir, meta_kf.get("wrist_image_file", "") or "")
            kfs.append(V3Keyframe(
                kf_idx=i,
                frame_idx=fi,
                chunk_end_frame=ce,
                phase_type=str(raw.get("phase_type", "")),
                imitation_supervised=bool(raw.get("imitation_supervised", True)),
                S=str(raw.get("S", "")),
                S_pred=str(raw.get("S_pred", "")),
                A=str(raw.get("A", "")),
                A_correct=str(raw.get("A_correct") or ""),
                mode_marker=str(raw.get("mode_marker", "")),
                kf_type=str(meta_kf.get("type", "")),
                gripper_state=str(meta_kf.get("gripper_state", "")),
                ext_img_path=ext_p if os.path.exists(ext_p) else "",
                wrist_img_path=wrist_p if os.path.exists(wrist_p) else "",
            ))

        # Companion audit + tool-log
        audit_self: dict = {}
        sap = ann_p + ".audit.json"
        if os.path.exists(sap):
            try:
                audit_self = json.load(open(sap))
            except Exception:
                pass
        log_count = 0
        tlp = os.path.join(ep_dir, ".tool_audit.jsonl")
        if os.path.exists(tlp):
            try:
                with open(tlp) as f:
                    log_count = sum(1 for ln in f if ln.strip())
            except Exception:
                pass

        episode_id = str(meta.get("episode_id", os.path.basename(ep_dir)))
        out.append(V3Episode(
            short_id=_short_id(episode_id, ep_dir),
            episode_id=episode_id,
            ep_dir=ep_dir,
            task_instruction=str(meta.get("task_instruction", "")),
            description=str(ann.get("description", "")),
            n_frames=int(meta.get("n_frames", 0)),
            keyframes=kfs,
            audit_self=audit_self,
            audit_log_count=log_count,
            hint=ep_hint,
            outcome=ep_outcome,
        ))
    return out


# ───────────────────────────────────────────────────────────────────────────
# Anchor + image resolution
# ───────────────────────────────────────────────────────────────────────────

def current_anchor_idx(ep: V3Episode, frame_idx: int) -> int:
    """Return the kf list index whose frame_idx ≤ frame_idx (the most
    recent prior anchor). -1 if none (slider below first kf)."""
    chosen = -1
    for i, kf in enumerate(ep.keyframes):
        if kf.frame_idx <= frame_idx:
            chosen = i
        else:
            break
    return chosen


def _index_frame_dir(d: str) -> list[str]:
    """Return [frame0_path, frame1_path, ...] from a frames/<view>/fNNNN.jpg dir.
    Sparse / missing frames represented as empty strings to preserve indexing."""
    if not os.path.isdir(d):
        return []
    pairs: list[tuple[int, str]] = []
    for name in os.listdir(d):
        if not name.endswith(".jpg"):
            continue
        stem = name[:-4]
        if not stem.startswith("f"):
            continue
        try:
            n = int(stem[1:])
        except ValueError:
            continue
        pairs.append((n, os.path.join(d, name)))
    if not pairs:
        return []
    n_max = max(p[0] for p in pairs)
    out: list[str] = ["" for _ in range(n_max + 1)]
    for n, p in pairs:
        out[n] = p
    return out


def load_kf_image(path: str) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ───────────────────────────────────────────────────────────────────────────
# Optional full-video cache (decode MP4 → in-memory JPEG bytes)
# ───────────────────────────────────────────────────────────────────────────

def resolve_mp4_paths(raw_root: str, episode_id: str) -> tuple[str | None, str | None]:
    """(ext_mp4, wrist_mp4) absolute paths in the raw mirror for an episode,
    or None where unavailable. raw_root = DROID_RAW_ROOT mirror."""
    raw_root = os.path.expanduser(raw_root or "")
    h5_path = os.path.join(raw_root, episode_id)
    if not os.path.exists(h5_path):
        return None, None
    ep_root = os.path.dirname(h5_path)
    md_files = list(Path(ep_root).glob("metadata_*.json"))
    if not md_files:
        return None, None
    raw_meta = json.load(open(md_files[0]))
    wrist_serial = raw_meta.get("wrist_cam_serial", "")
    primary_ext = raw_meta.get("ext1_cam_serial", "") or raw_meta.get("ext2_cam_serial", "")
    rec_dir = os.path.join(ep_root, "recordings", "MP4")
    ext_mp4 = os.path.join(rec_dir, f"{primary_ext}.mp4") if primary_ext else None
    wrist_mp4 = os.path.join(rec_dir, f"{wrist_serial}.mp4") if wrist_serial else None
    return ext_mp4, wrist_mp4


class VideoCache:
    """Decode each ep's ext + wrist MP4 into in-memory JPEG byte lists.

    Memory budget: ~50KB/frame × ~300 frames × 2 views × N eps. For our
    5-ep local set (~225 frames avg) that's ~70 MB. Trivial.

    Lazy per-episode: first frame request triggers full decode + cache.
    """

    def __init__(self, droid_raw_root: str, jpeg_quality: int = 85):
        self.droid_raw_root = os.path.expanduser(droid_raw_root)
        self.jpeg_quality = int(jpeg_quality)
        self._by_episode_id: dict[str, dict] = {}  # episode_id → {"ext": [bytes], "wrist": [bytes]}
        self._missing: set[str] = set()  # episodes we already failed on

    def _resolve_mp4_paths(self, episode_id: str) -> tuple[str | None, str | None]:
        return resolve_mp4_paths(self.droid_raw_root, episode_id)

    def _decode_mp4(self, path: str | None) -> list[bytes]:
        if not path or not os.path.exists(path):
            return []
        cap = cv2.VideoCapture(path)
        out: list[bytes] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                out.append(buf.tobytes())
        cap.release()
        return out

    def ensure_loaded(self, episode_id: str, ep_dir: str | None = None) -> bool:
        if episode_id in self._by_episode_id:
            return True
        if episode_id in self._missing:
            return False
        # Path 1: pre-decoded JPEG folder (HF Space path) — read paths lazily
        if ep_dir:
            ext_dir = os.path.join(ep_dir, "frames", "ext")
            wrist_dir = os.path.join(ep_dir, "frames", "wrist")
            if os.path.isdir(ext_dir) or os.path.isdir(wrist_dir):
                self._by_episode_id[episode_id] = {
                    "ext_paths": _index_frame_dir(ext_dir),
                    "wrist_paths": _index_frame_dir(wrist_dir),
                }
                ne = len(self._by_episode_id[episode_id]["ext_paths"])
                nw = len(self._by_episode_id[episode_id]["wrist_paths"])
                logger.info("VideoCache: indexed JPEG dir for %s "
                            "(ext=%d, wrist=%d)", os.path.basename(ep_dir), ne, nw)
                return True
        # Path 2: in-memory MP4 decode
        ext_mp4, wrist_mp4 = self._resolve_mp4_paths(episode_id)
        if not (ext_mp4 or wrist_mp4):
            logger.warning("VideoCache: no JPEG dir or MP4s for %s",
                           episode_id)
            self._missing.add(episode_id)
            return False
        logger.info("VideoCache: decoding %s ...", episode_id)
        ext_frames = self._decode_mp4(ext_mp4)
        wrist_frames = self._decode_mp4(wrist_mp4)
        self._by_episode_id[episode_id] = {"ext": ext_frames, "wrist": wrist_frames}
        logger.info("  → %d ext + %d wrist frames cached",
                    len(ext_frames), len(wrist_frames))
        return True

    def get_frame(self, episode_id: str, frame_idx: int,
                  view: str, ep_dir: str | None = None) -> np.ndarray | None:
        if not self.ensure_loaded(episode_id, ep_dir):
            return None
        data = self._by_episode_id[episode_id]
        # JPEG-folder path
        if f"{view}_paths" in data:
            paths = data[f"{view}_paths"]
            if not (0 <= frame_idx < len(paths)):
                return None
            img = cv2.imread(paths[frame_idx])
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # In-memory bytes path
        frames = data.get(view, [])
        if not (0 <= frame_idx < len(frames)):
            return None
        arr = np.frombuffer(frames[frame_idx], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_frame_image(ep: V3Episode, frame_idx: int, view: str,
                    video_cache: VideoCache | None,
                    overlay: bool = False,
                    anchor_kf: "V3Keyframe | None" = None) -> np.ndarray | None:
    """Resolve the image for (ep, frame_idx, view). Tries video cache for
    an exact frame; falls back to nearest keyframe image with an overlay.

    When `overlay` is set (and view=="ext"), draws the pose-axis gizmo."""
    img = None
    if video_cache is not None:
        img = video_cache.get_frame(ep.episode_id, frame_idx, view,
                                    ep_dir=ep.ep_dir)
    if img is None:
        if not ep.keyframes:
            return None
        nearest_kf = min(ep.keyframes, key=lambda k: abs(k.frame_idx - frame_idx))
        path = (nearest_kf.ext_img_path if view == "ext" else nearest_kf.wrist_img_path)
        img = load_kf_image(path)
        if img is None:
            return None
        if nearest_kf.frame_idx != frame_idx:
            cv2.putText(
                img, f"[nearest kf: f={nearest_kf.frame_idx}]", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2,
            )
    if overlay and view == "ext" and ep.axis_overlay:
        img = _draw_pose_gizmo(img, ep, frame_idx, anchor_kf)
    return img


# ───────────────────────────────────────────────────────────────────────────
# Phase-ribbon timeline
# ───────────────────────────────────────────────────────────────────────────

def _phase_color(phase_type: str) -> str:
    """Stable color per phase_type (categorical palette)."""
    palette = [
        "#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756",
        "#72B7B2", "#EECA3B", "#9D755D", "#FF9DA6", "#79706E",
    ]
    return palette[hash(phase_type or "") % len(palette)]


def _assign_lanes(kfs: list[V3Keyframe]) -> dict[int, int]:
    """Greedy interval-graph lane assignment: kfs that overlap get
    different sub-lanes within their tier. Returns kf_idx → lane (0..k)."""
    # Sort by frame_idx; ties broken by chunk_end_frame
    order = sorted(kfs, key=lambda k: (k.frame_idx, k.chunk_end_frame))
    lane_end: list[int] = []  # rightmost chunk_end_frame seen on each lane
    assign: dict[int, int] = {}
    for kf in order:
        placed = False
        for k, end in enumerate(lane_end):
            if kf.frame_idx >= end:
                lane_end[k] = kf.chunk_end_frame
                assign[kf.kf_idx] = k
                placed = True
                break
        if not placed:
            assign[kf.kf_idx] = len(lane_end)
            lane_end.append(kf.chunk_end_frame)
    return assign


def render_phase_ribbon(ep: V3Episode, slider: int) -> "plt.Figure":
    """Stacked timeline with greedy lane allocation:
      • Top tier (imit=true): demo + A_correct chunks, one lane per
        non-overlapping group (lanes grow upward from y=1.0)
      • Bottom tier (imit=false): recovery branch, lanes grow downward
        from y=0.0
      • Faint full-demo bar at the very top
      • Vertical red line marks slider position
    """
    # Split kfs by tier; assign lanes within each tier independently
    true_kfs = [k for k in ep.keyframes if k.imitation_supervised]
    false_kfs = [k for k in ep.keyframes if not k.imitation_supervised]
    true_lane = _assign_lanes(true_kfs)
    false_lane = _assign_lanes(false_kfs)
    n_lanes_true = max(true_lane.values(), default=-1) + 1 or 1
    n_lanes_false = max(false_lane.values(), default=-1) + 1 or 1

    lane_height = 0.42
    lane_gap = 0.06
    # Anchor coordinates
    true_base = 1.0        # bottom of lane 0 on the success tier
    false_base = -0.5      # bottom of lane 0 on the recovery tier (grows down)

    # Figure height scales with total lanes
    fig_h = 1.8 + 0.45 * (n_lanes_true + n_lanes_false)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    T = max(1, ep.n_frames)

    # Full demo bar (top, faint)
    top_y = true_base + n_lanes_true * (lane_height + lane_gap) + 0.25
    ax.broken_barh([(0, T)], (top_y, 0.18), facecolors="#e0e0e0",
                   edgecolor="#999", linewidth=0.5)
    ax.text(-0.01 * T, top_y + 0.09, "demo", va="center", ha="right",
            fontsize=8, color="#444")

    legend_seen: set[str] = set()
    legend_handles: list[mpatches.Patch] = []

    def _draw_chunk(kf: V3Keyframe, y0: float, tier_edge: str):
        color = _phase_color(kf.phase_type)
        x0 = kf.frame_idx
        w = max(1, kf.chunk_end_frame - kf.frame_idx)
        ax.broken_barh([(x0, w)], (y0, lane_height),
                       facecolors=color, edgecolor=tier_edge,
                       linewidth=0.7, alpha=0.92)
        if w >= 0.03 * T:
            ax.text(x0 + w / 2, y0 + lane_height / 2,
                    f"{kf.phase_type}\nkf{kf.kf_idx}",
                    ha="center", va="center", fontsize=7, color="#111")
        else:
            # Tiny chunk — label just kf number outside
            ax.text(x0 + w / 2, y0 + lane_height + 0.04,
                    f"kf{kf.kf_idx}",
                    ha="center", va="bottom", fontsize=6, color="#111")
        if kf.phase_type and kf.phase_type not in legend_seen:
            legend_seen.add(kf.phase_type)
            legend_handles.append(
                mpatches.Patch(color=color, label=kf.phase_type)
            )

    # True tier — lane 0 sits at true_base, higher lanes stack upward
    for kf in true_kfs:
        lane = true_lane[kf.kf_idx]
        y0 = true_base + lane * (lane_height + lane_gap)
        _draw_chunk(kf, y0, "#222")

    # False tier — lane 0 sits at false_base, higher lanes stack downward
    for kf in false_kfs:
        lane = false_lane[kf.kf_idx]
        y0 = false_base - lane * (lane_height + lane_gap)
        _draw_chunk(kf, y0, "#a33")

    # Tier labels on the left
    true_center = true_base + (n_lanes_true * (lane_height + lane_gap)) / 2 - lane_gap / 2
    ax.text(-0.01 * T, true_center, "demo + A_correct", va="center",
            ha="right", fontsize=8, color="#222")
    if false_kfs:
        false_top = false_base + lane_height
        false_bot = false_base - (n_lanes_false - 1) * (lane_height + lane_gap)
        false_center = (false_top + false_bot) / 2
        ax.text(-0.01 * T, false_center, "recovery", va="center",
                ha="right", fontsize=8, color="#a33")

    # Recovery branch elbow at first divergence
    first_false = false_kfs[0] if false_kfs else None
    if first_false is not None:
        bx = first_false.frame_idx
        ax.plot([bx, bx], [true_base, false_base + lane_height],
                color="#a33", linewidth=1.0, linestyle=":")

    # Slider position
    bottom_y = false_base - (n_lanes_false - 1) * (lane_height + lane_gap) - 0.25
    ax.axvline(slider, color="#d62728", linewidth=1.6, alpha=0.85,
               ymin=0, ymax=1)
    ax.text(slider, top_y + 0.3, f"frame {slider}", color="#d62728",
            fontsize=8, ha="center")

    # Anchor indicator: ▼ on the anchor's actual lane
    anc = current_anchor_idx(ep, slider)
    if anc >= 0:
        akf = ep.keyframes[anc]
        if akf.imitation_supervised:
            ay = true_base + true_lane[akf.kf_idx] * (lane_height + lane_gap)
        else:
            ay = false_base - false_lane[akf.kf_idx] * (lane_height + lane_gap)
        ax.scatter([akf.frame_idx], [ay + lane_height + 0.02],
                   marker="v", color="#d62728", s=70, zorder=5)

    ax.set_xlim(-0.02 * T, T * 1.02)
    ax.set_ylim(bottom_y - 0.1, top_y + 0.45)
    ax.set_yticks([])
    ax.set_xlabel("frame_idx")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right",
                  ncol=min(5, len(legend_handles)), fontsize=7,
                  framealpha=0.85, bbox_to_anchor=(1.0, -0.05))
    fig.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────────────────
# Markdown panels
# ───────────────────────────────────────────────────────────────────────────

_OUTCOME_STYLE = {
    "success": ("#1f883d", "SUCCESS"),
    "failure": ("#cf222e", "FAILURE"),
    "unknown": ("#6e7781", "UNKNOWN"),
}


def _outcome_chip_html(outcome: str) -> str:
    color, label = _OUTCOME_STYLE.get(outcome, _OUTCOME_STYLE["unknown"])
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:6px;font-weight:600;font-size:0.85em;"
        f"letter-spacing:0.5px;'>{label}</span>"
    )


def _hint_panel_html(hint: str) -> str:
    if not hint.strip():
        return ""
    # Render hint with newlines preserved
    body = hint.replace("\n", "<br>")
    return (
        "<div style='background:#fff8c4;border-left:4px solid #d4a72c;"
        "padding:8px 12px;border-radius:4px;margin:6px 0;"
        "font-family:system-ui;font-size:0.92em;'>"
        "<b style='color:#7d4e00;'>💡 Human hint</b><br>"
        f"<span style='color:#3b2e00;'>{body}</span></div>"
    )


def _meta_md(ep: V3Episode) -> str:
    n_true = sum(1 for k in ep.keyframes if k.imitation_supervised)
    n_false = len(ep.keyframes) - n_true
    first_div = next((k.kf_idx for k in ep.keyframes
                      if not k.imitation_supervised), None)
    audit_line = ""
    if ep.audit_log_count or ep.audit_self:
        audit_line = (
            f"  •  tool_log: {ep.audit_log_count} calls"
            f"  •  self-audit: {'yes' if ep.audit_self else 'no'}"
        )
    chip = _outcome_chip_html(ep.outcome)
    hint_html = _hint_panel_html(ep.hint)
    return (
        f"{chip}  &nbsp; **Task:** {ep.task_instruction}\n\n"
        f"{hint_html}\n\n"
        f"**Description:** {ep.description}\n\n"
        f"_{len(ep.keyframes)} kfs  •  imit=true: {n_true}  •  "
        f"imit=false: {n_false}  •  first_diverge_kf: {first_div}{audit_line}_"
    )


_CARD_STYLES = {
    # role → (bg, border-accent, label-bg, label-fg, icon)
    "S":      ("#eef4fb", "#2b6cb0", "#2b6cb0", "#fff", "👁"),
    "S_pred": ("#f3eefa", "#7c4ec0", "#7c4ec0", "#fff", "🔮"),
    "A_correct": ("#ecf6ec", "#2f855a", "#2f855a", "#fff", "🎯"),
    "A":      ("#f3f4f6", "#6b7280", "#6b7280", "#fff", "📜"),
    "pose":   ("#fff7e6", "#b7791f", "#b7791f", "#fff", "Δ"),
}


def _card(role: str, title: str, body_html: str) -> str:
    bg, border, lbg, lfg, icon = _CARD_STYLES[role]
    return (
        f"<div style='background:{bg};border-left:4px solid {border};"
        f"border-radius:6px;padding:10px 14px 12px 14px;margin:8px 0;"
        f"font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'>"
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
        f"<span style='background:{lbg};color:{lfg};padding:2px 8px;"
        f"border-radius:4px;font-size:0.78em;font-weight:700;"
        f"letter-spacing:0.5px;'>{icon}&nbsp;&nbsp;{title}</span>"
        f"</div>"
        f"<div style='color:#1f2937;line-height:1.55;font-size:0.95em;'>"
        f"{body_html}</div>"
        f"</div>"
    )


def _pose_delta_html(d: dict | None) -> str:
    """Compact pose-delta as inline-grid HTML inside the pose card."""
    if not d:
        return ""
    def _fmt_triple(triple: dict) -> str:
        try:
            f = triple["forward"]; l = triple["left"]; u = triple["up"]
            return (
                f"<span style='font-family:ui-monospace,Menlo,Consolas,monospace;"
                f"font-size:0.92em;'>"
                f"fwd <b>{f:+.1f}</b>cm &nbsp; "
                f"left <b>{l:+.1f}</b>cm &nbsp; "
                f"up <b>{u:+.1f}</b>cm</span>"
            )
        except Exception:
            return "?"
    nfr = d.get("n_frames", "?")
    rows = [
        f"<div style='color:#7c4a1c;font-weight:600;margin-bottom:4px;'>"
        f"motion over phase ({nfr} frames)</div>"
    ]
    grip = d.get("gripper")
    if grip:
        rows.append(
            f"<div style='margin-bottom:4px;'><span style='color:#7c4a1c;"
            f"font-weight:600;width:60px;display:inline-block;'>gripper</span>"
            f"<span style='font-family:ui-monospace,Menlo,monospace;font-size:0.92em;"
            f"background:#fdf0d5;padding:1px 6px;border-radius:3px;'>{grip}</span></div>"
        )
    rows.append(
        f"<div><span style='color:#7c4a1c;font-weight:600;width:60px;"
        f"display:inline-block;'>robot</span> {_fmt_triple(d['delta_robot'])}</div>"
    )
    if d.get("delta_ee"):
        rows.append(
            f"<div><span style='color:#7c4a1c;font-weight:600;width:60px;"
            f"display:inline-block;'>wrist</span> {_fmt_triple(d['delta_ee'])}</div>"
        )
    rot_w = d.get("delta_rot_world", "")
    rot_e = d.get("delta_rot_ee", "")
    if rot_w or rot_e:
        rows.append(
            f"<div style='margin-top:4px;font-size:0.88em;color:#5b3a14;'>"
            f"rot robot <code>{rot_w}</code> &nbsp;•&nbsp; "
            f"rot wrist <code>{rot_e}</code></div>"
        )
    events = d.get("interaction_events_in_range") or []
    if events:
        ev_str = ", ".join(f"{e['type']}@f{e['frame_idx']}" for e in events)
        rows.append(
            f"<div style='margin-top:4px;font-size:0.88em;color:#5b3a14;'>"
            f"events inside: <b>{ev_str}</b></div>"
        )
    gg = d.get("gap_to_grasp"); gr = d.get("gap_to_release")
    bits = []
    if gg: bits.append(f"grasp at f{gg['target_frame']}")
    if gr: bits.append(f"release at f{gr['target_frame']}")
    if bits:
        rows.append(
            f"<div style='margin-top:4px;font-size:0.88em;color:#5b3a14;'>"
            f"next ▸ {'  •  '.join(bits)}</div>"
        )
    return "".join(rows)


def _escape_text(s: str) -> str:
    """Light escape so user-supplied text doesn't break our HTML."""
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_think(s: str) -> str:
    """Escape text, then render any inline <think>…</think> block as a styled
    callout so the chain-of-thought reads distinctly from the action."""
    t = _escape_text(s)
    t = t.replace(
        "&lt;think&gt;",
        "<span style='display:block;background:#f3f0ff;border-left:3px solid "
        "#a78bfa;padding:4px 8px;margin:3px 0;color:#5b21b6;font-style:italic;'>"
        "🧠 ")
    t = t.replace("&lt;/think&gt;", "</span>")
    return t.replace("\n", "<br>")


def _anchor_md(ep: V3Episode, anchor_idx: int, slider: int) -> str:
    if anchor_idx < 0:
        return (
            f"<div style='padding:14px;background:#f9fafb;border-radius:6px;"
            f"color:#6b7280;font-style:italic;'>"
            f"slider at frame {slider} — before first keyframe</div>"
        )
    kf = ep.keyframes[anchor_idx]
    phase_color = _phase_color(kf.phase_type)
    phase_len = kf.chunk_end_frame - kf.frame_idx
    delta = slider - kf.frame_idx

    if kf.imitation_supervised:
        imit_html = (
            "<span style='background:#d4edda;color:#155724;"
            "padding:1px 8px;border-radius:4px;font-size:0.82em;font-weight:600;'>"
            "✓ imitation_supervised</span>"
        )
    else:
        imit_html = (
            "<span style='background:#f8d7da;color:#721c24;"
            "padding:1px 8px;border-radius:4px;font-size:0.82em;font-weight:600;'>"
            "✗ DIVERGES (recovery / intervention)</span>"
        )

    is_think = ("<think>" in kf.A) or ("<think>" in kf.A_correct)
    if is_think:
        mode_html = (
            "<span style='background:#ede9fe;color:#5b21b6;padding:1px 8px;"
            "border-radius:4px;font-size:0.82em;font-weight:600;'>🧠 think</span>"
        )
    else:
        mode_html = (
            "<span style='background:#e5e7eb;color:#374151;padding:1px 8px;"
            "border-radius:4px;font-size:0.82em;font-weight:600;'>⚡ act</span>"
        )

    header = (
        f"<div style='font-family:system-ui;'>"
        f"<div style='font-size:1.05em;font-weight:700;margin-bottom:6px;'>"
        f"Anchor &nbsp;<span style='font-family:ui-monospace,Menlo,monospace;"
        f"background:#1f2937;color:#f9fafb;padding:2px 8px;border-radius:4px;'>"
        f"kf{kf.kf_idx:02d}</span> &nbsp;@&nbsp;frame {kf.frame_idx}"
        f"&nbsp;&nbsp;<span style='background:{phase_color};color:white;"
        f"padding:2px 10px;border-radius:4px;font-size:0.88em;font-weight:600;'>"
        f"{_escape_text(kf.phase_type)}</span></div>"
        f"<div style='font-size:0.86em;color:#4b5563;margin-bottom:4px;'>"
        f"phase <code>[{kf.frame_idx} → {kf.chunk_end_frame}]</code> "
        f"({phase_len} frames) &nbsp;•&nbsp; slider δ {delta:+d} from anchor</div>"
        f"<div style='font-size:0.86em;color:#4b5563;margin-bottom:8px;'>"
        f"{mode_html} &nbsp; {imit_html} &nbsp;•&nbsp; "
        f"kf_type <code>{_escape_text(kf.kf_type)}</code> &nbsp;•&nbsp; "
        f"gripper <code>{_escape_text(kf.gripper_state)}</code></div>"
        f"</div>"
    )

    parts = [header]
    if kf.pose_delta:
        parts.append(_card("pose", "MOTION  Δ", _pose_delta_html(kf.pose_delta)))
    parts.append(_card("S",      "S — scene now", _fmt_think(kf.S)))
    # begin/end brackets are S-only: A and S_pred are empty there → hide.
    if kf.A.strip():
        a_title = ("A — action (= BC target)" if kf.imitation_supervised
                   else "A — demo action (NOT imitated; world-model only)")
        parts.append(_card("A", a_title, _fmt_think(kf.A)))
    # A_correct only exists when the policy overrides the demo.
    if kf.A_correct.strip():
        parts.append(_card("A_correct", "A_correct — override (reasoned)",
                           _fmt_think(kf.A_correct)))
    if kf.S_pred.strip():
        parts.append(_card("S_pred", "S_pred — forecast at phase end",
                           _fmt_think(kf.S_pred)))
    if not (kf.A.strip() or kf.S_pred.strip()):
        parts.append("<div style='color:#6b7280;font-style:italic;font-size:0.85em;"
                     "padding:4px 0;'>bracket keyframe — S-only (no action target)</div>")
    return "".join(parts)


def _audit_md(ep: V3Episode) -> str:
    if not (ep.audit_self or ep.audit_log_count):
        return "_(no audit data — set --save-tool-log / subagent self-report off)_"
    sa = ep.audit_self
    lines = [f"**tool_audit.jsonl** rows: `{ep.audit_log_count}`  (ground truth)"]
    if sa:
        ir = sa.get("image_reads") or []
        tc = sa.get("tool_calls") or []
        cr = sa.get("chunk_end_revisions") or []
        kd = sa.get("key_decisions") or []
        oq = sa.get("open_questions") or []
        lines.append(f"**self-reported:** images={len(ir)}  "
                     f"tool_calls={len(tc)}  revisions={len(cr)}  "
                     f"key_decisions={len(kd)}")
        if kd:
            lines.append("\n**key decisions:**")
            for k in kd[:8]:
                lines.append(f"- kf{k.get('kf', '?')}: {k.get('decision', '')}  "
                             f"_— {k.get('why', '')}_")
        if cr:
            lines.append("\n**chunk_end revisions considered:**")
            for r in cr[:8]:
                lines.append(f"- kf{r.get('kf', '?')}: considered "
                             f"{r.get('considered', [])} → chose `{r.get('chose', '?')}`  "
                             f"_({r.get('why', '')})_")
        if oq:
            lines.append("\n**open questions:** " + " | ".join(str(x) for x in oq))
    return "\n".join(lines)


def _kf_table(ep: V3Episode) -> list[list]:
    return [[
        f"kf{kf.kf_idx:02d}",
        kf.frame_idx,
        kf.chunk_end_frame,
        kf.chunk_end_frame - kf.frame_idx,
        kf.phase_type,
        "✓" if kf.imitation_supervised else "✗",
        ("🧠 " if ("<think>" in kf.A or "<think>" in kf.A_correct) else "")
        + (_strip_think(kf.A_correct) or _strip_think(kf.A))[:88],
    ] for kf in ep.keyframes]


def _strip_think(s: str) -> str:
    """Drop the <think>…</think> block for compact previews."""
    import re as _re
    return _re.sub(r"<think>.*?</think>", "", s or "", flags=_re.S).strip()


# ───────────────────────────────────────────────────────────────────────────
# Optional vision-LLM hint completion (provider-agnostic, stdlib only)
# ───────────────────────────────────────────────────────────────────────────

_HINT_SYS = (
    "You help a human write a SHORT annotation hint for ONE robot-arm "
    "manipulation episode (training metadata). In 1-3 concise sentences state: "
    "what the task actually is (objects + goal); for a FAILURE, where and why it "
    "fails; and any perception traps (occlusions, look-alike objects). The "
    "dataset's task label is often wrong — trust the images. Continue and refine "
    "the human's draft into a finished hint. Output ONLY the hint text.")


def _jpeg_b64(img) -> str:
    import io, base64
    from PIL import Image
    if img is None:
        raise ValueError("no frame image")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _http_post_json(url: str, body: dict, headers: dict, timeout: int = 40) -> dict:
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llm_complete_hint(draft: str, task: str, outcome: str, imgs_b64: list) -> str:
    """Ask a fast vision LLM to finish the hint, grounded on the current frame(s).
    Configured by env: SSAA_LLM_PROVIDER (anthropic|openai; auto-detected from
    keys), SSAA_LLM_MODEL, SSAA_LLM_BASE_URL, SSAA_LLM_KEY (else ANTHROPIC_API_KEY
    / OPENAI_API_KEY). Works with a local Ollama (OpenAI-compatible) too."""
    prov = os.environ.get("SSAA_LLM_PROVIDER")
    if not prov:
        if os.environ.get("SSAA_LLM_BASE_URL"):
            prov = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            prov = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            prov = "openai"
    if not prov:
        raise RuntimeError("LLM not configured — set ANTHROPIC_API_KEY or "
                           "OPENAI_API_KEY (or SSAA_LLM_BASE_URL for a local model)")
    user = (f"Task label (often wrong): {task or 'unknown'}. Outcome: {outcome}. "
            f'Human draft so far: "{(draft or "").strip()}".\n'
            "Look at the current frame(s) and finish the hint.")
    key = os.environ.get("SSAA_LLM_KEY")
    # generous budget: reasoning models (e.g. mimo) spend tokens on a thinking block
    mx = int(os.environ.get("SSAA_LLM_MAX_TOKENS", "600"))
    if prov == "anthropic":
        model = os.environ.get("SSAA_LLM_MODEL", "claude-haiku-4-5-20251001")
        base = os.environ.get("SSAA_LLM_BASE_URL", "https://api.anthropic.com").rstrip("/")
        content = [{"type": "image", "source": {"type": "base64",
                    "media_type": "image/jpeg", "data": b}} for b in imgs_b64]
        content.append({"type": "text", "text": user})
        body = {"model": model, "max_tokens": mx, "temperature": 0.4,
                "system": _HINT_SYS, "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": key or os.environ["ANTHROPIC_API_KEY"],
                   "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _http_post_json(base + "/v1/messages", body, headers)
        # join text blocks; skip any `thinking` blocks reasoning models emit
        txt = "".join(c.get("text", "") for c in data.get("content", [])
                      if c.get("type") == "text")
        return txt.strip()
    # openai-compatible
    model = os.environ.get("SSAA_LLM_MODEL", "gpt-4o-mini")
    base = os.environ.get("SSAA_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    content = [{"type": "text", "text": user}]
    content += [{"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b}} for b in imgs_b64]
    body = {"model": model, "max_tokens": mx, "temperature": 0.4,
            "messages": [{"role": "system", "content": _HINT_SYS},
                         {"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {key or os.environ.get('OPENAI_API_KEY', '')}",
               "content-type": "application/json"}
    data = _http_post_json(base + "/chat/completions", body, headers)
    return data["choices"][0]["message"]["content"].strip()


# ───────────────────────────────────────────────────────────────────────────
# Gradio UI
# ───────────────────────────────────────────────────────────────────────────

def build_ui(episodes: list[V3Episode], port: int,
             video_cache: VideoCache | None = None,
             hints_path: str | None = None,
             raw_root: str | None = None) -> None:
    if not episodes:
        raise SystemExit("No v3 episodes loaded.")
    by_short: dict[str, V3Episode] = {ep.short_id: ep for ep in episodes}
    overlay_available = any(ep.axis_overlay for ep in episodes)
    hint_editable = bool(hints_path)
    raw_root = os.path.expanduser(
        raw_root or os.environ.get("DROID_RAW_ROOT") or "~/datasets/droid_raw/1.0.1")

    def _ep_video(ep: V3Episode, cam: str = "ext"):
        """(selected_mp4_path_or_None, markdown_with_links) for the raw mirror
        video. `cam` picks which camera feeds the inline player (ext|wrist)."""
        ext, wrist = resolve_mp4_paths(raw_root, ep.episode_id)
        if not (ext or wrist):
            return None, ("_Raw video not in the local mirror "
                          f"(`{raw_root}`) — claim/pull this episode first._")
        import urllib.parse as _u
        folder = os.path.dirname(ext or wrist)
        links = "  ·  ".join(
            f"[▶ {name} camera](/file={_u.quote(p)})"
            for name, p in (("ext", ext), ("wrist", wrist)) if p)
        chosen = (wrist if cam == "wrist" else ext) or ext or wrist
        return chosen, f"{links}\n\n**Folder:** `{folder}`"

    def _resolve_ep(short_id: str) -> V3Episode:
        return by_short.get(short_id, episodes[0])

    def _anchor_kf(ep: V3Episode, frame_idx: int) -> "V3Keyframe | None":
        a = current_anchor_idx(ep, frame_idx)
        return ep.keyframes[a] if a >= 0 else None

    init_ep = episodes[0]
    init_slider = init_ep.keyframes[0].frame_idx if init_ep.keyframes else 0
    init_anc = current_anchor_idx(init_ep, init_slider)
    init_ext = get_frame_image(init_ep, init_slider, "ext", video_cache)
    init_wrist = get_frame_image(init_ep, init_slider, "wrist", video_cache)
    init_fig = render_phase_ribbon(init_ep, init_slider)
    init_table = _kf_table(init_ep)

    with gr.Blocks(title="SSAA-v3 viewer") as demo:
        gr.Markdown("# SSAA-v3 annotation viewer"
                    + ("  *(hint-writing mode)*" if hint_editable else "  *(read-only)*"))

        with gr.Accordion("📖 What am I looking at? — schema & field meanings", open=False):
            gr.Markdown(_INTRO_MD)

        ep_dropdown = gr.Dropdown(
            choices=list(by_short.keys()), value=init_ep.short_id,
            label="episode", interactive=True,
        )
        meta_panel = gr.Markdown(_meta_md(init_ep))

        _init_vid, _init_vid_md = _ep_video(init_ep, "ext")
        with gr.Accordion("🎬 Raw video — click ▶ to play in the browser", open=False):
            cam_radio = gr.Radio(["ext", "wrist"], value="ext", label="camera",
                                  interactive=True)
            raw_video = gr.Video(value=_init_vid, label="full MP4",
                                 interactive=False, height=360)
            raw_video_links = gr.Markdown(_init_vid_md)

        if hint_editable:
            with gr.Accordion("✍️ Write hint (saved to hints.md)", open=True):
                hint_outcome = gr.Markdown(
                    f"This episode is a {_outcome_chip_html(init_ep.outcome)}")
                hint_box = gr.Textbox(
                    value=init_ep.hint, lines=4, label=None, show_label=False,
                    placeholder=("What is the task (object, goal)? For a failure: "
                                 "where/why it goes wrong. Perception traps "
                                 "(occlusions, look-alikes). Frame numbers OK."),
                    interactive=True,
                )
                with gr.Row():
                    hint_save = gr.Button("💾 Save hint", variant="primary", scale=0)
                    hint_complete = gr.Button("✨ Complete (LLM)", scale=0)
                    hint_default = gr.Button("✓ Default hint", scale=0)
                    hint_exclude = gr.Button("🚫 Mark unusable", scale=0)
                    hint_status = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=2):
                ext_img = gr.Image(label="ext camera", type="numpy",
                                   value=init_ext, interactive=False, height=400)
                wrist_img = gr.Image(label="wrist camera", type="numpy",
                                     value=init_wrist, interactive=False, height=240)
                slider = gr.Slider(
                    minimum=0,
                    maximum=max(1, init_ep.n_frames - 1),
                    value=init_slider, step=1,
                    label="frame_idx (scrub me)",
                    interactive=True,
                )
                overlay_chk = gr.Checkbox(
                    value=False, label="🧭 pose-axis overlay (wrist frame on ext view)",
                    info=("Draws the wrist coord-frame at the gripper: bright="
                          "current frame, faint=anchor's chunk_end, connectors="
                          "motion sweep. Placement is indicative (focal length "
                          "guessed; DROID ships no intrinsics)."),
                    visible=overlay_available, interactive=True,
                )
            with gr.Column(scale=2):
                anchor_panel = gr.Markdown(_anchor_md(init_ep, init_anc, init_slider))

        gr.Markdown("### Phase ribbon")
        ribbon_plot = gr.Plot(value=init_fig)

        with gr.Accordion("Audit (tool log + self-report)", open=False):
            audit_panel = gr.Markdown(_audit_md(init_ep))

        gr.Markdown("### All keyframes")
        table = gr.Dataframe(
            headers=["#", "frame", "chunk_end", "len", "phase_type", "imit", "action (truncated)"],
            value=init_table, interactive=False, wrap=True,
        )

        # ── Handlers ────────────────────────────────────────────────────
        def _on_episode(short_id, overlay):
            ep = _resolve_ep(short_id)
            s0 = ep.keyframes[0].frame_idx if ep.keyframes else 0
            anc = current_anchor_idx(ep, s0)
            return (
                gr.update(minimum=0, maximum=max(1, ep.n_frames - 1), value=s0),
                _meta_md(ep),
                get_frame_image(ep, s0, "ext", video_cache,
                                overlay=overlay, anchor_kf=_anchor_kf(ep, s0)),
                get_frame_image(ep, s0, "wrist", video_cache),
                _anchor_md(ep, anc, s0),
                render_phase_ribbon(ep, s0),
                _audit_md(ep),
                _kf_table(ep),
            )

        ep_dropdown.change(
            _on_episode, inputs=[ep_dropdown, overlay_chk],
            outputs=[slider, meta_panel, ext_img, wrist_img,
                     anchor_panel, ribbon_plot, audit_panel, table],
        )

        def _on_ep_video(short_id, cam):
            return _ep_video(_resolve_ep(short_id), cam)

        ep_dropdown.change(_on_ep_video, inputs=[ep_dropdown, cam_radio],
                           outputs=[raw_video, raw_video_links])

        def _on_cam(short_id, cam):
            return _ep_video(_resolve_ep(short_id), cam)[0]

        cam_radio.change(_on_cam, inputs=[ep_dropdown, cam_radio],
                         outputs=[raw_video])

        if hint_editable:
            def _load_hint(short_id):
                ep = _resolve_ep(short_id)
                return (ep.hint, "",
                        f"This episode is a {_outcome_chip_html(ep.outcome)}")

            ep_dropdown.change(_load_hint, inputs=[ep_dropdown],
                               outputs=[hint_box, hint_status, hint_outcome])

            def _save_hint(short_id, text):
                ep = _resolve_ep(short_id)
                key = os.path.basename(ep.ep_dir.rstrip("/"))
                try:
                    save_hint_to_md(hints_path, key, text)
                    ep.hint = (text or "").strip()
                    return f"✅ saved hint for `{key}`"
                except Exception as e:
                    return f"⚠️ save failed: {e}"

            def _complete_hint(short_id, frame_idx, draft):
                ep = _resolve_ep(short_id)
                fi = int(frame_idx)
                imgs = []
                for view in ("ext", "wrist"):
                    im = get_frame_image(ep, fi, view, video_cache)
                    if im is not None:
                        try:
                            imgs.append(_jpeg_b64(im))
                        except Exception:
                            pass
                if not imgs:
                    return gr.update(), "⚠️ no frame image to send"
                try:
                    txt = llm_complete_hint(draft, ep.task_instruction, ep.outcome, imgs)
                    return txt, "✨ suggestion ready — edit if needed, then 💾 Save"
                except Exception as e:
                    return gr.update(), f"⚠️ completion failed: {e}"

            hint_complete.click(_complete_hint,
                                inputs=[ep_dropdown, slider, hint_box],
                                outputs=[hint_box, hint_status])

            def _default_hint(short_id, draft):
                # Unconditionally OVERWRITE the box with a generic hint and save.
                ep = _resolve_ep(short_id)
                task = (ep.task_instruction or "").strip() or "simple task"
                txt = (f"Routine/clear task: {task}. "
                       "Annotate exactly what the images show.")
                save_hint_to_md(hints_path, os.path.basename(ep.ep_dir.rstrip("/")), txt)
                ep.hint = txt
                return txt, "✓ default hint saved"

            def _exclude_ep(short_id, draft):
                # Confirm this episode is unusable (tagged, excluded on push).
                ep = _resolve_ep(short_id)
                txt = (draft or "").strip()
                if not txt.startswith("[[EXCLUDE]]"):
                    txt = ("[[EXCLUDE]] " + txt).strip()
                save_hint_to_md(hints_path, os.path.basename(ep.ep_dir.rstrip("/")), txt)
                ep.hint = txt
                return txt, "🚫 marked unusable"

            # ── finish current ep → jump to the next not-yet-stored one ──────
            def _goto(short_id, overlay, status=""):
                ep = _resolve_ep(short_id)
                s0 = ep.keyframes[0].frame_idx if ep.keyframes else 0
                anc = current_anchor_idx(ep, s0)
                vid, vid_md = _ep_video(ep, "ext")
                return [
                    gr.update(value=ep.short_id),
                    gr.update(minimum=0, maximum=max(1, ep.n_frames - 1), value=s0),
                    _meta_md(ep),
                    get_frame_image(ep, s0, "ext", video_cache,
                                    overlay=overlay, anchor_kf=_anchor_kf(ep, s0)),
                    get_frame_image(ep, s0, "wrist", video_cache),
                    _anchor_md(ep, anc, s0),
                    render_phase_ribbon(ep, s0),
                    _audit_md(ep),
                    _kf_table(ep),
                    vid, vid_md,
                    gr.update(value="ext"),
                    ep.hint,
                    f"This episode is a {_outcome_chip_html(ep.outcome)}",
                    status,
                ]

            def _next_unfinished(cur):
                choices = list(by_short.keys())
                stored = parse_hints_md(hints_path) if hints_path else {}
                def done(s):
                    return os.path.basename(by_short[s].ep_dir.rstrip("/")) in stored
                order = (choices[choices.index(cur) + 1:] + choices[:choices.index(cur) + 1]
                         if cur in choices else choices)
                return next((s for s in order if not done(s)), None)

            def _finish(cur, overlay):
                nxt = _next_unfinished(cur)
                if nxt is None:
                    return _goto(cur, overlay, "✓ every episode now has a hint / tag")
                return _goto(nxt, overlay, "→ next unfinished episode")

            FULL_OUT = [ep_dropdown, slider, meta_panel, ext_img, wrist_img,
                        anchor_panel, ribbon_plot, audit_panel, table,
                        raw_video, raw_video_links, cam_radio, hint_box,
                        hint_outcome, hint_status]

            # save / default / mark-unusable all = "this ep is finished" → advance
            hint_save.click(_save_hint, inputs=[ep_dropdown, hint_box],
                            outputs=[hint_status]).then(
                _finish, inputs=[ep_dropdown, overlay_chk], outputs=FULL_OUT)
            hint_default.click(_default_hint, inputs=[ep_dropdown, hint_box],
                               outputs=[hint_box, hint_status]).then(
                _finish, inputs=[ep_dropdown, overlay_chk], outputs=FULL_OUT)
            hint_exclude.click(_exclude_ep, inputs=[ep_dropdown, hint_box],
                               outputs=[hint_box, hint_status]).then(
                _finish, inputs=[ep_dropdown, overlay_chk], outputs=FULL_OUT)

        def _on_slider(short_id, frame_idx, overlay):
            ep = _resolve_ep(short_id)
            fi = int(frame_idx)
            anc = current_anchor_idx(ep, fi)
            return (
                get_frame_image(ep, fi, "ext", video_cache,
                                overlay=overlay, anchor_kf=_anchor_kf(ep, fi)),
                get_frame_image(ep, fi, "wrist", video_cache),
                _anchor_md(ep, anc, fi),
                render_phase_ribbon(ep, fi),
            )

        slider.change(
            _on_slider, inputs=[ep_dropdown, slider, overlay_chk],
            outputs=[ext_img, wrist_img, anchor_panel, ribbon_plot],
        )

        def _on_overlay(short_id, frame_idx, overlay):
            ep = _resolve_ep(short_id)
            fi = int(frame_idx)
            return get_frame_image(ep, fi, "ext", video_cache,
                                   overlay=overlay, anchor_kf=_anchor_kf(ep, fi))

        overlay_chk.change(
            _on_overlay, inputs=[ep_dropdown, slider, overlay_chk],
            outputs=[ext_img],
        )

    print(f"[v3 viewer] launching on http://0.0.0.0:{port}")
    allowed = [raw_root] if os.path.isdir(raw_root) else []
    demo.queue().launch(server_name="0.0.0.0", server_port=port,
                        prevent_thread_lock=False, share=False,
                        allowed_paths=allowed)


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--images-dir", required=True,
                    help="dir containing per-ep subdirs (meta.json + jpegs + "
                         "annotation_<suffix>.json each)")
    ap.add_argument("--suffix", required=True,
                    help="annotation filename suffix; viewer scans "
                         "ep*/annotation_<suffix>.json")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--include-unannotated", action="store_true",
                    help="Also load ep dirs that have meta.json but no "
                         "annotation_<suffix>.json (preview mode for "
                         "hint-writing on unannotated episodes).")
    ap.add_argument("--hints", default=None,
                    help="Path to hints.md (markdown with `## <ep_dir_name>` "
                         "sections). When provided, hint text is rendered "
                         "in a yellow callout above each ep's description.")
    ap.add_argument("--pose-delta", action="store_true",
                    help="Pre-compute per-anchor pose_delta via tools.py "
                         "(needs raw h5 + MP4 reachable; controlled by "
                         "DROID_RAW_ROOT env). Renders robot+wrist motion "
                         "vectors in the anchor panel.")
    ap.add_argument("--pose-overlay", action="store_true",
                    help="Load per-frame wrist-frame axis projection "
                         "(axis_overlay.json, else live via pose_overlay.py + h5). "
                         "Enables the optional pose-axis gizmo toggle on the ext view.")
    ap.add_argument("--load-video", action="store_true",
                    help="Decode each ep's MP4 streams into memory so the "
                         "slider shows the actual frame (not nearest keyframe). "
                         "Requires --droid-raw-root. ~50 KB/frame × 2 views × N eps.")
    ap.add_argument("--droid-raw-root", default=None,
                    help="Path to the raw DROID dataset root "
                         "(contains LAB/{success,failure}/DATE/.../trajectory.h5). "
                         "Defaults to $DROID_RAW_ROOT or "
                         "~/datasets/droid_raw/1.0.1.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    eps = load_v3_episodes(args.images_dir, args.suffix,
                           max_episodes=args.max_episodes,
                           include_unannotated=args.include_unannotated,
                           hints_path=args.hints)
    n_ann = sum(1 for e in eps if e.description != "(no annotation yet — preview for hint-writing)")
    n_pre = len(eps) - n_ann
    n_hint = sum(1 for e in eps if e.hint)
    print(f"[v3 viewer] loaded {len(eps)} eps from {args.images_dir} "
          f"(suffix={args.suffix}; {n_ann} annotated"
          + (f", {n_pre} preview" if n_pre else "")
          + (f"; {n_hint} have hints" if args.hints else "")
          + ")")
    if args.pose_delta:
        print(f"[v3 viewer] pre-computing pose deltas for {len(eps)} eps ...")
        for ep in eps:
            _fill_pose_deltas(ep)
    if args.pose_overlay:
        print(f"[v3 viewer] loading pose-axis overlays for {len(eps)} eps ...")
        for ep in eps:
            _fill_axis_overlay(ep)
        n_ov = sum(1 for e in eps if e.axis_overlay)
        print(f"[v3 viewer] pose-axis overlay: {n_ov}/{len(eps)} eps ready")
    if not eps:
        sys.exit(f"No annotation_{args.suffix}.json files under {args.images_dir}.")

    raw_root = (args.droid_raw_root
                or os.environ.get("DROID_RAW_ROOT")
                or "~/datasets/droid_raw/1.0.1")
    video_cache = None
    if args.load_video:
        video_cache = VideoCache(raw_root)
        # Eagerly initialize each ep — uses frames/<view>/fNNNN.jpg if present,
        # else falls back to in-memory MP4 decode under DROID_RAW_ROOT.
        print(f"[v3 viewer] loading video cache "
              f"(prefer ep_dir/frames/; fallback MP4 under {video_cache.droid_raw_root}) ...")
        for ep in eps:
            video_cache.ensure_loaded(ep.episode_id, ep_dir=ep.ep_dir)
        n_ok = len(video_cache._by_episode_id)
        print(f"[v3 viewer] video cache: {n_ok}/{len(eps)} eps ready "
              f"(missing: {len(eps) - n_ok})")

    build_ui(eps, args.port, video_cache=video_cache, hints_path=args.hints,
             raw_root=raw_root)


if __name__ == "__main__":
    main()
