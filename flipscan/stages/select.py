"""Stage 5: select — pick the best frame per cluster + build a debug contact sheet."""

from __future__ import annotations

import cv2
import numpy as np

from ..workspace import Workspace
from .score import scores_by_frame_id


def composite(rec: dict, norms: dict, w: dict) -> float:
    """Weighted product of normalized sub-scores; higher is better."""
    sharp = min(rec["sharpness"] / norms["sharp_hi"], 1.0) if norms["sharp_hi"] else 0.0
    flat = rec["flatness"]
    occl = 1.0 - min(rec["occlusion"] * 4.0, 1.0)  # 25% thumb coverage zeroes it
    motion = 1.0 / (1.0 + rec["motion"] / norms["motion_med"]) if norms["motion_med"] else 1.0
    return (
        max(sharp, 1e-6) ** w["w_sharpness"]
        * max(flat, 1e-6) ** w["w_flatness"]
        * max(occl, 1e-6) ** w["w_occlusion"]
        * max(motion, 1e-6) ** w["w_motion"]
    )


def frame_path(ws: Workspace, frame_id: str):
    vid, frame = frame_id.split("_", 1)
    return ws.frames_dir(vid) / f"{frame}.jpg"


def run(ws: Workspace, cfg: dict, log=print) -> None:
    scores = scores_by_frame_id(ws)
    w = cfg["score"]

    all_sharp = [r["sharpness"] for r in scores.values()]
    all_motion = [r["motion"] for r in scores.values() if r["motion"] > 0]
    norms = {
        "sharp_hi": float(np.percentile(all_sharp, 95)) if all_sharp else 0.0,
        "motion_med": float(np.median(all_motion)) if all_motion else 1.0,
    }

    for page in ws.manifest["pages"]:
        if not page.get("cluster_frames") or page.get("patched_source"):
            continue  # photo-added or patched pages keep their capture
        ranked = sorted(
            page["cluster_frames"],
            key=lambda fid: composite(scores[fid], norms, w),
            reverse=True,
        )
        page["canonical"] = ranked[0]
        page["fallback"] = ranked[1] if len(ranked) > 1 else None
        best = scores[ranked[0]]
        page["scores"] = {
            "composite": round(composite(best, norms, w), 4),
            "sharpness": best["sharpness"],
            "flatness": best["flatness"],
            "occlusion": best["occlusion"],
            "motion": best["motion"],
        }

    # restore cached transcriptions for pages whose best capture is unchanged
    # (re-clustering renumbers pages; the cache is keyed by capture identity)
    from .transcribe import cache_key, load_cache
    cache = load_cache(ws)
    restores = []  # read all contents first: targets may overwrite sources
    for page in ws.manifest["pages"]:
        if page.get("md"):
            continue
        rec = cache.get(cache_key(page) or "")
        if rec and (ws.root / rec["md"]).exists():
            restores.append(
                (page, rec, (ws.root / rec["md"]).read_text(encoding="utf-8")))
    for page, rec, content in restores:
        target = f"pages/{page['id']}.md"
        (ws.root / target).write_text(content, encoding="utf-8")
        page.update({k: rec[k] for k in
                     ("printed_number", "confidence", "regions", "flags",
                      "transcribed_by")})
        page["md"] = target

    # deletions persist across re-clustering, keyed by capture identity
    from .transcribe import cache_key as _ck
    deleted = set(ws.manifest.get("deleted_captures", []))
    if deleted:
        for page in ws.manifest["pages"]:
            if _ck(page) in deleted:
                page["status"] = "deleted"

    # bottom-percentile composite scores get flagged suspect
    composites = [p["scores"]["composite"] for p in ws.manifest["pages"]
                  if p.get("scores") and p.get("status") != "deleted"]
    if composites:
        cutoff = float(np.percentile(composites, cfg["cluster"]["suspect_score_percentile"]))
        for p in ws.manifest["pages"]:
            if (p.get("scores") and p["scores"]["composite"] < cutoff * 0.5
                    and not p.get("suspect_ignored")):
                p["status"] = "suspect"  # well below the low tail

    ws.save()
    sheet = contact_sheet(ws)
    log(f"  canonical frames chosen for {len(ws.manifest['pages'])} pages")
    log(f"  contact sheet: {sheet}")
    ws.stage_done("select")


def contact_sheet(ws: Workspace, thumb_w: int = 240, cols: int = 8):
    """Grid of canonical frames with page ids — eyeball cluster quality here."""
    pages = ws.manifest["pages"]
    if not pages:
        return None
    rotations = {v["id"]: v.get("rotate", 0) for v in ws.manifest["videos"]}
    thumbs = []
    for p in pages:
        if p.get("side") == "right" or p.get("status") == "deleted":
            continue  # one thumbnail per capture, not per spread half
        if p.get("canonical"):
            img = cv2.imread(str(frame_path(ws, p["canonical"])), cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None and rotations.get(p["canonical"].split("_")[0]) == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
        elif p.get("color") and (ws.root / p["color"]).exists():
            img = cv2.imread(str(ws.root / p["color"]), cv2.IMREAD_REDUCED_COLOR_2)
        else:
            img = None
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        t = cv2.resize(img, (thumb_w, int(img.shape[0] * scale)))
        score = p["scores"].get("composite") if p.get("scores") else None
        label = f"{p['id']} {score:.2f}" if score is not None else p["id"]
        color = (0, 0, 255) if p["status"] == "suspect" else (0, 200, 0)
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), color, 2)
        cv2.putText(t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        thumbs.append(t)
    if not thumbs:
        return None
    th = max(t.shape[0] for t in thumbs)
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        row = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in row]
        while len(row) < cols:
            row.append(np.full((th, thumb_w, 3), 30, np.uint8))
        rows.append(cv2.hconcat(row))
    out = ws.work_file("contact_sheet.jpg")
    cv2.imwrite(str(out), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
