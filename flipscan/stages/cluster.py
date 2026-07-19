"""Stage 4: cluster — group frames into page identities.

What real footage taught us (dense text pages defeat perceptual hashing —
same-page and different-page distances overlap):

- WITHIN a video, the reliable page-boundary signal is a TURN EVENT: a
  sustained run of high-motion frames. Segments separated by a real turn are
  different pages; separated by a brief wobble, the same page.
- ACROSS videos, the reliable identity signal is the PRINTED PAGE NUMBER read
  at transcription. Videos' page sequences are simply concatenated here;
  after transcription, pages sharing a printed number are deduplicated (best
  capture wins) and everything is ordered by number. So you can keep adding
  videos until every page is covered.
- A capture that shows an open two-page spread yields ONE page: the vision
  model is instructed to transcribe only the flat, readable side (the curled
  side gets its own flat capture in another pass and dedupes by number).
"""

from __future__ import annotations

import statistics

from ..imaging import majority_hash
from ..workspace import Workspace
from .score import load_scores


def _rest_mask(records: list[dict], spike_factor: float) -> tuple[list[bool], float]:
    motions = [r["motion"] for r in records[1:]]
    if not motions:
        return [True] * len(records), 0.0
    med = statistics.median(motions)
    threshold = max(med * spike_factor, 1e-3)
    mask = [(i == 0 or r["motion"] < threshold) and r["flatness"] > 0.05
            for i, r in enumerate(records)]
    return mask, threshold


def segment_pages(records: list[dict], spike_factor: float,
                  turn_min_frames: int = 4, min_seg: int = 2) -> list[list[int]]:
    """Split a video's frames into per-page index groups.

    Consecutive at-rest segments merge into one page unless the gap between
    them contains a sustained high-motion run (>= turn_min_frames) — a page
    turn. Tiny high-motion blips are hand wobble, not turns.
    """
    mask, _ = _rest_mask(records, spike_factor)

    pages: list[list[int]] = []
    current: list[int] = []
    gap_run = 0
    for i, at_rest in enumerate(mask):
        if at_rest:
            if gap_run >= turn_min_frames and current:
                pages.append(current)
                current = []
            gap_run = 0
            current.append(i)
        else:
            gap_run += 1
    if current:
        pages.append(current)

    # drop obvious debris: very short AND clearly above the rest noise floor
    rest_motions = [records[i]["motion"] for p in pages for i in p
                    if records[i]["motion"] > 0]
    floor = statistics.median(rest_motions) if rest_motions else 0.0
    return [
        p for p in pages
        if len(p) >= min_seg
        and (len(p) >= 4
             or statistics.fmean(records[i]["motion"] for i in p) <= 2.0 * floor)
    ]


def run(ws: Workspace, cfg: dict, log=print) -> None:
    min_frames = cfg["cluster"]["min_cluster_frames"]
    turn_min = cfg["cluster"].get("turn_min_frames", 4)

    old_pages = ws.manifest["pages"]
    pinned = [p for p in old_pages if p.get("pinned") or p.get("role") == "cover"]
    patched = [p for p in old_pages
               if p.get("patched_source") and p not in pinned and p.get("cluster_frames")]

    entries: list[dict] = []
    for video in ws.manifest["videos"]:
        vid = video["id"]
        records = load_scores(ws, vid)
        clusters = segment_pages(records, cfg["cluster"]["motion_spike_factor"],
                                 turn_min_frames=turn_min)
        seq = [
            {
                "video": vid,
                "frames": [f"{vid}_{records[i]['frame']}" for i in c],
                "hash": majority_hash([int(records[i]["phash"], 16) for i in c]),
                "suspect": len(c) < min_frames,
            }
            for c in clusters
        ]
        if video.get("direction") == "reverse":
            seq.reverse()
        log(f"  {vid}: {len(clusters)} page captures")
        entries.extend(seq)

    pages = []
    for idx, c in enumerate(entries):
        pages.append({
            "id": f"p{idx:04d}",
            "video": c["video"],
            "cluster_frames": c["frames"],
            "canonical": None,
            "scores": {},
            "status": "suspect" if c["suspect"] else "ok",
            "printed_number": None,
        })

    # re-attach patched replacements (they share frames with a rebuilt page)
    for pp in patched:
        own = set(pp["cluster_frames"])
        for i, p in enumerate(pages):
            if own & set(p["cluster_frames"]):
                pp["cluster_frames"] = p["cluster_frames"]
                pages[i] = pp
                break
        else:
            pages.append(pp)

    for pp in pinned:
        if pp.get("pinned") == "start" or (pp.get("role") == "cover"
                                           and pp.get("pinned") != "end"):
            pages.insert(0, pp)
        else:
            pages.append(pp)

    ws.manifest["pages"] = pages

    warnings: list[str] = []
    expected = ws.manifest["book"].get("expected_pages")
    detected = sum(1 for p in pages if not p.get("role"))
    if expected and detected != expected:
        warnings.append(
            f"found {detected} pages so far, expected {expected} — duplicates "
            f"collapse and order settles once transcription reads page numbers")
    for w in warnings:
        log(f"  WARNING: {w}")
    ws.stage_done("cluster", page_count=len(pages), warnings=warnings)
    log(f"  {len(pages)} page captures total (duplicates merge after "
        f"transcription via printed page numbers)")
