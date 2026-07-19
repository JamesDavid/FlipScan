"""Stage 4: cluster — group frames into page identities, then merge across videos.

Per video:
1. Rest frames = motion below median * spike_factor (turns are big spikes).
2. Consecutive rest frames form segments (a page lying flat between turns).
3. Adjacent segments whose majority pHash is within threshold merge into one
   cluster (page wobbled or paused twice).

Across videos:
- Any number of videos, shot in any direction, covering any subset of pages.
  Each video's clusters are matched against the pages found so far by pHash;
  a match means the same page was captured again, and its frames join the page
  so `select` can pick the best capture overall. Unmatched clusters are
  inserted in sequence position. Orientation of each additional video is
  inferred from the match order.
- Printed page numbers (read at transcription) refine the final order, so you
  can keep adding videos until every page is covered.
"""

from __future__ import annotations

import statistics

from ..imaging import hamming, majority_hash
from ..workspace import Workspace
from .score import load_scores


def _segments(records: list[dict], spike_factor: float,
              min_seg: int = 3) -> list[list[int]]:
    """Indices of consecutive at-rest frames.

    Short segments whose motion sits well above the rest-frame noise floor are
    mid-turn debris (motion briefly dipping under the spike threshold) and are
    dropped; short but genuinely still segments survive (flagged suspect later).
    """
    motions = [r["motion"] for r in records[1:]]  # first frame has motion 0
    if not motions:
        return []
    med = statistics.median(motions)
    threshold = max(med * spike_factor, 1e-3)
    segs, cur = [], []
    for i, r in enumerate(records):
        at_rest = (i == 0 or r["motion"] < threshold) and r["flatness"] > 0.05
        if at_rest:
            cur.append(i)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)

    long_segs = [s for s in segs if len(s) >= min_seg]
    rest_motions = [records[i]["motion"] for s in long_segs for i in s
                    if records[i]["motion"] > 0]
    noise_floor = statistics.median(rest_motions) if rest_motions else 0.0
    typical_len = statistics.median(len(s) for s in long_segs) if long_segs else min_seg
    return [
        s for s in segs
        if len(s) >= 0.5 * typical_len
        or statistics.fmean(records[i]["motion"] for i in s) <= 1.5 * noise_floor
    ]


def _cluster_video(records: list[dict], cfg: dict) -> list[dict]:
    """Merge rest segments into page clusters by pHash similarity."""
    threshold = cfg["cluster"]["hash_threshold"]
    segs = _segments(records, cfg["cluster"]["motion_spike_factor"])
    clusters: list[dict] = []
    for seg in segs:
        seg_hash = majority_hash([int(records[i]["phash"], 16) for i in seg])
        if clusters and hamming(seg_hash, clusters[-1]["hash"]) <= threshold:
            clusters[-1]["idx"].extend(seg)
        else:
            clusters.append({"idx": list(seg), "hash": seg_hash})
    for c in clusters:
        c["hash"] = majority_hash([int(records[i]["phash"], 16) for i in c["idx"]])
    return clusters


def _best_match(cluster: dict, merged: list[dict], threshold: int) -> int | None:
    best_i, best_d = None, threshold + 1
    for mi, m in enumerate(merged):
        d = hamming(cluster["hash"], m["hash"])
        if d < best_d:
            best_i, best_d = mi, d
    return best_i


def auto_merge(sequences: list[list[dict]], threshold: int) -> list[dict]:
    """Merge per-video cluster sequences into one page order.

    Each cluster dict: {"frames": [frame ids], "hash": int, "suspect": bool}.
    Matched clusters pool their frames (best capture wins at select time);
    unmatched clusters are inserted where their sequence neighbors landed.
    A video whose matches run backwards is auto-reversed.
    """
    merged: list[dict] = []
    for seq in sequences:
        if not merged:
            merged = [dict(c) for c in seq]
            continue

        matched = [(si, mi) for si, c in enumerate(seq)
                   if (mi := _best_match(c, merged, threshold)) is not None]
        if len(matched) >= 2:
            inc = sum(1 for a, b in zip(matched, matched[1:]) if b[1] > a[1])
            dec = sum(1 for a, b in zip(matched, matched[1:]) if b[1] < a[1])
            if dec > inc:
                seq = list(reversed(seq))

        cursor = len(merged)  # default: unmatched pages go to the end
        for si, c in enumerate(seq):
            mi = _best_match(c, merged, threshold)
            if mi is not None:
                cursor = mi  # place any unmatched predecessors before this match
                break

        for c in seq:
            mi = _best_match(c, merged, threshold)
            if mi is not None:
                merged[mi]["frames"] = merged[mi]["frames"] + c["frames"]
                merged[mi]["suspect"] = merged[mi]["suspect"] and c["suspect"]
                cursor = mi + 1
            else:
                merged.insert(cursor, dict(c))
                cursor += 1
    return merged


def run(ws: Workspace, cfg: dict, log=print) -> None:
    min_frames = cfg["cluster"]["min_cluster_frames"]
    threshold = cfg["cluster"]["hash_threshold"]

    old_pages = ws.manifest["pages"]
    # manual pages survive re-clustering: photos added via addpage (pinned) and
    # patched replacements (matched back onto their cluster below)
    pinned = [p for p in old_pages if p.get("pinned") or p.get("role") == "cover"]
    patched = [p for p in old_pages
               if p.get("patched_source") and p not in pinned and p.get("cluster_frames")]

    sequences: list[list[dict]] = []
    for video in ws.manifest["videos"]:
        vid = video["id"]
        records = load_scores(ws, vid)
        clusters = _cluster_video(records, cfg)
        log(f"  {vid}: {len(clusters)} page clusters")
        seq = [
            {
                "frames": [f"{vid}_{records[i]['frame']}" for i in c["idx"]],
                "hash": c["hash"],
                "suspect": len(c["idx"]) < min_frames,
            }
            for c in clusters
        ]
        if video.get("direction") == "reverse":
            seq.reverse()
        sequences.append(seq)

    merged = auto_merge(sequences, threshold)

    pages = []
    for idx, c in enumerate(merged):
        pages.append({
            "id": f"p{idx:04d}",
            "cluster_frames": c["frames"],
            "canonical": None,
            "scores": {},
            "status": "suspect" if c["suspect"] else "ok",
            "printed_number": None,
        })

    # re-attach patched replacements: a patched page owns the merged cluster it
    # shares frames with
    for pp in patched:
        own = set(pp["cluster_frames"])
        for i, p in enumerate(pages):
            if own & set(p["cluster_frames"]):
                pp["cluster_frames"] = p["cluster_frames"]
                pages[i] = pp
                break
        else:
            pages.append(pp)

    # re-attach pinned photo pages (covers etc.)
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
        warnings.append(f"found {detected} pages, expected {expected}")
    for w in warnings:
        log(f"  WARNING: {w}")
    ws.stage_done("cluster", page_count=len(pages), warnings=warnings)
    log(f"  {len(pages)} pages total, "
        f"{sum(1 for p in pages if p['status'] == 'suspect')} suspect")
