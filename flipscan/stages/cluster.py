"""Stage 4: cluster — group frames into page identities, per video, then parity-merge.

Approach per video:
1. Rest frames = motion below median * spike_factor (turns are big spikes).
2. Consecutive rest frames form segments (a page lying flat between turns).
3. Adjacent segments whose majority pHash is within threshold merge into one
   cluster (page wobbled or paused twice).
4. Clusters in chronological order, direction-normalized, map to the video's
   parity sequence (odd: 1,3,5... / even: 2,4,6... / all: 1,2,3...).
Then zip parities into the full page order and flag suspects.
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


def _cluster_video(records: list[dict], cfg: dict) -> list[list[int]]:
    """Merge rest segments into page clusters by pHash similarity."""
    threshold = cfg["cluster"]["hash_threshold"]
    segs = _segments(records, cfg["cluster"]["motion_spike_factor"])
    clusters: list[list[int]] = []
    cluster_hashes: list[int] = []
    for seg in segs:
        seg_hash = majority_hash([int(records[i]["phash"], 16) for i in seg])
        if clusters and hamming(seg_hash, cluster_hashes[-1]) <= threshold:
            clusters[-1].extend(seg)
            cluster_hashes[-1] = majority_hash(
                [int(records[i]["phash"], 16) for i in clusters[-1]]
            )
        else:
            clusters.append(list(seg))
            cluster_hashes.append(seg_hash)
    return clusters


def run(ws: Workspace, cfg: dict, log=print) -> None:
    min_frames = cfg["cluster"]["min_cluster_frames"]

    # per-video cluster sequences, direction-normalized to ascending page order
    sequences: dict[str, list[dict]] = {}
    for video in ws.manifest["videos"]:
        vid = video["id"]
        records = load_scores(ws, vid)
        clusters = _cluster_video(records, cfg)
        log(f"  {vid}: {len(clusters)} page clusters "
            f"({video['pages']} pages, direction={video['direction']})")
        seq = [
            {
                "video": vid,
                "frames": [f"{vid}_{records[i]['frame']}" for i in c],
                "suspect": len(c) < min_frames,
            }
            for c in clusters
        ]
        if video["direction"] == "reverse":
            seq.reverse()
        sequences[video["pages"]] = seq

    # merge parities into full page order
    warnings: list[str] = []
    if "odd" in sequences and "even" in sequences:
        odd, even = sequences["odd"], sequences["even"]
        if abs(len(odd) - len(even)) > 1:
            warnings.append(
                f"parity cluster counts differ by {abs(len(odd) - len(even))} "
                f"(odd={len(odd)}, even={len(even)}) — expect a merge misalignment"
            )
        merged = []
        for i in range(max(len(odd), len(even))):
            if i < len(odd):
                merged.append(odd[i])
            if i < len(even):
                merged.append(even[i])
    elif "all" in sequences:
        merged = sequences["all"]
    else:
        # single parity video only — half a book, but proceed
        merged = next(iter(sequences.values()))
        warnings.append("only one parity captured; book is missing half its pages")

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
    ws.manifest["pages"] = pages

    expected = ws.manifest["book"].get("expected_pages")
    if expected and abs(len(pages) - expected) > 0:
        warnings.append(f"found {len(pages)} pages, expected {expected}")
    for w in warnings:
        log(f"  WARNING: {w}")
    ws.stage_done("cluster", page_count=len(pages), warnings=warnings)
    log(f"  {len(pages)} pages total, "
        f"{sum(1 for p in pages if p['status'] == 'suspect')} suspect")
