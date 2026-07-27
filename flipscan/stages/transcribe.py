"""Stage 7: transcribe — vision LLM per page, with hybrid escalation and
printed-page-number monotonicity checking (the most reliable gap detector)."""

from __future__ import annotations

import json
import re

from ..backends import get_backend, needs_escalation
from ..workspace import Workspace

# transcription results cached by capture identity, so re-clustering (e.g. after
# adding another video) only re-transcribes pages whose best frame changed
CACHE_FILE = "transcriptions.json"


def cache_key(page: dict) -> str | None:
    base = page.get("patched_source") or page.get("canonical")
    if base and page.get("side"):
        return f"{base}|{page['side']}"  # spread halves share the canonical frame
    return base


def load_cache(ws: Workspace) -> dict:
    path = ws.work_file(CACHE_FILE)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(ws: Workspace, cache: dict) -> None:
    with open(ws.work_file(CACHE_FILE), "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _merge_regions(old: list | None, fresh: list) -> list:
    """A fresh transcription re-detects figure regions from the page image, but
    a user's own close-up (own_image) or hand-placed crop (user_crop) is NOT
    something the model can reproduce — it's a separate photo or a deliberate
    box. Preserve those at their original index (the index is baked into the
    figure filename, e.g. pXXXX_b.png) and only take the model's regions for
    the slots the user never touched. Without this, a retry-OCR or any
    re-transcribe silently destroys re-acquired figure close-ups."""
    old = old or []
    if not any(r.get("own_image") or r.get("user_crop") for r in old):
        return fresh
    merged = list(fresh or [])
    for i, r in enumerate(old):
        if r.get("own_image") or r.get("user_crop"):
            while len(merged) <= i:
                merged.append({"type": "figure", "bbox_norm": [0, 0, 1, 1],
                               "caption": "", "deleted": True})
            merged[i] = r
    return merged


def _write_result(ws: Workspace, page: dict, result: dict, backend_name: str) -> None:
    if "error" in result:
        page["status"] = "suspect"
        page["transcribe_error"] = result["error"]
        return
    md_path = ws.dir("pages") / f"{page['id']}.md"
    md_path.write_text(result["markdown"], encoding="utf-8")
    page["md"] = f"pages/{page['id']}.md"
    if not page.get("number_manual"):  # a user-entered number outranks the model
        page["printed_number"] = result["page_number_printed"]
    page["confidence"] = result["confidence"]
    page["regions"] = _merge_regions(page.get("regions"), result["regions"])
    page["flags"] = result["flags"]
    page["transcribed_by"] = backend_name
    page.pop("transcribe_error", None)
    # multi_column alone is layout info, not a quality problem — the model
    # reads both columns; the page just deserves a reading-order glance
    real_flags = [f for f in result["flags"] if f != "multi_column"]
    if ((result["confidence"] == "low" or real_flags)
            and not page.get("suspect_ignored")):
        page["status"] = "suspect"


def _is_pinned_start(p: dict) -> bool:
    return p.get("pinned") == "start" or (p.get("role") == "cover"
                                          and p.get("pinned") != "end")


def reorder_by_printed(pages: list[dict]) -> list[dict]:
    """Sort pages by printed page number where known; unnumbered pages keep
    their position via interpolated keys. Pinned pages (covers etc.) stay put."""
    start = [p for p in pages if _is_pinned_start(p)]
    end = [p for p in pages if p.get("pinned") == "end"]
    middle = [p for p in pages if p not in start and p not in end]

    n = len(middle)
    nums = [p.get("printed_number") for p in middle]
    if not any(v is not None for v in nums):
        return pages  # nothing to order by

    # an unnumbered page's best position comes from its OWN video's capture
    # order (two opposite-direction passes interleave by page number, so a
    # global predecessor may belong to the other pass entirely)
    def video_key(p) -> float | None:
        vid, f = p.get("video"), _frame_no(p)
        if vid is None or f is None:
            return None
        mates = sorted((q for q in middle
                        if q.get("video") == vid and _frame_no(q) is not None
                        and q.get("printed_number") is not None),
                       key=_frame_no)
        before = [q for q in mates if _frame_no(q) < f]
        after = [q for q in mates if _frame_no(q) > f]
        a = before[-1]["printed_number"] if before else None
        b = after[0]["printed_number"] if after else None
        if a is not None and b is not None:
            return (a + b) / 2
        if a is not None:
            return a + 0.5 if not before or a >= (b or a) else a - 0.5
        if b is not None:
            return b - 0.5
        return None

    keys: list[float] = []
    first_num = next(float(v) for v in nums if v is not None)
    for i in range(n):
        if nums[i] is not None:
            keys.append(float(nums[i]))
            continue
        vk = video_key(middle[i])
        if vk is not None:
            keys.append(vk + i * 1e-6)  # stable within ties
        elif keys:
            keys.append(keys[-1] + 0.001)
        else:
            keys.append(first_num - 0.001 * (n - i))  # leading unnumbered pages
    order = sorted(range(n), key=lambda i: keys[i])  # stable
    return start + [middle[i] for i in order] + end


def capture_rank(p: dict, mtime_of=None) -> tuple:
    """How good a capture is, for picking dedupe winners: a deliberate photo
    (patched) beats any video frame; among photos, the most recent retake
    wins; then transcription confidence, then frame score."""
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    patched = 1 if p.get("patched_source") else 0
    mt = mtime_of(p) if (patched and mtime_of) else 0.0
    return (patched, mt, conf_rank.get(p.get("confidence"), 0),
            (p.get("scores") or {}).get("composite", 0.0))


def _reset_dup(p: dict) -> None:
    if p["status"] == "duplicate":
        p["status"] = "patched" if p.get("patched_source") else "ok"


def dedupe_by_printed(pages: list[dict], log=print, mtime_of=None) -> int:
    """Pages sharing a printed page number are the same page captured more
    than once (multiple videos, a re-detected rest, or a wizard retake next
    to the old capture). Keep the best capture, mark the rest
    status="duplicate" (excluded from the book, shown in review)."""
    groups: dict[tuple, list[dict]] = {}
    for p in pages:
        n = p.get("printed_number")
        # PDF pages are unique-by-position; their numbers may repeat across
        # concatenated works, so never merge them by printed number
        if (n is None or p.get("role") or p.get("status") == "deleted"
                or p.get("source") == "pdf"):
            continue
        groups.setdefault((n,), []).append(p)

    deduped = 0
    for group in groups.values():
        # a previous run may have marked duplicates; re-decide from scratch —
        # except pages the user explicitly rescued ("not a duplicate") or
        # manually marked duplicate (their call stands)
        group = [p for p in group
                 if not p.get("dedupe_exempt") and not p.get("manual_duplicate")]
        for p in group:
            _reset_dup(p)
        if len(group) < 2:
            continue
        group.sort(key=lambda p: capture_rank(p, mtime_of), reverse=True)
        for p in group[1:]:
            p["status"] = "duplicate"
            deduped += 1
    if deduped:
        log(f"  merged {deduped} duplicate captures (same printed page number; "
            f"best capture kept)")
    return deduped


def dedupe_by_content(pages: list[dict], get_text, log=print,
                      mtime_of=None) -> int:
    """Adjacent pages with near-identical transcriptions are the same physical
    page captured twice where one copy lost its number — the printed-number
    dedupe can't see those. Keep the best capture, hide the twin."""
    from difflib import SequenceMatcher

    def squash(pid_page):
        t = get_text(pid_page) or ""
        return re.sub(r"[^a-z0-9]", "", t.lower())[:2500]

    body = _body_pages(pages)
    deduped = 0
    for a, b in zip(body, body[1:]):
        if (a["status"] == "duplicate" or b["status"] == "duplicate"
                or a.get("dedupe_exempt") or b.get("dedupe_exempt")
                or a.get("manual_duplicate") or b.get("manual_duplicate")):
            continue
        sa, sb = squash(a), squash(b)
        if len(sa) < 300 or len(sb) < 300:
            continue
        if SequenceMatcher(None, sa, sb).ratio() < 0.92:
            continue
        loser = min(a, b, key=lambda p: capture_rank(p, mtime_of))
        loser["status"] = "duplicate"
        loser["content_duplicate"] = True
        deduped += 1
    if deduped:
        log(f"  merged {deduped} near-identical page pairs (same content, "
            f"one copy unnumbered)")
    return deduped


def _body_pages(pages: list[dict]) -> list[dict]:
    return [p for p in pages
            if not p.get("role") and not p.get("pinned")
            and p.get("status") not in ("duplicate", "deleted")]


def _frame_no(page: dict) -> int | None:
    fid = page.get("canonical")
    if not fid or "_f" not in fid:
        return None
    try:
        return int(fid.rsplit("_f", 1)[1])
    except ValueError:
        return None


def sanitize_numbers_by_video(pages: list[dict], log=print) -> int:
    """Within one video, capture order is ground truth: printed numbers must run
    monotonically (ascending or descending). A number that breaks its video's
    sequence — e.g. reading '266' between neighbors 204 and 208 — is a misread:
    reject it so position-based inference can assign the real number (206).

    Keeps the maximum-weight monotonic subsequence per video (user-entered
    numbers are effectively unremovable anchors); everything else is cleared."""
    by_video: dict[str, list[dict]] = {}
    for p in _body_pages(pages):
        if p.get("printed_number") is None or _frame_no(p) is None:
            continue
        by_video.setdefault(p.get("video") or "?", []).append(p)

    rejected = 0
    for vid, group in by_video.items():
        group.sort(key=_frame_no)
        nums = [p["printed_number"] for p in group]
        weights = [1000 if p.get("number_manual") else 1 for p in group]
        n = len(group)
        if n < 3:
            continue

        def best_monotonic(sign: int, wts: list[int]) -> tuple[int, list[int]]:
            # O(n^2) weighted longest non-decreasing (sign=1) / non-increasing
            best = list(wts)
            prev = [-1] * n
            for i in range(n):
                for j in range(i):
                    if (nums[i] - nums[j]) * sign >= 0 and best[j] + wts[i] > best[i]:
                        best[i] = best[j] + wts[i]
                        prev[i] = j
            end = max(range(n), key=lambda i: best[i])
            keep, i = [], end
            while i != -1:
                keep.append(i)
                i = prev[i]
            return best[end], keep

        def pick(wts):
            asc_w, asc_keep = best_monotonic(1, wts)
            desc_w, desc_keep = best_monotonic(-1, wts)
            return set(asc_keep if asc_w >= desc_w else desc_keep)

        keep = pick(weights)
        if any(w > 1 for w in weights):
            # a manual anchor must never cost the rest of the video its numbers:
            # if honoring it rejects noticeably more pages than ignoring it,
            # the anchor itself is the outlier — keep it on its own page only
            unweighted = pick([1] * n)
            if len(unweighted) > len(keep) + 3:
                log("  WARNING: a manually-set page number conflicts with its "
                    "video's sequence — keeping it, but not letting it reject "
                    "other pages")
                keep = unweighted | {i for i in range(n)
                                     if group[i].get("number_manual")}
        for i, p in enumerate(group):
            if i not in keep and not p.get("number_manual"):
                p["printed_number"] = None
                p["number_rejected"] = True
                p.pop("number_inferred", None)
                rejected += 1
    if rejected:
        log(f"  rejected {rejected} printed numbers that break their video's "
            f"page order (misreads) — re-inferring from neighbors")
    return rejected


def video_parity(pages: list[dict]) -> dict[str, int | None]:
    """One flip pass captures every other page: detect each video's page-number
    parity (0=even, 1=odd) when >80% of its read numbers agree; None if mixed."""
    by_video: dict[str, list[int]] = {}
    for p in _body_pages(pages):
        n = p.get("printed_number")
        if n is not None and n >= 1 and not p.get("number_inferred"):
            by_video.setdefault(p.get("video") or "?", []).append(n % 2)
    out: dict[str, int | None] = {}
    for vid, bits in by_video.items():
        if len(bits) >= 5:
            frac = sum(bits) / len(bits)
            out[vid] = 1 if frac > 0.8 else 0 if frac < 0.2 else None
        else:
            out[vid] = None
    return out


def infer_from_video_order(pages: list[dict], log=print) -> int:
    """Fill unnumbered pages from their CAPTURE-ORDER neighbors in the same
    video — the strongest signal there is. Neighbors 117, ?, 113 (descending,
    step 2) pin the unknown to 115 even when the page landed far away in the
    global order after its misread number was rejected."""
    by_video: dict[str, list[dict]] = {}
    for p in _body_pages(pages):
        if _frame_no(p) is not None:
            by_video.setdefault(p.get("video") or "?", []).append(p)

    inferred = 0
    for group in by_video.values():
        group.sort(key=_frame_no)
        numbered = [i for i, p in enumerate(group)
                    if p.get("printed_number") is not None]
        for a, b in zip(numbered, numbered[1:]):
            unknown = group[a + 1:b]
            if not unknown:
                continue
            span = group[b]["printed_number"] - group[a]["printed_number"]
            slots = len(unknown) + 1
            if span != 0 and span % slots == 0 and 1 <= abs(span // slots) <= 2:
                step = span // slots
                for k, p in enumerate(unknown, 1):
                    p["printed_number"] = group[a]["printed_number"] + k * step
                    p["number_inferred"] = True
                    p.pop("number_rejected", None)
                    inferred += 1
    if inferred:
        log(f"  inferred {inferred} page numbers from video capture order")
    return inferred


def infer_missing_numbers(pages: list[dict], log=print) -> int:
    """Assign printed numbers to unnumbered pages when their position makes it
    unambiguous: one unknown page sitting between printed 52 and 54 must be 53.
    Runs of unknowns that exactly fill a gap get numbered sequentially; runs
    that exceed the gap are contradictions — those pages go suspect."""
    body = _body_pages(pages)
    numbered_idx = [i for i, p in enumerate(body)
                    if p.get("printed_number") is not None]
    inferred = 0
    for a, b in zip(numbered_idx, numbered_idx[1:]):
        unknown = body[a + 1:b]
        if not unknown:
            continue
        span = body[b]["printed_number"] - body[a]["printed_number"]
        slots = len(unknown) + 1
        # uniform-step fit: step 1 = every page captured; step 2 = one flat
        # side per pass (204, ?, 208 -> 206); larger steps are too ambiguous
        if span > 0 and span % slots == 0 and 1 <= span // slots <= 2:
            step = span // slots
            parity = video_parity(pages)
            for k, p in enumerate(unknown, 1):
                value = body[a]["printed_number"] + k * step
                vp = parity.get(p.get("video") or "?")
                if vp is not None and value % 2 != vp:
                    continue  # this video only holds the other parity
                p["printed_number"] = value
                p["number_inferred"] = True
                inferred += 1
        elif 0 < span <= len(unknown):
            for p in unknown:  # more captures than pages fit here
                if p["status"] == "ok":
                    p["status"] = "suspect"
                p["number_conflict"] = True
    if inferred:
        log(f"  inferred printed numbers for {inferred} unnumbered pages "
            f"from their neighbors")
    return inferred


def compute_missing_pages(pages: list[dict]) -> list[int]:
    """Printed numbers absent from the captured range = pages never captured.
    Zero/negative numbers are user-assigned front-matter ordering (title page,
    TOC, foreword before page 1) — not part of the printed range."""
    nums = sorted({p["printed_number"] for p in _body_pages(pages)
                   if p.get("printed_number") is not None
                   and p["printed_number"] >= 1})
    if len(nums) < 2:
        return []
    return sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))


def format_ranges(nums: list[int]) -> str:
    """[6,7,14,22,23,24] -> '6-7, 14, 22-24'"""
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(out)


def check_printed_numbers(pages: list[dict]) -> list[str]:
    """Printed page numbers must increase monotonically in cluster order;
    violations point at missed/duplicated pages or a parity merge misalignment."""
    warnings = []
    prev_num, prev_id = None, None
    for p in pages:
        n = p.get("printed_number")
        if n is None or p.get("status") in ("duplicate", "deleted"):
            continue
        if prev_num is not None and n <= prev_num:
            warnings.append(
                f"printed page numbers not monotonic: {prev_id} has {prev_num}, "
                f"then {p['id']} has {n} — likely missed/duplicated page or merge error"
            )
            p["status"] = "suspect"
        prev_num, prev_id = n, p["id"]
    return warnings


def _cache_page(ws: Workspace, page: dict) -> None:
    key = cache_key(page)
    if key and page.get("md"):
        cache = load_cache(ws)
        cache[key] = {
            "md": page["md"], "printed_number": page.get("printed_number"),
            "confidence": page.get("confidence"), "regions": page.get("regions"),
            "flags": page.get("flags"), "transcribed_by": page.get("transcribed_by"),
        }
        save_cache(ws, cache)


def _run_incremental(ws: Workspace, backend, todo, by_id, log) -> dict[str, dict]:
    """Transcribe page by page, persisting after each one — a multi-hour local
    run must survive interruption (it resumes where it stopped).

    With ollama_concurrency > 1, N requests run in flight at once (only useful
    when the Ollama server itself allows parallel slots via OLLAMA_NUM_PARALLEL;
    otherwise they just queue server-side). Results still persist on the main
    thread as each page completes."""
    results = {}
    workers = getattr(backend, "concurrency", 1)

    def one(item):
        return backend.transcribe([item], log=lambda m: None)[item[0]]

    def record(item, r, i):
        results[item[0]] = r
        _write_result(ws, by_id[item[0]], r, backend.name)
        _cache_page(ws, by_id[item[0]])
        ws.save()
        log(f"  {backend.name}: {item[0]} "
            f"({'ok' if 'error' not in r else 'FAILED'}) [{i}/{len(todo)}]")

    def safe_record(item, r, i):
        try:
            record(item, r, i)
        except Exception as e:  # bookkeeping hiccup must not sink a long run
            log(f"  WARNING: failed to record {item[0]}: {e}")

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(one, item): item for item in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                item = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:  # a worker crash must not sink the run
                    r = {"error": str(e)}
                safe_record(item, r, i)
    else:
        for i, item in enumerate(todo, 1):
            safe_record(item, one(item), i)
    return results


def reconcile(ws: Workspace, pages: list[dict], log=print) -> list[dict]:
    """Post-transcription bookkeeping: reject numbers that break their video's
    capture order, collapse duplicates, order by number, infer unnumbered pages
    from neighbors, and compute the never-captured page list.

    Self-healing: every pass starts from the pristine model-read numbers in the
    transcription cache (manual numbers excepted), so no rejection or inference
    is ever permanent state — a bad edit or a bug can't poison the manifest."""
    cache = load_cache(ws)
    for p in pages:
        if p.get("role"):
            continue
        if p.get("number_manual"):
            # a hand-entered number is by definition neither misread nor
            # inferred — drop stale flags from before it became manual
            # (the later passes re-set them if a real conflict remains)
            for k in ("number_inferred", "number_rejected", "number_conflict"):
                p.pop(k, None)
            continue
        rec = cache.get(cache_key(p) or "")
        if rec is not None and p.get("md"):
            p["printed_number"] = rec.get("printed_number")
        elif p.get("number_inferred"):
            p["printed_number"] = None
        p.pop("number_inferred", None)
        p.pop("number_rejected", None)
        p.pop("number_conflict", None)
    def mtime_of(p: dict) -> float:
        src = ws.root / (p.get("patched_source") or "")
        try:
            return src.stat().st_mtime
        except OSError:
            return 0.0

    _md_cache: dict[str, str] = {}

    def get_text(p: dict) -> str:
        if p["id"] not in _md_cache:
            f = ws.root / (p.get("md") or "")
            _md_cache[p["id"]] = (f.read_text(encoding="utf-8")
                                  if p.get("md") and f.exists() else "")
        return _md_cache[p["id"]]

    # PDF-imported pages are AUTHORITATIVE order: a PDF is already in book
    # order and each page is unique, so printed numbers are informational only
    # and may legitimately repeat (e.g. several papers concatenated, each
    # starting at page 1). The video-capture reconcile (which identifies and
    # merges pages BY printed number) would wrongly collapse those — so when
    # the book is PDF-sourced, keep the pages exactly as imported.
    body = [p for p in pages if not p.get("role")
            and p.get("status") not in ("deleted",)]
    if body and all(p.get("source") == "pdf" for p in body):
        ws.manifest["pages"] = pages          # import order = book order
        ws.manifest["missing_pages"] = []     # repeating numbers -> no gap list
        ws.save()
        log("  PDF book: kept authoritative page order (printed numbers may "
            "repeat across concatenated works)")
        return pages

    sanitize_numbers_by_video(pages, log)
    infer_from_video_order(pages, log)  # strongest signal: capture order
    dedupe_by_printed(pages, log, mtime_of=mtime_of)
    pages = reorder_by_printed(pages)
    dedupe_by_content(pages, get_text, log, mtime_of=mtime_of)
    infer_missing_numbers(pages, log)   # cross-video leftovers
    pages = reorder_by_printed(pages)
    ws.manifest["pages"] = pages
    ws.manifest["missing_pages"] = compute_missing_pages(pages)
    ws.save()
    return pages


def run(ws: Workspace, cfg: dict, log=print) -> None:
    pages = ws.manifest["pages"]
    todo = [(p["id"], ws.root / p["llm_image"]) for p in pages
            if not p.get("md") and p.get("llm_image") and p.get("role") != "cover"
            and p.get("status") != "deleted"]
    by_id = {p["id"]: p for p in pages}
    if not todo:
        log("  all pages already transcribed")
    else:
        provider = cfg["provider"]["name"]
        if provider == "hybrid":
            local = get_backend({**cfg, "provider": {**cfg["provider"], "name": "ollama"}})
            results = _run_incremental(ws, local, todo, by_id, log)
            escalate_on = cfg["provider"]["escalate_on"]
            escalate = [pid for pid, r in results.items()
                        if needs_escalation(r, escalate_on)]
            target = cfg["provider"].get("escalate_to", "anthropic")
            from ..backends import anthropic_enabled
            if escalate and target == "anthropic" and not anthropic_enabled(cfg):
                log(f"  hybrid: {len(escalate)} pages would escalate, but the "
                    f"Anthropic API is disabled in settings — keeping local results")
                escalate = []
            if escalate:
                log(f"  hybrid: escalating {len(escalate)} pages to {target}")
                remote = get_backend(
                    {**cfg, "provider": {**cfg["provider"], "name": target}})
                for pid, r in remote.transcribe(
                        [(pid, ws.root / by_id[pid]["llm_image"]) for pid in escalate],
                        log).items():
                    if "error" not in r:
                        by_id[pid]["status"] = "ok"
                        _write_result(ws, by_id[pid], r, target)
                        _cache_page(ws, by_id[pid])
                ws.save()
        elif provider == "anthropic":
            backend = get_backend(cfg)  # Batches API is inherently all-at-once
            for pid, r in backend.transcribe(todo, log).items():
                _write_result(ws, by_id[pid], r, backend.name)
                _cache_page(ws, by_id[pid])
            ws.save()
        else:
            backend = get_backend(cfg)
            _run_incremental(ws, backend, todo, by_id, log)

    pages = reconcile(ws, pages, log)
    missing = ws.manifest["missing_pages"]

    warnings = check_printed_numbers(pages)
    if missing:
        warnings.append(f"printed pages never captured: {format_ranges(missing)} "
                        f"— film those pages (any direction) and Add video, or "
                        f"add photos of them")
    for w in warnings:
        log(f"  WARNING: {w}")
    failed = [p["id"] for p in pages if p.get("transcribe_error")]
    if failed:
        log(f"  {len(failed)} pages failed transcription: {', '.join(failed)}")
    ws.manifest.pop("isbn_detected", None)   # re-scan for an ISBN next detail view
    ws.stage_done("transcribe", warnings=warnings, failed=failed)
