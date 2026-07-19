"""Stage 7: transcribe — vision LLM per page, with hybrid escalation and
printed-page-number monotonicity checking (the most reliable gap detector)."""

from __future__ import annotations

import json

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


def _write_result(ws: Workspace, page: dict, result: dict, backend_name: str) -> None:
    if "error" in result:
        page["status"] = "suspect"
        page["transcribe_error"] = result["error"]
        return
    md_path = ws.dir("pages") / f"{page['id']}.md"
    md_path.write_text(result["markdown"], encoding="utf-8")
    page["md"] = f"pages/{page['id']}.md"
    page["printed_number"] = result["page_number_printed"]
    page["confidence"] = result["confidence"]
    page["regions"] = result["regions"]
    page["flags"] = result["flags"]
    page["transcribed_by"] = backend_name
    page.pop("transcribe_error", None)
    if result["confidence"] == "low" or result["flags"]:
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

    # unnumbered pages ride along with the page before them
    keys: list[float] = []
    first_num = next(float(v) for v in nums if v is not None)
    for i in range(n):
        if nums[i] is not None:
            keys.append(float(nums[i]))
        elif keys:
            keys.append(keys[-1] + 0.001)
        else:
            keys.append(first_num - 0.001 * (n - i))  # leading unnumbered pages
    order = sorted(range(n), key=lambda i: keys[i])  # stable
    return start + [middle[i] for i in order] + end


def dedupe_by_printed(pages: list[dict], log=print) -> int:
    """Pages sharing a printed page number are the same page captured more
    than once (multiple videos, or a re-detected rest). Keep the best capture,
    mark the rest status="duplicate" (excluded from the book, shown in review)."""
    groups: dict[tuple, list[dict]] = {}
    for p in pages:
        n = p.get("printed_number")
        if n is None or p.get("role") or p.get("status") == "patched":
            continue
        groups.setdefault((n,), []).append(p)

    conf_rank = {"high": 2, "medium": 1, "low": 0}
    deduped = 0
    for group in groups.values():
        # a previous run may have marked duplicates; re-decide from scratch
        for p in group:
            if p["status"] == "duplicate":
                p["status"] = "ok"
        if len(group) < 2:
            continue
        group.sort(key=lambda p: (conf_rank.get(p.get("confidence"), 0),
                                  (p.get("scores") or {}).get("composite", 0.0)),
                   reverse=True)
        for p in group[1:]:
            p["status"] = "duplicate"
            deduped += 1
    if deduped:
        log(f"  merged {deduped} duplicate captures (same printed page number; "
            f"best capture kept)")
    return deduped


def check_printed_numbers(pages: list[dict]) -> list[str]:
    """Printed page numbers must increase monotonically in cluster order;
    violations point at missed/duplicated pages or a parity merge misalignment."""
    warnings = []
    prev_num, prev_id = None, None
    for p in pages:
        n = p.get("printed_number")
        if n is None or p.get("status") == "duplicate":
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
                record(item, r, i)
    else:
        for i, item in enumerate(todo, 1):
            record(item, one(item), i)
    return results


def run(ws: Workspace, cfg: dict, log=print) -> None:
    pages = ws.manifest["pages"]
    todo = [(p["id"], ws.root / p["llm_image"]) for p in pages
            if not p.get("md") and p.get("llm_image") and p.get("role") != "cover"]
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
            if escalate:
                log(f"  hybrid: escalating {len(escalate)} pages to anthropic")
                remote = get_backend(
                    {**cfg, "provider": {**cfg["provider"], "name": "anthropic"}})
                for pid, r in remote.transcribe(
                        [(pid, ws.root / by_id[pid]["llm_image"]) for pid in escalate],
                        log).items():
                    if "error" not in r:
                        by_id[pid]["status"] = "ok"
                        _write_result(ws, by_id[pid], r, "anthropic")
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

    dedupe_by_printed(pages, log)
    reordered = reorder_by_printed(pages)
    if [p["id"] for p in reordered] != [p["id"] for p in pages]:
        log("  reordered pages by printed page numbers")
        ws.manifest["pages"] = pages = reordered
    ws.save()

    warnings = check_printed_numbers(pages)
    for w in warnings:
        log(f"  WARNING: {w}")
    failed = [p["id"] for p in pages if p.get("transcribe_error")]
    if failed:
        log(f"  {len(failed)} pages failed transcription: {', '.join(failed)}")
    ws.stage_done("transcribe", warnings=warnings, failed=failed)
