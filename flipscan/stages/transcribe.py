"""Stage 7: transcribe — vision LLM per page, with hybrid escalation and
printed-page-number monotonicity checking (the most reliable gap detector)."""

from __future__ import annotations

from ..backends import get_backend, needs_escalation
from ..workspace import Workspace


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


def check_printed_numbers(pages: list[dict]) -> list[str]:
    """Printed page numbers must increase monotonically in cluster order;
    violations point at missed/duplicated pages or a parity merge misalignment."""
    warnings = []
    prev_num, prev_id = None, None
    for p in pages:
        n = p.get("printed_number")
        if n is None:
            continue
        if prev_num is not None and n <= prev_num:
            warnings.append(
                f"printed page numbers not monotonic: {prev_id} has {prev_num}, "
                f"then {p['id']} has {n} — likely missed/duplicated page or merge error"
            )
            p["status"] = "suspect"
        prev_num, prev_id = n, p["id"]
    return warnings


def run(ws: Workspace, cfg: dict, log=print) -> None:
    pages = ws.manifest["pages"]
    todo = [(p["id"], ws.root / p["llm_image"]) for p in pages if not p.get("md")]
    by_id = {p["id"]: p for p in pages}
    if not todo:
        log("  all pages already transcribed")
    else:
        provider = cfg["provider"]["name"]
        if provider == "hybrid":
            local = get_backend({**cfg, "provider": {**cfg["provider"], "name": "ollama"}})
            results = local.transcribe(todo, log)
            escalate_on = cfg["provider"]["escalate_on"]
            escalate = [pid for pid, r in results.items()
                        if needs_escalation(r, escalate_on)]
            for pid, r in results.items():
                _write_result(ws, by_id[pid], r, "ollama")
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
        else:
            backend = get_backend(cfg)
            for pid, r in backend.transcribe(todo, log).items():
                _write_result(ws, by_id[pid], r, backend.name)
        ws.save()

    warnings = check_printed_numbers(pages)
    for w in warnings:
        log(f"  WARNING: {w}")
    failed = [p["id"] for p in pages if p.get("transcribe_error")]
    if failed:
        log(f"  {len(failed)} pages failed transcription: {', '.join(failed)}")
    ws.stage_done("transcribe", warnings=warnings, failed=failed)
