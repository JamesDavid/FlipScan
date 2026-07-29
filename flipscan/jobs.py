"""A small, durable job queue backed by SQLite.

Why this exists: heavy work (the pipeline, chapter proofreads, page re-reads)
used to run either in a daemon thread inside the web server or — worse —
directly inside an HTTP request. Both die the moment the process restarts or
the request drops, so long tasks silently never finished. This queue makes
jobs durable rows in SQLite: a single background worker pulls them one at a
time, streams progress to a log table, and any job left `running` when the
process died is requeued on the next startup, so work actually resumes.

No broker, no Redis — one SQLite file and one worker thread. Handlers are
registered by the app (they close over ws_for/load_config) and dispatched by
`kind`. Cancellation is cooperative: handlers are handed a `should_cancel`
callback and a `log` callback.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# terminal states a job can end in
DONE, ERROR, CANCELED = "done", "error", "canceled"
_TERMINAL = {DONE, ERROR, CANCELED}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  project     TEXT    NOT NULL,
  kind        TEXT    NOT NULL,
  params      TEXT    NOT NULL DEFAULT '{}',
  label       TEXT    NOT NULL DEFAULT '',
  status      TEXT    NOT NULL DEFAULT 'queued',
  created_at  REAL    NOT NULL,
  started_at  REAL,
  finished_at REAL,
  error       TEXT,
  result      TEXT,
  cancel      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS job_logs(
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  ts     REAL    NOT NULL,
  line   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project);
CREATE INDEX IF NOT EXISTS idx_logs_job     ON job_logs(job_id, id);
"""


class JobCanceled(Exception):
    """Raised inside a handler (via the log/cancel hooks) to abort cleanly."""


# handler signature: (project, params, log, should_cancel) -> None
Handler = Callable[[str, dict, Callable[[str], None], Callable[[], bool]], Any]


class JobQueue:
    def __init__(self, db_path: str | Path,
                 lane_caps: dict[str, int] | None = None,
                 kind_lanes: dict[str, str] | None = None) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()          # one sqlite conn per thread
        self._handlers: dict[str, Handler] = {}
        self._wake = threading.Event()           # enqueue -> wake a worker
        self._stopping = threading.Event()
        self._workers: list[threading.Thread] = []
        # Concurrency is per LANE, not per kind. Every kind that mutates the
        # shared manifest belongs to the serial "main" lane (cap 1) so only one
        # runs at a time and they never race on manifest.json. Manifest-safe
        # kinds get their own parallel lane — chapter proofreads write their own
        # per-chapter file, so they run several at once. The worker pool has one
        # thread per unit of total lane capacity so the concurrency is real.
        self.lane_caps = dict(lane_caps or {"main": 1})
        self.kind_lanes = dict(kind_lanes or {})   # kind -> lane; default "main"
        self.default_lane = "main"
        self.pool_size = max(1, sum(self.lane_caps.values()))
        self._claim_lock = threading.Lock()      # serialize claim decisions
        self._conn().executescript(_SCHEMA)
        self._migrate()

    def _lane(self, kind: str) -> str:
        return self.kind_lanes.get(kind, self.default_lane)

    def _lane_cap(self, lane: str) -> int:
        return self.lane_caps.get(lane, 1)

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created."""
        c = self._conn()
        cols = {r["name"] for r in c.execute("PRAGMA table_info(jobs)").fetchall()}
        if "result" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN result TEXT")

    # -- connection ---------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            # autocommit (isolation_level=None); WAL + busy_timeout let the
            # worker write while request threads read without lock errors
            c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    # -- registration / worker lifecycle -----------------------------------

    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    def start_worker(self) -> None:
        if any(w.is_alive() for w in self._workers):
            return
        self._workers = []
        for i in range(self.pool_size):
            t = threading.Thread(target=self._loop, name=f"flipscan-jobs-{i}",
                                 daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    # -- enqueue / query ----------------------------------------------------

    def enqueue(self, project: str, kind: str, params: dict | None = None,
                label: str = "") -> int:
        import json
        cur = self._conn().execute(
            "INSERT INTO jobs(project, kind, params, label, status, created_at) "
            "VALUES(?,?,?,?, 'queued', ?)",
            (project, kind, json.dumps(params or {}), label, time.time()))
        self._wake.set()
        return int(cur.lastrowid)

    def _row(self, job_id: int) -> dict | None:
        r = self._conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None

    def job(self, job_id: int) -> dict | None:
        r = self._row(job_id)
        if r:
            import json
            try:
                r["params"] = json.loads(r["params"])
            except Exception:
                r["params"] = {}
            if r.get("result"):
                try:
                    r["result"] = json.loads(r["result"])
                except Exception:
                    pass
        return r

    def list(self, project: str | None = None, limit: int = 50) -> list[dict]:
        if project:
            rows = self._conn().execute(
                "SELECT id,project,kind,label,status,created_at,started_at,"
                "finished_at,error FROM jobs WHERE project=? ORDER BY id DESC LIMIT ?",
                (project, limit)).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT id,project,kind,label,status,created_at,started_at,"
                "finished_at,error FROM jobs ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def active(self, project: str, kinds: tuple[str, ...] | None = None) -> dict | None:
        """The queued/running job for a project (optionally of given kinds) —
        used to show 'running' and to reject a duplicate run with 409."""
        q = ("SELECT * FROM jobs WHERE project=? AND status IN('queued','running')")
        args: list = [project]
        if kinds:
            q += " AND kind IN(%s)" % ",".join("?" * len(kinds))
            args += list(kinds)
        q += " ORDER BY id DESC LIMIT 1"
        r = self._conn().execute(q, args).fetchone()
        return dict(r) if r else None

    def latest(self, project: str, kind: str | None = None) -> dict | None:
        q = "SELECT * FROM jobs WHERE project=?"
        args: list = [project]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY id DESC LIMIT 1"
        r = self._conn().execute(q, args).fetchone()
        return dict(r) if r else None

    def logs(self, job_id: int, after: int = 0, limit: int = 2000) -> list[dict]:
        rows = self._conn().execute(
            "SELECT id, ts, line FROM job_logs WHERE job_id=? AND id>? "
            "ORDER BY id LIMIT ?", (job_id, after, limit)).fetchall()
        return [dict(r) for r in rows]

    def request_cancel(self, job_id: int) -> bool:
        c = self._conn()
        r = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not r or r["status"] in _TERMINAL:
            return False
        if r["status"] == "queued":  # not started — cancel outright
            c.execute("UPDATE jobs SET status=?, finished_at=? WHERE id=?",
                      (CANCELED, time.time(), job_id))
        else:                        # running — ask it to stop cooperatively
            c.execute("UPDATE jobs SET cancel=1 WHERE id=?", (job_id,))
        return True

    # -- startup recovery ---------------------------------------------------

    def requeue_orphans(self) -> int:
        """Any job still 'running' when the process starts was orphaned by a
        crash/restart. Put it back on the queue so it resumes (stages are
        idempotent, so re-running skips completed work)."""
        c = self._conn()
        orphans = c.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
        for r in orphans:
            self._log(r["id"], "[queue] resuming after a server restart")
            c.execute("UPDATE jobs SET status='queued', started_at=NULL, "
                      "cancel=0 WHERE id=?", (r["id"],))
        return len(orphans)

    # -- worker internals ---------------------------------------------------

    def _log(self, job_id: int, line: str) -> None:
        self._conn().execute(
            "INSERT INTO job_logs(job_id, ts, line) VALUES(?,?,?)",
            (job_id, time.time(), line))

    def _claim(self) -> dict | None:
        # Serialize claim decisions across the worker pool so the per-lane
        # running counts stay consistent: claim the oldest queued job whose
        # lane is below its concurrency cap.
        with self._claim_lock:
            c = self._conn()
            running: dict[str, int] = {}
            for row in c.execute("SELECT kind, count(*) n FROM jobs "
                                 "WHERE status='running' GROUP BY kind"):
                lane = self._lane(row["kind"])
                running[lane] = running.get(lane, 0) + row["n"]
            # A pipeline run holds its lane for an hour or more. When the lane is
            # free and several jobs are waiting, let the short interactive jobs
            # (a single re-OCR, a page re-read) go FIRST rather than sit behind a
            # queued pipeline — FIFO is preserved within each priority tier.
            for r in c.execute("SELECT * FROM jobs WHERE status='queued' "
                               "ORDER BY (kind='pipeline') ASC, id ASC"):
                lane = self._lane(r["kind"])
                if running.get(lane, 0) >= self._lane_cap(lane):
                    continue                      # this lane is at capacity
                upd = c.execute(
                    "UPDATE jobs SET status='running', started_at=?, cancel=0 "
                    "WHERE id=? AND status='queued'", (time.time(), r["id"]))
                if upd.rowcount == 1:
                    return dict(r)
            return None

    def _canceled(self, job_id: int) -> bool:
        r = self._conn().execute("SELECT cancel FROM jobs WHERE id=?",
                                 (job_id,)).fetchone()
        return bool(r and r["cancel"])

    def _finish(self, job_id: int, status: str, error: str | None = None,
                result: Any = None) -> None:
        import json
        res = None
        if result is not None:
            try:
                res = json.dumps(result)
            except (TypeError, ValueError):
                res = None
        self._conn().execute(
            "UPDATE jobs SET status=?, finished_at=?, error=?, result=? WHERE id=?",
            (status, time.time(), error, res, job_id))

    def _loop(self) -> None:
        import json
        while not self._stopping.is_set():
            job = self._claim()
            if job is None:
                self._wake.wait(timeout=2.0)
                self._wake.clear()
                continue
            jid = job["id"]
            handler = self._handlers.get(job["kind"])
            log = lambda line, _j=jid: self._log(_j, str(line))
            should_cancel = lambda _j=jid: self._canceled(_j)
            if handler is None:
                self._finish(jid, ERROR, f"no handler for kind {job['kind']!r}")
                continue
            try:
                params = json.loads(job["params"] or "{}")
            except Exception:
                params = {}
            try:
                if should_cancel():
                    raise JobCanceled()
                ret = handler(job["project"], params, log, should_cancel)
                self._finish(jid, CANCELED if should_cancel() else DONE,
                             result=ret)
            except JobCanceled:
                self._log(jid, "[queue] canceled")
                self._finish(jid, CANCELED)
            except Exception as e:                       # noqa: BLE001
                self._log(jid, f"[queue] ERROR: {e}")
                self._log(jid, traceback.format_exc())
                self._finish(jid, ERROR, str(e))
