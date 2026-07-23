"""Per-book workspace directory + manifest.json access.

Layout:
    mybook/
      manifest.json
      config.toml        (optional, user-editable)
      videos/            copied source videos
      frames/<vid>/      extracted JPEG frames
      work/              scores, cluster data, contact sheets, preprocessed images
      pages/             per-page markdown
      figures/           cropped figure images
      review/            generated review HTML
      out/               built EPUB/PDF
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

SUBDIRS = ["videos", "frames", "work", "pages", "figures", "review", "out"]

# Pipeline stage order; `run` executes these left to right.
STAGES = [
    "extract",
    "score",
    "cluster",
    "select",
    "preprocess",
    "transcribe",
    "figures",
    "assemble",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self._manifest: dict[str, Any] | None = None

    # -- creation ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path | str,
        videos: list[dict[str, Any]],
        title: str | None = None,
        expected_pages: int | None = None,
        book: dict[str, Any] | None = None,
    ) -> "Workspace":
        ws = cls(root)
        ws.root.mkdir(parents=True, exist_ok=True)
        if ws.manifest_path.exists():
            raise FileExistsError(f"{ws.manifest_path} already exists; workspace is initialized")
        for d in SUBDIRS:
            (ws.root / d).mkdir(exist_ok=True)
        book_meta = {"title": title, "expected_pages": expected_pages}
        if book:                        # author / isbn / publisher / year / ...
            book_meta.update({k: v for k, v in book.items() if v is not None})
        ws._manifest = {
            "version": MANIFEST_VERSION,
            "created_at": _now(),
            "book": book_meta,
            "videos": videos,
            "stages": {},
            "pages": [],
        }
        ws.save()
        return ws

    @classmethod
    def open(cls, root: Path | str) -> "Workspace":
        ws = cls(root)
        if not ws.manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {ws.root} — run `flipscan init` first"
            )
        return ws

    # -- manifest ---------------------------------------------------------

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            with open(self.manifest_path, encoding="utf-8") as f:
                self._manifest = json.load(f)
        return self._manifest

    def save(self) -> None:
        import time

        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        # Windows: replace fails with PermissionError while another handle
        # (GUI progress polling) has the manifest open — retry briefly
        for attempt in range(6):
            try:
                tmp.replace(self.manifest_path)
                return
            except PermissionError:
                time.sleep(0.05 * (2 ** attempt))
        tmp.replace(self.manifest_path)

    # -- stage bookkeeping ------------------------------------------------

    def stage_status(self, stage: str) -> str:
        return self.manifest["stages"].get(stage, {}).get("status", "pending")

    def stage_done(self, stage: str, **extra: Any) -> None:
        self.manifest["stages"][stage] = {"status": "done", "completed_at": _now(), **extra}
        self.save()

    def stage_reset(self, stage: str) -> None:
        """Reset a stage and everything downstream of it."""
        if stage in STAGES:
            for s in STAGES[STAGES.index(stage):]:
                self.manifest["stages"].pop(s, None)
        else:
            self.manifest["stages"].pop(stage, None)
        self.save()

    # -- paths ------------------------------------------------------------

    def dir(self, name: str) -> Path:
        return self.root / name

    def frames_dir(self, video_id: str) -> Path:
        return self.root / "frames" / video_id

    def work_file(self, name: str) -> Path:
        return self.root / "work" / name

    # -- pages ------------------------------------------------------------

    def page(self, page_id: str) -> dict[str, Any] | None:
        for p in self.manifest["pages"]:
            if p["id"] == page_id:
                return p
        return None

    # -- video import -----------------------------------------------------

    def import_video(self, src: Path, video_id: str) -> Path:
        """Copy a source video into the workspace; returns workspace-relative dest."""
        dest = self.root / "videos" / f"{video_id}{src.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(src, dest)
        return dest.relative_to(self.root)
