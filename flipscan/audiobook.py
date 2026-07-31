"""Audiobook output: proofed book text -> local TTS -> chaptered .m4b.

The narration source is the same text the EPUB uses (accepted proofed chapters
substitute in), flattened from markdown to clean spoken prose. Synthesis runs
per chapter through a local voice model (Chatterbox TTS, MIT — supports
zero-shot voice cloning from a short reference sample); each chapter's wav is
cached in work/audiobook/ keyed by a text hash, so an interrupted or re-run
build only synthesizes what changed. ffmpeg then packs the wavs into an .m4b
with chapter markers, the book's cover as artwork, and title/author metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path
from typing import Callable

from .ffmpeg import _find
from .workspace import Workspace

SAMPLE_RATE = 24000          # Chatterbox output rate
CHUNK_CHARS = 400            # per-generate text budget (short = stable prosody)
CHUNK_GAP_S = 0.25           # silence between chunks
PARA_GAP_S = 0.65            # silence between paragraphs
CHAPTER_TAIL_S = 1.2         # silence at each chapter end


# ---------------- markdown -> narration text

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SUP = re.compile(r"<sup>.*?</sup>|<sub>.*?</sub>", re.S)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPH = re.compile(r"(\*\*|__|\*|_|`)(.+?)\1")
_HEAD = re.compile(r"^#{1,6}\s+", re.M)
_NOTE = re.compile(r"^>\s*⟦.*?⟧\s*$", re.M | re.S)


def md_to_narration(md: str) -> str:
    """Flatten a chapter's markdown into clean prose for the voice: images,
    tables, production notes and markup vanish; headings and paragraphs stay
    as paragraph breaks (which become audible pauses)."""
    t = md
    t = _NOTE.sub("", t)
    t = _IMG.sub("", t)
    t = _SUP.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _TAG.sub("", t)
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("|") or re.match(r"^[-*_]{3,}$", s):
            continue                       # tables/rules aren't narratable
        s = _HEAD.sub("", s) if s.startswith("#") else s
        s = s.lstrip("> ").strip()
        if s:
            lines.append(s)
    t = "\n".join(lines)
    t = _EMPH.sub(r"\2", t)
    t = t.replace("—", ", ").replace("–", " to ")   # audible, not silent, dashes
    t = re.sub(r"\[\[region-\d+\]\]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


_SENT = re.compile(r"(?<=[.!?…])\s+")


def chunk_paragraph(par: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Sentence-packed chunks under the model's comfortable budget; an
    over-long sentence is split at commas, then hard-wrapped as a last resort."""
    out: list[str] = []
    buf = ""
    for sent in _SENT.split(par.strip()):
        if not sent:
            continue
        while len(sent) > max_chars:       # pathological sentence
            cut = sent.rfind(",", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            piece, sent = sent[:cut + 1].strip(), sent[cut + 1:].strip()
            if buf:
                out.append(buf)
                buf = ""
            out.append(piece)
        if len(buf) + len(sent) + 1 > max_chars and buf:
            out.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return out


def narration_chapters(ws: Workspace) -> list[tuple[str, str]]:
    """(title, narration_text) per chapter — accepted proofed text substituted
    exactly like the EPUB build, so proofreading carries into the narration."""
    from .proofread import chapter_hash, chapters, load_proof
    out = []
    for i, (title, md) in enumerate(chapters(ws)):
        d = load_proof(ws, i)
        if (d and d.get("status") == "accepted" and d.get("proofed_md")
                and d.get("base_hash") == chapter_hash(md)):
            md = d["proofed_md"]
        text = md_to_narration(md)
        if text:
            out.append((title, text))
    return out


# ---------------- synthesis engine (lazy import; the model is ~2 GB)

class ChatterboxEngine:
    """Chatterbox TTS (resemble-ai, MIT). Default built-in voice, or zero-shot
    cloning from a short reference sample. Loaded lazily so the module imports
    without torch installed."""

    def __init__(self, cfg: dict):
        a = cfg.get("audiobook", {})
        self.voice = (a.get("voice_sample") or "").strip() or None
        self.exaggeration = float(a.get("exaggeration", 0.5))
        self.cfg_weight = float(a.get("cfg_weight", 0.5))
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from chatterbox.tts import ChatterboxTTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = ChatterboxTTS.from_pretrained(device=device)
        return self._model

    @property
    def sr(self) -> int:
        return self._model.sr if self._model else SAMPLE_RATE

    def speak(self, text: str):
        """One chunk -> 1-D float tensor."""
        m = self._load()
        kwargs = dict(exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
        if self.voice:
            kwargs["audio_prompt_path"] = self.voice
        return m.generate(text, **kwargs).squeeze(0)


def synthesize_chapter(engine: ChatterboxEngine, text: str, out_wav: Path,
                       log: Callable[[str], None] = print,
                       should_cancel: Callable[[], bool] = lambda: False) -> None:
    """Whole chapter -> one 16-bit PCM wav (paragraph pauses included)."""
    import torch
    import torchaudio
    pieces = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = [(pi, c) for pi, par in enumerate(paragraphs)
              for c in chunk_paragraph(par)]
    gap = lambda s: torch.zeros(int(engine.sr * s))
    last_par = None
    for n, (pi, chunk) in enumerate(chunks):
        if should_cancel():
            from .jobs import JobCanceled
            raise JobCanceled()
        if pieces:
            pieces.append(gap(PARA_GAP_S if pi != last_par else CHUNK_GAP_S))
        pieces.append(engine.speak(chunk).cpu())
        last_par = pi
        if (n + 1) % 10 == 0 or n + 1 == len(chunks):
            log(f"    {n + 1}/{len(chunks)} chunks")
    pieces.append(gap(CHAPTER_TAIL_S))
    audio = torch.cat(pieces).unsqueeze(0)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav), audio, engine.sr,
                    encoding="PCM_S", bits_per_sample=16)


# ---------------- m4b assembly

def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _esc_meta(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def assemble_m4b(ws: Workspace, wavs: list[tuple[str, Path]], out: Path,
                 log: Callable[[str], None] = print) -> None:
    """wavs [(chapter_title, wav_path)] -> .m4b with chapter markers, cover
    art, and book metadata."""
    adir = wavs[0][1].parent
    book = ws.manifest["book"]
    listing = adir / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for _, p in wavs),
        encoding="utf-8")
    meta = [";FFMETADATA1",
            f"title={_esc_meta(book.get('title') or ws.root.name)}",
            f"artist={_esc_meta(book.get('author') or '')}",
            "genre=Audiobook"]
    t = 0.0
    for title, p in wavs:
        dur = _wav_seconds(p)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000",
                 f"START={int(t * 1000)}", f"END={int((t + dur) * 1000)}",
                 f"title={_esc_meta(title)}"]
        t += dur
    metafile = adir / "chapters.ffmeta"
    metafile.write_text("\n".join(meta) + "\n", encoding="utf-8")

    cover = next((p for p in ws.manifest["pages"]
                  if p.get("role") == "cover" and p.get("color")
                  and (ws.root / p["color"]).exists()), None)
    cmd = [_find("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(listing),
           "-i", str(metafile)]
    if cover:
        cmd += ["-i", str(ws.root / cover["color"])]
    cmd += ["-map", "0:a", "-map_metadata", "1",
            "-c:a", "aac", "-b:a", "96k", "-ac", "1"]
    if cover:
        cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
    cmd += ["-f", "ipod", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg m4b assembly failed: {r.stderr[-400:]}")
    log(f"  audiobook: {out.name} ({t / 3600:.1f} h, {len(wavs)} chapters)")


# ---------------- top-level build (durable-job entry point)

def build_audiobook(ws: Workspace, cfg: dict, out: Path,
                    log: Callable[[str], None] = print,
                    should_cancel: Callable[[], bool] = lambda: False) -> Path:
    chs = narration_chapters(ws)
    if not chs:
        raise RuntimeError("no narratable text — run the pipeline first")
    a = dict(cfg.get("audiobook", {}))
    # a voice sample saved on the project wins; else the configured/global one
    proj_voice = ws.root / "voice-sample.wav"
    voice = str(proj_voice) if proj_voice.exists() \
        else (a.get("voice_sample") or "").strip()
    a["voice_sample"] = voice
    engine = ChatterboxEngine({**cfg, "audiobook": a})
    adir = ws.work_file("audiobook")
    adir.mkdir(parents=True, exist_ok=True)

    wavs: list[tuple[str, Path]] = []
    for i, (title, text) in enumerate(chs):
        wav = adir / f"ch{i:03d}.wav"
        sig = adir / f"ch{i:03d}.json"
        # cache key: chapter text + the voice/params that shaped it
        key = hashlib.sha1(json.dumps(
            [text, voice, engine.exaggeration, engine.cfg_weight]
        ).encode()).hexdigest()
        if wav.exists() and sig.exists() and \
                json.loads(sig.read_text()).get("key") == key:
            log(f"  [{i + 1}/{len(chs)}] {title[:46]!r}: cached")
        else:
            log(f"  [{i + 1}/{len(chs)}] {title[:46]!r}: "
                f"synthesizing {len(text) / 1000:.1f}k chars…")
            synthesize_chapter(engine, text, wav, log, should_cancel)
            sig.write_text(json.dumps({"key": key, "title": title}))
        wavs.append((title, wav))
    assemble_m4b(ws, wavs, out, log)
    return out
