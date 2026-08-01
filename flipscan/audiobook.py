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
INTRO_GAP_S = 1.4            # silence after the spoken chapter announcement
CHAPTER_TAIL_S = 1.2         # silence at each chapter end


# ---------------- markdown -> narration text

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SUP = re.compile(r"<sup>.*?</sup>|<sub>.*?</sub>", re.S)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPH = re.compile(r"(\*\*|__|\*|_|`)(.+?)\1")
_HEAD = re.compile(r"^#{1,6}\s+", re.M)
_NOTE = re.compile(r"^>\s*⟦.*?⟧\s*$", re.M | re.S)


def md_to_narration(md: str, title: str | None = None) -> str:
    """Flatten a chapter's markdown into clean prose for the voice: images,
    tables, production notes and markup vanish; headings and paragraphs stay
    as paragraph breaks (which become audible pauses). When `title` is given,
    a leading line that just repeats it is dropped — the chapter announcement
    is spoken separately (with a longer pause) by synthesize_chapter."""
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
    t = t.strip()
    if title:
        first, _, rest = t.partition("\n")
        if first.strip().rstrip(".").lower() == title.strip().rstrip(".").lower():
            t = rest.strip()
    return t


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
        text = md_to_narration(md, title=title)
        if text:
            out.append((title, text))
    return out


# ---------------- synthesis engine (lazy import; the model is ~2 GB)

_SHARED_MODEL = None    # one Chatterbox instance per process (it's ~3.4 GB VRAM)
_BUILTIN_CONDS = None   # the model's default-narrator conditioning, as loaded
_COND_CACHE: dict[str, object] = {}   # voice path -> prepared conditioning


def _shared_model():
    global _SHARED_MODEL, _BUILTIN_CONDS
    if _SHARED_MODEL is None:
        import torch
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SHARED_MODEL = ChatterboxTTS.from_pretrained(device=device)
        _BUILTIN_CONDS = _SHARED_MODEL.conds
    return _SHARED_MODEL


class ChatterboxEngine:
    """Chatterbox TTS (resemble-ai, MIT). Default built-in voice, or zero-shot
    cloning from a short reference sample. Loaded lazily so the module imports
    without torch installed; the underlying model is shared process-wide so an
    audiobook build and a quick voice preview never double-load it."""

    def __init__(self, cfg: dict):
        a = cfg.get("audiobook", {})
        self.voice = (a.get("voice_sample") or "").strip() or None
        self.exaggeration = float(a.get("exaggeration", 0.5))
        self.cfg_weight = float(a.get("cfg_weight", 0.5))
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = _shared_model()
        return self._model

    @property
    def sr(self) -> int:
        return self._model.sr if self._model else SAMPLE_RATE

    def speak(self, text: str, voice: str | None = None):
        """One chunk -> 1-D float tensor. `voice` overrides the engine default
        for this chunk (full-cast narration switches voices per quote).

        Chatterbox keeps the LAST prompt's conditioning on the model, so a
        call without a prompt after a character quote would silently stay in
        the character's voice — the narrator would never come back. We manage
        conditioning explicitly: prepare each voice once, cache it, and set
        the right one before every generate."""
        m = self._load()
        v = (voice if voice is not None else self.voice) or ""
        conds = _COND_CACHE.get(v)
        if conds is None:
            if v:
                m.prepare_conditionals(v, exaggeration=self.exaggeration)
                conds = m.conds
            else:
                conds = _BUILTIN_CONDS
            _COND_CACHE[v] = conds
        m.conds = conds
        return m.generate(text, exaggeration=self.exaggeration,
                          cfg_weight=self.cfg_weight).squeeze(0)


TARGET_LUFS = -20.0   # broadcast-ish speech level all pieces normalize to


def _condition_piece(wav, sr: int):
    """Clean one synthesized piece before concatenation: short edge fades kill
    the clicks heard at voice-switch seams, and loudness normalization evens
    out level differences between voices (each reference sample records at its
    own level, so cast quotes otherwise jump louder/quieter than narration)."""
    import torch
    wav = wav.clone()
    n = int(sr * 0.008)
    if wav.numel() > 2 * n:
        ramp = torch.linspace(0.0, 1.0, n)
        wav[:n] = wav[:n] * ramp
        wav[-n:] = wav[-n:] * ramp.flip(0)
    if wav.numel() >= sr // 2:            # meter needs ≥400 ms
        try:
            import pyloudnorm
            loud = pyloudnorm.Meter(sr).integrated_loudness(
                wav.numpy().astype("float64"))
            if loud > -70.0:              # not silence
                gain = 10 ** ((TARGET_LUFS - loud) / 20)
                wav = wav * min(gain, 8.0)
        except Exception:
            pass                          # normalization is a nicety
    return wav.clamp(-0.98, 0.98)


def synthesize_chapter(engine: ChatterboxEngine, text: str, out_wav: Path,
                       log: Callable[[str], None] = print,
                       should_cancel: Callable[[], bool] = lambda: False,
                       announce: str | None = None,
                       segments: list[tuple[str, str]] | None = None) -> None:
    """Whole chapter -> one 16-bit PCM wav (paragraph pauses included). With
    `announce`, the chapter title is spoken first as its own line, followed by
    a longer beat of silence — the audiobook convention that makes chapter
    starts audible in the recording itself, not just in the player's menu.

    `segments` enables full-cast narration: [(voice_path, seg_text)] runs in
    order, '' = the engine's default narrator. Segment boundaries usually fall
    mid-paragraph (a quote inside a sentence), so crossing one uses the short
    chunk gap, not a paragraph pause."""
    import torch
    import torchaudio
    if segments is None:
        segments = [("", text)]
    # (voice, paragraph_id, chunk) — paragraph ids number across the whole
    # chapter so pause logic survives segmentation
    chunks: list[tuple[str, int, str]] = []
    pid = 0
    for si, (voice, seg) in enumerate(segments):
        paras = [p for p in seg.split("\n\n") if p.strip()]
        for j, par in enumerate(paras):
            if j > 0:
                pid += 1
            for c in chunk_paragraph(par):
                chunks.append((voice, pid, c))
        # a segment boundary is mid-flow — do NOT advance the paragraph id
    pieces = []
    gap = lambda s: torch.zeros(int(engine.sr * s))
    if announce:
        line = announce.strip()
        pieces.append(_condition_piece(
            engine.speak(line if line[-1:] in ".!?" else line + ".").cpu(),
            engine.sr))
        pieces.append(gap(INTRO_GAP_S))
    last_par = None
    for n, (voice, pi, chunk) in enumerate(chunks):
        if should_cancel():
            from .jobs import JobCanceled
            raise JobCanceled()
        if pieces:
            pieces.append(gap(PARA_GAP_S if pi != last_par else CHUNK_GAP_S))
        pieces.append(_condition_piece(
            engine.speak(chunk, voice=voice or None).cpu(), engine.sr))
        last_par = pi
        if (n + 1) % 10 == 0 or n + 1 == len(chunks):
            log(f"    {n + 1}/{len(chunks)} chunks")
    pieces.append(gap(CHAPTER_TAIL_S))
    audio = torch.cat(pieces).unsqueeze(0)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav), audio, engine.sr,
                    encoding="PCM_S", bits_per_sample=16)


PREVIEW_LINE = ("This is how your audiobook will sound. A clear evening, "
                "a good chair, and a story worth hearing.")


def resolve_voice(ws: Workspace, global_dir: Path | None,
                  name: str) -> Path | None:
    """A voice name -> its wav. The book's own voices/ folder wins (🪄-generated
    character voices are book-scoped), then the shared library (recorded or
    uploaded voices, usable everywhere)."""
    name = (name or "").strip()
    if not name:
        return None
    for d in (ws.root / "voices", global_dir):
        if d is not None:
            f = d / f"{name}.wav"
            if f.exists():
                return f
    return None


def synthesize_preview(cfg: dict, text: str, voice_path: str,
                       out_wav: Path) -> Path:
    """A few seconds of a voice reading `text` — cheap enough to audition
    every voice before committing hours of synthesis to a full book."""
    import torch
    import torchaudio
    engine = ChatterboxEngine(cfg)
    text = (text or PREVIEW_LINE).strip()[:300]
    pieces = []
    for chunk in chunk_paragraph(text):
        pieces.append(_condition_piece(
            engine.speak(chunk, voice=voice_path or None).cpu(), engine.sr))
        pieces.append(torch.zeros(int(engine.sr * CHUNK_GAP_S)))
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav), torch.cat(pieces).unsqueeze(0), engine.sr,
                    encoding="PCM_S", bits_per_sample=16)
    return out_wav


# ---------------- m4b assembly

def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _esc_meta(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def assemble_m4b(ws: Workspace, wavs: list[tuple[str, Path]], out: Path,
                 speed: float = 1.0,
                 log: Callable[[str], None] = print) -> None:
    """wavs [(chapter_title, wav_path)] -> .m4b with chapter markers, cover
    art, and book metadata. `speed` > 1 time-compresses the narration (ffmpeg
    atempo, pitch preserved) — applied here at assembly, so re-rendering the
    same chapters at a different speed is seconds, not hours."""
    speed = max(0.5, min(3.0, float(speed or 1.0)))
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
        dur = _wav_seconds(p) / speed     # markers land on the sped-up timeline
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
    cmd += ["-map", "0:a", "-map_metadata", "1"]
    if speed != 1.0:
        cmd += ["-filter:a", f"atempo={speed}"]
    cmd += ["-c:a", "aac", "-b:a", "96k", "-ac", "1"]
    if cover:
        cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
    cmd += ["-f", "ipod", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg m4b assembly failed: {r.stderr[-400:]}")
    log(f"  audiobook: {out.name} ({t / 3600:.1f} h"
        + (f" at {speed}x" if speed != 1.0 else "") + f", {len(wavs)} chapters)")


# ---------------- top-level build (durable-job entry point)

ESTIMATE_CHARS_PER_MIN = 1216   # measured: 62.4k chars -> 51.3 min narration
ESTIMATE_SYNTH_FACTOR = 1.11    # measured: 56.8 min GPU for 51.3 min audio


def estimate(ws: Workspace) -> dict:
    """Rough numbers for the UI, from the narration text length: minutes of
    audio at 1x, and minutes of GPU synthesis (which speed settings do NOT
    change — speed is applied after synthesis)."""
    chars = sum(len(t) for _, t in narration_chapters(ws))
    audio_min = chars / ESTIMATE_CHARS_PER_MIN
    return {"chars": chars,
            "audio_min": round(audio_min, 1),
            "synth_min": round(audio_min * ESTIMATE_SYNTH_FACTOR, 1)}


def build_audiobook(ws: Workspace, cfg: dict, out: Path,
                    voice: str | None = None, speed: float = 1.0,
                    use_cast: bool = False, voices_dir: Path | None = None,
                    chapters: list[int] | None = None, head_chars: int = 0,
                    log: Callable[[str], None] = print,
                    should_cancel: Callable[[], bool] = lambda: False) -> Path:
    """`voice` is a path to a reference sample from the voice library (cloned
    narration), or None/'' for the engine's built-in narrator; a configured
    audiobook.voice_sample is the fallback when no explicit choice is made.
    With `use_cast`, quotes attributed by the cast analysis are spoken in each
    character's assigned library voice (voices_dir resolves the names).
    `chapters` limits the build to those chapter indices (a sample build) —
    their wavs land in the same cache, so the sample's synthesis is banked
    toward the eventual full build."""
    chs = narration_chapters(ws)
    if not chs:
        raise RuntimeError("no narratable text — run the pipeline first")
    a = dict(cfg.get("audiobook", {}))
    voice = (voice if voice is not None
             else (a.get("voice_sample") or "")).strip()
    a["voice_sample"] = voice
    engine = ChatterboxEngine({**cfg, "audiobook": a})
    adir = ws.work_file("audiobook")
    adir.mkdir(parents=True, exist_ok=True)

    cast, voice_of = None, {}
    if use_cast:
        from .casting import load_cast
        cast = load_cast(ws)
        if cast is None:
            raise RuntimeError("no cast analysis — run Analyze characters first")
        for cname, c in (cast.get("characters") or {}).items():
            p = resolve_voice(ws, voices_dir, c.get("voice") or "")
            if p is not None:
                voice_of[cname] = str(p)
        log(f"  cast: {len(voice_of)} character(s) with voices — "
            + (", ".join(f"{n} -> {Path(p).stem}" for n, p in voice_of.items())
               or "none (all narrator)"))

    def chapter_quotes(i: int, title: str) -> list[dict]:
        rows = (cast or {}).get("chapters") or []
        if i < len(rows) and rows[i].get("title") == title:
            return rows[i].get("quotes") or []
        return next((r.get("quotes") or [] for r in rows
                     if r.get("title") == title), [])

    wavs: list[tuple[str, Path]] = []
    for i, (title, text) in enumerate(chs):
        if chapters is not None and i not in chapters:
            continue
        head = ""
        if head_chars > 0:
            # quick audition: only the chapter's opening, cut at a paragraph
            # boundary; cached under its own slot so the full chapter's audio
            # is never clobbered
            parts, total = [], 0
            for par in text.split("\n\n"):
                parts.append(par)
                total += len(par)
                if total >= head_chars:
                    break
            text = "\n\n".join(parts)
            head = "-head"
        segments = None
        if voice_of:
            from .casting import locate_quotes, segments_for
            segments = segments_for(
                text, locate_quotes(text, chapter_quotes(i, title)), voice_of)
        wav = adir / f"ch{i:03d}{head}.wav"
        sig = adir / f"ch{i:03d}{head}.json"
        # cache key: chapter text + announcement + the voice/params behind it
        # + the resolved cast segmentation. A changed voice sample re-keys every
        # chapter (hash the sample bytes, not the path).
        def _vhash(p: str) -> str:
            return (hashlib.sha1(Path(p).read_bytes()).hexdigest()
                    if p and Path(p).exists() else "")
        key = hashlib.sha1(json.dumps(
            [text, title, _vhash(voice), engine.exaggeration, engine.cfg_weight,
             [(v, _vhash(v), len(t)) for v, t in (segments or [])]]
        ).encode()).hexdigest()
        if wav.exists() and sig.exists() and \
                json.loads(sig.read_text()).get("key") == key:
            log(f"  [{i + 1}/{len(chs)}] {title[:46]!r}: cached")
        else:
            nseg = sum(1 for v, _ in (segments or []) if v)
            log(f"  [{i + 1}/{len(chs)}] {title[:46]!r}: "
                f"synthesizing {len(text) / 1000:.1f}k chars"
                + (f", {nseg} cast quote run(s)" if nseg else "") + "…")
            synthesize_chapter(engine, text, wav, log, should_cancel,
                               announce=title, segments=segments)
            sig.write_text(json.dumps({"key": key, "title": title}))
        wavs.append((title, wav))
    assemble_m4b(ws, wavs, out, speed=speed, log=log)
    return out
