# FlipScan

Turn an iPhone slow-motion video of thumbing through a book into a clean EPUB. FlipScan extracts frames from 120/240 fps flip-through videos, scores and clusters them to identify individual pages, picks the best frame per page, transcribes each page with a vision LLM (local Ollama, Anthropic API, or a hybrid of both), and assembles the result into an EPUB — surfacing any pages it couldn't capture well as a "reshoot list" instead of silently dropping them.

See [SPEC.md](SPEC.md) for the full architecture.

## Capture protocol (two passes, recommended)

1. **Pass 1**: flip back-to-front, camera framed on the side of the spread that lays flat — this captures the odd pages, each appearing flat and unoccluded.
2. **Pass 2**: flip front-to-back, same framing — captures the even pages.

Tips:
- Keep the book at the same angle and distance between passes.
- Use 240 fps slow-mo if your phone supports it; more frames = more chances to catch each page at rest.
- Expect stiff pages near the covers to need individual reshoots — the review step will list them.

Single-video mode (`--pages all`) is supported as a quick-and-dirty path.

## Quickstart

### Docker (recommended)

```sh
docker compose up
# drop your videos into ./books, open http://localhost:8321
```

### Dev install

```sh
pip install -e .
# requires ffmpeg on PATH
```

## CLI reference

_(populated as features land — see implementation status below)_

```
flipscan init DIR --video V [--pages odd|even|all] [--direction forward|reverse] [--expected-pages N]
flipscan run DIR [--provider ollama|anthropic|hybrid] [--model NAME] [--ollama-url URL]
flipscan review DIR
flipscan patch DIR --page ID IMG
flipscan build DIR -o book.epub [--format epub|pdf|pdf-facsimile]
flipscan ui [--port 8321]
```

## Configuration

`config.toml` in the workspace (or global); env vars override: `FLIPSCAN_OLLAMA_URL`, `FLIPSCAN_ANTHROPIC_API_KEY`, `FLIPSCAN_PROVIDER`.

## Implementation status

- [x] M0 Repo setup
- [x] M1 Workspace + manifest + init/extract
- [x] M2 Scoring + clustering + selection (+ contact sheet) — validated on synthetic two-pass video (`tools/make_test_video.py`); validate on a real book video before trusting thresholds
- [x] M3 Preprocess (crop/perspective)
- [ ] M4 Transcribe (backends + JSON schema)
- [ ] M5 Assemble + EPUB
- [ ] M6 Review HTML + patch flow
- [ ] M7 Figures pipeline
- [ ] M8 PDF outputs
- [ ] M9 GUI (`flipscan ui`)
- [ ] M10 Docker packaging
- [ ] M11 Dewarp, tuning, polish
