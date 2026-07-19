# FlipScan — Architecture Spec (v0.1)

Turn an iPhone slow-motion video of thumbing through a book into a clean EPUB, using frame analysis + a vision LLM.

## Goals

- Input: one or more slow-mo videos (120/240 fps) of a book being flipped front-to-back or back-to-front.
- Output: an EPUB with correct page order, markdown-quality text, tables rendered as markdown, and figures cropped from the source frames and embedded as images.
- Human-in-the-loop: the tool must surface pages it couldn't capture well and produce a "reshoot list" rather than silently dropping content.

## Non-goals (v1)

- No GUI app; CLI + a generated HTML review page.
- No on-device iPhone component. Video arrives via AirDrop/Files.
- No handwriting, no multi-column academic layout optimization (flag, don't solve).

## Stack

- **Python 3.11+**, single package `flipscan/`
- **ffmpeg** (subprocess) for frame extraction
- **OpenCV + numpy** for scoring, dedupe, cropping, dewarp
- **imagehash** (perceptual hashing) for page clustering
- **Pluggable vision backends** for transcription:
  - **Ollama** (default): user's existing Ollama server on the local network — just an HTTP endpoint, no setup/Docker/provisioning in scope. `base_url` set in `config.toml` (e.g. `http://192.168.x.x:11434`). Gemma 4 with max vision token budget (1120) for dense text pages.
  - **Anthropic API**: higher accuracy; **Batches API** for cost (50% discount, fine since this is offline processing)
- **ebooklib** (or pandoc if installed) for EPUB assembly
- **Click** for the CLI; **Jinja2** for the HTML review page

## Pipeline stages

Each stage reads/writes to a per-book workspace directory and updates a single `manifest.json`. Stages are idempotent and resumable.

```
ingest → extract → score → cluster → select → preprocess → transcribe → figures → assemble → build
```

### 1. Ingest
- Recommended capture protocol: **two passes** — e.g. one video back-to-front capturing odd pages, one front-to-back capturing even pages. In each pass, the camera targets the side of the spread that lays flat, full-frame. Every page then gets a flat, unoccluded appearance.
- `flipscan init mybook/ --video oddpass.mov --pages odd --direction reverse --video evenpass.mov --pages even --direction forward`
- Single-video mode (`--pages all`) remains supported as the quick-and-dirty path.
- Copies/links videos, records metadata (fps as recorded — iPhone slow-mo containers report playback fps; use `ffprobe` on the stream, not container), creates workspace.

### 2. Extract
- ffmpeg dumps all frames as JPEG q=2 into `frames/` with source-video + frame-index naming.
- Optionally pre-filter with ffmpeg's scene/blur filters to cut volume, but default to all frames — disk is cheap, missed pages are not.

### 3. Score
For every frame compute and store in manifest:
- **Sharpness**: variance of Laplacian on a center crop.
- **Flatness proxy**: detect the page quadrilateral (largest bright contour); score how close it is to a rectangle and how large/centered it is.
- **Occlusion**: skin-tone blob detection over the page region (thumb coverage %).
- **Motion**: frame-diff vs neighbors; page-at-rest frames have low diff.
- Composite score = weighted product. Weights in `config.toml`, tunable.

### 4. Cluster (page identity)
- Runs **per video**; each video's clusters map to its declared parity sequence (odd: 1,3,5,… or even: 2,4,6,…), direction-normalized per its `--direction`.
- Perceptual hash (pHash) each frame's page region (the flat-side page is the region of interest; ignore the mid-turn side of the spread).
- Walk frames chronologically; a new cluster starts when hash distance from current cluster centroid exceeds threshold AND motion has spiked and settled (page-turn event).
- **Parity merge**: zip the two sequences into full page order. Cross-checks: cluster counts should differ by ≤1 between videos; printed page numbers (stage 7) must interleave monotonically with correct parity — a wrong-parity printed number is a hard error pointing at a merge misalignment.
- **Gap detection**: clusters lasting < N frames or whose best score is below threshold are flagged `suspect`. If the user supplies `--expected-pages 320`, warn on count mismatch.

### 5. Select
- Best-scoring frame per cluster becomes the page's canonical frame.
- Second-best kept as fallback for the review UI.

### 6. Preprocess
- Crop to page quad, perspective-correct (homography), optional dewarp for curl (start with simple cylindrical model; page-dewarp-style optimization later if needed).
- Grayscale-ish contrast normalization for the LLM copy; keep the full-color corrected frame for figure crops.
- Downscale LLM copy to ~1600px long edge (token cost vs legibility sweet spot; configurable).

### 7. Transcribe (vision LLM)
- A `TranscriptionBackend` interface with two implementations:
  - `OllamaBackend`: POST to `{base_url}/api/chat` with base64 image, `model: gemma4` (size variant configurable), sequential or small-N concurrent requests.
  - `AnthropicBackend`: Batches API, e.g. `claude-sonnet-4-6`.
- **Escalation mode** (`provider = "hybrid"`): run everything through local Gemma 4 first; pages returning `confidence: low`, malformed JSON, or flagged regions get re-run through Anthropic. Best cost/quality tradeoff for a 300-page book.
- Both backends use the same prompt and must return strict JSON (validate + one retry on parse failure; local models fail JSON more often, so keep the schema flat):
  ```json
  {
    "markdown": "...",
    "page_number_printed": 143,
    "confidence": "high|medium|low",
    "regions": [
      {"type": "figure|table_as_image|photo", "bbox_norm": [x0,y0,x1,y1], "caption": "..."}
    ],
    "flags": ["cut_off_text", "blur", "multi_column"]
  }
  ```
- Tables that are cleanly readable → markdown tables inline. Complex tables → flagged as `table_as_image` region.
- `page_number_printed` is a cross-check against cluster order — mismatches (non-monotonic sequence) are strong signals of a missed/duplicated page. This is the most reliable gap detector in the whole system.

### 8. Figures
- For each region: bbox from the LLM is treated as approximate. Expand by 5–10%, then snap to content using edge/whitespace analysis on the full-res corrected frame. Crop from the color frame, save to `figures/`, insert `![caption](...)` at the marked position in the markdown.
- If a figure's source frame is low quality, add the page to the reshoot list with reason `figure_quality`.

### 9. Assemble
- Concatenate page markdown in order; heal hyphenated words split across pages; merge paragraphs continuing across page breaks (heuristic: page ends mid-sentence + next page starts lowercase).
- Detect chapter headings (LLM already emits `#`/`##`); build EPUB TOC from them.
- Strip running headers/footers and printed page numbers (LLM instructed to omit; assembler double-checks repeated first/last lines).

### 10. Build + Review
- `flipscan build --format epub|pdf|pdf-facsimile` (repeatable flag; default epub):
  - **epub**: via ebooklib, embedded figure images, metadata from CLI flags or detected title page.
  - **pdf** (reflowed): rendered from the same markdown via weasyprint (or pandoc if present). Cheap once EPUB works.
  - **pdf-facsimile**: corrected page images as pages, with the per-page transcription embedded as an invisible OCR text layer (pikepdf/reportlab) → searchable, layout-exact, robust to transcription errors. Best fallback when quality is marginal.
- `flipscan review` → static HTML: side-by-side canonical frame vs rendered markdown per page, suspect pages highlighted, and a **reshoot list** ("re-photograph pages 87, 143, 200–201 individually"). `flipscan patch --page 143 photo.jpg` slots a replacement photo through preprocess→transcribe→assemble.

## Manifest schema (abridged)

```json
{
  "book": {"title": "...", "reverse": true, "expected_pages": null},
  "videos": [{"path": "...", "fps_actual": 240}],
  "pages": [
    {
      "id": "p0142",
      "cluster_frames": ["v0_f18211", "..."],
      "canonical": "v0_f18240",
      "scores": {"sharpness": 812.4, "occlusion": 0.02},
      "status": "ok|suspect|patched|missing",
      "printed_number": 143,
      "md": "pages/p0142.md",
      "figures": ["figures/p0142_a.png"]
    }
  ]
}
```

## CLI surface

```
flipscan init DIR --video V [--reverse] [--expected-pages N]
flipscan run DIR [--provider ollama|anthropic|hybrid] [--model NAME] [--ollama-url URL]
                            # extract→…→transcribe (resumable)
flipscan review DIR         # HTML QA page + reshoot list
flipscan patch DIR --page ID IMG
flipscan build DIR -o book.epub [--format epub|pdf|pdf-facsimile]
flipscan ui [--port 8321]   # optional local web GUI
```

## Optional GUI (`flipscan ui`)

A local web app (FastAPI backend + one-page frontend), replacing the static review HTML. Core layout:

- **Project sidebar**: list of workspaces; New Project (pick video files, set reverse/expected-pages).
- **Pipeline panel**: stage-by-stage status from `manifest.json` (extract → … → build), Run/Resume buttons, live progress via SSE.
- **Files panel**: tabs for Inputs (videos), Frames (contact sheet of canonical frames), Pages (frame vs rendered markdown side-by-side, inline markdown editing, suspect pages highlighted), Figures.
- **Reshoot workflow**: reshoot list with drag-and-drop replacement photo → runs the patch flow.
- **Output panel**: build buttons per format, download links.

The GUI is a thin client over the same stage functions the CLI calls — no logic lives in the UI layer. CLI remains fully sufficient without it.

## Packaging: Docker

Ship as a single image so "easy to run" = one compose command. (The user's Ollama server stays outside Docker — it's just a network endpoint.)

- **Dockerfile**: `python:3.11-slim` base + `ffmpeg` via apt + the package. One image serves both CLI and GUI (`ENTRYPOINT ["flipscan"]`, default `CMD ["ui", "--host", "0.0.0.0"]`).
- **docker-compose.yml**:
  - volume mount `./books:/data` for workspaces (videos in, EPUB/PDF out — everything persists on the host)
  - `ports: "8321:8321"` for the GUI
  - env vars: `FLIPSCAN_OLLAMA_URL`, `FLIPSCAN_ANTHROPIC_API_KEY`, `FLIPSCAN_PROVIDER` (env overrides `config.toml`)
- **Reaching the LAN Ollama box**: its LAN IP works directly from inside the container on default bridge networking. If Ollama ever runs on the Docker host itself, use `host.docker.internal` (Docker Desktop on Windows/macOS).
- CLI usage through the same image: `docker compose run --rm flipscan run /data/mybook --provider hybrid`
- Non-Docker install (`pip install -e .` + system ffmpeg) remains supported for development.

## Risks / open questions

1. **Cluster boundary errors** are the main correctness risk. Mitigations: printed-page-number monotonicity check (stage 7) and expected-page-count check. Tune thresholds on a real test video early.
2. **Pages that never appear flat** — largely solved by the two-pass parity protocol; residual cases (stiff early/late pages near the covers) go to the reshoot loop.
3. **Cost/quality**: local Gemma 4 is free but will misread more dense text than Claude; hybrid mode caps API spend to only the pages that need it. Benchmark both on ~10 real pages early to set the escalation threshold.
4. **Local JSON reliability**: smaller local models drift from strict JSON schemas. Keep the schema flat, validate hard, retry once, then escalate.
5. **Copyright**: personal-use digitization of owned books only; no DRM circumvention involved.

## Repository setup (Claude Code, step 0)

- Init a git repo and create a **private** GitHub repo using `gh` (already authenticated on the dev machine): `git init && gh repo create flipscan --private --source=. --push`
- Sensible `.gitignore` from the start: workspaces/`books/`, `frames/`, `.env`, `__pycache__`, build artifacts. **Never commit videos, extracted frames, or API keys.**
- Commit at the end of each implementation-order step with a descriptive message, and push.

### README.md requirements
The README is the single source of documentation and must be kept current as features land. It must cover:
- What FlipScan does (one paragraph) and the two-pass capture protocol, with tips (steady book angle between passes; expect stiff pages near covers to need reshoots)
- Quickstart: Docker (`docker compose up`, drop video in `./books`, open the GUI) and non-Docker dev install
- Full CLI reference for every subcommand and flag
- Configuration: `config.toml` schema and env var overrides (Ollama URL, provider, model, API key)
- Backends: Ollama/Gemma 4 vs Anthropic vs hybrid escalation, and how to tune the escalation threshold
- Pipeline overview with the stage diagram and what each stage writes to the workspace
- Review/reshoot/patch workflow
- Output formats (epub, pdf, pdf-facsimile) and when to prefer each
- Troubleshooting: common failure modes (cluster miscounts, parity mismatches, blurry pages) and what to do

## Implementation order

0. Repo setup per above (`gh repo create` private, `.gitignore`, README skeleton); update README as each step lands
1. Workspace + manifest + `init`/`extract` (thin ffmpeg wrapper)
2. Scoring + clustering + selection, with a debug contact-sheet output to eyeball cluster quality **← validate on a real test video before building anything downstream**
3. Preprocess (crop/perspective; defer dewarp)
4. Transcribe with Batches + JSON schema
5. Assemble + EPUB
6. Review HTML + patch flow
7. Figures pipeline
8. PDF outputs (facsimile first — it's independent of markdown quality; reflowed second)
9. GUI (`flipscan ui`) wrapping the existing stage functions
10. Dockerfile + compose (thin packaging step once GUI exists)
11. Dewarp, tuning, polish
