"""Configuration: defaults <- workspace config.toml <- environment variables."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "provider": {
        "name": "ollama",  # ollama | anthropic | hybrid
        "ollama_url": "http://localhost:11434",
        "ollama_model": "gemma4",
        "ollama_num_predict": 4096,
        "ollama_think": False,  # thinking wastes minutes/page on transcription
        "anthropic_model": "claude-sonnet-4-6",
        "ollama_concurrency": 2,
        # hybrid: escalate to anthropic when local result matches any of these
        "escalate_on": ["low_confidence", "malformed_json", "flags"],
    },
    "extract": {
        "jpeg_quality": 2,  # ffmpeg -qscale:v
    },
    "score": {
        # weights for the composite score (weighted product exponents)
        "w_sharpness": 1.0,
        "w_flatness": 1.0,
        "w_occlusion": 1.0,
        "w_motion": 1.0,
        "center_crop": 0.6,  # fraction of frame used for sharpness
    },
    "cluster": {
        "hash_threshold": 10,  # pHash distance for patched-page reattachment
        "min_cluster_frames": 3,  # smaller clusters flagged suspect
        "motion_spike_factor": 2.5,  # motion must exceed median * factor between clusters
        "turn_min_frames": 4,  # sustained high-motion run = page turn (shorter = wobble)
        "suspect_score_percentile": 10,  # clusters whose best score is in the bottom N% flagged
    },
    "preprocess": {
        "llm_long_edge": 1600,
        "quad_pad": 0.025,  # expand the page crop so edge content (page numbers) survives
        "isolate_page": True,  # edge-density crop to the flat readable page
        "mask_clutter": False,  # experimental: isolate the book, hide desk clutter
        "dewarp": False,
    },
    "transcribe": {
        "max_retries": 1,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


ENV_OVERRIDES = {
    "FLIPSCAN_PROVIDER": ("provider", "name"),
    "FLIPSCAN_OLLAMA_URL": ("provider", "ollama_url"),
    "FLIPSCAN_OLLAMA_MODEL": ("provider", "ollama_model"),
    "FLIPSCAN_ANTHROPIC_MODEL": ("provider", "anthropic_model"),
}


def global_config_path() -> Path:
    """Global config shared by all projects: $FLIPSCAN_ROOT/config.toml
    (the GUI's projects folder), or ~/.flipscan/config.toml otherwise."""
    root = os.environ.get("FLIPSCAN_ROOT")
    base = Path(root) if root else Path.home() / ".flipscan"
    return base / "config.toml"


def load_config(workspace: Path | None = None) -> dict[str, Any]:
    """Merged config: defaults <- global config <- workspace config.toml <- env."""
    cfg = DEFAULTS
    gp = global_config_path()
    if gp.exists():
        with open(gp, "rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))
    if workspace is not None:
        toml_path = Path(workspace) / "config.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                cfg = _deep_merge(cfg, tomllib.load(f))
    for env, (section, key) in ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            cfg = _deep_merge(cfg, {section: {key: val}})
    return cfg


def _toml_value(v: Any) -> str:
    import json as _json
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return _json.dumps(str(v))  # valid TOML basic string


def save_global_config(sections: dict[str, dict[str, Any]]) -> Path:
    """Persist settings to the global config file (whole-file rewrite)."""
    path = global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# FlipScan global settings (edited by the GUI settings panel)"]
    for section, values in sections.items():
        vals = {k: v for k, v in values.items() if v not in (None, "")}
        if not vals:
            continue
        lines.append(f"\n[{section}]")
        for k, v in vals.items():
            lines.append(f"{k} = {_toml_value(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
