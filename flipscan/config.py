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
        "hash_threshold": 10,  # pHash Hamming distance to open a new cluster
        "min_cluster_frames": 3,  # smaller clusters flagged suspect
        "motion_spike_factor": 2.5,  # motion must exceed median * factor between clusters
        "suspect_score_percentile": 10,  # clusters whose best score is in the bottom N% flagged
    },
    "preprocess": {
        "llm_long_edge": 1600,
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


def load_config(workspace: Path | None = None) -> dict[str, Any]:
    """Merged config for a workspace. Env vars win over config.toml over defaults."""
    cfg = DEFAULTS
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
