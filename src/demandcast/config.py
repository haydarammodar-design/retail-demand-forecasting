"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config and resolve relative paths against the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for section, key in [
        ("data", "raw_dir"),
        ("data", "processed_dir"),
        ("output", "reports_dir"),
        ("output", "figures_dir"),
    ]:
        p = Path(cfg[section][key])
        cfg[section][key] = p if p.is_absolute() else PROJECT_ROOT / p
        cfg[section][key].mkdir(parents=True, exist_ok=True)
    return cfg
