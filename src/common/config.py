"""Configuration loading and path helpers shared by every pipeline stage."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def path_for(cfg: dict, key: str) -> Path:
    """Return an (existing) directory for one of the keys under `paths`."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
