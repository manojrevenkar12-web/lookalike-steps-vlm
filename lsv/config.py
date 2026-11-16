from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, root: Path = ROOT) -> dict:
    return yaml.safe_load((root / "config" / name).read_text())
