"""What the product is given: video, and for labelled recordings the labels a person drew. It never sees the world."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Video:
    key: str
    frames: np.ndarray                     # (T, H, W, 3) uint8
    options: list[str]                     # the steps in the station's standard work, plus idle
    labels: np.ndarray | None = None       # (T,) catalog step index, for labelled recordings
    hands: np.ndarray | None = None        # (T, 2, 4) labelled hand boxes
    hand_visible: np.ndarray | None = None
    tool: np.ndarray | None = None
    tool_visible: np.ndarray | None = None
