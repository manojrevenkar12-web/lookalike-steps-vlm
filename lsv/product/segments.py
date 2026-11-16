"""From frame labels to steps in time: smoothing, segments, and 'when did step X happen?' answered from the dense
labels."""
from __future__ import annotations

import numpy as np

from .samples import runs


def smooth(pred: np.ndarray, min_len: int = 2) -> np.ndarray:
    """Runs shorter than min_len frames are absorbed by the run before them."""
    out = pred.copy()
    for lab, a, b in runs(out):
        if b - a + 1 < min_len and a > 0:
            out[a: b + 1] = out[a - 1]
    return out


def locate(pred: np.ndarray, step: int, lo: int, hi: int) -> tuple[int, int] | None:
    """The longest run of `step` overlapping frames lo..hi, clipped to them."""
    best = None
    for lab, a, b in runs(pred):
        if lab != step or b < lo or a > hi:
            continue
        a, b = max(a, lo), min(b, hi)
        if best is None or b - a > best[1] - best[0]:
            best = (a, b)
    return best
