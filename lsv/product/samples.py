"""Turning video into questions for the video-language model: a clip around every frame, each frame shown with the
two hand crops beside it."""
from __future__ import annotations

import numpy as np

from .data import Video
from .vlm import Recog

GREY = 128


def window(n: int, t: int, length: int) -> np.ndarray:
    """Frame indices of a clip around t (t at length // 2), repeated at the edges of the recording."""
    return np.clip(np.arange(t - length // 2, t - length // 2 + length), 0, n - 1)


def canvas(frames: np.ndarray, crops: np.ndarray | None) -> np.ndarray:
    """(T, H, W, 3) and (2, T, c, c, 3) to (T, H, W + H // 2, 3): the frame, then a column with the left hand's crop
    on top and the right hand's below, each scaled to H // 2. Without crops the column is grey."""
    T, H, W, _ = frames.shape
    out = np.full((T, H, W + H // 2, 3), GREY, np.uint8)
    out[:, :, :W] = frames
    if crops is not None:
        c = crops.shape[2]
        assert c == H // 2, "crops must be half the frame height"
        out[:, : H // 2, W:] = crops[0]
        out[:, H // 2:, W:] = crops[1]
    return out


def recog(v: Video, crops: np.ndarray | None, length: int, frames: list[int], steps: list[str], with_answer: bool) -> list[Recog]:
    full = canvas(v.frames, crops)
    n = len(v.frames)
    out = []
    for t in frames:
        ans = steps[int(v.labels[t])] if (with_answer and v.labels is not None) else None
        out.append(Recog(full, window(n, t, length), v.options, ans))
    return out


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, first frame, last frame) of every run of one label."""
    out, s = [], 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[s]:
            out.append((int(labels[s]), s, t - 1))
            s = t
    return out
