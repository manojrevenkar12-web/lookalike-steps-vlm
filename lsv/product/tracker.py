"""Hand tracks from detections, and the hand crops the video-language model reads.

- `OverlapTracker` is the usual baseline: class-agnostic hand boxes associated frame to frame by overlap with a
  constant-velocity prediction. When the hands meet, it can hand an identity to the other hand.
- `SideTracker` keeps two identities, the left and the right hand, and assigns each detection by the detector's side
  probability and overlap together. A hand's side changes only when it is seen on the other side for several frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> float:
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.prod(np.clip(rb - lt, 0, None)))
    ua = float(np.prod(np.clip(a[2:] - a[:2], 0, None)) + np.prod(np.clip(b[2:] - b[:2], 0, None))) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    misses: int = 0


class OverlapTracker:
    def __init__(self, min_score: float, min_iou: float, max_misses: int, hand_class: int = 0):
        self.min_score, self.min_iou, self.max_misses, self.hand_class = min_score, min_iou, max_misses, hand_class
        self.tracks: list[Track] = []
        self.next_id = 0

    def update(self, dets: list) -> dict[int, np.ndarray]:
        boxes = [d[2] for d in dets if d[0] == self.hand_class and d[1] >= self.min_score]
        pred = [t.box + t.vel for t in self.tracks]
        out: dict[int, np.ndarray] = {}
        used = set()
        if pred and boxes:
            cost = np.array([[1 - iou(p, b) for b in boxes] for p in pred])
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if 1 - cost[r, c] >= self.min_iou:
                    t = self.tracks[r]
                    t.vel = 0.5 * t.vel + 0.5 * (boxes[c] - t.box)
                    t.box, t.misses = boxes[c], 0
                    used.add(c)
                    out[t.id] = t.box
        for t in self.tracks:
            if t.id not in out:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for c, b in enumerate(boxes):
            if c not in used:
                self.tracks.append(Track(self.next_id, b))
                out[self.next_id] = b
                self.next_id += 1
        return out


class SideTracker:
    def __init__(self, min_score: float, hysteresis: int):
        self.min_score, self.hyst = min_score, hysteresis
        self.box: list[np.ndarray | None] = [None, None]
        self.flip = [0, 0]

    def update(self, dets: list) -> dict[int, np.ndarray]:
        cand = [d for d in dets if d[0] in (0, 1) and d[1] >= self.min_score]
        out: dict[int, np.ndarray] = {}
        if not cand:
            return out
        # cost of giving detection j the identity k: the detector's doubt about the side, less the overlap with k's box
        cost = np.zeros((2, len(cand)))
        for k in range(2):
            for j, d in enumerate(cand):
                p_side = d[3][k] / max(d[3][0] + d[3][1], 1e-6)
                ov = iou(self.box[k], d[2]) if self.box[k] is not None else 0.0
                cost[k, j] = (1 - p_side) - 0.5 * ov
        ri, ci = linear_sum_assignment(cost)
        for k, j in zip(ri, ci):
            d = cand[j]
            said = int(d[0])
            if said != k and self.box[k] is not None and iou(self.box[k], d[2]) > 0.2:
                self.flip[k] += 1  # the detector disagrees; trust continuity unless it keeps disagreeing
                if self.flip[k] < self.hyst:
                    self.box[k] = d[2]
                    out[k] = d[2]
                    continue
            self.flip[k] = 0
            self.box[k] = d[2]
            out[k] = d[2]
        return out


def track(dets_per_frame: list[list], tracker) -> list[dict[int, np.ndarray]]:
    return [tracker.update(d) for d in dets_per_frame]


def crops(frames: np.ndarray, tracks: list[dict[int, np.ndarray]], half: int, out: int) -> np.ndarray:
    """(2, T, out, out, 3): windows around the left and right hand, following the track; the last box is kept when a
    hand is not seen, and the window starts at the bench centre's bottom if it was never seen."""
    T, H, W, _ = frames.shape
    res = np.zeros((2, T, out, out, 3), np.uint8)
    last = [np.array([W * 0.35, H * 0.85]), np.array([W * 0.65, H * 0.85])]
    pad = cv2.copyMakeBorder(frames.reshape(T * H, W, 3), 0, 0, 0, 0, cv2.BORDER_REPLICATE).reshape(T, H, W, 3)
    for t in range(T):
        for k in range(2):
            b = tracks[t].get(k)
            if b is not None:
                last[k] = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            cx, cy = np.clip(last[k], half, [W - half, H - half]).astype(int)
            win = pad[t, cy - half: cy + half, cx - half: cx + half]
            res[k, t] = cv2.resize(win, (out, out), interpolation=cv2.INTER_LINEAR)
    return res


def idf1(truth: np.ndarray, visible: np.ndarray, tracks: list[dict[int, np.ndarray]], min_iou: float = 0.3) -> dict:
    """Identity F1 over a recording: truth (T, 2, 4) boxes of the left and right hand, predicted identities matched to
    true ones once for the whole recording (the IDF1 of Ristani et al.)."""
    ids = sorted({i for fr in tracks for i in fr})
    gt_n = int(visible.sum())
    pr_n = sum(len(fr) for fr in tracks)
    if not ids:
        return {"idf1": 0.0, "id_switches": 0, "recall": 0.0}
    m = np.zeros((2, len(ids)))
    for t, fr in enumerate(tracks):
        for j, i in enumerate(ids):
            if i in fr:
                for k in range(2):
                    if visible[t, k] and iou(truth[t, k], fr[i]) >= min_iou:
                        m[k, j] += 1
    ri, ci = linear_sum_assignment(-m)
    idtp = m[ri, ci].sum()
    switches, last = 0, [None, None]
    found = 0
    for t, fr in enumerate(tracks):
        for k in range(2):
            if not visible[t, k]:
                continue
            best = max(((iou(truth[t, k], b), i) for i, b in fr.items()), default=(0, None))
            if best[0] >= min_iou:
                found += 1
                if last[k] is not None and best[1] != last[k]:
                    switches += 1
                last[k] = best[1]
    return {"idf1": float(2 * idtp / max(gt_n + pr_n, 1)), "id_switches": switches, "recall": found / max(gt_n, 1)}
