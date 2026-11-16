"""Scores against the truth: frame accuracy, look-alike accuracy, segment quality (edit score, F1@50), 'when did it
happen' (R@0.5 over windows), tracking identity (IDF1)."""
from __future__ import annotations

import numpy as np

from ..product.samples import runs
from ..product.segments import locate, smooth


def edit_score(pred: np.ndarray, truth: np.ndarray) -> float:
    a = [lab for lab, _, _ in runs(pred)]
    b = [lab for lab, _, _ in runs(truth)]
    d = np.zeros((len(a) + 1, len(b) + 1))
    d[:, 0], d[0, :] = np.arange(len(a) + 1), np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return float(100 * (1 - d[-1, -1] / max(len(a), len(b), 1)))


def f1_at(pred: np.ndarray, truth: np.ndarray, overlap: float = 0.5) -> float:
    P, T = runs(pred), runs(truth)
    used = set()
    tp = 0
    for lab, a, b in P:
        best, bj = 0.0, None
        for j, (l2, c, d) in enumerate(T):
            if l2 != lab or j in used:
                continue
            inter = max(0, min(b, d) - max(a, c) + 1)
            iou = inter / (max(b, d) - min(a, c) + 1)
            if iou > best:
                best, bj = iou, j
        if best >= overlap:
            tp += 1
            used.add(bj)
    prec, rec = tp / max(len(P), 1), tp / max(len(T), 1)
    return float(100 * 2 * prec * rec / max(prec + rec, 1e-9))


def grounding(pred: np.ndarray, truth: np.ndarray, steps: list[str], window: int = 24) -> list[float]:
    """For every non-idle step instance, the window of `window` frames centred on it: IoU of the located run."""
    out = []
    for lab, a, b in runs(truth):
        if steps[lab] == "idle":
            continue
        mid = (a + b) // 2
        lo, hi = max(0, mid - window // 2), min(len(truth) - 1, mid + window // 2 - 1)
        if a < lo or b > hi:
            continue
        got = locate(pred, lab, lo, hi)
        if got is None:
            out.append(0.0)
            continue
        inter = max(0, min(got[1], b) - max(got[0], a) + 1)
        out.append(inter / (max(got[1], b) - min(got[0], a) + 1))
    return out


def score(probs_list: list[np.ndarray], truths: list[np.ndarray], steps: list[str], pair_steps: set[int]) -> dict:
    acc, pair, edit, f1, ious = [], [], [], [], []
    for p, t in zip(probs_list, truths):
        pred = smooth(p.argmax(1))
        acc.append((pred == t).mean())
        m = np.isin(t, list(pair_steps))
        pair.append((pred[m] == t[m]).mean() if m.any() else np.nan)
        edit.append(edit_score(pred, t))
        f1.append(f1_at(pred, t))
        ious += grounding(pred, t, steps)
    r = lambda x: round(float(np.nanmean(x)), 4)  # noqa: E731
    return {"frame_accuracy": r(acc), "lookalike_accuracy": r(pair), "edit": round(float(np.mean(edit)), 1),
            "f1_50": round(float(np.mean(f1)), 1), "grounding_r50": r(np.array(ious) >= 0.5), "grounding_miou": r(ious)}
