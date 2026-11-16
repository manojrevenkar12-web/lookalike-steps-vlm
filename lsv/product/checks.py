"""Checks that hold weak answers back before a customer sees them.

- Per frame: the edge reports a step only when it is confident enough and, if its two best guesses are a look-alike
  pair, sure enough between them. Both limits are set on the station's validation recording so the reported frames
  reach a target accuracy there.
- Per station: once every few seconds a clip goes to the cloud model for a second opinion. When the edge disagrees
  with it clearly more often than it did on the validation recording, or reports far fewer frames than there, the
  station is flagged as weak. Both limits are relative to validation: the edge and the cloud model never agree on
  every frame (they differ at step boundaries), so an absolute limit would flag healthy stations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Gate:
    min_confidence: float
    min_pair_margin: float
    pair_index: set[tuple[int, int]]

    def accept(self, probs: np.ndarray) -> np.ndarray:
        """(T, K) probabilities to a boolean (T,): which frames the edge may report."""
        order = np.argsort(-probs, 1)
        p1 = probs[np.arange(len(probs)), order[:, 0]]
        p2 = probs[np.arange(len(probs)), order[:, 1]]
        pair = np.array([(a, b) in self.pair_index for a, b in zip(order[:, 0], order[:, 1])])
        return (p1 >= self.min_confidence) & (~pair | (p1 - p2 >= self.min_pair_margin))


def pair_index(steps: list[str], lookalike: list[list[str]]) -> set[tuple[int, int]]:
    out = set()
    for a, b in lookalike:
        i, j = steps.index(a), steps.index(b)
        out |= {(i, j), (j, i)}
    return out


def calibrate(probs: np.ndarray, labels: np.ndarray, pairs: set[tuple[int, int]], target: float) -> tuple[Gate, dict]:
    """The gate with the highest coverage whose accepted frames reach `target` accuracy on validation."""
    best = None
    correct = probs.argmax(1) == labels
    for c in np.linspace(0.3, 0.99, 24):
        for m in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            g = Gate(float(c), float(m), pairs)
            acc_mask = g.accept(probs)
            if acc_mask.sum() == 0:
                continue
            acc, cov = correct[acc_mask].mean(), acc_mask.mean()
            if acc >= target and (best is None or cov > best[1]["coverage"]):
                best = (g, {"coverage": float(cov), "accuracy": float(acc)})
    if best is None:  # the target cannot be reached: report only the most confident frames
        g = Gate(0.99, 0.6, pairs)
        m = g.accept(probs)
        best = (g, {"coverage": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else 0.0})
    return best


@dataclass
class Monitor:
    every: int            # frames between second opinions
    window: int           # second opinions per decision
    max_disagreement: float  # validation disagreement plus the margin
    min_coverage: float      # validation coverage times the allowed ratio

    def run(self, edge_probs: np.ndarray, accepted: np.ndarray, cloud_probs: np.ndarray) -> dict:
        """cloud_probs holds the cloud model's answer at frames 0, every, 2 * every, ... Returns the first frame at which
        the station is flagged (or None) and why."""
        samples = list(range(0, len(edge_probs), self.every))
        disagree = [int(edge_probs[t].argmax() != cloud_probs[i].argmax()) for i, t in enumerate(samples)]
        for k in range(self.window, len(samples) + 1):
            t = samples[k - 1]
            d = float(np.mean(disagree[k - self.window: k]))
            cov = float(accepted[max(0, t - self.every * self.window): t + 1].mean())
            if d > self.max_disagreement:
                return {"flagged_at": t, "why": f"edge and cloud disagree on {d:.0%} of the last {self.window} clips"}
            if cov < self.min_coverage:
                return {"flagged_at": t, "why": f"the edge reports {cov:.0%} of frames, under {self.min_coverage:.0%}"}
        return {"flagged_at": None, "why": ""}
