import numpy as np

from lsv.eval.metrics import edit_score, f1_at, grounding


def test_segment_metrics():
    t = np.array([0] * 5 + [1] * 5 + [2] * 5)
    assert edit_score(t, t) == 100 and f1_at(t, t) == 100
    p = np.array([0] * 5 + [2] * 5 + [2] * 5)
    assert edit_score(p, t) < 100 and f1_at(p, t) < 100


def test_grounding_from_dense_labels():
    steps = ["idle", "a", "b"]
    t = np.array([0] * 10 + [1] * 6 + [0] * 10)
    assert grounding(t, t, steps) == [1.0]
    shifted = np.array([0] * 12 + [1] * 6 + [0] * 8)
    assert 0.4 < grounding(shifted, t, steps)[0] < 1.0
