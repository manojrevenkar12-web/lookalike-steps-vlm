import numpy as np
import torch

from lsv.product import checks as CK
from lsv.product.student import Student, predict


def test_the_student_never_looks_ahead():
    torch.manual_seed(0)
    m = Student(5, 16, 3).eval()
    f = torch.rand(1, 12, 3, 64, 64)
    c = torch.rand(1, 12, 2, 3, 32, 32)
    a = m(f, c)
    f2 = f.clone()
    f2[:, 8:] = 0
    b = m(f2, c)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)


def test_predictions_stay_in_the_allowed_steps():
    m = Student(5, 16, 2)
    p = predict(m, np.zeros((6, 64, 64, 3), np.uint8), np.zeros((2, 6, 32, 32, 3), np.uint8), np.array([1, 1, 0, 0, 1], bool))
    assert np.allclose(p.sum(1), 1) and (p[:, 2:4] == 0).all()


def test_the_gate_reaches_its_target_on_validation():
    g = np.random.default_rng(0)
    labels = g.integers(0, 4, 2000)
    conf = g.random(2000)
    probs = np.full((2000, 4), 0.0)
    right = conf > 0.3  # confident answers are right, unsure ones are random
    pred = np.where(right, labels, g.integers(0, 4, 2000))
    probs[np.arange(2000), pred] = 0.25 + 0.75 * conf
    probs += (1 - probs.sum(1, keepdims=True)) / 4
    gate, info = CK.calibrate(probs, labels, set(), 0.97)
    acc = gate.accept(probs)
    assert (pred[acc] == labels[acc]).mean() >= 0.97 and info["coverage"] > 0.5


def test_a_lookalike_tie_is_held_back():
    pairs = CK.pair_index(["a", "b", "c"], [["a", "b"]])
    gate = CK.Gate(0.4, 0.2, pairs)
    probs = np.array([[0.5, 0.45, 0.05], [0.5, 0.05, 0.45]])
    assert list(gate.accept(probs)) == [False, True]


def test_the_monitor_flags_disagreement_and_not_agreement():
    eye = np.eye(3)
    edge = eye[np.zeros(300, int)]
    mon = CK.Monitor(every=50, window=3, max_disagreement=0.25, min_coverage=0.5)
    same = mon.run(edge, np.ones(300, bool), eye[np.zeros(6, int)])
    other = mon.run(edge, np.ones(300, bool), eye[np.ones(6, int)])
    assert same["flagged_at"] is None and other["flagged_at"] == 100
