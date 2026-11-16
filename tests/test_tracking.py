import numpy as np
import torch

from lsv.product import detr as D
from lsv.product import tracker as TR


def box(cx, cy, s=6):
    return np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], float)


def det(cls, b, p_left):
    return (cls, 0.9, b, np.array([p_left, 1 - p_left, 0.0]))


def test_side_tracker_keeps_identities_when_the_hands_meet():
    t = TR.SideTracker(0.5, 3)
    tracks = []
    for k in range(20):  # the hands move towards each other and overlap in the middle
        lx, rx = 20 + k, 44 - k
        tracks.append(t.update([det(1, box(rx, 30), 0.1), det(0, box(lx, 30), 0.9)]))
    assert all(np.allclose((fr[0][0] + fr[0][2]) / 2, 20 + k) for k, fr in enumerate(tracks))


def test_overlap_tracker_can_swap_and_idf1_sees_it():
    truth = np.array([[box(20 + k, 30), box(44 - k, 30)] for k in range(24)])
    vis = np.ones((24, 2), bool)
    swapped = [{0: truth[k, 0], 1: truth[k, 1]} if k < 12 else {0: truth[k, 1], 1: truth[k, 0]} for k in range(24)]
    perfect = [{0: truth[k, 0], 1: truth[k, 1]} for k in range(24)]
    assert TR.idf1(truth, vis, perfect)["idf1"] == 1.0
    s = TR.idf1(truth, vis, swapped)
    assert s["idf1"] < 0.6 and s["id_switches"] == 2


def test_detr_matching_and_loss():
    torch.manual_seed(0)
    m = D.Detr(3, 32, 4, 1, 2)
    x = torch.rand(2, 3, 64, 64)
    tg = [(torch.tensor([0, 2]), torch.tensor([[0.3, 0.5, 0.1, 0.1], [0.6, 0.4, 0.1, 0.1]])), (torch.zeros(0, dtype=torch.long), torch.zeros(0, 4))]
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    first = None
    for _ in range(30):
        loss = D.set_loss(m(x), tg, 3)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.isclose(D.giou_pairs(a, a), torch.ones(1))


def test_crops_follow_the_track_and_keep_the_last_place():
    frames = np.zeros((3, 64, 64, 3), np.uint8)
    frames[:, 10:14, 50:54] = 255
    tracks = [{1: box(52, 12, 4)}, {}, {1: box(52, 12, 4)}]
    c = TR.crops(frames, tracks, 9, 32)
    assert c.shape == (2, 3, 32, 32, 3)
    assert c[1, 1].max() == 255  # not seen in frame 1: the window stays where the hand was
