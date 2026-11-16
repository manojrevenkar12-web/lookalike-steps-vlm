import numpy as np


def test_recordings_repeat_and_hold_every_customer_step(small_world, steps):
    a = small_world.recording(small_world.customer, "normal", 0)
    b = small_world.recording(small_world.customer, "normal", 0)
    assert np.array_equal(a.frames, b.frames) and np.array_equal(a.labels, b.labels)
    assert a.frames.shape == (100, 64, 64, 3)
    seen = {steps[i] for i in a.labels}
    assert {"fasten_a", "insert_clip"} <= seen


def test_lookalike_steps_differ_in_few_pixels(small_world, steps):
    """Fastening at slot a and at slot b: the mean frames differ, but only in a small part of the picture."""
    r = small_world.recording(small_world.customer, "normal", 1)
    fa = r.frames[r.labels == steps.index("fasten_a")].astype(float).mean(0)
    fb = r.frames[r.labels == steps.index("fasten_b")].astype(float).mean(0)
    diff = np.abs(fa - fb).mean(-1)
    assert diff.max() > 20 and (diff > 20).mean() < 0.05


def test_hand_boxes_follow_the_hands(small_world):
    r = small_world.recording(small_world.customer, "normal", 2)
    w = r.hands[..., 2] - r.hands[..., 0]
    assert ((w > 3) & (w < 12)).mean() > 0.95
    assert r.hand_visible.mean() > 0.95


def test_conditions_change_the_picture(small_world):
    n = small_world.recording(small_world.customer, "normal", 3)
    c = small_world.recording(small_world.customer, "camera", 3)
    v = small_world.recording(small_world.customer, "variant", 3)
    assert not np.array_equal(n.frames, c.frames)
    assert v.meta["variant"] and not n.meta["variant"]
