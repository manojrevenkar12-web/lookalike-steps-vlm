import copy

import numpy as np
import pytest
import torch

from lsv.product import train as T
from lsv.product.samples import canvas, window
from lsv.product.vlm import VLM, Recog, patchify


def clip(t=8, h=64, w=96, seed=0):
    return (np.random.default_rng(seed).random((t, h, w, 3)) * 255).astype(np.uint8)


def test_patches_are_the_ones_the_qwen_processor_makes():
    pytest.importorskip("torchvision")
    from transformers import Qwen2VLVideoProcessor
    fr = clip()
    mine, grid = patchify(fr, 8, 2, 2)
    proc = Qwen2VLVideoProcessor(do_resize=False, image_mean=[0.5] * 3, image_std=[0.5] * 3, patch_size=8, merge_size=2, temporal_patch_size=2)
    theirs = proc(videos=[torch.from_numpy(fr).permute(0, 3, 1, 2)], return_tensors="pt", do_sample_frames=False)
    assert tuple(theirs["video_grid_thw"][0].tolist()) == grid
    assert torch.allclose(theirs["pixel_values_videos"], mine, atol=1e-5)


def test_answers_stay_inside_the_station_options(steps, model_cfg):
    v = VLM(steps, model_cfg["vlm"])
    opts = ["idle", "fasten_a", "fasten_b"]
    fr = clip()
    p = v.step_probs([Recog(fr, np.arange(8), opts), Recog(fr, np.arange(8), steps)])
    assert np.isclose(p.sum(1), 1).all()
    assert (p[0][[steps.index(s) for s in steps if s not in opts]] == 0).all()


def test_lookalike_loss_adds_to_the_language_loss(steps, model_cfg):
    v = VLM(steps, model_cfg["vlm"])
    s = [Recog(clip(), np.arange(8), steps, "fasten_a")]
    base = v.loss(s).item()
    with_pairs = v.loss(s, {"fasten_a": ["fasten_b"]}, 1.0).item()
    assert with_pairs > base


def test_lora_trains_few_parameters_and_merges_exactly(steps, model_cfg):
    v = VLM(steps, model_cfg["vlm"])
    full = v.parameters()
    lv = copy.deepcopy(v)
    n = T.add_lora(lv, model_cfg["finetune"]["lora"])
    assert 0 < n < full / 5
    T.train_vlm(lv, [Recog(clip(seed=i), np.arange(8), steps, "fasten_b") for i in range(4)], 1, 0.01, 4, 0, log=lambda *_: None)
    s = [Recog(clip(seed=9), np.arange(8), steps)]
    before = lv.step_probs(s)
    T.merge_lora(lv)
    assert np.allclose(before, lv.step_probs(s), atol=1e-4)


def test_clips_are_views_with_crops_beside_the_frame():
    fr = clip(t=20, w=64)
    crops = (np.ones((2, 20, 32, 32, 3)) * [[[[[10]]]], [[[[200]]]]]).astype(np.uint8)
    c = canvas(fr, crops)
    assert c.shape == (20, 64, 96, 3)
    assert (c[:, :32, 64:] == 10).all() and (c[:, 32:, 64:] == 200).all()
    assert list(window(20, 0, 8)) == [0, 0, 0, 0, 0, 1, 2, 3]
