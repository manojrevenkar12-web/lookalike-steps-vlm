import numpy as np
import torch

from lsv.product.ddp import distributed_grads
from lsv.product.vlm import VLM, Recog


def test_two_ranks_give_the_whole_batch_gradient(steps, model_cfg):
    cfg = model_cfg["vlm"]
    v = VLM(steps, cfg, seed=1)
    g = np.random.default_rng(0)
    samples = [Recog((g.random((8, 64, 96, 3)) * 255).astype(np.uint8), np.arange(8), steps, steps[i % len(steps)]) for i in range(4)]
    v.loss(samples).backward()
    single = {n: p.grad for n, p in v.model.named_parameters() if p.grad is not None}
    dist = distributed_grads(steps, cfg, v.model.state_dict(), samples, world=2)
    assert set(single) == set(dist)
    for n in single:
        assert torch.allclose(single[n], dist[n], atol=1e-5, rtol=1e-4), n
