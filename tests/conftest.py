from __future__ import annotations

import pytest
import torch

from lsv.config import load

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def catalog():
    return load("catalog.yaml")


@pytest.fixture(scope="session")
def steps(catalog):
    return list(catalog["steps"])


@pytest.fixture(scope="session")
def model_cfg():
    return load("model.yaml")


@pytest.fixture(scope="session")
def small_world(catalog):
    from lsv.world.world import World
    cfg = load("world.yaml")
    cfg["seconds"] = 20
    cfg["pretrain"]["stations"] = 2
    return World(catalog, cfg)
