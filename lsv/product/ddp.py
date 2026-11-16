"""Data-parallel fine-tuning with DistributedDataParallel. On the build box the ranks are CPU processes on gloo; on a
GPU node the same code runs one rank per GPU on nccl (`torchrun --nproc-per-node N`). The test checks that two ranks
produce the gradient one process gets from the whole batch."""
from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .vlm import VLM


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup(rank: int, world: int, port: int) -> str:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    return backend


def wrap(vlm: VLM) -> VLM:
    """Wraps the model so backward averages gradients across ranks."""
    vlm.model = DDP(vlm.model)
    return vlm


def _grad_worker(rank: int, world: int, port: int, steps, cfg, state, samples, out) -> None:
    torch.set_num_threads(1)
    setup(rank, world, port)
    vlm = VLM(steps, cfg)
    vlm.model.load_state_dict(state)
    wrap(vlm)
    vlm.loss(samples[rank::world]).backward()
    if rank == 0:
        # numpy, not tensors: a tensor sent through a queue shares memory with a process about to exit
        out.put({n: p.grad.detach().numpy().copy() for n, p in vlm.model.module.named_parameters() if p.grad is not None})
    dist.barrier()
    dist.destroy_process_group()


def distributed_grads(steps: list[str], cfg: dict, state: dict, samples: list, world: int = 2) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_grad_worker, args=(r, world, port, steps, cfg, state, samples, q)) for r in range(world)]
    for p in procs:
        p.start()
    grads = {n: torch.from_numpy(g) for n, g in q.get(timeout=600).items()}
    for p in procs:
        p.join(timeout=600)
    return grads
