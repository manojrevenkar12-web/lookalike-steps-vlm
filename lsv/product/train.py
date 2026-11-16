"""Training loops. Every loop is seeded, single-threaded by default and runs the same under DistributedDataParallel
(`ddp.py`), so one recipe serves the CPU build box and a multi-GPU node."""
from __future__ import annotations

import numpy as np
import torch

from . import detr as D
from .data import Video


def train_detr(videos: list[Video], sides: bool, cfg: dict, stride: int = 2, log=print) -> D.Detr:
    torch.manual_seed(cfg["seed"])
    classes = 3 if sides else 2
    model = D.Detr(classes, cfg["dim"], cfg["queries"], cfg["enc_layers"], cfg["dec_layers"], videos[0].frames.shape[1])
    items = [(v, t) for v in videos for t in range(0, len(v.frames), stride)]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = cfg["epochs"] * (len(items) // cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=max(steps, 1), pct_start=0.1)
    g = np.random.default_rng(cfg["seed"])
    size = videos[0].frames.shape[1]
    for ep in range(cfg["epochs"]):
        order = g.permutation(len(items))
        model.train()
        tot = 0.0
        for i in range(0, len(order) - cfg["batch"] + 1, cfg["batch"]):
            batch = [items[j] for j in order[i: i + cfg["batch"]]]
            x = torch.from_numpy(np.stack([v.frames[t] for v, t in batch])).float().div(255).permute(0, 3, 1, 2)
            tg = [D.targets_for(v.hands[t], v.hand_visible[t], v.tool[t], bool(v.tool_visible[t]), sides, size) for v, t in batch]
            loss = D.set_loss(model(x), tg, classes)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            sched.step()
            tot += loss.item()
        log(f"detr sides={sides} epoch {ep + 1}: loss {tot / max(1, len(order) // cfg['batch']):.3f}")
    return model


def train_vlm(vlm, samples: list, epochs: int, lr: float, batch: int, seed: int, pairs: dict | None = None,
              pair_weight: float = 0.0, log=print, ddp=None) -> list[float]:
    """Fine-tunes whatever parameters of vlm.model require gradients. With ddp=(rank, world), each rank takes its
    share of every batch and gradients are averaged by DistributedDataParallel."""
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n_steps = epochs * (len(samples) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=max(n_steps, 1), pct_start=0.1)
    g = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        vlm.model.train()
        order = g.permutation(len(samples))
        tot, k = 0.0, 0
        for i in range(0, len(order) - batch + 1, batch):
            chunk = [samples[j] for j in order[i: i + batch]]
            if ddp is not None:
                rank, world = ddp
                chunk = chunk[rank::world]
            loss = vlm.loss(chunk, pairs, pair_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            k += 1
        losses.append(tot / max(k, 1))
        log(f"vlm epoch {ep + 1}/{epochs}: loss {losses[-1]:.3f}")
    return losses


def add_lora(vlm, cfg: dict) -> int:
    """Wraps the model in LoRA adapters on attention and MLP projections of both towers; returns trainable params."""
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj", "gate_proj", "up_proj", "down_proj"])
    vlm.model = get_peft_model(vlm.model, lc)
    return sum(p.numel() for p in vlm.model.parameters() if p.requires_grad)


def merge_lora(vlm) -> None:
    vlm.model = vlm.model.merge_and_unload()
