"""The edge student: a small frame encoder (full frame and the two hand crops) and a causal temporal convolution that
labels each frame from the last 16 frames. It learns from the video-language model's soft answers on recordings
nobody labelled, plus the few labelled ones, so a camera box can run what the cloud model knows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cnn(chans: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(chans, chans[1:]):
        layers += [nn.Conv2d(a, b, 3, 2, 1), nn.BatchNorm2d(b), nn.ReLU()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Frame (B, 3, 64, 64) and crops (B, 2, 3, 32, 32) in [0, 1] to a feature vector. Positions are kept (no global
    pooling): which slot the tool is at is a question of where."""

    def __init__(self, dim: int):
        super().__init__()
        self.full = _cnn([3, 16, 32, 32])          # 64 -> 8
        self.crop = _cnn([3, 16, 16])              # 32 -> 8
        self.head = nn.Sequential(nn.Linear(32 * 8 * 8 + 2 * 16 * 8 * 8, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, frame: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
        a = self.full(frame * 2 - 1).flatten(1)
        b = self.crop(crops.flatten(0, 1) * 2 - 1).reshape(frame.shape[0], -1)
        return self.head(torch.cat([a, b], 1))


class TCN(nn.Module):
    """Causal dilated convolutions over features (B, dim, T) to step logits (B, K, T)."""

    def __init__(self, dim: int, steps: int, layers: int):
        super().__init__()
        self.dil = [2 ** i for i in range(layers)]
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, 2, dilation=d) for d in self.dil])
        self.out = nn.Conv1d(dim, steps, 1)

    @property
    def window(self) -> int:
        return sum(self.dil) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # before the first frame, the stream is taken to have shown the first frame: the edge runtime starts the same way
        for d, c in zip(self.dil, self.convs):
            x = x + F.relu(c(F.pad(x, (d, 0), mode="replicate")))
        return self.out(x)


class Student(nn.Module):
    def __init__(self, steps: int, dim: int, layers: int):
        super().__init__()
        self.enc = Encoder(dim)
        self.tcn = TCN(dim, steps, layers)

    def forward(self, frames: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
        """frames (B, T, 3, 64, 64), crops (B, T, 2, 3, 32, 32) -> logits (B, T, K)."""
        B, T = frames.shape[:2]
        f = self.enc(frames.flatten(0, 1), crops.flatten(0, 1)).reshape(B, T, -1)
        return self.tcn(f.transpose(1, 2)).transpose(1, 2)


def to_tensors(frames: np.ndarray, crops: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, 64, 64, 3) uint8 and (2, T, 32, 32, 3) uint8 to (T, 3, 64, 64) and (T, 2, 3, 32, 32) floats."""
    f = torch.from_numpy(frames).float().div(255).permute(0, 3, 1, 2)
    c = torch.from_numpy(crops).float().div(255).permute(1, 0, 4, 2, 3)
    return f, c


def train(videos: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]], steps: int, cfg: dict,
          log=print) -> Student:
    """videos: (frames, crops, hard labels or None, teacher probs (T, K) or None-like zeros, allowed-steps mask (K,)).
    Loss: KL to the teacher at temperature cfg.temperature where there is a teacher, cross-entropy where there are labels."""
    torch.manual_seed(cfg["seed"])
    model = Student(steps, cfg["dim"], cfg["tcn_layers"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    seq = min(64, min(len(v[0]) for v in videos))
    chunks = [(i, s) for i, v in enumerate(videos) for s in range(0, len(v[0]) - seq + 1, seq // 2)]
    tens = [to_tensors(v[0], v[1]) for v in videos]
    g = np.random.default_rng(cfg["seed"])
    T = cfg["temperature"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * ((len(chunks) + 7) // 8), pct_start=0.1)
    for ep in range(cfg["epochs"]):
        model.train()
        order = g.permutation(len(chunks))
        tot = 0.0
        for b in range(0, len(order), 8):
            sel = [chunks[j] for j in order[b: b + 8]]
            fr = torch.stack([tens[i][0][s: s + seq] for i, s in sel])
            cr = torch.stack([tens[i][1][s: s + seq] for i, s in sel])
            logits = model(fr, cr)
            mask = torch.stack([torch.from_numpy(videos[i][4]) for i, _ in sel])[:, None, :]
            logits = logits.masked_fill(~mask, -1e4)
            loss = torch.zeros(())
            for k, (i, s) in enumerate(sel):
                lab, soft = videos[i][2], videos[i][3]
                if soft is not None:
                    p = torch.from_numpy(soft[s: s + seq]).clamp(min=1e-6)
                    p = (p.log() / T).masked_fill(~mask[k], -1e4).softmax(-1)
                    loss = loss + T * T * F.kl_div(F.log_softmax(logits[k] / T, -1), p, reduction="batchmean")
                if lab is not None:
                    loss = loss + F.cross_entropy(logits[k], torch.from_numpy(lab[s: s + seq]))
            loss = loss / len(sel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"student epoch {ep + 1}: loss {tot / max(1, len(order) // 8):.3f}")
    return model


@torch.no_grad()
def predict(model: Student, frames: np.ndarray, crops: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    model.eval()
    f, c = to_tensors(frames, crops)
    logits = model(f[None], c[None])[0]
    return logits.masked_fill(~torch.from_numpy(allowed)[None], -1e4).softmax(-1).numpy()
