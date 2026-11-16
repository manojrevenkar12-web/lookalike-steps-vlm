"""A small DETR for hands and tools: a convolutional stem, a transformer encoder over the feature map, learned object
queries decoded by a transformer decoder, and set prediction trained with Hungarian matching. Each decoder layer is
supervised, as in DETR.

`sides=True` asks it to tell the left hand from the right hand, which it can see from the arm: an arm reaches in from
its own shoulder. Tracking then keeps identities by side instead of by overlap, which is where an overlap tracker
swaps the two hands when they meet at the fixture.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise generalised IoU of xyxy boxes, (N, M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None] - inter
    iou = inter / union.clamp(min=1e-6)
    lt2, rb2 = torch.min(a[:, None, :2], b[None, :, :2]), torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb2 - lt2).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-6)


class Detr(nn.Module):
    def __init__(self, classes: int, dim: int, queries: int, enc_layers: int, dec_layers: int, size: int = 64):
        super().__init__()
        self.classes, self.size = classes, size
        # stride 2 features for where exactly a box is, stride 4 tokens for the transformer: a hand is 6 pixels wide
        self.fine = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 32, 3, 1, 1), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(32, dim, 3, 2, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU())
        n = (size // 4) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, dim * 2, 0.0, batch_first=True), enc_layers)
        self.dec = nn.ModuleList([nn.TransformerDecoderLayer(dim, 4, dim * 2, 0.0, batch_first=True) for _ in range(dec_layers)])
        # anchor boxes as queries (the DAB-DETR idea): each query starts from a box on a grid, its positional
        # embedding is computed from that box, and every decoder layer refines the box; this converges far faster
        # than DETR's free queries on a small budget
        side = int(np.ceil(np.sqrt(queries)))
        grid = [((i % side + 0.5) / side, (i // side + 0.5) / side, 0.15, 0.15) for i in range(queries)]
        self.anchor = nn.Parameter(torch.logit(torch.tensor(grid, dtype=torch.float32).clamp(1e-3, 1 - 1e-3))[None])
        self.anchor_pos = nn.Sequential(nn.Linear(4 * 16, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, classes + 1)
        # each layer looks at the fine features at a few points around its current box (the Deformable DETR idea)
        # before moving the box
        self.points = 4
        self.offsets = nn.ModuleList([nn.Linear(dim, self.points * 2) for _ in range(dec_layers)])
        self.box = nn.ModuleList([nn.Sequential(nn.Linear(dim + self.points * 32, dim), nn.GELU(), nn.Linear(dim, 4)) for _ in range(dec_layers)])
        for b in self.box:
            nn.init.zeros_(b[-1].weight)
            nn.init.zeros_(b[-1].bias)

    @staticmethod
    def _sine(b: torch.Tensor) -> torch.Tensor:
        freq = 2 ** torch.arange(8, dtype=b.dtype) * torch.pi
        a = b[..., None] * freq
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, 3, H, W) in [0, 1]. Returns (class logits, boxes cxcywh in [0, 1]) after every decoder layer."""
        fine = self.fine(x * 2 - 1)
        f = self.stem(fine).flatten(2).transpose(1, 2) + self.pos
        mem = self.enc(f)
        ref = self.anchor.expand(x.shape[0], -1, -1)  # logits of cx cy w h
        B, Q = x.shape[0], ref.shape[1]
        h = torch.zeros(B, Q, mem.shape[-1])
        out = []
        for layer, offs, box in zip(self.dec, self.offsets, self.box):
            r = ref.sigmoid()
            h = layer(h + self.anchor_pos(self._sine(r)), mem)
            hn = self.norm(h)
            pts = r[..., None, :2] + offs(hn).view(B, Q, self.points, 2).tanh() * r[..., None, 2:]
            sampled = F.grid_sample(fine, pts * 2 - 1, align_corners=False)  # (B, C, Q, points)
            ref = ref + box(torch.cat([hn, sampled.permute(0, 2, 3, 1).reshape(B, Q, -1)], -1))
            out.append((self.cls(hn), ref.sigmoid()))
            ref = ref.detach()
        return out


def match(logits: torch.Tensor, boxes: torch.Tensor, t_cls: torch.Tensor, t_box: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if len(t_cls) == 0:
        return np.array([], int), np.array([], int)
    prob = logits.softmax(-1)[:, t_cls]
    cost = -prob + 5 * torch.cdist(boxes, t_box, p=1) - 2 * giou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(t_box))
    return linear_sum_assignment(cost.detach().numpy())


def giou_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalised IoU of matched xyxy boxes, (N,)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    inter = (torch.min(a[:, 2:], b[:, 2:]) - torch.max(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    union = area_a + area_b - inter
    hull = (torch.max(a[:, 2:], b[:, 2:]) - torch.min(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    return inter / union.clamp(min=1e-6) - (hull - union) / hull.clamp(min=1e-6)


def set_loss(outs: list, targets: list[tuple[torch.Tensor, torch.Tensor]], classes: int) -> torch.Tensor:
    """Hungarian-matched classification, L1 and GIoU losses, summed over the decoder layers."""
    total = 0.0
    weight = torch.ones(classes + 1)
    weight[-1] = 0.1
    for logits, boxes in outs:
        tgt_cls = torch.full(logits.shape[:2], classes, dtype=torch.long)
        bi, qi_all, gt = [], [], []
        with torch.no_grad():
            for i, (tc, tb) in enumerate(targets):
                qi, ti = match(logits[i], boxes[i], tc, tb)
                if len(qi):
                    tgt_cls[i, qi] = tc[ti]
                    bi += [i] * len(qi)
                    qi_all += list(qi)
                    gt.append(tb[ti])
        loss = F.cross_entropy(logits.reshape(-1, classes + 1), tgt_cls.reshape(-1), weight=weight)
        if gt:
            pb, gb = boxes[torch.tensor(bi), torch.tensor(qi_all)], torch.cat(gt)
            l1 = F.l1_loss(pb, gb, reduction="sum") / len(gb)
            gl = (1 - giou_pairs(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(gb))).mean()
            loss = loss + 5 * l1 + 2 * gl
        total = total + loss
    return total


def targets_for(hands: np.ndarray, hand_vis: np.ndarray, tool: np.ndarray, tool_vis: bool, sides: bool, size: int):
    """Boxes (xyxy pixels) to DETR targets. Classes with sides: 0 left hand, 1 right hand, 2 tool; without: 0 hand, 1 tool."""
    cls, bx = [], []
    for k in range(2):
        if hand_vis[k]:
            cls.append(k if sides else 0)
            bx.append(hands[k])
    if tool_vis:
        cls.append(2 if sides else 1)
        bx.append(tool)
    if not bx:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, 4)
    b = np.clip(np.array(bx, np.float32), 0, size) / size
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    return torch.tensor(cls), torch.from_numpy(c)


@torch.no_grad()
def detect(model: Detr, frames: np.ndarray, batch: int = 128) -> list[list[tuple[int, float, np.ndarray]]]:
    """Per frame: (class, score, xyxy pixels) for every query, best class first. No threshold here; the tracker decides."""
    model.eval()
    res = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i: i + batch]).float().div(255).permute(0, 3, 1, 2)
        logits, boxes = model(x)[-1]
        p = logits.softmax(-1)[..., :-1]
        score, cls = p.max(-1)
        xyxy = box_cxcywh_to_xyxy(boxes) * model.size
        for j in range(len(x)):
            res.append([(int(cls[j, q]), float(score[j, q]), xyxy[j, q].numpy(), p[j, q].numpy()) for q in range(p.shape[1])])
    return res
