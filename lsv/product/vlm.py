"""The video-language model: Qwen2.5-VL from transformers, its video patches built the way its processor builds them,
a prompt for step recognition, and the answer read back as probabilities.

The video it reads is the camera frame with two insets beside it: the left and the right hand, cut out around the
tracked hands at full resolution. That is where the look-alike steps differ.

The model reads; code decides. A step answer is the next-token distribution restricted to the steps the station's
standard work contains, so the model can never answer with a step the station does not have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from .tokenizer import build as build_tokenizer

MEAN, STD = 0.5, 0.5


@dataclass
class Recog:
    """One question. The clip is a view into a recording (no copy): source[idx] is (T, H, W, 3), the camera frame
    with the two hand crops beside it."""
    source: np.ndarray
    idx: np.ndarray
    options: list[str]
    answer: str | None = None

    @property
    def frames(self) -> np.ndarray:
        return self.source[self.idx]


def patchify(frames: np.ndarray, patch: int, tp: int, merge: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Frames (T, H, W, 3) uint8 to Qwen2-VL patches (gt*gh*gw, 3*tp*patch*patch), in the processor's order."""
    x = torch.from_numpy(frames).float().div(255).sub(MEAN).div(STD).permute(0, 3, 1, 2)  # T C H W
    t, c, h, w = x.shape
    gt, gh, gw = t // tp, h // patch, w // patch
    x = x.reshape(gt, tp, c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gt * gh * gw, c * tp * patch * patch)
    return x, (gt, gh, gw)


class VLM:
    def __init__(self, steps: list[str], cfg: dict, seed: int = 0):
        self.cfg, self.steps = cfg, steps
        self.max_frames = 32
        self.tok = build_tokenizer(steps, self.max_frames)
        v, tx = cfg["vision"], cfg["text"]
        ids = {k: self.tok.convert_tokens_to_ids(k) for k in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
        mc = Qwen2_5_VLConfig(
            text_config=dict(vocab_size=len(self.tok), hidden_size=tx["hidden_size"], intermediate_size=tx["intermediate_size"],
                             num_hidden_layers=tx["num_hidden_layers"], num_attention_heads=tx["num_attention_heads"],
                             num_key_value_heads=tx["num_key_value_heads"], max_position_embeddings=1024,
                             rope_scaling={"type": "mrope", "mrope_section": tx["mrope_section"]}, tie_word_embeddings=True,
                             bos_token_id=None, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id),
            vision_config=dict(depth=v["depth"], hidden_size=v["hidden_size"], intermediate_size=v["intermediate_size"],
                               num_heads=v["num_heads"], out_hidden_size=tx["hidden_size"], patch_size=v["patch_size"],
                               spatial_merge_size=v["spatial_merge_size"], temporal_patch_size=v["temporal_patch_size"],
                               in_channels=3, fullatt_block_indexes=[v["depth"] - 1], window_size=64),
            image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
            vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"])
        torch.manual_seed(seed)
        self.model = Qwen2_5_VLForConditionalGeneration(mc)
        self.step_ids = torch.tensor(self.tok.convert_tokens_to_ids(steps))
        self.num_ids = torch.tensor(self.tok.convert_tokens_to_ids([str(i) for i in range(self.max_frames)]))

    # ------------------------------------------------------------------ prompts
    def _video(self, frames: np.ndarray, pv: list, grids: list) -> str:
        v = self.cfg["vision"]
        p, g = patchify(frames, v["patch_size"], v["temporal_patch_size"], v["spatial_merge_size"])
        pv.append(p)
        grids.append(g)
        n = g[0] * g[1] * g[2] // v["spatial_merge_size"] ** 2
        return "<|vision_start|> " + " ".join(["<|video_pad|>"] * n) + " <|vision_end|>"

    def _recog_text(self, s: Recog, pv: list, grids: list) -> tuple[str, str | None]:
        parts = ["full view left hand right hand", self._video(s.frames, pv, grids),
                 "question which step ? options :", " | ".join(s.options), "answer :"]
        return " ".join(parts), s.answer

    def batch(self, samples: list, train: bool) -> dict:
        pv: list = []
        grids: list = []
        seqs, masks, ans_pos = [], [], []
        for s in samples:
            prompt, ans = self._recog_text(s, pv, grids)
            p_ids = self.tok(prompt, add_special_tokens=False).input_ids
            a_ids = self.tok(ans, add_special_tokens=False).input_ids + [self.tok.eos_token_id] if (train and ans) else []
            seqs.append(p_ids + a_ids)
            masks.append([0] * len(p_ids) + [1] * len(a_ids))
            ans_pos.append(len(p_ids) - 1)
        n = max(map(len, seqs))
        ids = torch.full((len(seqs), n), self.tok.pad_token_id)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        labels = torch.full((len(seqs), n), -100)
        for i, (sq, m) in enumerate(zip(seqs, masks)):
            ids[i, : len(sq)] = torch.tensor(sq)
            att[i, : len(sq)] = 1
            lab = torch.tensor([t if k else -100 for t, k in zip(sq, m)])
            labels[i, : len(sq)] = lab
        out = {"input_ids": ids, "attention_mask": att, "pixel_values_videos": torch.cat(pv), "video_grid_thw": torch.tensor(grids)}
        return {"inputs": out, "labels": labels, "answer_pos": torch.tensor(ans_pos)}

    # ------------------------------------------------------------------ training losses
    def loss(self, samples: list, pairs: dict[str, list[str]] | None = None, pair_weight: float = 0.0) -> torch.Tensor:
        b = self.batch(samples, train=True)
        out = self.model(**b["inputs"])
        logits = out.logits[:, :-1]
        target = b["labels"][:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
        if pairs and pair_weight > 0:
            # the look-alike loss: at the answer position, the true step against the steps it is confused with
            extra, n = 0.0, 0
            for i, s in enumerate(samples):
                if s.answer in pairs:
                    group = [s.answer] + pairs[s.answer]
                    ids = torch.tensor(self.tok.convert_tokens_to_ids(group))
                    z = out.logits[i, b["answer_pos"][i], ids]
                    extra = extra + F.cross_entropy(z[None], torch.zeros(1, dtype=torch.long))
                    n += 1
            if n:
                loss = loss + pair_weight * extra / n
        return loss

    # ------------------------------------------------------------------ reading answers
    @torch.no_grad()
    def step_probs(self, samples: list[Recog], batch: int = 64) -> np.ndarray:
        """P(step) over the full catalog, zero outside each sample's options."""
        self.model.eval()
        out = np.zeros((len(samples), len(self.steps)), np.float32)
        for i in range(0, len(samples), batch):
            chunk = samples[i: i + batch]
            b = self.batch(chunk, train=False)
            logits = self.model(**b["inputs"]).logits
            z = logits[torch.arange(len(chunk)), b["answer_pos"]][:, self.step_ids]
            for j, s in enumerate(chunk):
                allowed = torch.tensor([st in s.options for st in self.steps])
                zz = z[j].masked_fill(~allowed, float("-inf"))
                out[i + j] = F.softmax(zz, -1).numpy()
        return out

    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
