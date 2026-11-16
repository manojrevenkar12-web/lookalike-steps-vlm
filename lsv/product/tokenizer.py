"""A word-level tokenizer as a Hugging Face fast tokenizer: the shop-floor vocabulary, step names, frame numbers and
the vision tokens Qwen2.5-VL expects."""
from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

SPECIAL = ["<pad>", "<eos>", "<unk>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>"]
WORDS = ["question", "which", "step", "?", "options", "answer", ":", "when", "full", "left", "right", "hand", "view", "|"]


def build(steps: list[str], max_frames: int) -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(SPECIAL + WORDS + list(steps) + [str(i) for i in range(max_frames)])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
