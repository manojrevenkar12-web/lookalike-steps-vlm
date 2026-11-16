"""What runs on the camera box: ONNX Runtime and NumPy only, no PyTorch. Frame in, step out, with the hand tracker,
the crops, the int8 encoder, the temporal head and the gate that holds weak answers back."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .checks import Gate
from .tracker import SideTracker, crops


def _session(path: Path) -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


class EdgePipeline:
    def __init__(self, model_dir: Path, steps: list[str], allowed: np.ndarray, gate: Gate, window: int,
                 crop_half: int, crop: int, min_score: float, hysteresis: int, int8: bool = True):
        self.det = _session(model_dir / "detector.onnx")
        self.enc = _session(model_dir / ("encoder_int8.onnx" if int8 else "encoder_fp32.onnx"))
        self.tmp = _session(model_dir / "temporal.onnx")
        self.steps, self.allowed, self.gate = steps, allowed, gate
        self.window, self.half, self.crop = window, crop_half, crop
        self.tracker = SideTracker(min_score, hysteresis)
        self.feats: deque = deque(maxlen=window)
        self.last_box: dict = {}

    @classmethod
    def from_dir(cls, model_dir: Path, int8: bool = True) -> "EdgePipeline":
        """Loads the models and the station's settings written next to them (edge.json)."""
        c = json.loads((model_dir / "edge.json").read_text())
        gate = Gate(c["min_confidence"], c["min_pair_margin"], {tuple(p) for p in c["pairs"]})
        return cls(model_dir, c["steps"], np.array(c["allowed"]), gate, c["window"], c["crop_half"], c["crop"],
                   c["min_score"], c["hysteresis"], int8)

    def detections(self, frame: np.ndarray) -> list:
        probs, boxes = self.det.run(None, {"frame": frame[None].transpose(0, 3, 1, 2).astype(np.float32) / 255})
        p = probs[0, :, :-1]
        cls, score = p.argmax(1), p.max(1)
        cx, cy, w, h = boxes[0].T * frame.shape[0]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        return [(int(c), float(s), b, pp) for c, s, b, pp in zip(cls, score, xyxy, p)]

    def step(self, frame: np.ndarray) -> dict:
        tr = self.tracker.update(self.detections(frame))
        self.last_box.update(tr)  # a hand not seen this frame keeps its last crop window, as in training
        c = crops(frame[None], [dict(self.last_box)], self.half, self.crop)  # (2, 1, c, c, 3)
        f = frame[None].transpose(0, 3, 1, 2).astype(np.float32) / 255
        cr = c.transpose(1, 0, 4, 2, 3).astype(np.float32) / 255
        feat = self.enc.run(None, {"frame": f, "crops": cr})[0][0]
        if not self.feats:
            self.feats.extend([feat] * self.window)  # a new stream starts as if the first frame had lasted
        self.feats.append(feat)
        logits = self.tmp.run(None, {"window": np.stack(self.feats, 1)[None].astype(np.float32)})[0][0]
        logits = np.where(self.allowed, logits, -1e4)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        ok = bool(self.gate.accept(p[None])[0])
        k = int(p.argmax())
        return {"step": self.steps[k] if ok else None, "best_guess": self.steps[k], "confidence": round(float(p[k]), 3),
                "reported": ok, "hands": {("left" if i == 0 else "right"): [round(float(v), 1) for v in b] for i, b in tr.items()}}
