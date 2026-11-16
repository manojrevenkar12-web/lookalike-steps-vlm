"""Exports the edge models to ONNX: the hand detector, the student's frame encoder (quantised to int8 with static
calibration) and its temporal head, which takes the last `window` features and labels the newest frame."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from .detr import Detr
from .student import Student


class _LastFrame(nn.Module):
    def __init__(self, tcn):
        super().__init__()
        self.tcn = tcn

    def forward(self, x):  # (1, dim, window) -> (1, K)
        return self.tcn(x)[:, :, -1]


class _DetrLast(nn.Module):
    def __init__(self, m: Detr):
        super().__init__()
        self.m = m

    def forward(self, x):
        logits, boxes = self.m(x)[-1]
        return logits.softmax(-1), boxes


class _Reader(CalibrationDataReader):
    def __init__(self, frames: np.ndarray, crops: np.ndarray):
        self.items = iter([{"frame": frames[i: i + 1], "crops": crops[i: i + 1]} for i in range(len(frames))])

    def get_next(self):
        return next(self.items, None)


def write_settings(out: Path, steps: list[str], allowed: np.ndarray, gate, window: int, crop_half: int, crop: int,
                   min_score: float, hysteresis: int) -> Path:
    import json
    p = out / "edge.json"
    p.write_text(json.dumps({"steps": steps, "allowed": [bool(a) for a in allowed], "min_confidence": gate.min_confidence,
                             "min_pair_margin": gate.min_pair_margin, "pairs": sorted([list(map(int, x)) for x in gate.pair_index]),
                             "window": window, "crop_half": crop_half, "crop": crop, "min_score": min_score,
                             "hysteresis": hysteresis}, indent=1))
    return p


def export(student: Student, detr: Detr, out: Path, calib_frames: np.ndarray, calib_crops: np.ndarray) -> dict[str, Path]:
    """calib_frames (N, 3, 64, 64) float32 in [0, 1], calib_crops (N, 2, 3, 32, 32)."""
    out.mkdir(parents=True, exist_ok=True)
    student.eval()
    detr.eval()
    p = {k: out / f"{k}.onnx" for k in ("detector", "encoder_fp32", "encoder_int8", "temporal")}
    kw = dict(opset_version=17, dynamo=False)
    torch.onnx.export(_DetrLast(detr), torch.zeros(1, 3, 64, 64), p["detector"], input_names=["frame"], output_names=["probs", "boxes"], **kw)
    torch.onnx.export(student.enc, (torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 3, 32, 32)), p["encoder_fp32"],
                      input_names=["frame", "crops"], output_names=["feature"], **kw)
    w = student.tcn.window
    torch.onnx.export(_LastFrame(student.tcn), torch.zeros(1, student.enc.head[-1].out_features, w), p["temporal"],
                      input_names=["window"], output_names=["logits"], **kw)
    quantize_static(str(p["encoder_fp32"]), str(p["encoder_int8"]), _Reader(calib_frames, calib_crops),
                    quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    for f in p.values():
        onnx.checker.check_model(str(f))
    return p
