"""The camera box's HTTP face: frames in, steps out. ONNX Runtime only. Video stays on the box; what leaves it is the
step, the confidence and whether the step was reported."""
from __future__ import annotations

import base64
import binascii
import threading
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .edge import EdgePipeline


class Frames(BaseModel):
    frames_b64: str = Field(description="raw uint8 RGB frames, base64, shape (n, size, size, 3)")
    n: int = Field(ge=1, le=600)


def create_app(model_dir: Path, int8: bool = True) -> FastAPI:
    app = FastAPI(title="lookalike-steps-vlm edge", version="1.0")
    pipes: dict[str, EdgePipeline] = {}
    lock = threading.Lock()
    counts: Counter[str] = Counter()
    size = 64

    def pipe(station: str) -> EdgePipeline:
        if station not in pipes:
            pipes[station] = EdgePipeline.from_dir(model_dir, int8)
        return pipes[station]

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "models": sorted(p.name for p in model_dir.glob("*.onnx"))}

    @app.post("/stations/{station}/frames")
    def frames(station: str, body: Frames) -> list[dict]:
        try:
            raw = base64.b64decode(body.frames_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(422, f"frames_b64 is not base64: {e}") from e
        if len(raw) != body.n * size * size * 3:
            raise HTTPException(422, f"expected {body.n} frames of {size}x{size}x3 bytes, got {len(raw)} bytes")
        arr = np.frombuffer(raw, np.uint8).reshape(body.n, size, size, 3)
        with lock:
            p = pipe(station)
            out = [p.step(f) for f in arr]
        counts["frames"] += len(out)
        counts["reported"] += sum(o["reported"] for o in out)
        return out

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return "".join(f'lsv_frames_total{{kind="{k}"}} {v}\n' for k, v in sorted(counts.items()))

    return app
