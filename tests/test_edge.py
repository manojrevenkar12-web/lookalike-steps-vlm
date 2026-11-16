import base64

import numpy as np
import onnxruntime as ort
import pytest
import torch
from fastapi.testclient import TestClient

from lsv.product import checks as CK
from lsv.product.detr import Detr
from lsv.product.edge import EdgePipeline
from lsv.product.export import export, write_settings
from lsv.product.service import create_app
from lsv.product.student import Student, to_tensors

STEPS = ["idle", "fasten_a", "fasten_b", "unload"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    st, de = Student(len(STEPS), 32, 3).eval(), Detr(3, 32, 4, 1, 1).eval()
    g = np.random.default_rng(0)
    f, c = to_tensors((g.random((16, 64, 64, 3)) * 255).astype(np.uint8), (g.random((2, 16, 32, 32, 3)) * 255).astype(np.uint8))
    out = tmp_path_factory.mktemp("onnx")
    export(st, de, out, f.numpy(), c.numpy())
    gate = CK.Gate(0.0, 0.0, CK.pair_index(STEPS, [["fasten_a", "fasten_b"]]))
    write_settings(out, STEPS, np.ones(len(STEPS), bool), gate, st.tcn.window, 9, 32, 0.0, 3)
    return st, de, out, f, c


def test_onnx_models_compute_what_pytorch_computes(exported):
    st, de, out, f, c = exported
    det = ort.InferenceSession(str(out / "detector.onnx"))
    probs, boxes = det.run(None, {"frame": f[:1].numpy()})
    logits, tb = de(f[:1])[-1]
    assert np.allclose(probs, logits.softmax(-1).detach().numpy(), atol=1e-4) and np.allclose(boxes, tb.detach().numpy(), atol=1e-4)
    enc = ort.InferenceSession(str(out / "encoder_fp32.onnx"))
    feats = np.stack([enc.run(None, {"frame": f[i: i + 1].numpy(), "crops": c[i: i + 1].numpy()})[0][0] for i in range(16)])
    with torch.no_grad():
        ref = st.enc(f, c).numpy()
    assert np.allclose(feats, ref, atol=1e-4)
    tmp = ort.InferenceSession(str(out / "temporal.onnx"))
    w = st.tcn.window
    last = tmp.run(None, {"window": feats[-w:].T[None].astype(np.float32)})[0][0]
    with torch.no_grad():
        full = st.tcn(torch.from_numpy(ref).T[None])[0, :, -1].numpy()
    assert np.allclose(last, full, atol=1e-4)


def test_int8_encoder_is_smaller_and_close(exported):
    _, _, out, f, c = exported
    assert (out / "encoder_int8.onnx").stat().st_size < (out / "encoder_fp32.onnx").stat().st_size / 2
    a = ort.InferenceSession(str(out / "encoder_fp32.onnx")).run(None, {"frame": f[:4].numpy()[:1], "crops": c[:1].numpy()})[0]
    b = ort.InferenceSession(str(out / "encoder_int8.onnx")).run(None, {"frame": f[:4].numpy()[:1], "crops": c[:1].numpy()})[0]
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.95


def test_pipeline_and_service(exported):
    _, _, out, _, _ = exported
    pipe = EdgePipeline.from_dir(out)
    r = pipe.step(np.zeros((64, 64, 3), np.uint8))
    assert r["best_guess"] in STEPS and set(r) >= {"step", "confidence", "reported", "hands"}
    client = TestClient(create_app(out))
    frames = (np.random.default_rng(1).random((5, 64, 64, 3)) * 255).astype(np.uint8)
    res = client.post("/stations/S1/frames", json={"frames_b64": base64.b64encode(frames.tobytes()).decode(), "n": 5})
    assert res.status_code == 200 and len(res.json()) == 5
    assert client.post("/stations/S1/frames", json={"frames_b64": "@@", "n": 1}).status_code == 422
    assert client.post("/stations/S1/frames", json={"frames_b64": base64.b64encode(b"123").decode(), "n": 1}).status_code == 422
    assert 'lsv_frames_total{kind="frames"} 5' in client.get("/metrics").text
