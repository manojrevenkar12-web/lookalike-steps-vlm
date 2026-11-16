"""Runs every design on one world and scores it against the truth. Stages are cached under runs/<label>/ so a report
can be rebuilt without retraining; delete the folder to start over."""
from __future__ import annotations

import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from ..config import ROOT, load
from ..product import checks as CK
from ..product import detr as D
from ..product import samples as S
from ..product import student as ST
from ..product import tracker as TR
from ..product import train as T
from ..product.data import Video
from ..product.vlm import VLM
from ..world.world import Recording, World
from .metrics import score


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Cache:
    def __init__(self, folder: Path, log):
        self.folder, self.log = folder, log
        folder.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, fn):
        p = self.folder / f"{name}.pkl"
        if p.exists():
            return pickle.loads(p.read_bytes())
        t = time.time()
        v = fn()
        p.write_bytes(pickle.dumps(v))
        self.log(f"[{name}] {time.time() - t:.0f} s")
        return v


def to_video(r: Recording, options: list[str], labelled: bool) -> Video:
    return Video(r.key, r.frames, options, r.labels if labelled else None, r.hands if labelled else None,
                 r.hand_visible if labelled else None, r.tool if labelled else None, r.tool_visible if labelled else None)


def evaluate(world_file: str, label: str, profile: str = "full", root: Path = ROOT, log=print, with_curve: bool = True) -> dict:
    torch.set_num_threads(2)
    catalog, designs = load("catalog.yaml", root), load("designs.yaml", root)
    prof = designs["profiles"][profile]
    wcfg = merge(load(world_file, root), prof.get("world", {}))
    mc = merge(load("model.yaml", root), prof.get("model", {}))
    steps = list(catalog["steps"])
    pairs_idx = CK.pair_index(steps, catalog["lookalike"])
    pair_steps = {i for ab in pairs_idx for i in ab}
    pairs = {}
    for a, b in catalog["lookalike"]:
        pairs.setdefault(a, []).append(b)
        pairs.setdefault(b, []).append(a)
    world = World(catalog, wcfg)
    cache = Cache(root / "runs" / f"{label}-{profile}", log)
    vc = mc["vlm"]
    started = time.time()

    def opts(st) -> list[str]:
        return ["idle"] + sorted(set(st.standard_work))

    pre = [to_video(world.recording(st, "normal", i), opts(st), True)
           for st in world.pretrain_stations for i in range(wcfg["pretrain"]["recordings"])]
    sets = world.customer_sets()
    copt = opts(world.customer)
    allowed = np.array([s in copt for s in steps])
    cust = {k: [to_video(r, copt, k in ("labelled", "validation") or k.startswith("test/")) for r in v] for k, v in sets.items()}
    tests = {k[5:]: v for k, v in cust.items() if k.startswith("test/")}
    labelled = cust["labelled"]

    # ------------------------------------------------------------------ hands
    det_cfg = mc["tracker"]["detr"]
    detr_side = cache.get("detr_side", lambda: T.train_detr(pre + labelled, True, det_cfg, det_cfg["stride"], log))
    detr_plain = cache.get("detr_plain", lambda: T.train_detr(pre + labelled, False, det_cfg, det_cfg["stride"], log))
    tc = mc["tracker"]

    def side_tracks(v: Video):
        return TR.track(D.detect(detr_side, v.frames), TR.SideTracker(tc["min_score"], tc["side_hysteresis"]))

    all_videos = pre + [v for vs in cust.values() for v in vs]
    tracks = cache.get("tracks", lambda: {v.key: side_tracks(v) for v in all_videos})
    crops = {v.key: TR.crops(v.frames, tracks[v.key], vc["crop_half"], vc["crop"]) for v in all_videos}

    def tracking_scores():
        out = {}
        for cond, vs in tests.items():
            rows = {"side": [], "overlap": []}
            for v, r in zip(vs, sets[f"test/{cond}"]):
                rows["side"].append(TR.idf1(r.hands, r.hand_visible, tracks[v.key]))
                ov = TR.track(D.detect(detr_plain, v.frames), TR.OverlapTracker(tc["min_score"], **tc["iou_baseline"]))
                rows["overlap"].append(TR.idf1(r.hands, r.hand_visible, ov))
            minutes = sum(len(v.frames) for v in vs) / catalog["fps"] / 60
            out[cond] = {k: {"idf1": round(float(np.mean([x["idf1"] for x in xs])), 4),
                             "id_switches_per_min": round(sum(x["id_switches"] for x in xs) / minutes, 2),
                             "recall": round(float(np.mean([x["recall"] for x in xs])), 4)} for k, xs in rows.items()}
        return out
    tracking = cache.get("tracking_scores", tracking_scores)

    # ------------------------------------------------------------------ the base video-language model
    def pretrain():
        g = np.random.default_rng(mc["pretrain"]["seed"])
        smp = []
        for v in pre:
            idx = list(range(0, len(v.frames), mc["pretrain"]["stride"]))
            with_c, without = [t for t in idx if g.uniform() >= vc["crop_dropout"]], None
            without = [t for t in idx if t not in set(with_c)]
            smp += S.recog(v, crops[v.key], vc["clip_frames"], with_c, steps, True)
            smp += S.recog(v, None, vc["clip_frames"], without, steps, True)
        vlm = VLM(steps, vc, seed=mc["pretrain"]["seed"])
        p = mc["pretrain"]
        losses = T.train_vlm(vlm, smp, p["epochs"], p["lr"], p["batch"], p["seed"], log=log)
        return {"state": vlm.model.state_dict(), "losses": losses, "samples": len(smp), "params": vlm.parameters()}
    base = cache.get("vlm_base", pretrain)

    def fresh() -> VLM:
        v = VLM(steps, vc)
        v.model.load_state_dict(base["state"])
        return v

    ft = mc["finetune"]

    def finetune(spec: dict, recs: list[Video]) -> dict:
        vlm = fresh()
        info = {"trainable": vlm.parameters()}
        if spec["tune"] != "none":
            smp = []
            for v in recs:
                idx = list(range(0, len(v.frames), ft["stride"]))
                if spec["pairs"]:
                    idx += [t for t in idx if steps[int(v.labels[t])] in pairs] * (ft["pair_oversample"] - 1)
                smp += S.recog(v, crops[v.key] if spec["crops"] else None, vc["clip_frames"], idx, steps, True)
            hp = ft[spec["tune"]]
            if spec["tune"] == "lora":
                info["trainable"] = T.add_lora(vlm, hp)
            info["losses"] = T.train_vlm(vlm, smp, hp["epochs"], hp["lr"], ft["batch"], ft["seed"],
                                         pairs if spec["pairs"] else None, ft["pair_weight"] if spec["pairs"] else 0.0, log=log)
            if spec["tune"] == "lora":
                T.merge_lora(vlm)
        info["state"] = vlm.model.state_dict()
        return info

    def probs(state: dict, vs: list[Video], use_crops: bool) -> list[np.ndarray]:
        vlm = fresh()
        vlm.model.load_state_dict(state)
        return [vlm.step_probs(S.recog(v, crops[v.key] if use_crops else None, vc["clip_frames"], list(range(len(v.frames))), steps, False))
                for v in vs]

    vlm_results, teacher_probs = {}, {}
    for name, spec in designs["designs"].items():
        model = cache.get(f"model_{name}", lambda spec=spec: finetune(spec, labelled))

        def run_probs(model=model, spec=spec):
            return {k: probs(model["state"], vs, spec["crops"]) for k, vs in [("validation", cust["validation"]),
                                                                                 ("unlabelled", cust["unlabelled"])] + list(tests.items())}
        out = cache.get(f"probs_{name}", run_probs)
        vlm_results[name] = {"trainable_params": int(model["trainable"]),
                             "conditions": {c: score(out[c], [v.labels for v in tests[c]], steps, pair_steps) for c in tests}}
        if name == designs["teacher"]:
            teacher_probs = out

    curve = {}
    if with_curve:
        tspec = designs["designs"][designs["teacher"]]
        for n in prof.get("label_curve", designs["label_curve"]):
            if n > len(labelled):
                continue
            if n == len(labelled):
                curve[n] = vlm_results[designs["teacher"]]["conditions"]["normal"]
                continue
            m = cache.get(f"curve_{n}", lambda n=n: finetune(tspec, labelled[:n]))
            curve[n] = score(probs(m["state"], tests["normal"], True), [v.labels for v in tests["normal"]], steps, pair_steps)

    # ------------------------------------------------------------------ the edge student
    sc = mc["student"]

    def student_data(with_teacher: bool):
        data = [(v.frames, crops[v.key], v.labels, None, allowed) for v in labelled]
        if with_teacher:  # the recordings nobody labelled, with the cloud model's soft answers
            data += [(v.frames, crops[v.key], None, p, allowed) for v, p in zip(cust["unlabelled"], teacher_probs["unlabelled"])]
        return data
    scratch = cache.get("student_scratch", lambda: ST.train(student_data(False), len(steps), sc, log))
    edge = cache.get("student_edge", lambda: ST.train(student_data(True), len(steps), sc, log))
    edge_probs = {}
    edge_results = {}
    for name, m in (("scratch_student", scratch), ("edge_student", edge)):
        out = {k: [ST.predict(m, v.frames, crops[v.key], allowed) for v in vs]
               for k, vs in [("validation", cust["validation"])] + list(tests.items())}
        edge_results[name] = {"conditions": {c: score(out[c], [v.labels for v in tests[c]], steps, pair_steps) for c in tests},
                              "params": int(sum(p.numel() for p in m.parameters()))}
        if name == "edge_student":
            edge_probs = out
    agreement = {c: round(float(np.mean([(a.argmax(1) == b.argmax(1)).mean() for a, b in zip(edge_probs[c], teacher_probs[c])])), 4)
                 for c in tests}

    # ------------------------------------------------------------------ checks
    ck = mc["checks"]
    gate, gate_val = CK.calibrate(edge_probs["validation"][0], cust["validation"][0].labels, pairs_idx, ck["target_accuracy"])
    every = ck["monitor_every_s"] * catalog["fps"]
    val_disagree = float((edge_probs["validation"][0].argmax(1) != teacher_probs["validation"][0].argmax(1)).mean())
    mon = CK.Monitor(every, ck["monitor_window"], val_disagree + ck["disagreement_margin"], gate_val["coverage"] * ck["min_coverage_ratio"])
    reported, monitor_rows = {}, []
    for c, vs in tests.items():
        acc_r, cov = [], []
        for v, ep, tp in zip(vs, edge_probs[c], teacher_probs[c]):
            a = gate.accept(ep)
            correct = ep.argmax(1) == v.labels
            acc_r.append(correct[a].mean() if a.any() else np.nan)
            cov.append(a.mean())
            res = mon.run(ep, a, tp[::every])
            weak = bool(correct.mean() < ck["weak_below"])
            monitor_rows.append({"recording": v.key, "condition": c, "edge_accuracy": round(float(correct.mean()), 4), "weak": weak,
                                 "flagged": res["flagged_at"] is not None, "flagged_after_s": None if res["flagged_at"] is None else res["flagged_at"] / catalog["fps"],
                                 "why": res["why"]})
        reported[c] = {"reported_accuracy": round(float(np.nanmean(acc_r)), 4), "coverage": round(float(np.mean(cov)), 4)}
    weak = [r for r in monitor_rows if r["weak"]]
    healthy = [r for r in monitor_rows if not r["weak"]]
    monitor = {"validation_disagreement": round(val_disagree, 4), "max_disagreement": round(mon.max_disagreement, 4),
               "min_coverage": round(mon.min_coverage, 4), "weak_recordings": len(weak), "weak_flagged": round(sum(r["flagged"] for r in weak) / max(len(weak), 1), 4) if weak else None,
               "healthy_recordings": len(healthy), "healthy_flagged": round(sum(r["flagged"] for r in healthy) / max(len(healthy), 1), 4),
               "rows": monitor_rows}

    # ------------------------------------------------------------------ edge runtime
    def edge_bench():
        from ..product.edge import EdgePipeline
        from ..product.export import export, write_settings
        out_dir = root / "runs" / f"{label}-{profile}" / "onnx"
        cal = cust["unlabelled"][0]
        f, c = ST.to_tensors(cal.frames[::3], crops[cal.key][:, ::3])
        paths = export(edge, detr_side, out_dir, f.numpy(), c.numpy())
        write_settings(out_dir, steps, allowed, gate, edge.tcn.window, vc["crop_half"], vc["crop"], tc["min_score"], tc["side_hysteresis"])
        res = {}
        for int8 in (True, False):
            v = tests["normal"][0]
            times = []
            for _ in range(3):  # the median of three passes over a recording, each after a warm-up frame
                pipe = EdgePipeline.from_dir(out_dir, int8=int8)
                pipe.step(v.frames[0])
                pipe = EdgePipeline.from_dir(out_dir, int8=int8)
                t0 = time.perf_counter()
                outs = [pipe.step(fr) for fr in v.frames]
                times.append(1000 * (time.perf_counter() - t0) / len(v.frames))
            ms = float(np.median(times))
            guess = np.array([steps.index(o["best_guess"]) for o in outs])
            res["int8" if int8 else "fp32"] = {"ms_per_frame": round(ms, 2), "frame_accuracy": round(float((guess == v.labels).mean()), 4),
                                               "agrees_with_pytorch": round(float((guess == edge_probs["normal"][0].argmax(1)).mean()), 4)}
        res["model_kb"] = {k: round(p.stat().st_size / 1024, 1) for k, p in paths.items()}
        return res
    edge_runtime = cache.get("edge_runtime", edge_bench)

    def showcase():
        """What the dashboard draws: a recording's truth, cloud and edge labels frame by frame, and one frame of each
        look-alike step as the cloud model sees it (frame plus hand crops)."""
        import base64

        import cv2
        out = {"recordings": [], "frames": {}}
        for c in ("normal", "camera"):
            v = tests[c][0]
            ep, tp = edge_probs[c][0], teacher_probs[c][0]
            out["recordings"].append({"key": v.key, "condition": c, "truth": v.labels.tolist(), "cloud": tp.argmax(1).tolist(),
                                      "edge": ep.argmax(1).tolist(), "reported": gate.accept(ep).astype(int).tolist()})
        v = tests["normal"][0]
        full = S.canvas(v.frames, crops[v.key])
        for name in ("fasten_a", "fasten_b", "torque_a", "insert_clip", "remove_clip"):
            idx = np.where(v.labels == steps.index(name))[0]
            if len(idx):
                t = int(idx[len(idx) // 2])
                img = cv2.resize(full[t], (full.shape[2] * 4, full.shape[1] * 4), interpolation=cv2.INTER_NEAREST)
                ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                out["frames"][name] = base64.b64encode(png.tobytes()).decode()
        return out

    st = world.customer
    report = {
        "label": label, "profile": profile, "seconds": round(time.time() - started),
        "world": {"seed": wcfg["seed"], "customer": st.name, "slot_gap": st.slot_gap, "standard_work": st.standard_work,
                  "pretrain_stations": len(world.pretrain_stations), "pretrain_recordings": len(pre),
                  "frames_pretrain": int(sum(len(v.frames) for v in pre)), "labelled_recordings": len(labelled),
                  "unlabelled_recordings": len(cust["unlabelled"]), "test_recordings": {c: len(v) for c, v in tests.items()},
                  "seconds_per_recording": wcfg["seconds"], "fps": catalog["fps"], "frame": catalog["frame"]},
        "base": {"params": int(base["params"]), "samples": base["samples"], "losses": [round(x, 4) for x in base["losses"]]},
        "vlm": vlm_results, "curve": {str(k): v for k, v in curve.items()}, "edge": edge_results,
        "agreement_edge_cloud": agreement, "tracking": tracking, "gate_calibration": {**gate_val, "min_confidence": round(gate.min_confidence, 3),
                                                                                     "min_pair_margin": gate.min_pair_margin},
        "reported": reported, "monitor": monitor, "edge_runtime": edge_runtime, "steps": steps,
        "lookalike": catalog["lookalike"], "showcase": showcase(),
    }
    return json.loads(json.dumps(report, default=float))
