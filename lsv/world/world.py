"""The simulated shop floor: stations, workers doing their standard work with deviations, and an overhead camera.
This module holds the truth (which step every frame shows, where each hand and tool is). The product never imports
it; the evaluation compares the product's answers with it.

The steps that look alike are built into the pixels:
- fastening at slot a or slot b differs by a few pixels of tool position;
- fastening and the torque check at slot a differ by the tool (length, width, colour);
- inserting and removing the clip pass through the same frames in opposite directions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..keyed import rng as keyed

WORLD = 96
REGIONS = {  # where furniture may stand, in the 96 x 96 bench
    "housing_bin": ((8, 26), (14, 40)),
    "screw_tray": ((68, 88), (12, 30)),
    "clip_tray": ((8, 24), (50, 66)),
    "fixture": ((40, 56), (36, 50)),
    "output": ((72, 88), (56, 74)),
    "tool_rest": ((70, 84), (34, 46)),
}


@dataclass
class Station:
    name: str
    layout: dict[str, np.ndarray]
    slot_gap: float
    driver: tuple[float, float, float]
    wrench: tuple[float, float, float]
    housing: tuple[float, float, float]
    bench: tuple[float, float, float]
    standard_work: list[str]

    def slots(self, variant: bool) -> tuple[np.ndarray, np.ndarray]:
        f = self.layout["fixture"] + (np.array([3.0, 2.0]) if variant else 0.0)
        return f + [-self.slot_gap / 2, -2], f + [self.slot_gap / 2, -2]


@dataclass
class Segment:
    step: str
    start: float
    end: float
    cycle: int


@dataclass
class Recording:
    station: str
    condition: str
    index: int
    frames: np.ndarray        # (T, 64, 64, 3) uint8
    labels: np.ndarray        # (T,) step index in the catalog
    hands: np.ndarray         # (T, 2, 4) boxes x0 y0 x1 y1 in image pixels; 0 is the left hand, 1 the right
    hand_visible: np.ndarray  # (T, 2)
    tool: np.ndarray          # (T, 4) the held tool's box
    tool_visible: np.ndarray  # (T,)
    segments: list[Segment]
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.station}/{self.condition}/{self.index}"


def make_station(name: str, spec: dict, catalog_steps: list[str], seed: str) -> Station:
    r = keyed(seed, "station", name)
    layout = {k: np.array([r.uniform(*xr), r.uniform(*yr)]) for k, (xr, yr) in REGIONS.items()}
    layout["rest_left"] = np.array([r.uniform(28, 38), 88.0])
    layout["rest_right"] = np.array([r.uniform(58, 68), 88.0])
    gap = spec["slot_gap"] if np.isscalar(spec["slot_gap"]) else r.uniform(*spec["slot_gap"])
    if "tool_colours" in spec and np.ndim(spec["tool_colours"]) == 3:
        drv, wr = spec["tool_colours"][int(r.integers(len(spec["tool_colours"])))]
    else:
        drv, wr = spec["tool_colours"]
    housing = spec["housing"] if np.ndim(spec["housing"]) == 1 else spec["housing"][int(r.integers(len(spec["housing"])))]
    bench = tuple(float(v) for v in r.uniform(0.35, 0.6, 3))
    sw = spec["standard_work"]
    assert all(s in catalog_steps for s in sw), sw
    return Station(name, layout, float(gap), tuple(drv), tuple(wr), tuple(housing), bench, list(sw))


def schedule(st: Station, cfg: dict, seconds: float, r: np.random.Generator) -> list[Segment]:
    durs, dev = cfg["durations"], cfg["deviations"]
    speed = float(r.uniform(*cfg["worker_speed"]))
    t = float(r.uniform(0, 1.5))
    segs = [Segment("idle", 0.0, t, -1)] if t > 0 else []
    k = 0
    while t < seconds:
        steps = list(st.standard_work)
        if r.uniform() < dev["skip"]:
            cand = [i for i, s in enumerate(steps) if s in ("fasten_b", "torque_a", "inspect")]
            if cand:
                del steps[int(r.choice(cand))]
        if r.uniform() < dev["swap"]:
            i = int(r.integers(2, len(steps) - 2))
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
        pause_at = int(r.integers(1, len(steps))) if r.uniform() < dev["pause"] else -1
        for i, s in enumerate(steps):
            if i == pause_at:
                d = float(r.uniform(2, 5))
                segs.append(Segment("idle", t, t + d, k))
                t += d
            d = durs[s] * speed * float(np.exp(r.normal(0, cfg["duration_jitter"])))
            segs.append(Segment(s, t, t + d, k))
            t += d
        gap = float(r.uniform(*durs["idle_between_cycles"]))
        segs.append(Segment("idle", t, t + gap, -1))
        t += gap
        k += 1
    return segs


def _ease(p: float) -> float:
    return 0.5 - 0.5 * np.cos(np.pi * min(max(p, 0.0), 1.0))


class Camera:
    def __init__(self, cond: dict, r: np.random.Generator, size: int):
        draw = lambda a: float(r.uniform(*a)) if a[1] > a[0] else float(a[0])  # noqa: E731
        th, s = np.deg2rad(draw(cond["rotation_deg"])), draw(cond["scale"])
        dx, dy = draw(cond["shift"]), draw(cond["shift"])
        c = WORLD / 2
        src = np.float32([[0, 0], [WORLD, 0], [WORLD, WORLD], [0, WORLD]])
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        dst = ((src - c) @ rot.T * s + c + [dx, dy]) * (size / WORLD)
        self.H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        self.gain, self.noise, self.size = draw(cond["gain"]), draw(cond["noise"]), size

    def points(self, pts: np.ndarray) -> np.ndarray:
        p = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1) @ self.H.T
        return p[..., :2] / p[..., 2:3]

    def box(self, centre: np.ndarray, half: float) -> np.ndarray:
        corners = centre + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        q = self.points(corners)
        return np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def render(st: Station, segs: list[Segment], cond_name: str, cond: dict, seconds: float, fps: int, size: int,
           step_index: dict[str, int], r: np.random.Generator, index: int) -> Recording:
    cam = Camera(cond, r, size)
    L = st.layout
    variant = bool(cond["variant"])
    housing = np.array([0.55, 0.62, 0.78]) if variant else np.array(st.housing)
    slot_a, slot_b = st.slots(variant)
    fix = L["fixture"]
    clip_pt = fix + [0, 6]
    sleeve = np.array([[0.15, 0.25, 0.55], [0.2, 0.2, 0.22], [0.55, 0.15, 0.12]][int(r.integers(3))])
    glove = np.array([[0.85, 0.65, 0.5], [0.6, 0.6, 0.62], [0.1, 0.1, 0.12]][int(r.integers(3))])
    bench_r = keyed("bench", st.name)
    base = np.ones((WORLD, WORLD, 3), np.float32) * np.array(st.bench, np.float32)
    base += cv2.GaussianBlur(bench_r.normal(0, 0.04, (WORLD, WORLD)).astype(np.float32), (0, 0), 6, sigmaY=0.8)[..., None]
    for key, col, (w, h) in [("housing_bin", (0.25, 0.3, 0.35), (14, 12)), ("screw_tray", (0.3, 0.3, 0.3), (12, 10)),
                             ("clip_tray", (0.28, 0.32, 0.28), (12, 10)), ("output", (0.2, 0.32, 0.22), (16, 14)),
                             ("fixture", (0.22, 0.22, 0.24), (22, 16))]:
        x, y = L[key]
        cv2.rectangle(base, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), col, -1)
    cx, cy = L["clip_tray"]
    for i in range(4):
        cv2.rectangle(base, (int(cx - 4 + i * 2), int(cy - 1)), (int(cx - 3 + i * 2), int(cy + 1)), (0.2, 0.75, 0.3), -1)
    n = int(seconds * fps)
    frames = np.zeros((n, size, size, 3), np.uint8)
    labels = np.zeros(n, np.int64)
    hands_b = np.zeros((n, 2, 4))
    hand_vis = np.ones((n, 2), bool)
    tool_b = np.zeros((n, 4))
    tool_vis = np.zeros(n, bool)
    passes = []
    for _ in range(r.poisson(float(r.uniform(*cond["passers_per_minute"])) * seconds / 60)):
        passes.append((r.uniform(0, seconds), r.uniform(2, 4), r.uniform(20, 76), r.choice([-1, 1])))
    seg_i, start_l, start_r = 0, L["rest_left"].copy(), L["rest_right"].copy()
    end_l, end_r = start_l.copy(), start_r.copy()
    clip_on = False
    for f in range(n):
        t = f / fps
        while seg_i < len(segs) - 1 and segs[seg_i].end <= t:
            start_l, start_r = end_l.copy(), end_r.copy()
            prev = segs[seg_i].step
            if prev == "insert_clip":
                clip_on = True
            if prev in ("remove_clip", "unload"):
                clip_on = False
            seg_i += 1
        s = segs[seg_i]
        p = (t - s.start) / max(s.end - s.start, 1e-6)
        hold_l, hold_r = fix + [-10, 4], fix + [10, 5]
        tool_kind, tool_tip = None, None
        clip_in_hand = False
        show_clip = clip_on
        housing_at, scale = (fix.copy(), 1.0)
        if s.step == "idle":
            tl, tr = L["rest_left"], L["rest_right"]
            housing_at = fix.copy() if s.cycle >= 0 else None
        elif s.step == "pick_housing":
            tl, tr = L["housing_bin"], L["rest_right"]
            housing_at = None if p < 0.5 else start_l + (tl - start_l) * _ease(p / 0.45) + [0, -3]
        elif s.step == "place_housing":
            tl, tr = fix + [-2, 0], L["rest_right"]
            housing_at = (start_l + (tl - start_l) * _ease(p / 0.45) + [0, -3]) if p < 0.7 else fix.copy()
        elif s.step == "pick_screw":
            tl, tr = hold_l, L["screw_tray"]
        elif s.step in ("fasten_a", "fasten_b", "torque_a"):
            slot = slot_b if s.step == "fasten_b" else slot_a
            tl, tr = hold_l, slot + [4, 7]
            if 0.25 <= p <= 0.92:
                tool_kind, tool_tip = ("wrench" if s.step == "torque_a" else "driver"), slot
        elif s.step in ("insert_clip", "remove_clip"):
            tray = L["clip_tray"]
            a, b = (tray, clip_pt) if s.step == "insert_clip" else (clip_pt, tray)
            tr = hold_r
            if p < 0.35:
                tl = start_l + (a - start_l) * _ease(p / 0.35)
            else:
                tl = a + (b - a) * _ease((p - 0.35) / 0.45)
            # the left hand ends where the motion ends; the clip rides in it between the two places
            clip_in_hand = 0.38 <= p <= 0.8
            show_clip = (clip_on and s.step == "remove_clip" and p < 0.38) or (s.step == "insert_clip" and p > 0.8)
            end_l, end_r = (b.copy(), tr.copy())
        elif s.step == "inspect":
            tl, tr = fix + [-11, -8], fix + [11, -8]
            housing_at, scale = fix + [0, -6], 1.25
        elif s.step == "unload":
            tl, tr = L["output"], L["rest_right"]
            housing_at = None if p > 0.75 else start_l + (tl - start_l) * _ease(p / 0.45) + [0, -3]
        else:
            raise ValueError(s.step)
        jit = r.normal(0, 0.5, (2, 2))
        if s.step in ("insert_clip", "remove_clip"):
            left = np.asarray(tl, float) + jit[0]
        else:
            left = start_l + (np.asarray(tl, float) - start_l) * _ease(p / 0.45) + jit[0]
            end_l = np.asarray(tl, float).copy()
        right = start_r + (np.asarray(tr, float) - start_r) * _ease(p / 0.45) + jit[1]
        end_r = np.asarray(tr, float).copy()
        img = base.copy()
        if housing_at is not None:
            w, h = 16 * scale, 10 * scale
            x, y = housing_at
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), tuple(housing.tolist()), -1)
            for sl in (slot_a, slot_b):
                q = housing_at + (sl - fix) * scale
                cv2.circle(img, _pt(q), 1, (0.3, 0.3, 0.3), -1)
            if show_clip:
                q = housing_at + (clip_pt - fix) * scale
                cv2.rectangle(img, (int(q[0] - 2), int(q[1] - 1)), (int(q[0] + 2), int(q[1] + 1)), (0.2, 0.75, 0.3), -1)
        # tools resting on the bench unless one is in use
        tr_rest = L["tool_rest"]
        if tool_kind != "driver":
            cv2.line(img, _pt(tr_rest), _pt(tr_rest + [6, 0]), st.driver, 1)
        if tool_kind != "wrench":
            cv2.line(img, _pt(tr_rest + [0, 6]), _pt(tr_rest + [9, 6]), st.wrench, 2)
        for i, (hnd, sh) in enumerate(((left, L["rest_left"] + [-8, 14]), (right, L["rest_right"] + [8, 14]))):
            cv2.line(img, _pt(sh), _pt(hnd), tuple(sleeve.tolist()), 6)
        if tool_kind is not None:
            length, width, col = (9, 1, st.driver) if tool_kind == "driver" else (12, 2, st.wrench)
            tip = tool_tip
            cv2.line(img, _pt(tip), _pt(tip + [length * 0.5, length * 0.8]), col, width)
        for hnd in (left, right):
            cv2.ellipse(img, _pt(hnd), (4, 3), 0, 0, 360, tuple(glove.tolist()), -1)
        if clip_in_hand:
            cv2.rectangle(img, (int(left[0] - 2), int(left[1] - 4)), (int(left[0] + 2), int(left[1] - 2)), (0.2, 0.75, 0.3), -1)
        for t0, dur, y, d in passes:
            if t0 <= t <= t0 + dur:
                x = -20 + (t - t0) / dur * (WORLD + 40)
                c = np.array([x if d > 0 else WORLD - x, y])
                cv2.ellipse(img, _pt(c), (14, 9), 0, 0, 360, (0.18, 0.18, 0.2), -1)
                cv2.circle(img, _pt(c), 5, (0.12, 0.09, 0.07), -1)
                for k, hnd in enumerate((left, right)):
                    if abs(hnd[0] - c[0]) < 14 and abs(hnd[1] - c[1]) < 9:
                        hand_vis[f, k] = False
        im = cv2.warpPerspective(img, cam.H, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        im = im * cam.gain + r.normal(0, cam.noise, im.shape)
        frames[f] = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        labels[f] = step_index[s.step]
        hands_b[f, 0], hands_b[f, 1] = cam.box(left, 4.5), cam.box(right, 4.5)
        for k in range(2):
            c = hands_b[f, k]
            if c[2] < 0 or c[3] < 0 or c[0] >= size or c[1] >= size:
                hand_vis[f, k] = False
        if tool_kind is not None:
            tool_b[f] = cam.box(tool_tip + [2.0, 3.5], 4.0)
            tool_vis[f] = True
    meta = {"gain": cam.gain, "noise": cam.noise, "passers": len(passes), "variant": variant}
    return Recording(st.name, cond_name, index, frames, labels, hands_b, hand_vis, tool_b, tool_vis, segs, meta)


class World:
    """Builds every recording of one world file. Recordings are made on demand and are the same on every call."""

    def __init__(self, catalog: dict, cfg: dict):
        self.catalog, self.cfg = catalog, cfg
        self.steps = list(catalog["steps"])
        self.step_index = {s: i for i, s in enumerate(self.steps)}
        seed = cfg["seed"]
        pre = cfg["pretrain"]
        pool = cfg["standard_work_pool"]
        self.pretrain_stations = [
            make_station(f"P{i + 1}", {**pre, "standard_work": pool[i % len(pool)]}, self.steps, seed) for i in range(pre["stations"])]
        c = cfg["customer"]
        self.customer = make_station(c["name"], {**c, "standard_work": cfg["customer_standard_work"]}, self.steps, seed)

    def recording(self, st: Station, condition: str, index: int) -> Recording:
        r = keyed(self.cfg["seed"], "recording", st.name, condition, index)
        segs = schedule(st, self.cfg, self.cfg["seconds"], r)
        return render(st, segs, condition, self.cfg["conditions"][condition], self.cfg["seconds"], self.catalog["fps"],
                      self.catalog["frame"], self.step_index, r, index)

    def pretrain(self) -> list[Recording]:
        n = self.cfg["pretrain"]["recordings"]
        return [self.recording(st, "normal", i) for st in self.pretrain_stations for i in range(n)]

    def customer_sets(self) -> dict[str, list[Recording]]:
        c, st = self.cfg["customer"], self.customer
        i = 0
        out: dict[str, list[Recording]] = {}

        def take(k: int, cond: str = "normal") -> list[Recording]:
            nonlocal i
            recs = [self.recording(st, cond, i + j) for j in range(k)]
            i += k
            return recs
        out["labelled"] = take(c["labelled"])
        out["validation"] = take(c["validation"])
        out["unlabelled"] = take(c["unlabelled"])
        for cond in self.cfg["conditions"]:
            out[f"test/{cond}"] = take(c["test"], cond)
        return out
