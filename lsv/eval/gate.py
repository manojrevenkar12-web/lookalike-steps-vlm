"""The ship gate: config/gate.yaml, written before any full result, checked line by line against a report."""
from __future__ import annotations

from ..config import ROOT, load


def check(report: dict, root=ROOT) -> tuple[bool, list[dict]]:
    g = load("gate.yaml", root)
    teacher = load("designs.yaml", root)["teacher"]
    c = g["station_condition"]
    mon = report["monitor"]
    lines = [
        ("Look-alike steps right, cloud model", report["vlm"][teacher]["conditions"][c]["lookalike_accuracy"], ">=", g["min_lookalike_accuracy_cloud"]),
        ("Frames right, edge model", report["edge"]["edge_student"]["conditions"][c]["frame_accuracy"], ">=", g["min_frame_accuracy_edge"]),
        ("Reported frames right, edge model after its gate", report["reported"][c]["reported_accuracy"], ">=", g["min_reported_accuracy_edge"]),
        ("Frames the edge reports", report["reported"][c]["coverage"], ">=", g["min_coverage_edge"]),
        ("Hand identity F1", report["tracking"][c]["side"]["idf1"], ">=", g["min_hand_idf1"]),
        ("Edge milliseconds per frame (int8, one thread)", report["edge_runtime"]["int8"]["ms_per_frame"], "<=", g["max_edge_ms_per_frame"]),
        ("Weak recordings flagged by the monitor", mon["weak_flagged"], ">=", g["min_weak_recordings_flagged"]),
        ("Healthy recordings flagged by the monitor", mon["healthy_flagged"], "<=", g["max_healthy_recordings_flagged"]),
    ]
    out = []
    for name, value, op, limit in lines:
        ok = value is not None and value == value and (value >= limit if op == ">=" else value <= limit)
        out.append({"line": name, "value": value, "op": op, "limit": limit, "ok": bool(ok)})
    return all(x["ok"] for x in out), out
