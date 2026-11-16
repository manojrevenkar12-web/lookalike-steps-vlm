"""python -m lsv.eval.cli evaluate [--profile quick] | gate | heldout | readme [--check] | serve"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT

REPORTS = ROOT / "reports"
LOCK = ROOT / "config" / "heldout.lock"


def write(name: str, data: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    p = REPORTS / name
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return p


def read(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def sha(*names: str) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update((ROOT / "config" / n).read_bytes())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lsv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="every design on the dev world; writes reports/dev.json (full) or reports/quick.json")
    e.add_argument("--profile", default="full", choices=["full", "quick"])
    g = sub.add_parser("gate", help="check config/gate.yaml against a report")
    g.add_argument("--report", default="dev.json")
    sub.add_parser("heldout", help="the held-out station, once; refuses after config/heldout.lock exists")
    r = sub.add_parser("readme", help="README.md from the reports")
    r.add_argument("--check", action="store_true")
    s = sub.add_parser("serve", help="the edge service on the exported models of the dev run")
    s.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    if a.cmd == "evaluate":
        from .run import evaluate
        rep = evaluate("world.yaml", "dev", a.profile, with_curve=True)
        print("wrote", write("dev.json" if a.profile == "full" else "quick.json", rep))
        return 0
    if a.cmd == "gate":
        from .gate import check
        rep = read(a.report)
        if rep is None:
            print(f"no reports/{a.report}; run evaluate first", file=sys.stderr)
            return 2
        ok, lines = check(rep)
        for x in lines:
            print(f"{'PASS' if x['ok'] else 'FAIL'}  {x['line']}: {x['value']} {x['op']} {x['limit']}")
        print("GATE PASSED" if ok else "GATE FAILED")
        return 0 if ok else 1
    if a.cmd == "heldout":
        if LOCK.exists():
            print(f"the held-out station was already run: {LOCK.read_text().strip()}", file=sys.stderr)
            return 3
        from .gate import check
        from .run import evaluate
        rep = evaluate("heldout.yaml", "heldout", "full", with_curve=False)
        ok, lines = check(rep)
        rep["gate"] = {"passed": ok, "lines": lines}
        write("heldout.json", rep)
        LOCK.write_text(json.dumps({"ran": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "settings_sha": sha("gate.yaml", "model.yaml", "designs.yaml", "heldout.yaml"), "passed": ok}) + "\n")
        for x in lines:
            print(f"{'PASS' if x['ok'] else 'FAIL'}  {x['line']}: {x['value']} {x['op']} {x['limit']}")
        print("HELD-OUT GATE PASSED" if ok else "HELD-OUT GATE FAILED")
        return 0
    if a.cmd == "readme":
        from .readme import render
        text = render(read("dev.json"), read("heldout.json"))
        if a.check:
            same = (ROOT / "README.md").exists() and (ROOT / "README.md").read_text() == text
            print("README.md matches the reports" if same else "README.md is stale: run python -m lsv.eval.cli readme")
            return 0 if same else 1
        (ROOT / "README.md").write_text(text)
        print("wrote README.md")
        return 0
    if a.cmd == "serve":
        import uvicorn

        from ..product.service import create_app
        uvicorn.run(create_app(ROOT / "runs" / "dev-full" / "onnx", read("dev.json")), host="0.0.0.0", port=a.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
