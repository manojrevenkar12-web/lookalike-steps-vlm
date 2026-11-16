"""The rules of AGENTS.md that code can check."""
import ast
import json
import subprocess
import sys

from lsv.config import ROOT


def imports(path):
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


def test_the_product_never_reads_the_truth():
    for f in (ROOT / "lsv" / "product").glob("*.py"):
        for m in imports(f):
            assert "world" not in m and "eval" not in m, f"{f.name} imports {m}"


def test_the_world_does_not_read_the_product_settings():
    for f in (ROOT / "lsv" / "world").glob("*.py"):
        text = f.read_text()
        for name in ("model.yaml", "designs.yaml", "gate.yaml"):
            assert name not in text
        assert not any("product" in m for m in imports(f))


def test_the_edge_runtime_has_no_pytorch():
    code = "import sys, lsv.product.edge, lsv.product.service; assert 'torch' not in sys.modules, 'torch loaded'"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_readme_is_generated_from_the_reports():
    from lsv.eval import cli
    from lsv.eval.readme import render
    if (ROOT / "README.md").exists() and cli.read("dev.json"):
        assert (ROOT / "README.md").read_text() == render(cli.read("dev.json"), cli.read("heldout.json"))


def test_settings_were_not_moved_after_the_held_out_run():
    from lsv.eval.cli import LOCK, sha
    if LOCK.exists():
        assert json.loads(LOCK.read_text())["settings_sha"] == sha("gate.yaml", "model.yaml", "designs.yaml", "heldout.yaml")
