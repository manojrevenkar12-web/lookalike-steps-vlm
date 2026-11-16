# Working in lookalike-steps-vlm

Fast feedback first: `python -m pytest -q` runs the world, model, LoRA, data-parallel, tracking, student, checks,
ONNX, service and invariant tests in a few minutes on CPU. `python -m lsv.eval.cli evaluate --profile quick` runs the
whole pipeline on a sliver of the data in about a minute. `cd web && npm test && npm run typecheck` for the dashboard.
After any change to the product, the world or the settings run

```bash
python -m lsv.eval.cli evaluate && python -m lsv.eval.cli gate && python -m lsv.eval.cli readme
```

and commit the regenerated `reports/dev.json` and `README.md`. Stages are cached in `runs/dev-full/`; delete the
stage files a change affects (or the folder) before running again. CI checks that the README matches the reports and
runs the quick profile; torch is not bitwise reproducible across machines, so CI does not compare full reports.

## Layout

| path | what it is | may import |
|---|---|---|
| `lsv/world` | the simulated stations, workers, camera and conditions; the truth of every frame | `lsv/keyed.py` |
| `lsv/product` | the video-language model, LoRA, the look-alike loss, DETR, the trackers, the edge student, export, the edge runtime, the checks, the service | nothing outside `lsv/product` |
| `lsv/product/edge.py`, `service.py`, `checks.py`, `tracker.py` | what runs on the camera box | NumPy, SciPy, OpenCV, ONNX Runtime: never PyTorch |
| `lsv/eval` | runs every design, scores against the truth, the gate, the README, the CLI | everything |
| `web` | the dashboard: React, TypeScript, Motion, feature-first, reads `reports/dev.json` | the report |
| `config` | catalog (steps, look-alike pairs), world files, model settings, designs, the gate | |

## Invariants (checked in `tests/test_invariants.py` and the tests)

1. **The truth stays in the world.** `lsv/product` never imports `lsv/world` or `lsv/eval`. It gets frames, the
   station's standard work and, for labelled recordings, the labels a person drew.
2. **The world does not read the product's settings.** `lsv/world` reads the catalog and its world file only.
3. **Same luck for every design.** Every recording is drawn from a key naming it (`lsv/keyed.py`); every design is
   trained on the same labelled recordings and tested on the same new ones.
4. **The model reads and code decides.** The model's answer is a distribution over the steps the station has; the
   gate decides what is reported, the monitor decides when a station is flagged.
5. **The edge has no PyTorch.** The camera box runs ONNX Runtime; the ONNX models compute what PyTorch computes (tested).
6. **Settings in config.** Model sizes, training, thresholds and designs live in `config/`; the dashboard's colours,
   sizes and durations in `web/src/shared/theme.css` and `tokens.ts`.
7. **Generated reports.** README numbers come from `reports/`.

## The held-out rule

`config/heldout.yaml` (station H1) is run once (`python -m lsv.eval.cli heldout`) with the code and settings dev
ended on. It writes `config/heldout.lock` and `reports/heldout.json`; findings go into `docs/heldout.md`. Fixes are
made and measured on dev; the held-out station is not run again and `config/gate.yaml` is not moved.
