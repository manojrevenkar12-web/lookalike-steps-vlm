# SKILLS.md

## Onboard a new station

Add its world (layout, slot gap, tools, standard work) as a world file, or in production collect a few labelled
recordings. The steps the model may answer are the station's standard work plus idle. Fine-tune with
`designs.teacher` (LoRA, look-alike loss, crops), distil the edge student on unlabelled recordings, calibrate the gate
on one validation recording. `label_curve` in `config/designs.yaml` shows how many labelled recordings it needs.

## Add a look-alike pair

Add it to `lookalike` in `config/catalog.yaml`. The look-alike loss, the oversampling, the gate's margin and the
metric all read that list. If the world does not make the pair hard yet, add the cue in `render` in
`lsv/world/world.py` and a test in `tests/test_world.py`.

## Swap in a released checkpoint

`VLM` in `lsv/product/vlm.py` builds `Qwen2_5_VLForConditionalGeneration` from a config. For a released checkpoint,
load it with `from_pretrained`, use its processor and tokenizer, keep `patchify` (tested against
`Qwen2VLVideoProcessor`) or the processor itself, and keep the answer restricted to the station's steps. Train with
`torchrun --nproc-per-node N`; `lsv/product/ddp.py` wraps the model in DistributedDataParallel.

## Change what the edge runs

`lsv/product/student.py` (model), `export.py` (ONNX and int8), `edge.py` (runtime). `tests/test_edge.py` checks that
the ONNX models compute what PyTorch computes; keep PyTorch out of `edge.py`, `service.py`, `checks.py`, `tracker.py`.

## Change the checks

`lsv/product/checks.py`, thresholds in `config/model.yaml` under `checks`. The report's monitor table shows every
recording, whether it was weak by the truth and whether it was flagged.
