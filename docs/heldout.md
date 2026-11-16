# The held-out station

Station H1 (`config/heldout.yaml`, seed `heldout-2026`) was run once on 2026-11-16 with the code and settings dev
ended on (`config/heldout.lock`). It is harder than dev on purpose, and was written before any full result: slots
5 units apart instead of 6, a driver and a torque wrench close in colour,
a standard work of 13 steps with two torque checks, and its own eight pretraining stations.

## The gate failed, as it did on dev

| Line | Dev | Held-out | Limit | Held-out |
|---|---:|---:|---|---|
| Look-alike steps right, cloud model | 0.81 | 0.582 | >= 0.9 | fail |
| Frames right, edge model | 0.831 | 0.806 | >= 0.88 | fail |
| Reported frames right, edge model after its gate | 0.935 | 0.935 | >= 0.95 | fail |
| Frames the edge reports | 0.349 | 0.329 | >= 0.7 | fail |
| Hand identity F1 | 0.707 | 0.658 | >= 0.8 | fail |
| Edge milliseconds per frame (int8, one thread) | 3.86 | 3.95 | <= 5.0 | pass |
| Weak recordings flagged by the monitor | 0.5 | 0.929 | >= 0.8 | pass |
| Healthy recordings flagged by the monitor | 0.333 | 1 | <= 0.1 | fail |

## What held

- **The edge model.** 80.6% of frames right on H1 against
  83.1% on dev, F1@50 87.9.
  It is far better than the cloud model it was distilled from on H1 (64.7%):
  its temporal context over 16 frames and the labelled recordings in its loss carry it where the teacher's clip-level
  answers are weak. Distillation here works as a way to use unlabelled video, not only as compression.
- **Runtime.** 3.95 ms a frame in int8 on one thread.
- **Identities.** The left/right DETR kept hand identity F1 at 0.66, against
  0.15 for overlap tracking.
- **The monitor caught weak recordings**: 93% of 14. But only one H1 recording was
  healthy by the truth (edge accuracy of 85% or more), and it was flagged too, so the monitor flags almost everything
  on H1. That is the right call for a station whose edge model gets a fifth of its frames wrong, and not evidence that it
  can tell a good recording from a weak one.

## What did not hold

| Design | Frames, dev | Frames, H1 | Look-alike, dev | Look-alike, H1 |
|---|---:|---:|---:|---:|
| `vlm_zero_shot` | 47.3% | 37.6% | 47.2% | 32.4% |
| `vlm_full` | 72.2% | 53.8% | 62.4% | 41.6% |
| `vlm_lora` | 68.3% | 58.2% | 55.4% | 55.9% |
| `vlm_lora_pairs` | 72.1% | 54.6% | 63.0% | 46.3% |
| `vlm_lora_pairs_crops` | 80.4% | 64.7% | 81.0% | 58.2% |
| `scratch_student` | 79.0% | 76.4% | 82.7% | 71.3% |
| `edge_student` | 83.1% | 80.6% | 83.5% | 76.5% |

1. **The look-alike loss alone made H1 worse** (46.3% against
   55.9% for plain LoRA). With slots 5 units apart the full frame
   carries almost no signal for slot a against slot b, and pushing the model to separate them on that frame fits
   noise. Together with the hand crops it helps (58.2%):
   the loss needs the zoomed evidence to work with.
2. **The cloud model is too small for this station.** About a million parameters, trained on CPU, reaches
   64.7% of frames. This is the limit the README names, and the
   first thing a GPU and a released checkpoint would change.
3. **The edge gate is calibrated on one validation recording**, so it reports 32.9% of frames at
   93.5% right, short of the 95% it was set for.

## What changes next, measured on dev

- Fine-tune on every frame of the labelled recordings instead of every second one, and weight the look-alike loss by
  whether crops are present.
- Calibrate the edge gate on more than one recording (part of the unlabelled set, labelled by the cloud model where it
  is confident), and judge stations against their own validation, not one fixed accuracy.
- Improve box precision of the hand detector (more training, a finer sampling level); identity is lost mostly through
  boxes a few pixels off on 6-pixel hands, not through swaps.

The held-out station is not run again, and `config/gate.yaml` is not moved.
