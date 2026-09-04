# Production / Reproducibility Runbook

This document describes how to run **CVRKD-IQA** (this repository) in a
reproducible, operationally safe way: pinned environments, canonical data layout,
deterministic training runs, checkpoint hygiene, evaluation, and serving the trained
student model behind an inference API.

It complements the quick start in the [README](README.md).

---

## 1. Scope & assumptions

| Item | Assumption |
| --- | --- |
| Runtime | Python 3.9, Linux, NVIDIA GPU with CUDA |
| Training data | KADID-10K (distorted) + DIV2K (HQ references) |
| Eval data | LIVE, CSIQ, TID2013, KonIQ-10K, LIVEC, BID, PIQ23 |
| Weights | Stored under `model_zoo/`, loaded by path from CLI flags |
| Tracking | Config snapshot auto-saved to `checkpoint_DistillationIQA/setting.txt` |

The reference IQA baselines in `models/` (`CNNIQA`, `HyperIQA`, `WaDIQaM`, …) are
research comparators, **not** part of the production student pipeline.

---

## 2. Environment

Use an isolated, reproducible environment and install exact wheels for the GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Pin exact torch builds for your CUDA driver, e.g. cu118:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

For a fully hermetic setup, containerize:

```dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "test_DistillationIQA_single.py"]
```

> Freeze **all** transitive deps before a release: `pip freeze > constraints.txt`
> and install with `pip install -c constraints.txt`.

---

## 3. Data

Datasets are gitignored; download them once and keep the layout stable.

### Canonical layout

```
dataset/
├── kadid10k/
│   ├── dmos.csv
│   └── images/
├── DIV2K_ref/
│   ├── train_HR/      # HQ references used during training
│   └── val_HR/        # HQ references used during testing
├── LIVE/  CSIQ/  TID2013/  LIVEC/  koniq-10k/  BID/  PIQ23/
```

Dataset root paths are hard-coded in each script's `folder_path` / `img_num`
dicts (e.g. `train_DistillationIQA.py`, `test_DistillationIQA.py`) — keep the
default `./dataset/<NAME>` layout or edit those dicts consistently.

### Integrity

Before a training run, verify every index range in the script matches the number of
samples you downloaded (see the `img_num` dicts) and that each CSV/MAT label file is
present:

```bash
ls dataset/kadid10k/dmos.csv dataset/LIVE/dmos_realigned.mat \
   dataset/TID2013/mos_with_names.txt
```

Missing a label file fails silently later — check logs for empty epochs.

---

## 4. Training — reproducible runs

### Configuration

- Every knob is an argparse flag in `option_train_DistillationIQA*.py`.
- The effective config is snapshotted automatically to
  `checkpoint_DistillationIQA/setting.txt` — store it with the run artifacts.
- Checkpoints are written each epoch as
  `checkpoint_DistillationIQA/models/Distillation_inner_<epoch>_saved_model.pth`.

### Recommended run template

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --nproc_per_node=1 train_DistillationIQA.py \
    --train_dataset kadid10k \
    --test_dataset tid2013 \
    --self_patch_num 10 \
    --patch_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --lr 2e-5 \
    --feature_stacking False \
    --repeatable_loss False
```

### Determinism

The training loop seeds only the train-index shuffle, so bit-exact reproducibility
requires explicit seeding. Before `solver.train()` add:

```python
import random, numpy, torch
random.seed(seed); numpy.random.seed(seed); torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Record the seed and the exact commit (`git rev-parse HEAD`) alongside the setting.txt
snapshot in your experiment log. Note: the test code draws **random crops**
(`RandomCrop`) per image at inference, so average over several runs when comparing models.

### Run bookkeeping

```
runs/<run_id>/
├── setting.txt            # auto-snapshot (copy after run starts)
├── seed.txt               # manual
├── git_commit.txt         # manual
├── models/                # checkpoints per epoch
└── logs/                  # train/val acc + loss curves (auto-written)
```

Training already writes loss curves to `checkpoint_DistillationIQA/log/*.txt`
(`train_acc`, `test_acc`, `pred_loss`, `feature_loss`, `loss`). Keep the newest
best checkpoint per fold and prune the rest.

---

## 5. Checkpoint / weight registry

| Role | Expected path (CLI default) |
| --- | --- |
| FR-teacher | `model_zoo/FR_teacher_cross_dataset.pth` |
| NAR-student | `model_zoo/NAR_student_cross_dataset.pth` |

Rules:

- `*.pth` / `*.pt` are gitignored — weights are **never** committed to the repo.
- State dicts are loaded via the model's tolerant `_load_state_dict`; a dimension
  mismatch on a key fails loudly (good). Check that the loaded dict matches the
  variant you instantiate (`org` vs `stackingV1` vs `stackingV2`) — they are **not**
  interchangeable.
- For releases, publish weights with metadata: model variant, patch size, distillation
  layer, epoch, and the exact metrics from the run that produced them.

---

## 6. Evaluation

### Cross-dataset benchmark

```bash
python test_DistillationIQA.py        # runs LIVE + LIVEC folds by default
```

- Reports **SRCC / PLCC / KRCC**. PLCC can be computed after a 4-parameter logistic
  fit to the MOS scale (`--use_fitting_prcc_srcc`, default `True`); SRCC/KRCC are
  monotonic and need no fitting.
- Predictions and ground truth are dumped to `pred_*.txt` / `gt_*.txt`.

### Single-image scoring

Prefer the class API in `test_DistillationIQA_single.py` (import it; it is a module,
not a CLI):

```python
from test_DistillationIQA_single import DistillationIQASolver

solver = DistillationIQASolver(
    student_address="./model_zoo/NAR_student_cross_dataset.pth",
    net_mode="org",                       # org | stackingV1 | stackingV2
)
score = solver.cvr_on_single_image("lq.png", "ref_hq.png")  # content-variant ref OK
```

The returned score is on the model's arbitrary scale. Map it to a target MOS range
with the logistic fit in `tools.py::convert_obj_score` before exposing it to users.

### Minimum acceptance bar

Compare every candidate model against the previous best on the **same** folds of all
held-out sets and require an improvement (or no regression) on SRCC before promotion.
Average ≥ 3 repetitions to damp the random-crop variance.

---

## 7. Serving (inference API)

The student forward pass is: LQ image + a content-variant HQ reference →
quality score. Wrap it behind an idempotent API; below is a minimal FastAPI sketch
(concept only — no new files are added by this document).

```python
# app.py (illustrative)
import io
from fastapi import FastAPI, UploadFile
from PIL import Image
from test_DistillationIQA_single import DistillationIQASolver

app = FastAPI()
solver = DistillationIQASolver("model_zoo/NAR_student_cross_dataset.pth", "org")

@app.post("/score")
def score(lq: UploadFile, ref: UploadFile):
    lq_img, ref_img = Image.open(lq.file).convert("RGB"), Image.open(ref.file).convert("RGB")
    lq_img.save("/tmp/lq.png"); ref_img.save("/tmp/ref.png")
    return {"quality_score": solver.cvr_on_single_image("/tmp/lq.png", "/tmp/ref.png")}
```

Operational checklist:

- **Model warm-up & single instance:** load the checkpoint once at startup; don't
  reload per request.
- **Batch requests** (server needs higher throughput) by routing many LQ/ref pairs
  through `solver.test`, which already batches patches per image.
- **GPU isolation:** one worker process per GPU; set `CUDA_VISIBLE_DEVICES`.
- **Input policy:** fixed patch size (default `224`), reject/`resize=True` for
  smaller images, define your MOS-scale mapping response contract.
- **Observability:** log score distribution, p95 latency, and input dimensions;
  alert on GPU OOM (patches are shaped `[1, self_patch_num, 3, H, W]` per image).
- **Fallbacks:** return a clear error for unreadable images; cap concurrent requests
  to available VRAM.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| Empty epochs or 0 samples | Dataset layout / label file mismatch — check `img_num` vs. downloaded files |
| `RuntimeError: dimension mismatch ... in checkpoint` | Checkpoint built for a different variant (`org`/`stackingV1`/`stackingV2`) or `self_patch_num` |
| Different scores run-to-run | RandomCrop at test time — average ≥ 3 runs or use fixed center patches |
| Slow first epoch | ResNet-50 ImageNet weights are downloaded via `model_zoo.load_url` at first use — cache them |
| OOM on inference | Reduce `--self_patch_num` or batch images one at a time |
| Settings written to wrong path | `--checkpoint_dir` default `./checkpoint_DistillationIQA/`; set per-run with the run id |

---

## 9. Release checklist

- [ ] `requirements.txt` resolved to a frozen `constraints.txt` on the target GPU image.
- [ ] Datasets verified (layout + index counts), documented hashes stored off-repo.
- [ ] Best student checkpoint published to `model_zoo/` (gitignored) or external storage
      with full metadata (variant, epoch, config, metrics, commit, seed).
- [ ] Config snapshot (`setting.txt`) and git commit SHA archived with the run.
- [ ] Cross-dataset eval reproduced ≥ 3×; mean ± std reported.
- [ ] Single-image smoke test passed on synthetic and authentic samples.
- [ ] Serving path tested for warm start, error handling, and MOS-scale mapping.
- [ ] LICENSE/attribution for CVRKD-IQA, HyperIQA, and all benchmark datasets confirmed.

---

## 10. References

- Paper: Yin et al., *Content-Variant Reference Image Quality Assessment via Knowledge
  Distillation*, arXiv:2202.13123.
- Base repo: [CVRKD-IQA](https://github.com/guanghaoyin/CVRKD-IQA) · [HyperIQA](https://github.com/SSL92/hyperIQA)
- Benchmark datasets: KADID-10K, DIV2K, LIVE, CSIQ, TID2013, KonIQ-10K, LIVEC, BID, PIQ23.
