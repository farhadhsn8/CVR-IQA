<div align="center">

# CVR-IQA / CVRKD-IQA

**Content-Variant Reference Image Quality Assessment via Knowledge Distillation**

<!-- TEMPLATE: About section for GitHub
Short description (About): -->
> Content-variant reference IQA via knowledge distillation. A full-reference
> teacher transfers high-quality distribution priors to a student that only needs a
> **non-pixel-aligned / content-variant reference**, so quality scores no longer
> require a pixel-perfect reference image.

![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<!-- TEMPLATE: GitHub Topics / tags (comma-separated) -->
`image-quality-assessment` `no-reference-iqa` `full-reference-iqa` `knowledge-distillation` `pytorch` `computer-vision` `mlp-mixer` `resnet` `deep-learning` `kadid10k` `live-challenge` `tid2013` `csiq` `koniq10k` `ntire`

</div>

---

## About

CVRKD-IQA is a **content-variant reference** (CVR) image quality assessment method
that relaxes the pixel-alignment requirement of classic full-reference (FR) metrics.
Instead of feeding a pixel-aligned pristine copy, the model takes a high-quality image of
**arbitrary but similar content** and measures the distribution difference between the
degraded (LQ) image and that high-quality (HQ) reference.

Two-stage **knowledge distillation** makes this possible:

1. An **FR-teacher** is trained on pixel-aligned HQ/LQ pairs and learns strong
   HQ–LQ distribution differences across the network.
2. A **NAR-student** (non-aligned reference) is trained with only content-variant
   references while its intermediate features are distilled from the frozen teacher.

At inference the student needs **no aligned reference** and is robust to content
variation, making it practical where only "some good image of the same scene" exists.

### This repository

This is a working fork/extension of the original [CVRKD-IQA](https://github.com/guanghaoyin/CVRKD-IQA)
with research variants kept runnable out of the box:

| Variant | Entry point | Notes |
| --- | --- | --- |
| FR-teacher | `train_DistillationIQA_FR.py` | Full-reference teacher (aligned refs) |
| NAR-student (distilled) | `train_DistillationIQA.py` | Core distillation pipeline |
| `--feature_stacking` | both train scripts | Multi-scale feature stacking (`stackingV2`) |
| `--repeatable_loss` | both train scripts | Patch-consistency regularizer |
| make-ref student | `train_DistillationIQA_makeRef.py` | Student synthesizes its own reference (`DistillationIQANet_makeRef`) |

### Architecture

- **Backbone:** frozen ImageNet-pretrained ResNet-50, multi-scale features (256/512/1024/2048).
- **Difference features:** per-scale 1×1 convs + `HQ − LQ` residual features.
- **Encoders:** two MLP-Mixer towers (`MLP_encoder_lq`, `MLP_encoder_diff`) on 7×7 grids.
- **Head:** lightweight regression MLP (`RegressionFCNet`) → quality score.
- **Distillation:** teacher's intermediate encoder features supervise the student via
  MSE (`feature_loss`) combined with the student's prediction loss.

Model code: `models/DistillationIQA.py`. Reference IQA baselines used during development
(`CNNIQA`, `HyperIQA`, `WaDIQaM`, `TRIQ`, ...) are kept under `models/`.

---

## Repository layout

```
.
├── models/            # Model zoo (DistillationIQA.py + reference IQA baselines)
├── folders/           # Per-dataset Folder classes (LIVE, CSIQ, TID2013, ...)
├── dataloaders/       # DataLoader wrappers (LQ, aligned LQ/HQ, diff-content HQ)
├── option_train_DistillationIQA*.py   # CLI args / config for each stage
├── train_DistillationIQA_FR.py        # 1) train FR teacher
├── train_DistillationIQA.py           # 2) distill NAR student
├── train_DistillationIQA_makeRef.py   #    make-ref variant
├── test_DistillationIQA.py            #    cross-dataset evaluation
├── test_DistillationIQA_single.py     #    single-image scoring API
├── tools.py              # MOS-scale nonlinear fitting (logistic regression)
├── model_zoo/            # pretrained checkpoints (gitignored *.pth)
├── dataset/              # datasets (gitignored, see README data prep)
├── requirements.txt
└── PRODUCTION.md         # production / reproducibility runbook
```

---

## Requirements

```
Python 3.9 · PyTorch ≥ 1.10 · torchvision ≥ 0.11
```

```bash
pip install -r requirements.txt
# CUDA build of torch is recommended, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Data preparation

All datasets are expected under `./dataset/` (gitignored). See
[PRODUCTION.md → Data](PRODUCTION.md#data) for the canonical layout.

- **Training — synthetic:** [KADID-10K](http://database.mmsp-kn.de/kadid-10k-database.html).
- **Training refs — HQ:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) train HR images.
- **Testing refs — HQ:** DIV2K validation HR images.
- **Testing:** [LIVE](http://live.ece.utexas.edu/index.php), [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/),
  [TID2013](http://www.ponomarenko.info/tid2013.htm), [KonIQ-10K](http://database.mmsp-kn.de/koniq-10k-database.html)
  (authentic), LIVEC, BID, PIQ23.

---

## Training

Every stage is configured through its `option_*.py` CLI parser; all hyper-parameters
(batch size, patch size, learning rate, distillation layer, teacher/student paths, …)
are argparse flags. The active configuration is written to
`checkpoint_DistillationIQA/setting.txt` for reproducibility.

### 1. Train the FR-teacher

```bash
python train_DistillationIQA_FR.py --self_patch_num 10 --patch_size 224
```

### 2. Distill the NAR-student from a frozen teacher

```bash
python train_DistillationIQA.py --self_patch_num 10 --patch_size 224
```

Optional research variants:

```bash
# feature stacking mode (stackingV2)
python train_DistillationIQA.py --feature_stacking True

# patch-consistency (repeatable) loss
python train_DistillationIQA.py --repeatable_loss True

# make-ref student (student synthesizes its reference)
python train_DistillationIQA_makeRef.py --self_patch_num 10 --patch_size 224
```

Pretrained checkpoints (optional) go under `model_zoo/`:
`model_zoo/FR_teacher_cross_dataset.pth`, `model_zoo/NAR_student_cross_dataset.pth`.

---

## Evaluation

### Cross-dataset benchmark

```bash
python test_DistillationIQA.py
```

Reports **SRCC / PLCC / KRCC** (optionally logistic-fitted PLCC via `--use_fitting_prcc_srcc`)
per dataset. Raw + aggregated predictions are dumped to `pred_*.txt` / `gt_*.txt`.

### Single image

```python
from test_DistillationIQA_single import DistillationIQASolver

solver = DistillationIQASolver(student_address="./model_zoo/NAR_student_cross_dataset.pth",
                               net_mode="org")   # "org" | "stackingV1" | "stackingV2"
score = solver.cvr_on_single_image(lq_path="img_lq.png", ref_path="img_ref_hq.png")
print(score)
```

> Note: quality scores live on an arbitrary scale; fit to your MOS range (logistic
> mapping in `tools.py`) before reporting them on a reference scale.

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{yin2022content,
  title={Content-Variant Reference Image Quality Assessment via Knowledge Distillation},
  author={Yin, Guanghao and Wang, Wei and Yuan, Zehuan and Han, Chuchu and Ji, Wei and Sun, Shouqian and Wang, Changhu},
  journal={arXiv preprint arXiv:2202.13123},
  year={2022}
}
```

---

## Acknowledgements

The framework builds on [CVRKD-IQA](https://github.com/guanghaoyin/CVRKD-IQA) and
[HyperIQA](https://github.com/SSL92/hyperIQA). Please check their licenses before reuse.

---

## License

[MIT](LICENSE) (original copyright held by the CVRKD-IQA authors).
