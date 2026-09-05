# 🍅 Tomato Disease Detection — CNN Project

Classifies tomato leaf images into 9 diseases + healthy, using 8 CNN architectures
trained on the PlantVillage tomato subset (16,011 images) — then does the three
things a classifier alone cannot:

- **Knows when not to answer.** A 10-way softmax is a closed world: show it a
  coffee cup and it returns a confident disease. An abstention layer
  (`utils/ood.py`) refuses images that are not tomato leaves, with a reason.
- **Says what to do about it.** `treatments.py` turns each class into a plan —
  pathogen, urgency, spray programme with dosages, and the expensive mistakes to
  avoid — scaled by how much of the leaf is actually symptomatic.
- **Reports the honest accuracy.** `field_eval.py` measures the model on your own
  field photographs next to the lab number, because 99.8% on PlantVillage is a
  claim about laboratory conditions, not about diagnosing disease.

**Status:** code complete and verified end to end (`python smoke_test.py` → 20/20).
Training must run on a CUDA GPU — see [Running on another device](#running-on-another-device).

---

## Quick start on a new machine

```bash
pip install -r requirements.txt

# Point at the dataset (skip if PlantVillage/ sits next to the code)
set TOMATO_DATA_DIR=D:\path\to\PlantVillage        # Windows
export TOMATO_DATA_DIR=/path/to/PlantVillage       # Linux/macOS

python check_env.py --bench    # 1. deps, dataset, GPU + time estimate
python smoke_test.py           # 2. full pipeline on a tiny subset (~2 min)
python train.py                # 3. train all 8 models
python evaluate.py --model all # 4. test-set metrics + confusion matrices
python calibrate_ood.py --model all   # 5. tune the abstention thresholds
python field_eval.py           # 6. accuracy on YOUR photos vs the lab number
python app.py                  # 7. web UI at http://localhost:5000
```

Steps 5 and 6 are what separate this from a benchmark script — see
[Trust, advice, and the honest number](#trust-advice-and-the-honest-number).

---

## Project structure

```
tomato_disease_cnn/
├── config.py                # All configuration (paths, models, hyperparameters)
├── console_setup.py         # UTF-8 console fix (Windows cp1252 safety)
├── check_env.py             # Preflight: deps / dataset / GPU / speed benchmark
├── smoke_test.py            # End-to-end pipeline test on a tiny subset
├── train.py                 # Trains all 8 models (AMP, early stopping, resume)
├── evaluate.py              # Test-set metrics, confusion matrices, comparison
├── calibrate_ood.py         # Fits temperature + abstention thresholds
├── field_eval.py            # Lab vs. real-field accuracy, risk-coverage curve
├── treatments.py            # Agronomic knowledge base (what to actually do)
├── predict.py               # Single-image prediction + verdict + treatment
├── camera_app.py            # Real-time webcam detection (OpenCV window)
├── app.py                   # Flask web app (upload, camera stream, Grad-CAM)
├── models/model_builder.py  # The 8 CNN architectures
├── utils/
│   ├── dataset.py           # Loading, stratified split, augmentation
│   ├── visualize.py         # Training curves, confusion matrix, comparison chart
│   ├── gradcam.py           # Grad-CAM heatmaps
│   ├── ood.py               # Abstention layer (quality + energy + Mahalanobis)
│   └── severity.py          # Symptomatic leaf area from Excess Green
├── templates/index.html     # Web UI
├── PlantVillage/            # Dataset (10 class folders)
├── field_data/              # YOUR field photographs (see its README)
├── saved_models/            # Checkpoints  (<Model>_best.pth, <Model>_last.pth)
├── results/                 # Metrics, plots, JSON histories
└── logs/                    # TensorBoard logs
```

## Disease classes (9 diseases + healthy)

| # | Class | Images |
|---|-------|--------|
| 0 | Target Spot | 1,404 |
| 1 | Mosaic Virus | 373 |
| 2 | Yellow Leaf Curl Virus | 3,208 |
| 3 | Bacterial Spot | 2,127 |
| 4 | Early Blight | 1,000 |
| 5 | Healthy | 1,591 |
| 6 | Late Blight | 1,909 |
| 7 | Leaf Mold | 952 |
| 8 | Septoria Leaf Spot | 1,771 |
| 9 | Spider Mites (two-spotted) | 1,676 |

Classes are imbalanced (373 → 3,208), so the train/val/test split is **stratified**:
each class is split 70/15/15 independently.

## The 8 models

| Model | Params | Notes |
|-------|--------|-------|
| CustomCNN | 9.8M | 5-block CNN from scratch — baseline |
| ResNet50 | 24.6M | Transfer learning |
| EfficientNetB0 | 4.0M | Best accuracy/size trade-off |
| MobileNetV3 | 4.2M | Fastest inference — use for camera |
| VGG16 | 134.3M | Heavy; batch size auto-drops to 16 |
| AlexNet | 57.0M | Classic 2012 architecture — fast, useful comparison point |
| UNet | 8.2M | From scratch; encoder/decoder + skip connections, batch auto-drops to 16 |
| DenseNet121 | 7.0M | Transfer learning; strong on fine leaf textures |

Every model is selectable everywhere by name — `train.py`, `evaluate.py`,
`predict.py`, `camera_app.py`, `app.py` and Grad-CAM all read the same registry,
so nothing else needs editing to use the new ones:

```bash
python train.py --model AlexNet
python train.py --model UNet
python train.py --model DenseNet121
```

Disable any you don't want in `MODELS_TO_TRAIN` (config.py) — set it to `False`.

---

## Trust, advice, and the honest number

### 1. Knowing when not to answer

A softmax over 10 classes cannot say "that is not a tomato leaf" — every image
gets forced into a disease. `utils/ood.py` wraps the classifier in an abstention
layer that combines four independent kinds of evidence:

| Signal | What it catches | Reference |
|---|---|---|
| Image quality | Non-vegetation, blur, under/over-exposure — before the model runs | — |
| Max softmax + entropy + margin | Ordinary uncertainty between classes | Hendrycks & Gimpel, ICLR 2017 |
| Free energy `-logsumexp(logits)` | Logit magnitude collapse on unfamiliar input | Liu et al., NeurIPS 2020 |
| Mahalanobis distance | Features landing outside every learned class cluster | Lee et al., NeurIPS 2018 |

Signals are grouped into **evidence families** and each family casts at most one
vote — `msp`, `margin` and `entropy` are three readings of the same softmax
vector, so letting them vote separately would turn one piece of evidence into an
instant rejection. One unhappy family → `review`; two or more → `reject`. A
frame with no vegetation in it is rejected outright, whatever the model thinks.

Thresholds are measured, not guessed:

```bash
python calibrate_ood.py --model all
python calibrate_ood.py --ood-dir not_leaves/   # adds AUROC + FPR@95TPR
```

This fits a temperature by minimising NLL (Guo et al., ICML 2017) and sets each
threshold at the quantile retaining 95% of genuine validation images, saving
`results/ood_calibration_<model>.json`. Without that file the detector still
runs on conservative fallbacks and reports `calibrated: false` everywhere —
including in the web UI footer, so an uncalibrated demo never looks like a tuned
one.

**Two findings worth knowing**, both surfaced by calibration on MobileNetV3:

- **T = 0.385, i.e. below 1.** Temperature scaling *sharpens* these models
  rather than softening them, because label smoothing 0.1 during training leaves
  them systematically *under*-confident. ECE drops 0.0914 → 0.0005.
- **Three thresholds hit their clamp.** At 99.96% validation accuracy the
  confidence distribution is nearly a point mass at 1.0, so the 95%-TPR quantile
  of MSP lands at 0.9999 — an artifact of a validation set with almost no hard
  examples, not an operating point. `config.OOD_THRESHOLD_LIMITS` bounds it, and
  calibration prints a warning whenever a clamp binds. This is a property of
  PlantVillage, which is exactly what section 3 measures.

Measured behaviour on MobileNetV3 (200 dataset leaves, 20 per class):
**165 accept / 29 review / 6 reject** — 97% coverage — while synthetic non-leaf
images are rejected 4/4.

### 2. Saying what to do about it

`treatments.py` holds a curated entry per class: pathogen and type, how to
confirm it by eye, how it spreads, urgency 1–5 with an action window, contagion
risk, yield loss if untreated, and three tiers of response (organic, chemical
with active ingredients and per-litre dosages plus FRAC/IRAC groups, cultural)
along with a `do_not` list of expensive mistakes — such as spraying fungicide at
a virus, or hitting spider mites with a pyrethroid and triggering a worse flare.

Advice is scaled by severity. `utils/severity.py` measures the symptomatic share
of the leaf using Excess Green on normalised chromaticity, `2g - r - b`, which
is roughly shading-invariant. That matters: thresholding raw HSV green scores a
*healthy* PlantVillage leaf at ~29% symptomatic, because leaf shadow and
specular highlight fall outside the green band. Excess Green scores the same
leaves at ~1.6%. Median symptomatic area by class:

| Class | Median | Class | Median |
|---|---|---|---|
| Healthy | 1.6% | Spider Mites | 10.6% |
| Bacterial Spot | 4.1% | Late Blight | 12.4% |
| Septoria | 6.2% | Target Spot | 13.3% |
| | | Early Blight | 13.8% |
| | | Leaf Mold | 17.2% |

The two viruses damage leaves through curling and stunting rather than dead
tissue, so they are returned with `area_based: false` rather than a misleadingly
low number.

An image the abstention layer refuses gets **no** treatment plan — advising a
spray programme for a photo you just declined to diagnose would defeat the point.

### 3. The honest number

PlantVillage is a laboratory dataset: single detached leaves, even lighting,
uniform backgrounds. 99.79% on its test split is a claim about those conditions.
`field_eval.py` measures the gap on photographs you take yourself.

```bash
# put your photos in field_data/<class name>/  — see field_data/README.md
python field_eval.py --model all
python field_eval.py --unknown-dir not_leaves/   # outright rejection rate
```

It reports lab accuracy beside field accuracy and the gap between them, plus
**selective accuracy**: how accurate the model is on only the images the
abstention layer accepted, and what coverage that cost. A useful detector raises
accuracy faster than it loses coverage; the generated risk-coverage curve shows
whether it does. A subset of classes is fine, and misspelled folders are flagged
with the closest valid name.

Outputs: `results/field_results.json`, `field_vs_lab.png`,
`field_risk_coverage.png`, plus a per-model report and confusion matrix.

### New endpoints

| Route | Purpose |
|---|---|
| `POST /predict` | Now returns `verdict`, `severity` and `treatment` alongside the prediction |
| `GET /treatment/<class>` | Treatment plan for any class, no image needed |
| `POST /severity` | Severity number plus the segmentation mask it came from |
| `GET /ood/status` | Whether thresholds are calibrated or fallback |

---

## Running on another device

Training on CPU is not practical. Measured on a Ryzen 5 7530U (no CUDA):

| Model | CPU speed | 30 epochs |
|-------|-----------|-----------|
| CustomCNN | 16.3 s/step | ~60 h |
| ResNet50 | 9.8 s/step | ~36 h |
| EfficientNetB0 | 4.6 s/step | ~17 h |
| MobileNetV3 | 2.5 s/step | ~9 h |
| VGG16 | 12.7 s/step | ~93 h |
| AlexNet | 1.5 s/step | ~6 h |
| UNet | 8.8 s/step | ~64 h |
| DenseNet121 | 9.2 s/step | ~34 h |
| **Total** | | **~319 h (13.3 days)** |

(Measured with `python check_env.py --bench`; run it on your own machine for a
local estimate.)

On a modern CUDA GPU the same run is roughly **3–6 hours** total, since mixed
precision (`USE_AMP`) switches on automatically.

### Option A — a machine with an NVIDIA GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python check_env.py --bench     # confirm the GPU is detected
python train.py
```

Nothing else needs editing — `DEVICE`, mixed precision (`USE_AMP`), `pin_memory`
and the cuDNN autotuner all switch on by themselves once CUDA is visible.
`check_env.py` prints what it detected; if it still says `cpu`, the installed
torch is the CPU build — reinstall from the CUDA index above.

If you hit `CUDA out of memory`, lower the batch size without touching code:

```bash
set TOMATO_BATCH_SIZE=16        # Windows
export TOMATO_BATCH_SIZE=16     # Linux/macOS
```

VGG16 and UNet are additionally capped at 16, and the cap scales down with you.

### Option B — Google Colab (free GPU)

Upload the project folder (or clone it), then in a cell:

```python
!pip install -q -r /content/tomato_disease_cnn/requirements.txt
%cd /content/tomato_disease_cnn
import os; os.environ["TOMATO_DATA_DIR"] = "/content/PlantVillage"
!python check_env.py
!python train.py
```

Colab disconnects after a few hours — that is what `--resume` is for:

```bash
python train.py --resume          # picks up from each model's *_last.pth
python train.py --skip-existing   # don't retrain models already finished
```

### Environment variables (no code edits needed)

| Variable | Purpose |
|----------|---------|
| `TOMATO_DATA_DIR` | Path to the PlantVillage folder |
| `TOMATO_FIELD_DIR` | Path to your field photographs (default `./field_data`) |
| `TOMATO_DEVICE` | Force `cuda` / `cpu` / `mps` |
| `TOMATO_BATCH_SIZE` | Batch size — raise on a big GPU, lower on CUDA OOM |
| `TOMATO_NUM_WORKERS` | DataLoader worker processes |
| `TOMATO_MAX_PER_CLASS` | Cap images per class (quick trials) |

If `TOMATO_DATA_DIR` is unset, the dataset is looked up at `./PlantVillage`, then
`~/Desktop/PlantVillage`.

---

## Training details

- **Optimizer** AdamW, lr 1e-3, weight decay 1e-4
- **Scheduler** CosineAnnealingLR → 1e-6
- **Loss** CrossEntropy with label smoothing 0.1
- **Early stopping** patience 7 epochs on validation accuracy
- **Mixed precision** automatic on CUDA
- **Augmentation** flips, ±30° rotation, colour jitter, blur, random erasing
  (torchvision `transforms.v2` — no compiler toolchain needed)
- **Checkpoints** `<Model>_best.pth` (best val acc) and `<Model>_last.pth`
  (optimizer + scheduler + history, for `--resume`)

Monitor a run with TensorBoard:

```bash
tensorboard --logdir logs
```

## Usage

```bash
python train.py --model EfficientNetB0 --epochs 20 --lr 5e-4
python evaluate.py --model all
python predict.py --image leaf.jpg --model EfficientNetB0 --topk 3
python predict.py --image leaf.jpg --json          # machine-readable
python predict.py --image leaf.jpg --no-treatment  # label only
python predict.py --image leaf.jpg --no-ood        # skip abstention, always answer
python camera_app.py --model MobileNetV3     # Q quit, S save, M switch, SPACE pause
python app.py --port 5000
python utils/gradcam.py --image leaf.jpg --model EfficientNetB0
python utils/severity.py --image leaf.jpg --out mask.png
python calibrate_ood.py --model all
python field_eval.py --model all
```

After training, set `INFERENCE_MODEL` in `config.py` to the winner reported at the
end of `train.py`.

## Outputs

`results/` gets: `<Model>_history.json`, `<Model>_curves.png`, `<Model>_report.txt`,
`<Model>_confusion_matrix.png`, `model_comparison.json` (validation) and
`model_comparison_test.png` (test set).

After `calibrate_ood.py`: `ood_calibration_<Model>.json` (temperature,
thresholds, clamped/unclamped values, ECE before and after),
`ood_mahalanobis_<Model>.npz` (the fitted Gaussians) and
`ood_scores_<Model>.png` (score distributions with the operating point drawn on).

After `field_eval.py`: `field_results.json`, `field_vs_lab.png`,
`field_risk_coverage.png`, `field_report_<Model>.txt` and
`<Model>_field_confusion_matrix.png`.
