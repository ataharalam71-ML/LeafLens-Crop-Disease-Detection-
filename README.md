# 🍅 Tomato Disease Detection — CNN Project

Classifies tomato leaf images into 9 diseases + healthy, using 8 CNN architectures
trained on the PlantVillage tomato subset (16,011 images).

**Status:** code complete and verified end to end. Training must run on a CUDA GPU —
see [Running on another device](#running-on-another-device).

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
python app.py                  # 5. web UI at http://localhost:5000
```

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
├── predict.py               # Single-image prediction
├── camera_app.py            # Real-time webcam detection (OpenCV window)
├── app.py                   # Flask web app (upload, camera stream, Grad-CAM)
├── models/model_builder.py  # The 8 CNN architectures
├── utils/
│   ├── dataset.py           # Loading, stratified split, augmentation
│   ├── visualize.py         # Training curves, confusion matrix, comparison chart
│   └── gradcam.py           # Grad-CAM heatmaps
├── templates/index.html     # Web UI
├── PlantVillage/            # Dataset (10 class folders)
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
python camera_app.py --model MobileNetV3     # Q quit, S save, M switch, SPACE pause
python app.py --port 5000
python utils/gradcam.py --image leaf.jpg --model EfficientNetB0
```

After training, set `INFERENCE_MODEL` in `config.py` to the winner reported at the
end of `train.py`.

## Outputs

`results/` gets: `<Model>_history.json`, `<Model>_curves.png`, `<Model>_report.txt`,
`<Model>_confusion_matrix.png`, `model_comparison.json` (validation) and
`model_comparison_test.png` (test set).
