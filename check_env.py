"""
check_env.py  —  Preflight check. Run this FIRST on a new machine.

    python check_env.py            # deps + dataset + device report
    python check_env.py --bench    # also benchmark each model and estimate training time

Exits with code 0 if the machine is ready to train, 1 otherwise.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import console_setup  # noqa: F401  — safe output on cp1252 terminals

REQUIRED = [
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("numpy",        "numpy"),
    ("PIL",          "Pillow"),
    ("sklearn",      "scikit-learn"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
    ("cv2",          "opencv-python"),
    ("flask",        "flask"),
    ("tqdm",         "tqdm"),
    ("tensorboard",  "tensorboard"),
]


def hr(title=""):
    print(f"\n{'─' * 62}")
    if title:
        print(f"  {title}")
        print("─" * 62)


def check_deps() -> bool:
    hr("1. Dependencies")
    missing = []
    for module, package in REQUIRED:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "")
            print(f"  ✅  {package:<18} {version}")
        except Exception:
            print(f"  ❌  {package:<18} MISSING")
            missing.append(package)

    if missing:
        print(f"\n  Install with:  pip install {' '.join(missing)}")
        return False
    return True


def check_device():
    hr("2. Device")
    import torch
    import config

    print(f"  torch          : {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}          : {props.name} "
                  f"({props.total_memory / 1024**3:.1f} GB)")
    print(f"  Selected device: {config.DEVICE}")
    print(f"  Mixed precision: {config.USE_AMP}")
    print(f"  Dataloader workers: {config.NUM_WORKERS}")

    if config.DEVICE.type == "cpu":
        n_enabled = sum(1 for on in config.MODELS_TO_TRAIN.values() if on)
        print(f"\n  ⚠️  No GPU detected. Training {n_enabled} models on CPU will take days.")
        print("     Use a CUDA machine, or Google Colab (see README).")
    return config.DEVICE.type != "cpu"


def check_dataset() -> bool:
    hr("3. Dataset")
    import config
    from utils.dataset import build_samples, split_samples

    print(f"  Path: {config.DATA_DIR}")
    try:
        samples = build_samples(config.DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n  ❌  {e}")
        return False

    train, val, test = split_samples(samples)
    print(f"\n  Total : {len(samples)} images across {config.NUM_CLASSES} classes")
    print(f"  Train : {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    return True


def check_models() -> bool:
    hr("4. Model architectures")
    import config
    from models.model_builder import get_model, count_parameters

    ok = True
    for name, enabled in config.MODELS_TO_TRAIN.items():
        if not enabled:
            print(f"  ⏭  {name:<16} disabled in config")
            continue
        try:
            # pretrained=False → verifies the architecture without a weight download
            model = get_model(name, config.NUM_CLASSES, pretrained=False)
            total = sum(p.numel() for p in model.parameters())
            print(f"  ✅  {name:<16} {total/1e6:6.1f}M params")
            del model
        except Exception as e:
            print(f"  ❌  {name:<16} {type(e).__name__}: {e}")
            ok = False
    return ok


def benchmark():
    """Time one train step per model and extrapolate the full run."""
    hr("5. Speed benchmark (estimates full training time)")
    import torch
    import torch.nn as nn
    import config
    from models.model_builder import get_model
    from utils.dataset import build_samples, split_samples

    samples = build_samples(config.DATA_DIR, verbose=False)
    train_n = len(split_samples(samples)[0])

    device = config.DEVICE
    total_estimate = 0.0

    for name, enabled in config.MODELS_TO_TRAIN.items():
        if not enabled:
            continue
        bs = config.MODEL_BATCH_SIZE.get(name, config.BATCH_SIZE)
        try:
            model = get_model(name, config.NUM_CLASSES, pretrained=False).to(device)
            opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
            crit  = nn.CrossEntropyLoss()
            x = torch.randn(bs, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
            y = torch.randint(0, config.NUM_CLASSES, (bs,), device=device)

            # Warm-up (kernel autotune / lazy init)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                    loss = crit(model(x), y)
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()

            reps = 5
            t0 = time.time()
            for _ in range(reps):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                    loss = crit(model(x), y)
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_step = (time.time() - t0) / reps

            steps_per_epoch = train_n / bs
            # +~25% for validation pass and data loading overhead
            epoch_s = per_step * steps_per_epoch * 1.25
            model_s = epoch_s * config.NUM_EPOCHS
            total_estimate += model_s

            print(f"  {name:<16} {per_step*1000:7.0f} ms/step (bs={bs})  "
                  f"→ ~{epoch_s/60:5.1f} min/epoch  "
                  f"→ ~{model_s/3600:5.1f} h for {config.NUM_EPOCHS} epochs")

            del model, opt, x, y
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠️  {name:<16} benchmark failed: {type(e).__name__}: {e}")

    print(f"\n  Estimated total for all enabled models: ~{total_estimate/3600:.1f} hours")
    print("  (Upper bound — early stopping usually finishes well before max epochs.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark models and estimate training time")
    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  🍅  Tomato Disease CNN — environment check")
    print("═" * 62)

    if not check_deps():
        print("\n  ❌  Install the missing packages, then re-run.")
        sys.exit(1)

    has_gpu    = check_device()
    data_ok    = check_dataset()
    models_ok  = check_models()

    if args.bench and data_ok and models_ok:
        benchmark()

    hr("Summary")
    ready = data_ok and models_ok
    if ready and has_gpu:
        print("  ✅  Ready to train.   Run:  python train.py")
    elif ready:
        print("  ⚠️  Code is ready but there is no GPU — training will be very slow.")
        print("      Verify the pipeline first:  python smoke_test.py")
    else:
        print("  ❌  Not ready — fix the ❌ items above.")
    print()
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
