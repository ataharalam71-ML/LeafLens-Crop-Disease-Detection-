"""
train.py  —  Train multiple CNN models for tomato disease detection.

Usage:
    python train.py                          # train every model enabled in config
    python train.py --model EfficientNetB0   # train one model
    python train.py --model ResNet50 --epochs 20 --lr 0.0005
    python train.py --resume                 # continue an interrupted run
    python train.py --skip-existing          # don't retrain models already finished

Checkpoints per model:
    saved_models/<Model>_best.pth   → weights with the best validation accuracy
    saved_models/<Model>_last.pth   → full state (optimizer/scheduler/history) for --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from models.model_builder import get_model, count_parameters
from utils.dataset import get_dataloaders
from utils.visualize import plot_training_curves

# Dataloaders are expensive to build (full disk scan) — reuse per batch size.
_LOADER_CACHE: dict = {}


def get_cached_dataloaders(batch_size: int):
    if batch_size not in _LOADER_CACHE:
        _LOADER_CACHE[batch_size] = get_dataloaders(config.DATA_DIR, batch_size=batch_size)
    return _LOADER_CACHE[batch_size]


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total         += images.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled=False):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total         += images.size(0)

    return total_loss / total, total_correct / total


# ─── Main training loop ───────────────────────────────────────────────────────

def train_model(model_name: str, epochs: int, lr: float, resume: bool = False) -> float:
    batch_size = config.MODEL_BATCH_SIZE.get(model_name, config.BATCH_SIZE)

    print(f"\n{'═'*60}")
    print(f"  Model : {model_name}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Device: {config.DEVICE}  |  AMP: {config.USE_AMP}")
    print(f"{'═'*60}\n")

    # Data
    train_loader, val_loader, _ = get_cached_dataloaders(batch_size)

    # Model
    model = get_model(model_name, config.NUM_CLASSES).to(config.DEVICE)
    print(f"  Parameters — {count_parameters(model)}\n")

    # Loss + optimizer + scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(config.DEVICE.type, enabled=config.USE_AMP)

    # State
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_cnt = 0
    start_epoch  = 1

    best_ckpt = config.MODELS_DIR / f"{model_name}_best.pth"
    last_ckpt = config.MODELS_DIR / f"{model_name}_last.pth"

    # ── Resume from an interrupted run ──
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        history      = state["history"]
        best_val_acc = state["best_val_acc"]
        patience_cnt = state["patience_cnt"]
        start_epoch  = state["epoch"] + 1
        print(f"  ↻  Resumed from epoch {state['epoch']} "
              f"(best val_acc={best_val_acc:.4f})\n")
        if start_epoch > epochs:
            print(f"  ✔  {model_name} already completed {epochs} epochs.")
            return best_val_acc

    # TensorBoard
    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / model_name))
    t_model = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, scaler)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, config.DEVICE, amp_enabled=config.USE_AMP)
        scheduler.step()

        elapsed = time.time() - t0
        remaining = (epochs - epoch) * elapsed
        print(f"  Epoch {epoch:03d}/{epochs} "
              f"| Train Loss {train_loss:.4f} Acc {train_acc:.4f} "
              f"| Val Loss {val_loss:.4f} Acc {val_acc:.4f} "
              f"| {elapsed:.1f}s  (ETA {format_duration(remaining)})")

        # Record history first, so it is complete even if we stop early below.
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # TensorBoard logging
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best checkpoint
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_name":  model_name,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "class_names": config.CLASS_NAMES,
            }, best_ckpt)
            print(f"    ✅  New best saved  (val_acc={best_val_acc:.4f})")
        else:
            patience_cnt += 1

        # Save resumable state every epoch
        torch.save({
            "epoch":            epoch,
            "model_name":       model_name,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict() if config.USE_AMP else None,
            "history":          history,
            "best_val_acc":     best_val_acc,
            "patience_cnt":     patience_cnt,
            "class_names":      config.CLASS_NAMES,
        }, last_ckpt)

        if patience_cnt >= config.PATIENCE:
            print(f"\n  ⏹  Early stopping after {epoch} epochs "
                  f"(no improvement for {config.PATIENCE}).")
            break

    writer.close()

    # Save history
    hist_path = config.RESULTS_DIR / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    if history["train_loss"]:
        plot_training_curves(history, model_name, config.RESULTS_DIR)

    print(f"\n  🏆  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  ⏱  Model time: {format_duration(time.time() - t_model)}")
    print(f"  💾  Checkpoint: {best_ckpt}")
    return best_val_acc


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Tomato Disease CNN")
    parser.add_argument("--model",  type=str, default=None,
                        help="Model name (default: all enabled in config)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr",     type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from each model's *_last.pth checkpoint")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models that already have a best checkpoint")
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)

    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = [m for m, enabled in config.MODELS_TO_TRAIN.items() if enabled]

    print(f"\n  Dataset : {config.DATA_DIR}")
    print(f"  Device  : {config.DEVICE}")
    print(f"  Models  : {', '.join(models_to_run)}")

    comparison_path = config.RESULTS_DIR / "model_comparison.json"
    results = {}
    if comparison_path.exists():
        try:
            results = json.loads(comparison_path.read_text())
        except json.JSONDecodeError:
            results = {}

    t_start = time.time()
    for model_name in models_to_run:
        if args.skip_existing and (config.MODELS_DIR / f"{model_name}_best.pth").exists() \
                and model_name in results:
            print(f"\n  ⏭  Skipping {model_name} (already trained).")
            continue

        acc = train_model(model_name, args.epochs, args.lr, resume=args.resume)
        results[model_name] = round(acc, 4)

        # Write after every model so an interrupted run keeps its results.
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "═" * 40)
    print("  Final Results (validation accuracy)")
    print("═" * 40)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} →  {acc:.4f} ({acc*100:.2f}%)")
    print("═" * 40)
    print(f"  Total time: {format_duration(time.time() - t_start)}")

    if results:
        best = max(results, key=results.get)
        print(f"\n  🥇  Best model: {best}  "
              f"(set INFERENCE_MODEL = \"{best}\" in config.py)")
    print(f"  Comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
