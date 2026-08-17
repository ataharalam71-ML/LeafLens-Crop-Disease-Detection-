"""
evaluate.py  —  Test-set evaluation with full metrics + confusion matrix.

Usage:
    python evaluate.py                         # evaluates INFERENCE_MODEL
    python evaluate.py --model ResNet50
    python evaluate.py --model all             # evaluate all saved checkpoints
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

import config
from models.model_builder import get_model
from utils.dataset import get_dataloaders
from utils.visualize import plot_confusion_matrix, plot_model_comparison


_TEST_LOADER = None


def get_test_loader():
    """Built once — the test split is identical for every model."""
    global _TEST_LOADER
    if _TEST_LOADER is None:
        _, _, _TEST_LOADER = get_dataloaders(config.DATA_DIR)
    return _TEST_LOADER


@torch.no_grad()
def evaluate_model(model_name: str) -> dict:
    ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  ⚠️  Checkpoint not found: {ckpt_path}")
        return {}

    print(f"\n{'─'*50}")
    print(f"  Evaluating: {model_name}")

    # Load checkpoint
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE).eval()

    # Data (test split)
    test_loader = get_test_loader()

    all_preds, all_labels = [], []
    for images, labels in tqdm(test_loader, desc="  Testing"):
        images = images.to(config.DEVICE)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n  ✅  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    report = classification_report(
        all_labels, all_preds,
        target_names=config.DISPLAY_NAMES,
        digits=4,
    )
    print(report)

    # Save report
    out_txt = config.RESULTS_DIR / f"{model_name}_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nTest Accuracy: {acc:.4f}\n\n{report}")

    # Confusion matrix plot
    plot_confusion_matrix(
        all_labels, all_preds,
        config.DISPLAY_NAMES, model_name, config.RESULTS_DIR,
    )

    return {"model": model_name, "test_acc": round(acc, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.INFERENCE_MODEL,
                        help="Model name or 'all'")
    args = parser.parse_args()

    if args.model.lower() == "all":
        # Every model that actually has a trained checkpoint on disk.
        models = [m for m in config.MODELS_TO_TRAIN
                  if (config.MODELS_DIR / f"{m}_best.pth").exists()]
        if not models:
            print("  ❌  No trained checkpoints found. Run  python train.py  first.")
            return
    else:
        models = [args.model]

    summary = {}
    for m in models:
        res = evaluate_model(m)
        if res:
            summary[res["model"]] = res["test_acc"]

    if not summary:
        print("  ❌  Nothing evaluated.")
        return

    if len(summary) > 1:
        plot_model_comparison(summary, config.RESULTS_DIR,
                              metric_label="Test Accuracy",
                              filename="model_comparison_test.png")

    print("\n" + "═" * 40)
    print("  Test-set ranking")
    print("═" * 40)
    for name, acc in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} →  {acc:.4f} ({acc*100:.2f}%)")
    print("═" * 40)

    with open(config.RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {config.RESULTS_DIR / 'test_results.json'}")


if __name__ == "__main__":
    main()
