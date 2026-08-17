"""
utils/visualize.py  —  Training curves, confusion matrix, sample grid
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; remove for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, List], model_name: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", ms=4, label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-o", ms=4, label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].set_ylim([0, 1]); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / f"{model_name}_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Curves saved → {out}")


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names: List[str],
                          model_name: str, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()

    out = out_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrix → {out}")


# ─── Model comparison bar chart ───────────────────────────────────────────────

def plot_model_comparison(results: Dict[str, float], out_dir: Path,
                          metric_label: str = "Accuracy",
                          filename: str = "model_comparison.png"):
    names = sorted(results, key=results.get)
    accs  = [results[n] * 100 for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.55)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"{metric_label} (%)")
    ax.set_title(f"Model Comparison — {metric_label}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = out_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Comparison chart → {out}")
