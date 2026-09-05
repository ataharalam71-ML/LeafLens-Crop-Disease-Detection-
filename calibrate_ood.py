"""
calibrate_ood.py — Turn the OOD detector's thresholds into measured quantities.

Hand-picked thresholds are the weak point of most abstention layers: they are
tuned until the demo behaves, and nobody can say what they cost. This script
derives them instead.

What it does
------------
1. Runs the model over the validation split, capturing logits and penultimate
   features (the split the model never trained on, and never early-stopped on
   for these statistics).

2. Fits a temperature by minimising NLL (Guo et al., ICML 2017). A single scalar
   divided into the logits; it cannot change any prediction, only how confident
   the probabilities are. Expected Calibration Error before and after is printed,
   so the improvement is visible rather than asserted.

3. Sets every threshold at the quantile that retains OOD_TARGET_TPR (default
   95%) of genuine validation images. This is the operating point stated as a
   promise: "95 out of 100 real tomato leaves still get an answer". The
   remaining 5% is the cost, and it is deliberate rather than accidental.

4. Fits the class-conditional Gaussians for the Mahalanobis scorer.

5. If --ood-dir is given, scores a folder of genuinely out-of-distribution
   images and reports AUROC and FPR@95TPR per signal - the standard numbers the
   OOD literature reports, computed on your own data.

Usage:
    python calibrate_ood.py                              # INFERENCE_MODEL
    python calibrate_ood.py --model ResNet50
    python calibrate_ood.py --model all
    python calibrate_ood.py --ood-dir ood_samples/       # measure AUROC too
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import config
from models.model_builder import get_model
from utils.dataset import get_dataloaders, inference_transform, IMG_EXTENSIONS
from utils.ood import (MahalanobisScorer, FeatureTap, logit_scores,
                       assess_quality)

_LOADERS = None


def get_val_loader():
    """Built once; the split is identical for every model."""
    global _LOADERS
    if _LOADERS is None:
        _LOADERS = get_dataloaders(config.DATA_DIR)
    return _LOADERS[1]


# ==============================================================================
# Model + feature extraction
# ==============================================================================

def load_model(model_name: str):
    ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint for '{model_name}'.")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(config.DEVICE).eval()


@torch.no_grad()
def collect(model, loader, desc: str, max_images: int = 0):
    """
    Run the loader, returning (logits, features, labels) as numpy arrays.

    `max_images` stops early once that many images have been seen. The
    validation split is shuffled once at split time with a fixed seed, so the
    batches arrive class-mixed rather than class-ordered and an early stop is
    a representative subset, not the first two classes.

    Subsampling exists because calibration on CPU is dominated by the forward
    pass: VGG16 over the full 2,397-image split takes far longer than the
    quantiles need. A few hundred images already pin a 95th percentile tightly
    enough; the printed image count says what any given file was fitted on.
    """
    all_logits, all_feats, all_labels = [], [], []
    seen = 0
    total_batches = len(loader)
    if max_images:
        total_batches = min(total_batches,
                            -(-max_images // loader.batch_size))  # ceil

    with FeatureTap(model) as tap:
        for images, labels in tqdm(loader, desc=f"  {desc}", total=total_batches):
            images = images.to(config.DEVICE, non_blocking=True)
            logits = model(images)
            feats = tap.get()
            all_logits.append(logits.float().cpu().numpy())
            all_feats.append(feats.float().reshape(feats.shape[0], -1).cpu().numpy())
            all_labels.append(labels.numpy())

            seen += len(labels)
            if max_images and seen >= max_images:
                break

    return (np.concatenate(all_logits),
            np.concatenate(all_feats),
            np.concatenate(all_labels))


@torch.no_grad()
def collect_folder(model, folder: Path, desc: str):
    """Same, for a flat folder of loose images with no labels."""
    paths = [p for p in sorted(Path(folder).rglob("*"))
             if p.suffix.lower() in IMG_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No images found under {folder}")

    all_logits, all_feats, kept = [], [], []
    with FeatureTap(model) as tap:
        for path in tqdm(paths, desc=f"  {desc}"):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            tensor = inference_transform(img).unsqueeze(0).to(config.DEVICE)
            logits = model(tensor)
            feats = tap.get()
            all_logits.append(logits.float().cpu().numpy())
            all_feats.append(feats.float().reshape(1, -1).cpu().numpy())
            kept.append(path)
    return np.concatenate(all_logits), np.concatenate(all_feats), kept


# ==============================================================================
# Temperature scaling
# ==============================================================================

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    One scalar T minimising NLL of softmax(logits / T).

    Optimised with LBFGS on log-T so the temperature cannot go negative.
    """
    z = torch.from_numpy(logits).float()
    y = torch.from_numpy(labels).long()

    log_t = torch.zeros(1, requires_grad=True)      # T = exp(0) = 1
    optimiser = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(logits: np.ndarray, labels: np.ndarray,
                               temperature: float = 1.0, bins: int = 15) -> float:
    """
    Standard 15-bin ECE: mean gap between confidence and accuracy.

    0 means the model's stated confidence matches how often it is right.
    """
    z = torch.from_numpy(logits).float() / max(temperature, 1e-6)
    probs = F.softmax(z, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred.numpy() == labels).astype(np.float64)
    conf = conf.numpy()

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if not np.any(sel):
            continue
        ece += (sel.mean()) * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


# ==============================================================================
# Scoring helpers
# ==============================================================================

def score_matrix(logits: np.ndarray, temperature: float) -> dict:
    """Vectorised msp / margin / entropy / energy over a batch of logits."""
    z = torch.from_numpy(logits).float()
    scaled = z / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=1)

    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    msp = top2.values[:, 0]
    margin = (top2.values[:, 0] - top2.values[:, 1]) if probs.shape[1] > 1 \
        else torch.ones_like(msp)

    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / np.log(probs.shape[1])

    energy = -torch.logsumexp(z, dim=1)

    return {
        "msp": msp.numpy(),
        "margin": margin.numpy(),
        "entropy": entropy.numpy(),
        "energy": energy.numpy(),
    }


# Which tail of the in-distribution score is "suspicious".
# "low"  -> we reject when the score falls BELOW the threshold
# "high" -> we reject when the score rises ABOVE the threshold
REJECT_SIDE = {
    "msp": "low",
    "margin": "low",
    "entropy": "high",
    "energy": "high",
    "mahalanobis": "high",
}


def threshold_at_tpr(id_scores: np.ndarray, side: str, tpr: float) -> float:
    """Quantile of the in-distribution scores that keeps `tpr` of them."""
    if side == "low":
        return float(np.quantile(id_scores, 1.0 - tpr))
    return float(np.quantile(id_scores, tpr))


def _apply_clamps(raw: dict):
    """
    Hold each threshold inside config.OOD_THRESHOLD_LIMITS.

    Returns (clamped_thresholds, names_that_were_clamped). See the comment on
    OOD_THRESHOLD_LIMITS in config.py for why a 99.96%-accurate validation set
    produces thresholds that need this.
    """
    out, clamped = {}, []
    for name, value in raw.items():
        limits = config.OOD_THRESHOLD_LIMITS.get(name)
        if limits is None:
            out[name] = value
            continue
        lo, hi = limits
        bounded = float(min(max(value, lo), hi))
        out[name] = bounded
        if abs(bounded - value) > 1e-9:
            clamped.append(name)
    return out, clamped


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray, side: str) -> float:
    """
    AUROC for separating in-distribution from OOD.

    Oriented so that higher always means "more in-distribution", then measured
    with sklearn. 0.5 = the signal is useless, 1.0 = perfect separation.
    """
    from sklearn.metrics import roc_auc_score
    sign = 1.0 if side == "low" else -1.0
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([sign * id_scores, sign * ood_scores])
    return float(roc_auc_score(y, s))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray,
               side: str, tpr: float) -> float:
    """Share of OOD images that slip through at the chosen operating point."""
    thr = threshold_at_tpr(id_scores, side, tpr)
    if side == "low":
        return float((ood_scores >= thr).mean())
    return float((ood_scores <= thr).mean())


# ==============================================================================
# Main calibration for one model
# ==============================================================================

def calibrate(model_name: str, ood_dir: Path = None, plot: bool = True,
              max_images: int = 0) -> dict:
    print(f"\n{'='*62}\n  Calibrating OOD detector: {model_name}\n{'='*62}")

    model = load_model(model_name)
    val_loader = get_val_loader()

    print("\n  [1/5] Collecting validation logits + features")
    logits, feats, labels = collect(model, val_loader, "val", max_images)
    acc = float((logits.argmax(1) == labels).mean())
    subset = f" (subsampled from {len(val_loader.dataset)})" if max_images else ""
    print(f"        {len(labels)} images{subset}, {feats.shape[1]}-dim features, "
          f"val accuracy {acc*100:.2f}%")

    print("\n  [2/5] Fitting temperature")
    ece_before = expected_calibration_error(logits, labels, 1.0)
    temperature = fit_temperature(logits, labels)
    ece_after = expected_calibration_error(logits, labels, temperature)
    print(f"        T = {temperature:.4f}")
    print(f"        ECE {ece_before:.4f} -> {ece_after:.4f}")

    print("\n  [3/5] Setting thresholds at "
          f"{config.OOD_TARGET_TPR*100:.0f}% true-positive rate")
    id_scores = score_matrix(logits, temperature)

    print("\n  [4/5] Fitting Mahalanobis Gaussians")
    scorer = MahalanobisScorer.fit(
        feats, labels, config.NUM_CLASSES,
        shrinkage=config.OOD_MAHALANOBIS_SHRINKAGE)
    id_scores["mahalanobis"] = scorer.score_batch(feats)
    print(f"        shared covariance, shrinkage "
          f"{config.OOD_MAHALANOBIS_SHRINKAGE}, "
          f"median distance {np.median(id_scores['mahalanobis']):.1f}")

    raw_thresholds = {
        name: threshold_at_tpr(scores, REJECT_SIDE[name], config.OOD_TARGET_TPR)
        for name, scores in id_scores.items()
    }
    thresholds, clamped = _apply_clamps(raw_thresholds)

    print()
    for name, thr in thresholds.items():
        side = "below" if REJECT_SIDE[name] == "low" else "above"
        note = f"   <- clamped from {raw_thresholds[name]:.4f}" if name in clamped else ""
        print(f"        {name:<12} reject {side:>5} {thr:>10.4f}{note}")

    if clamped:
        print(f"\n        [!] {len(clamped)} threshold(s) hit their clamp. This "
              f"happens when the\n"
              f"            validation set is too easy to produce a meaningful "
              f"quantile:\n"
              f"            at {acc*100:.2f}% accuracy the confidence "
              f"distribution is nearly a\n"
              f"            point mass at 1.0, so the 95%-TPR quantile lands at "
              f"~0.9999 and\n"
              f"            would flag a good leaf seen at 98% confidence. The "
              f"clamped value\n"
              f"            is used instead. This is a property of PlantVillage, "
              f"not a bug -\n"
              f"            field_eval.py measures what it costs on real photos.")

    # -- optional: measure against real OOD images ---------------------------
    ood_report = None
    if ood_dir is not None:
        print(f"\n  [5/5] Scoring OOD folder: {ood_dir}")
        ood_logits, ood_feats, ood_paths = collect_folder(model, ood_dir, "ood")
        ood_scores = score_matrix(ood_logits, temperature)
        ood_scores["mahalanobis"] = scorer.score_batch(ood_feats)

        ood_report = {"n_images": len(ood_paths), "per_signal": {}}
        print(f"\n        {'signal':<14}{'AUROC':>8}{'FPR@95TPR':>12}")
        print(f"        {'-'*34}")
        for name, scores in ood_scores.items():
            side = REJECT_SIDE[name]
            a = auroc(id_scores[name], scores, side)
            f = fpr_at_tpr(id_scores[name], scores, side, config.OOD_TARGET_TPR)
            ood_report["per_signal"][name] = {"auroc": round(a, 4),
                                              "fpr_at_95tpr": round(f, 4)}
            print(f"        {name:<14}{a:>8.4f}{f:>12.2%}")

        # How the assembled detector behaves, not just each signal alone.
        quality_flags = [assess_quality(p)["flags"] for p in ood_paths]
        caught_by_quality = sum(1 for f in quality_flags if "not_vegetation" in f)
        ood_report["rejected_by_quality_gate"] = caught_by_quality
        print(f"\n        The vegetation gate alone rejects "
              f"{caught_by_quality}/{len(ood_paths)} "
              f"({caught_by_quality/len(ood_paths):.0%}) before the model runs.")
    else:
        print("\n  [5/5] No --ood-dir given; skipping AUROC measurement.")

    # -- persist --------------------------------------------------------------
    npz_path = config.RESULTS_DIR / f"ood_mahalanobis_{model_name}.npz"
    np.savez_compressed(npz_path, means=scorer.means, precision=scorer.precision)

    payload = {
        "model": model_name,
        "n_calibration_images": int(len(labels)),
        "subsampled": bool(max_images),
        "val_accuracy": round(acc, 4),
        "temperature": round(temperature, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "target_tpr": config.OOD_TARGET_TPR,
        "thresholds": {k: round(float(v), 6) for k, v in thresholds.items()},
        "thresholds_unclamped": {k: round(float(v), 6)
                                  for k, v in raw_thresholds.items()},
        "clamped": clamped,
        "reject_side": REJECT_SIDE,
        "mahalanobis_file": npz_path.name,
        "mahalanobis_shrinkage": config.OOD_MAHALANOBIS_SHRINKAGE,
        "feature_dim": int(feats.shape[1]),
        "ood_evaluation": ood_report,
    }

    out_path = config.RESULTS_DIR / f"ood_calibration_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved  ->  {out_path}")
    print(f"  Saved  ->  {npz_path}")

    if plot:
        _plot_distributions(model_name, id_scores, thresholds,
                            ood_scores if ood_dir is not None else None)

    return payload


def _plot_distributions(model_name: str, id_scores: dict, thresholds: dict,
                        ood_scores: dict = None):
    """Score histograms with the chosen operating point drawn on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["msp", "entropy", "margin", "energy", "mahalanobis"]
    names = [n for n in names if n in id_scores]

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        ax.hist(id_scores[name], bins=50, alpha=0.75,
                label="tomato leaves (val)", color="#2e8b57", density=True)
        if ood_scores is not None and name in ood_scores:
            ax.hist(ood_scores[name], bins=50, alpha=0.6,
                    label="out-of-distribution", color="#c0392b", density=True)
        ax.axvline(thresholds[name], color="black", ls="--", lw=1.6,
                   label=f"threshold {thresholds[name]:.3g}")
        ax.set_title(name, fontweight="bold")
        ax.set_yticks([])
        ax.legend(fontsize=7)

    fig.suptitle(
        f"OOD score distributions - {model_name}  "
        f"(thresholds at {config.OOD_TARGET_TPR*100:.0f}% TPR)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = config.RESULTS_DIR / f"ood_scores_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved  ->  {out}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Calibrate the OOD detector")
    ap.add_argument("--model", default=config.INFERENCE_MODEL,
                    help="Model name, or 'all' for every trained checkpoint")
    ap.add_argument("--ood-dir", default=None, type=str,
                    help="Folder of non-tomato-leaf images, to measure AUROC")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Calibrate on at most N validation images (0 = all). "
                         "Use ~800 to calibrate every model in minutes on CPU.")
    args = ap.parse_args()

    ood_dir = Path(args.ood_dir) if args.ood_dir else None
    if ood_dir is not None and not ood_dir.is_dir():
        print(f"  [X]  --ood-dir not found: {ood_dir}")
        return

    if args.model.lower() == "all":
        models = [m for m in config.MODELS_TO_TRAIN
                  if (config.MODELS_DIR / f"{m}_best.pth").exists()]
        if not models:
            print("  [X]  No trained checkpoints found. Run train.py first.")
            return
    else:
        models = [args.model]

    for name in models:
        try:
            calibrate(name, ood_dir, plot=not args.no_plot,
                      max_images=args.max_images)
        except FileNotFoundError as exc:
            print(f"  [!]  Skipping {name}: {exc}")

    print("\n  Done. The web app and predict.py pick these up automatically.")


if __name__ == "__main__":
    main()
