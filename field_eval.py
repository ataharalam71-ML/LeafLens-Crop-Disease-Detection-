"""
field_eval.py — The honest number: how the model does on real photographs.

PlantVillage is a laboratory dataset. Every image is a single detached leaf,
centred, evenly lit, on a uniform background. A model scoring 99.8% on its test
split has learned to separate ten classes *under those conditions*, and that is
a much smaller claim than "99.8% accurate at diagnosing tomato disease". The gap
between the two is where deployed plant-disease models fail, and it is invisible
if the lab number is the only one you report.

This script measures the gap. Point it at a folder of photographs you took
yourself, in a real field, on a phone, and it reports lab accuracy and field
accuracy side by side - along with what the OOD layer does about the difference.

Expected layout (a subset of classes is fine - only what you have):

    field_data/
        Tomato_Early_blight/      IMG_0413.jpg  IMG_0414.jpg  ...
        Tomato_healthy/           IMG_0501.jpg  ...
        Tomato_Late_blight/       ...

Folder names must match config.CLASS_NAMES. Anything else is skipped with a
warning naming the closest valid class.

What comes out
--------------
- Raw field accuracy, next to the PlantVillage test accuracy from the same model.
- Selective accuracy: accuracy on only the images the OOD layer accepted, and
  the coverage that came at. A useful detector raises accuracy faster than it
  loses coverage; if it does not, the report says so.
- A risk-coverage curve - the standard way to show a system knows what it does
  not know.
- Per-class breakdown and a confusion matrix on field data.
- Optional: --unknown-dir, a folder of things that are not tomato leaves at all,
  to report the outright rejection rate.

Usage:
    python field_eval.py                                # INFERENCE_MODEL
    python field_eval.py --model all
    python field_eval.py --data-dir my_photos/
    python field_eval.py --unknown-dir not_leaves/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import config
from models.model_builder import get_model
from utils.dataset import inference_transform, IMG_EXTENSIONS
from utils.ood import FeatureTap, OODDetector, REJECT as OOD_REJECT
from utils.visualize import plot_confusion_matrix


def resolve_field_dir(cli_value: str = None) -> Path:
    """CLI flag, then TOMATO_FIELD_DIR, then ./field_data."""
    if cli_value:
        return Path(cli_value).expanduser()
    env = os.environ.get("TOMATO_FIELD_DIR")
    if env:
        return Path(env).expanduser()
    return config.BASE_DIR / "field_data"


# ==============================================================================
# Loading field images
# ==============================================================================

def build_field_samples(field_dir: Path, verbose: bool = True):
    """
    Scan class subfolders. Unlike the training loader this accepts a SUBSET of
    classes - you will rarely have field photos of all ten, and demanding all
    ten would make the harness unusable exactly when it matters.
    """
    field_dir = Path(field_dir)
    if not field_dir.is_dir():
        raise FileNotFoundError(
            f"Field image folder not found: {field_dir}\n"
            f"Create it with one subfolder per class you photographed, e.g.\n"
            f"    {field_dir / 'Tomato_Early_blight'}/IMG_0413.jpg\n"
            f"Folder names must match config.CLASS_NAMES."
        )

    class_to_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}
    samples, found = [], {}

    for folder in sorted(field_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in class_to_idx:
            hint = _closest_class(folder.name)
            print(f"  [skip] '{folder.name}' is not a class name."
                  + (f" Did you mean '{hint}'?" if hint else ""))
            continue
        imgs = sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in IMG_EXTENSIONS)
        if not imgs:
            continue
        samples.extend((str(p), class_to_idx[folder.name]) for p in imgs)
        found[folder.name] = len(imgs)

    if not samples:
        raise FileNotFoundError(
            f"No images found under {field_dir}. Expected class subfolders "
            f"containing {sorted(IMG_EXTENSIONS)} files."
        )

    if verbose:
        print(f"\n  Field set: {len(samples)} images across "
              f"{len(found)} of {config.NUM_CLASSES} classes")
        for name, n in sorted(found.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<48} {n:>4}")
        missing = [c for c in config.CLASS_NAMES if c not in found]
        if missing:
            print(f"\n  Not represented ({len(missing)}): "
                  f"{', '.join(config.DISPLAY_NAMES[config.CLASS_NAMES.index(m)] for m in missing)}")
        if len(samples) < 50:
            print(f"\n  [!] {len(samples)} images is a small sample - treat the "
                  f"accuracy as indicative, not precise.")

    return samples, found


def _closest_class(name: str):
    """Cheap typo hint for a mislabelled folder."""
    import difflib
    match = difflib.get_close_matches(name, config.CLASS_NAMES, n=1, cutoff=0.6)
    return match[0] if match else None


# ==============================================================================
# Inference over the field set
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
def run_field(model, detector: OODDetector, samples, desc="field"):
    """
    One pass over the field images.

    Returns preds, labels, confidences, decisions and the per-image score used
    for the risk-coverage curve.
    """
    preds, labels, confs, decisions, scores = [], [], [], [], []

    with FeatureTap(model) as tap:
        for path, label in tqdm(samples, desc=f"  {desc}"):
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:                        # noqa: BLE001
                print(f"  [skip] {path}: {exc}")
                continue

            tensor = inference_transform(img).unsqueeze(0).to(config.DEVICE)
            logits = model(tensor)
            feats = tap.get()

            prob = F.softmax(logits, dim=1)[0]
            top_p, top_i = prob.max(dim=0)

            verdict = detector.evaluate(img, logits[0], feats)

            preds.append(int(top_i))
            labels.append(label)
            confs.append(float(top_p))
            decisions.append(verdict.decision)
            # Calibrated confidence drives the risk-coverage curve.
            scores.append(verdict.scores["msp"])

    return (np.array(preds), np.array(labels), np.array(confs),
            np.array(decisions), np.array(scores))


@torch.no_grad()
def run_unknown(model, detector: OODDetector, unknown_dir: Path):
    """Rejection rate on images that are not tomato leaves at all."""
    paths = [p for p in sorted(Path(unknown_dir).rglob("*"))
             if p.suffix.lower() in IMG_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No images under {unknown_dir}")

    outcomes = []
    with FeatureTap(model) as tap:
        for path in tqdm(paths, desc="  unknown"):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            tensor = inference_transform(img).unsqueeze(0).to(config.DEVICE)
            logits = model(tensor)
            verdict = detector.evaluate(img, logits[0], tap.get())
            outcomes.append(verdict.decision)

    outcomes = np.array(outcomes)
    return {
        "n_images": int(len(outcomes)),
        "rejected": int((outcomes == OOD_REJECT).sum()),
        "rejection_rate": round(float((outcomes == OOD_REJECT).mean()), 4),
        "answered_anyway": int((outcomes != OOD_REJECT).sum()),
    }


# ==============================================================================
# Metrics
# ==============================================================================

def risk_coverage(correct: np.ndarray, scores: np.ndarray, points: int = 25):
    """
    Accuracy as a function of how much of the set we agree to answer.

    Sort by confidence, then walk down: answering only the most confident 20%
    should be far more accurate than answering everything. A curve that stays
    flat means the confidence score carries no information about correctness,
    which is worth knowing before trusting an abstention layer.
    """
    order = np.argsort(-scores)
    ordered = correct[order]
    n = len(ordered)
    out = []
    for frac in np.linspace(1.0 / max(points, 1), 1.0, points):
        k = max(1, int(round(frac * n)))
        out.append({"coverage": round(k / n, 4),
                    "accuracy": round(float(ordered[:k].mean()), 4)})
    return out


def selective_metrics(correct: np.ndarray, decisions: np.ndarray) -> dict:
    """What the OOD layer actually bought, in accuracy and in coverage."""
    answered = decisions != OOD_REJECT
    n = len(correct)
    metrics = {
        "coverage": round(float(answered.mean()), 4),
        "n_answered": int(answered.sum()),
        "n_abstained": int((~answered).sum()),
        "accuracy_all": round(float(correct.mean()), 4),
        "accuracy_answered": (round(float(correct[answered].mean()), 4)
                              if answered.any() else None),
        "accuracy_abstained": (round(float(correct[~answered].mean()), 4)
                               if (~answered).any() else None),
        "decision_counts": {d: int((decisions == d).sum())
                            for d in sorted(set(decisions.tolist()))},
        "n_total": int(n),
    }
    if metrics["accuracy_answered"] is not None:
        metrics["accuracy_gain"] = round(
            metrics["accuracy_answered"] - metrics["accuracy_all"], 4)
    return metrics


# ==============================================================================
# Reporting
# ==============================================================================

def evaluate_model(model_name: str, samples, lab_acc, unknown_dir=None) -> dict:
    from sklearn.metrics import classification_report

    print(f"\n{'='*62}\n  Field evaluation: {model_name}\n{'='*62}")

    model = load_model(model_name)
    detector = OODDetector(model_name)
    if not detector.calibrated:
        print("  [!] OOD detector is running on fallback thresholds. "
              "Run  python calibrate_ood.py  for tuned ones.")

    preds, labels, confs, decisions, scores = run_field(model, detector, samples)
    correct = (preds == labels).astype(np.float64)

    field_acc = float(correct.mean())
    sel = selective_metrics(correct, decisions)
    curve = risk_coverage(correct, scores)

    lab_str = (f"{lab_acc*100:.2f}%" if lab_acc is not None
               else "(not evaluated yet - run evaluate.py)")
    print(f"\n  PlantVillage test accuracy : {lab_str}")
    print(f"  Field accuracy             : {field_acc*100:.2f}%")
    if lab_acc is not None:
        print(f"  Generalisation gap         : "
              f"{(lab_acc - field_acc)*100:+.2f} points")

    print(f"\n  With the OOD layer:")
    print(f"    answered      {sel['n_answered']}/{sel['n_total']} "
          f"({sel['coverage']*100:.0f}% coverage)")
    if sel["accuracy_answered"] is not None:
        print(f"    accuracy on answered  {sel['accuracy_answered']*100:.2f}% "
              f"({sel.get('accuracy_gain', 0)*100:+.2f} points)")
    if sel["accuracy_abstained"] is not None:
        print(f"    accuracy on abstained {sel['accuracy_abstained']*100:.2f}% "
              f"  <- what it declined to answer")

    # Per-class, restricted to classes actually present.
    present = sorted(set(labels.tolist()) | set(preds.tolist()))
    report = classification_report(
        labels, preds,
        labels=present,
        target_names=[config.DISPLAY_NAMES[i] for i in present],
        digits=4, zero_division=0,
    )
    print(f"\n{report}")

    # Artefacts
    out_txt = config.RESULTS_DIR / f"field_report_{model_name}.txt"
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Model: {model_name}\n")
        fh.write(f"PlantVillage test accuracy: "
                 f"{'%.4f' % lab_acc if lab_acc is not None else 'n/a'}\n")
        fh.write(f"Field accuracy: {field_acc:.4f}\n")
        if lab_acc is not None:
            fh.write(f"Generalisation gap: {lab_acc - field_acc:+.4f}\n")
        fh.write(f"OOD coverage: {sel['coverage']:.4f}\n")
        fh.write(f"Accuracy on answered: {sel['accuracy_answered']}\n\n")
        fh.write(report)
    print(f"  Report  ->  {out_txt}")

    if len(present) > 1:
        plot_confusion_matrix(
            labels, preds,
            [config.DISPLAY_NAMES[i] for i in present],
            f"{model_name}_field", config.RESULTS_DIR)

    result = {
        "model": model_name,
        "lab_accuracy": lab_acc,
        "field_accuracy": round(field_acc, 4),
        "gap": round(lab_acc - field_acc, 4) if lab_acc is not None else None,
        "n_field_images": int(len(labels)),
        "ood_calibrated": detector.calibrated,
        "selective": sel,
        "risk_coverage": curve,
    }

    if unknown_dir is not None:
        print(f"\n  Non-leaf rejection test: {unknown_dir}")
        unk = run_unknown(model, detector, unknown_dir)
        print(f"    rejected {unk['rejected']}/{unk['n_images']} "
              f"({unk['rejection_rate']*100:.0f}%)")
        if unk["answered_anyway"]:
            print(f"    [!] {unk['answered_anyway']} non-leaf images still got "
                  f"a diagnosis - tighten thresholds or recalibrate.")
        result["unknown_rejection"] = unk

    return result


# ==============================================================================
# Plots
# ==============================================================================

def plot_lab_vs_field(results: dict, out_dir: Path):
    """The headline chart: lab bar next to field bar, gap annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, r in results.items() if r.get("lab_accuracy") is not None]
    if not names:
        return
    names.sort(key=lambda n: -results[n]["field_accuracy"])

    lab = [results[n]["lab_accuracy"] * 100 for n in names]
    field = [results[n]["field_accuracy"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(names)), 5))
    b1 = ax.bar(x - width / 2, lab, width, label="PlantVillage (lab)",
                color="#7fb3d5", edgecolor="white")
    b2 = ax.bar(x + width / 2, field, width, label="Field photographs",
                color="#e59866", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", fontsize=9,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Lab accuracy vs. real field photographs",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "field_vs_lab.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart   ->  {out}")


def plot_risk_coverage(results: dict, out_dir: Path):
    """Accuracy vs. how much of the set the system agrees to answer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, res in results.items():
        curve = res.get("risk_coverage")
        if not curve:
            continue
        cov = [p["coverage"] * 100 for p in curve]
        acc = [p["accuracy"] * 100 for p in curve]
        ax.plot(cov, acc, marker="o", ms=3, lw=1.8, label=name)

    ax.set_xlabel("Coverage - share of images answered (%)")
    ax.set_ylabel("Accuracy on answered images (%)")
    ax.set_title("Risk-coverage: does abstaining actually help?",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = out_dir / "field_risk_coverage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart   ->  {out}")


# ==============================================================================
# CLI
# ==============================================================================

def load_lab_accuracies() -> dict:
    path = config.RESULTS_DIR / "test_results.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                       # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate on real field photographs, next to the lab number")
    ap.add_argument("--model", default=config.INFERENCE_MODEL,
                    help="Model name, or 'all' for every trained checkpoint")
    ap.add_argument("--data-dir", default=None,
                    help="Field image folder (default: ./field_data or $TOMATO_FIELD_DIR)")
    ap.add_argument("--unknown-dir", default=None,
                    help="Folder of non-tomato-leaf images, to test outright rejection")
    args = ap.parse_args()

    field_dir = resolve_field_dir(args.data_dir)
    print(f"\n  Field images: {field_dir}")

    try:
        samples, _ = build_field_samples(field_dir)
    except FileNotFoundError as exc:
        print(f"\n  [X]  {exc}")
        return

    unknown_dir = Path(args.unknown_dir) if args.unknown_dir else None
    if unknown_dir is not None and not unknown_dir.is_dir():
        print(f"  [X]  --unknown-dir not found: {unknown_dir}")
        return

    if args.model.lower() == "all":
        models = [m for m in config.MODELS_TO_TRAIN
                  if (config.MODELS_DIR / f"{m}_best.pth").exists()]
        if not models:
            print("  [X]  No trained checkpoints found. Run train.py first.")
            return
    else:
        models = [args.model]

    lab = load_lab_accuracies()
    results = {}
    for name in models:
        try:
            results[name] = evaluate_model(name, samples, lab.get(name), unknown_dir)
        except FileNotFoundError as exc:
            print(f"  [!]  Skipping {name}: {exc}")

    if not results:
        print("\n  [X]  Nothing evaluated.")
        return

    print(f"\n{'='*62}\n  Lab vs. field summary\n{'='*62}")
    print(f"  {'model':<18}{'lab':>9}{'field':>9}{'gap':>9}{'covg':>8}{'sel.acc':>10}")
    print(f"  {'-'*61}")
    for name, res in sorted(results.items(),
                            key=lambda kv: -kv[1]["field_accuracy"]):
        lab_s = f"{res['lab_accuracy']*100:.2f}" if res["lab_accuracy"] is not None else "n/a"
        gap_s = f"{res['gap']*100:+.2f}" if res["gap"] is not None else "n/a"
        sel = res["selective"]
        sel_s = (f"{sel['accuracy_answered']*100:.2f}"
                 if sel["accuracy_answered"] is not None else "n/a")
        print(f"  {name:<18}{lab_s:>9}{res['field_accuracy']*100:>9.2f}"
              f"{gap_s:>9}{sel['coverage']*100:>7.0f}%{sel_s:>10}")
    print(f"  {'='*61}")

    plot_lab_vs_field(results, config.RESULTS_DIR)
    plot_risk_coverage(results, config.RESULTS_DIR)

    out = config.RESULTS_DIR / "field_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results ->  {out}\n")


if __name__ == "__main__":
    main()
