"""
utils/ood.py — Knowing when NOT to answer.

A 10-way softmax is a closed world: every image it is ever shown gets forced
into one of ten tomato classes. Photograph a coffee cup, a potato leaf or your
hand and the model will not shrug - it will return a confident disease. That is
the single most damaging failure mode a field diagnosis tool can have, because
the wrong answer arrives with the same confidence bar as the right one.

This module wraps the classifier in an abstention layer. It answers a different
question from "which disease": it answers "should I be answering at all".

The four signals
----------------
1. Image quality (no model needed, runs first)
      vegetation share, blur (variance of Laplacian), exposure.
   Catches the honest failures - a photo of a wall, a dark blurry snap - and
   catches them with a reason a user can act on ("move closer", "hold still").

2. Maximum softmax probability + normalised entropy + top-1/top-2 margin
      The classic baseline (Hendrycks & Gimpel, ICLR 2017). Weak on its own
      because modern CNNs are overconfident, which is exactly why it is not
      used on its own here.

3. Free energy
      E(x) = -T * logsumexp(logits / T)
   From Liu et al., NeurIPS 2020. Unlike softmax, energy keeps the magnitude of
   the logits, which is the part that actually collapses on out-of-distribution
   input - softmax throws that away when it normalises. This is the strongest
   logit-based signal available without retraining.

4. Mahalanobis distance in feature space
      Class-conditional Gaussians fitted on penultimate features with a shared,
      shrinkage-regularised covariance (Lee et al., NeurIPS 2018). Measures
      "does this look like anything I was trained on" in representation space
      rather than in output space. Requires calibration; skipped if absent.

Decisions are made by vote, not by any one signal, so a single borderline score
downgrades the answer to REVIEW rather than throwing it away.

Calibration
-----------
Thresholds are not hand-tuned constants. `calibrate_ood.py` fits a temperature
on the validation split (Guo et al., ICML 2017) and then sets each threshold at
the quantile that retains OOD_TARGET_TPR (default 95%) of genuine validation
images. Without a calibration file the detector still runs, using the
conservative fallbacks below, and reports `calibrated: False` everywhere -
including in the UI, so an uncalibrated demo is never passed off as a tuned one.

Usage:
    from utils.ood import OODDetector
    det = OODDetector("EfficientNetB0")
    verdict = det.evaluate(pil_img, logits, features)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

import config

# ==============================================================================
# Decisions
# ==============================================================================

ACCEPT = "accept"    # answer normally
REVIEW = "review"    # answer, but flagged - one signal is unhappy
REJECT = "reject"    # refuse to answer - this is not a tomato leaf we know


# ==============================================================================
# 1. Image quality - runs before the model
# ==============================================================================

def assess_quality(image) -> dict:
    """
    Cheap, model-free checks on whether the photo is even usable.

    Returns a dict of measurements plus a `flags` list naming what is wrong.
    Flags: 'not_vegetation', 'blurry', 'too_dark', 'too_bright'.
    """
    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    # -- vegetation share: normalised Excess Green above zero --
    f = bgr.astype(np.float32)
    total = f.sum(axis=2, keepdims=True)
    total[total < 1e-6] = 1e-6
    nrm = f / total
    b, g, r = nrm[:, :, 0], nrm[:, :, 1], nrm[:, :, 2]
    exg = 2.0 * g - r - b
    vegetation = float((exg > 0.02).mean())

    # -- blur: variance of the Laplacian; low variance means no sharp edges --
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # -- exposure --
    brightness = float(gray.mean())
    saturation = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())

    flags: List[str] = []
    if vegetation < config.OOD_MIN_VEGETATION:
        flags.append("not_vegetation")
    if blur_score < config.OOD_MIN_SHARPNESS:
        flags.append("blurry")
    if brightness < config.OOD_MIN_BRIGHTNESS:
        flags.append("too_dark")
    if brightness > config.OOD_MAX_BRIGHTNESS:
        flags.append("too_bright")

    return {
        "vegetation": round(vegetation, 4),
        "sharpness": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "flags": flags,
    }


QUALITY_MESSAGES = {
    "not_vegetation": "This does not look like a plant leaf - only {pct:.0f}% of "
                      "the frame is vegetation.",
    "blurry":         "The photo is too blurry to diagnose - hold steady and refocus.",
    "too_dark":       "The photo is too dark - move into better light.",
    "too_bright":     "The photo is blown out - avoid direct glare on the leaf.",
}


# ==============================================================================
# 2-3. Logit-based scores
# ==============================================================================

def logit_scores(logits: torch.Tensor, temperature: float = 1.0) -> dict:
    """
    Confidence, entropy, margin and free energy from a single logit vector.

    Args:
        logits      : (num_classes,) or (1, num_classes) tensor.
        temperature : calibration temperature; 1.0 = uncalibrated.

    Energy is computed on the *raw* logits, deliberately. Temperature scaling
    exists to fix the probabilities; the energy score has its own temperature in
    the literature and rescaling it twice just moves the threshold around.
    """
    z = logits.detach().float().reshape(-1)
    scaled = z / max(temperature, 1e-6)

    probs = F.softmax(scaled, dim=0)
    top2 = torch.topk(probs, k=min(2, probs.numel()))

    msp = float(top2.values[0])
    margin = float(top2.values[0] - top2.values[1]) if probs.numel() > 1 else 1.0

    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    max_entropy = float(np.log(probs.numel()))
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

    energy = float(-torch.logsumexp(z, dim=0))

    return {
        "msp": round(msp, 4),
        "margin": round(margin, 4),
        "entropy": round(entropy_norm, 4),
        "energy": round(energy, 4),
    }


# ==============================================================================
# 4. Feature-space Mahalanobis
# ==============================================================================

class MahalanobisScorer:
    """
    Class-conditional Gaussians with a shared covariance over penultimate features.

    Score = min over classes of the squared Mahalanobis distance. Low means the
    representation sits inside a class cluster the model was trained on; high
    means the image landed somewhere in feature space with no training data.
    """

    def __init__(self, means: np.ndarray, precision: np.ndarray):
        self.means = means.astype(np.float64)          # (num_classes, dim)
        self.precision = precision.astype(np.float64)  # (dim, dim)

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray,
            num_classes: int, shrinkage: float = 0.1) -> "MahalanobisScorer":
        """
        Fit means and a shared, shrinkage-regularised inverse covariance.

        Shrinkage matters: the feature dimension (1280 for EfficientNet-B0) is
        the same order as the number of validation images, so the raw covariance
        is rank-deficient and its inverse is meaningless. Pulling it toward a
        scaled identity fixes the conditioning.
        """
        feats = features.astype(np.float64)
        dim = feats.shape[1]

        means = np.zeros((num_classes, dim), dtype=np.float64)
        centred = np.empty_like(feats)
        for c in range(num_classes):
            sel = labels == c
            if not np.any(sel):
                continue
            means[c] = feats[sel].mean(axis=0)
            centred[sel] = feats[sel] - means[c]

        cov = (centred.T @ centred) / max(len(feats) - num_classes, 1)
        cov = (1.0 - shrinkage) * cov + shrinkage * (np.trace(cov) / dim) * np.eye(dim)
        precision = np.linalg.pinv(cov)
        return cls(means, precision)

    def score(self, feature: np.ndarray) -> float:
        """Squared Mahalanobis distance to the nearest class centroid."""
        feature = np.asarray(feature, dtype=np.float64).reshape(-1)
        if feature.size != self.means.shape[1]:
            # Fail loudly rather than let numpy broadcast a batched or
            # wrong-architecture feature vector into a meaningless number.
            raise ValueError(
                f"Feature has {feature.size} dims, but this scorer was fitted "
                f"on {self.means.shape[1]}. Pass a single image's penultimate "
                f"features, and make sure the calibration matches the model.")
        delta = self.means - feature.reshape(1, -1)
        dists = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
        return float(dists.min())

    def score_batch(self, features: np.ndarray) -> np.ndarray:
        feats = features.astype(np.float64)
        out = np.empty(len(feats), dtype=np.float64)
        for i, f in enumerate(feats):
            delta = self.means - f.reshape(1, -1)
            out[i] = np.einsum("ij,jk,ik->i", delta, self.precision, delta).min()
        return out

    def to_dict(self) -> dict:
        return {"means": self.means.tolist(), "precision": self.precision.tolist()}


# ==============================================================================
# Penultimate feature extraction - one hook that works for all 8 architectures
# ==============================================================================

def last_linear(model: nn.Module) -> nn.Linear:
    """The final nn.Linear in the network - every model here ends in one."""
    found = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            found = module
    if found is None:
        raise ValueError("Model has no nn.Linear layer to hook.")
    return found


class FeatureTap:
    """
    Captures the input to the final Linear layer, i.e. the penultimate features.

    Hooking the classifier's input rather than a named backbone layer means the
    same code covers CustomCNN, ResNet50, EfficientNetB0, MobileNetV3, VGG16,
    AlexNet, UNet and DenseNet121 with no per-architecture special-casing.
    """

    def __init__(self, model: nn.Module):
        self.features: Optional[torch.Tensor] = None
        self._handle = last_linear(model).register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.features = inputs[0].detach()

    def get(self) -> torch.Tensor:
        if self.features is None:
            raise RuntimeError("No forward pass has run since the tap was attached.")
        return self.features

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ==============================================================================
# Verdict
# ==============================================================================

@dataclass
class Verdict:
    decision: str                        # ACCEPT / REVIEW / REJECT
    headline: str                        # one line for the user
    reasons: List[str] = field(default_factory=list)
    votes: int = 0                       # how many evidence FAMILIES objected
    families: List[str] = field(default_factory=list)   # which ones
    calibrated: bool = False
    scores: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return self.decision != REJECT

    def to_dict(self) -> dict:
        d = asdict(self)
        d["answerable"] = self.answerable
        return d


# ==============================================================================
# The detector
# ==============================================================================

class OODDetector:
    """
    Runtime abstention layer for one model.

    Loads `results/ood_calibration_<model>.json` when present. Without it the
    detector still works on conservative fallbacks and reports calibrated=False.
    """

    def __init__(self, model_name: str, calibration_dir: Path = None):
        self.model_name = model_name
        self.calibrated = False
        self.temperature = 1.0
        self.thresholds = dict(config.OOD_FALLBACK_THRESHOLDS)
        self.mahalanobis: Optional[MahalanobisScorer] = None
        self._load(calibration_dir or config.RESULTS_DIR)

    # -- calibration ----------------------------------------------------------

    def calibration_path(self, results_dir: Path) -> Path:
        return Path(results_dir) / f"ood_calibration_{self.model_name}.json"

    def _load(self, results_dir: Path):
        path = self.calibration_path(results_dir)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.temperature = float(data.get("temperature", 1.0))
            self.thresholds.update(data.get("thresholds", {}))

            # The Gaussians live in a sidecar .npz: a 1280x1280 precision matrix
            # is ~1.6M floats, which is fine as binary and absurd as JSON text.
            maha_file = data.get("mahalanobis_file")
            if maha_file:
                maha_path = Path(path).parent / maha_file
                if maha_path.exists():
                    with np.load(maha_path) as npz:
                        self.mahalanobis = MahalanobisScorer(
                            npz["means"], npz["precision"])
                else:
                    print(f"  [ood] Mahalanobis file missing: {maha_path.name} "
                          f"- continuing without that signal.")
            self.calibrated = True
        except Exception as exc:                       # noqa: BLE001
            print(f"  [ood] Ignoring unreadable calibration {path.name}: {exc}")

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, image, logits: torch.Tensor,
                 features: Optional[torch.Tensor] = None) -> Verdict:
        """
        Decide whether this prediction should be shown.

        Args:
            image    : PIL.Image / path / BGR array - for the quality checks.
            logits   : raw model output, (num_classes,) or (1, num_classes).
            features : optional penultimate features for the Mahalanobis test.
        """
        quality = assess_quality(image)
        scores = logit_scores(logits, self.temperature)

        # -- hard gate: not a plant at all -----------------------------------
        # This one does not vote. If the frame contains no vegetation there is
        # nothing to diagnose, and no amount of model confidence should override
        # that - a confident answer about a coffee cup is the exact failure this
        # module exists to prevent.
        if "not_vegetation" in quality["flags"]:
            return Verdict(
                decision=REJECT,
                headline="Not a tomato leaf",
                reasons=[QUALITY_MESSAGES["not_vegetation"].format(
                    pct=quality["vegetation"] * 100)],
                votes=config.OOD_REJECT_VOTES,
                families=["quality"],
                calibrated=self.calibrated,
                scores=scores,
                quality=quality,
            )

        # -- Mahalanobis needs computing before the signals are collected ----
        if self.mahalanobis is not None and features is not None:
            feat = features.detach().float().reshape(-1).cpu().numpy()
            scores["mahalanobis"] = round(float(self.mahalanobis.score(feat)), 2)

        # -- collect which individual signals object -------------------------
        objections = {}   # signal name -> reason string

        for flag in ("blurry", "too_dark", "too_bright"):
            if flag in quality["flags"]:
                objections[flag] = QUALITY_MESSAGES[flag]

        if scores["msp"] < self.thresholds["msp"]:
            objections["msp"] = (
                f"Confidence {scores['msp']*100:.1f}% is below the "
                f"{self.thresholds['msp']*100:.1f}% operating point.")

        if scores["entropy"] > self.thresholds["entropy"]:
            objections["entropy"] = (
                "The model is spread across several diseases rather than "
                "settling on one.")

        if scores["margin"] < self.thresholds["margin"]:
            objections["margin"] = (
                "The top two diseases are too close to separate reliably.")

        if scores["energy"] > self.thresholds["energy"]:
            objections["energy"] = (
                "Logit magnitude is outside the range seen in training - this "
                "image does not resemble the training data.")

        if ("mahalanobis" in scores
                and scores["mahalanobis"] > self.thresholds.get(
                    "mahalanobis", float("inf"))):
            objections["mahalanobis"] = (
                "The image sits far from every disease cluster the model learned.")

        # -- one vote per evidence family ------------------------------------
        # Three unhappy softmax statistics are one piece of evidence, not three.
        families, reasons = [], []
        for family, members in config.OOD_SIGNAL_FAMILIES.items():
            hit = [m for m in members if m in objections]
            if hit:
                families.append(family)
                reasons.extend(objections[m] for m in hit)

        votes = len(families)

        if votes >= config.OOD_REJECT_VOTES:
            decision = REJECT
            headline = "Cannot diagnose this image reliably"
        elif votes >= config.OOD_REVIEW_VOTES:
            decision = REVIEW
            headline = "Low-confidence diagnosis - verify before acting"
        else:
            decision, headline = ACCEPT, "Confident diagnosis"

        return Verdict(
            decision=decision,
            headline=headline,
            reasons=reasons,
            votes=votes,
            families=families,
            calibrated=self.calibrated,
            scores=scores,
            quality=quality,
        )

    # -- convenience ----------------------------------------------------------

    def status(self) -> dict:
        """What the UI shows about how this detector is configured."""
        return {
            "model": self.model_name,
            "calibrated": self.calibrated,
            "temperature": round(self.temperature, 4),
            "thresholds": {k: round(float(v), 4) for k, v in self.thresholds.items()},
            "mahalanobis": self.mahalanobis is not None,
        }


# ==============================================================================
# Helpers
# ==============================================================================

def _to_bgr(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        arr = cv2.imread(str(image))
        if arr is None:
            raise ValueError(f"Cannot read image: {image}")
        return arr
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image)}")


_DETECTORS: dict = {}


def get_detector(model_name: str) -> OODDetector:
    """Process-wide cache - calibration files are read once."""
    if model_name not in _DETECTORS:
        _DETECTORS[model_name] = OODDetector(model_name)
    return _DETECTORS[model_name]
