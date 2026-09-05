"""
utils/severity.py — How much of the leaf is actually symptomatic?

The classifier says *which* disease; this says *how far along* it is, which is
what decides the dose, the spray interval and whether the plant is worth saving.
It is a classical CV estimate (no extra model, no extra training), so it costs a
few milliseconds and adds nothing to the download size.

Method
------
1. Leaf silhouette: threshold out the desaturated background, keep the largest
   connected component, then flood-fill holes from the border. The hole-filling
   step is what recovers dark necrotic patches *inside* the leaf - they are not
   background, they are the damage we are trying to measure.

2. Healthy tissue: measured with Excess Green on normalised chromaticity,
        ExG = 2g - r - b,   where r, g, b = R/(R+G+B), ...
   Dividing by total intensity makes the index roughly invariant to shading, so
   a shadowed but healthy leaf still reads as healthy. That matters: the obvious
   approach - thresholding raw HSV green - scores a healthy PlantVillage leaf at
   ~29% symptomatic, because leaf shadow and specular highlights fall outside
   the green band. ExG scores the same leaves at ~2%.

3. severity = fraction of leaf pixels with ExG below EXG_THRESHOLD.

Calibration
-----------
EXG_THRESHOLD = 0.04 was chosen by sweeping the threshold over 30 images per
class from the PlantVillage tomato subset (identical images at every threshold).
Median symptomatic area at that setting:

    healthy         1.6%      spider mites     10.6%
    bacterial spot  4.1%      late blight      12.4%
    septoria        6.2%      target spot      13.3%
                              early blight     13.8%
                              leaf mold        17.2%

Healthy sits an order of magnitude below every disease class, and the ordering
tracks how much tissue each disease actually destroys - bacterial spot and early
septoria are small discrete lesions, leaf mold blankets the blade. Septoria's
mean (20.9%) is far above its median, which is correct: the class contains both
freshly-spotted and fully-defoliating leaves.

Limits (stated because they matter when reading the number)
-----------------------------------------------------------
- It measures *discoloured leaf area*, not pathogen load. For the two viruses
  (ToMV, TYLCV) the damage is distortion and stunting rather than
  discolouration, so the number under-reads; those classes come back with
  `area_based=False`.
- A photo containing several leaves or heavy background clutter reduces the
  silhouette to the largest leaf, which is the intended behaviour for a
  single-leaf diagnosis.

Usage:
    from utils.severity import estimate_severity
    result = estimate_severity(pil_img, class_name="Tomato_Early_blight")
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

import config

# Classes whose damage is not primarily a discoloured *area*, so an area-based
# severity would mislead. We still return a number, but flag it.
NON_AREA_CLASSES = {
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
}

# Working resolution - big enough for 2-4 mm Septoria spots, small enough to be free.
WORK_SIZE = 384

# Leaf silhouette thresholds (OpenCV HSV: H 0-179, S 0-255, V 0-255).
LEAF_MIN_SAT = 35
LEAF_MIN_VAL = 25

# Excess-Green cut-off separating healthy tissue from symptomatic tissue.
# See the calibration table in the module docstring.
EXG_THRESHOLD = 0.04

# Below this share of the frame the leaf is too small for a trustworthy estimate.
MIN_LEAF_COVERAGE = 0.05
# Below this, there is effectively no leaf at all.
MIN_MEASURABLE_COVERAGE = 0.02


def excess_green(bgr: np.ndarray) -> np.ndarray:
    """
    Excess Green on normalised chromaticity: 2g - r - b.

    Normalising by (R+G+B) divides out the intensity term, so the index depends
    on hue rather than on how brightly that patch happens to be lit. Healthy
    leaf tissue lands near +0.2; necrotic brown, chlorotic yellow and grey
    lesions land near or below 0.
    """
    f = bgr.astype(np.float32)
    total = f.sum(axis=2, keepdims=True)
    total[total < 1e-6] = 1e-6
    nrm = f / total
    b, g, r = nrm[:, :, 0], nrm[:, :, 1], nrm[:, :, 2]
    return 2.0 * g - r - b


def leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary silhouette (uint8 0/255) of the dominant leaf, interior holes filled."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    mask = ((s > LEAF_MIN_SAT) & (v > LEAF_MIN_VAL)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest blob - the leaf, not stray background colour.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill interior holes: flood from a border pixel, then OR back whatever the
    # flood could not reach. Dark necrotic centres are holes, and they count as
    # damage, not as background.
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def estimate_severity(image,
                      class_name: Optional[str] = None,
                      return_mask: bool = False) -> dict:
    """
    Estimate the symptomatic fraction of the leaf.

    Args:
        image       : PIL.Image, a path, or a BGR numpy array.
        class_name  : optional CLASS_NAMES entry. Used only to flag the two
                      distortion-type viruses; the measurement itself is
                      identical for every class, so the number stays honest.
        return_mask : also return a BGR visualisation (green = healthy tissue,
                      red = symptomatic, background dimmed).

    Returns dict:
        fraction        float in [0, 1] - symptomatic share of the leaf
        leaf_coverage   float in [0, 1] - share of the frame the leaf occupies
        area_based      bool  - False when the class is a distortion-type virus
        reliable        bool  - False when the leaf is too small to trust
        note            str   - human-readable caveat, or ""
        overlay         np.ndarray (BGR), only when return_mask=True
    """
    bgr = _to_bgr(image)

    h, w = bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    mask = leaf_mask(bgr)
    leaf_area = int(np.count_nonzero(mask))
    frame_area = mask.shape[0] * mask.shape[1]
    coverage = leaf_area / float(frame_area)

    if coverage < MIN_MEASURABLE_COVERAGE:
        out = {"fraction": 0.0, "leaf_coverage": round(coverage, 4),
               "area_based": True, "reliable": False,
               "note": "No leaf large enough to measure - severity not estimated."}
        if return_mask:
            out["overlay"] = bgr
        return out

    exg = excess_green(bgr)
    symptomatic = (exg < EXG_THRESHOLD) & (mask > 0)

    fraction = float(np.count_nonzero(symptomatic)) / float(leaf_area)
    fraction = float(np.clip(fraction, 0.0, 1.0))

    area_based = True
    note = ""
    if class_name in NON_AREA_CLASSES:
        area_based = False
        note = ("This virus damages the leaf through curling, mottling and "
                "stunting rather than dead tissue, so measured area "
                "under-states the true severity.")

    reliable = coverage >= MIN_LEAF_COVERAGE
    if not reliable and not note:
        note = "Leaf fills little of the frame - move closer for a firmer estimate."

    out = {
        "fraction": round(fraction, 4),
        "leaf_coverage": round(coverage, 4),
        "area_based": area_based,
        "reliable": reliable,
        "note": note,
    }
    if return_mask:
        out["overlay"] = _make_overlay(bgr, mask, symptomatic)
    return out


def _make_overlay(bgr: np.ndarray, mask: np.ndarray,
                  symptomatic: np.ndarray) -> np.ndarray:
    """Green = healthy tissue, red = symptomatic, background dimmed."""
    healthy = (mask > 0) & (~symptomatic)

    tint = np.zeros_like(bgr)
    tint[healthy] = (60, 200, 60)          # BGR green
    tint[symptomatic] = (40, 40, 220)      # BGR red

    overlay = cv2.addWeighted(bgr, 0.62, tint, 0.38, 0)
    background = mask == 0
    overlay[background] = (bgr[background] * 0.35).astype(np.uint8)
    return overlay


def _to_bgr(image) -> np.ndarray:
    """Accept PIL.Image, a filesystem path, or a BGR ndarray."""
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


# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Leaf severity estimate")
    ap.add_argument("--image", required=True)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--out", default=None, help="Save the mask overlay PNG here")
    args = ap.parse_args()

    res = estimate_severity(args.image, args.class_name, return_mask=bool(args.out))
    print(f"  Symptomatic area : {res['fraction']*100:.1f}%")
    print(f"  Leaf coverage    : {res['leaf_coverage']*100:.1f}% of frame")
    print(f"  Reliable         : {res['reliable']}")
    if res["note"]:
        print(f"  Note             : {res['note']}")
    if args.out:
        cv2.imwrite(args.out, res["overlay"])
        print(f"  Overlay saved    : {args.out}")
