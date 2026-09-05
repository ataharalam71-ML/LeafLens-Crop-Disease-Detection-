"""
predict.py  —  Predict disease from a single image file.

Prints the diagnosis, whether the system is willing to stand behind it
(see utils/ood.py), how far the infection has progressed, and what to do.

Usage:
    python predict.py --image leaf.jpg
    python predict.py --image leaf.jpg --model ResNet50
    python predict.py --image leaf.jpg --topk 3
    python predict.py --image leaf.jpg --no-treatment    # just the label
    python predict.py --image leaf.jpg --json            # machine-readable
"""

import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image

import config
import treatments
from models.model_builder import get_model
from utils.dataset import inference_transform
from utils.ood import FeatureTap, get_detector, ACCEPT, REVIEW, REJECT
from utils.severity import estimate_severity


def load_model(model_name: str):
    ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE).eval()
    return model


@torch.no_grad()
def predict(image_path: str, model_name: str = None, topk: int = 3,
            with_treatment: bool = True, check_ood: bool = True) -> dict:
    """
    Classify one image and decide whether the answer is worth reporting.

    Returns a dict with keys: model, top, topk, verdict, severity, treatment.
    `verdict` is None when check_ood is False.
    """
    model_name = model_name or config.INFERENCE_MODEL
    model = load_model(model_name)

    img = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(config.DEVICE)

    with FeatureTap(model) as tap:
        logits = model(tensor)
        features = tap.get()

    probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(min(topk, config.NUM_CLASSES))
    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append({
            "class": config.CLASS_NAMES[idx],
            "display": config.DISPLAY_NAMES[idx],
            "confidence": round(float(prob), 4),
        })

    verdict = None
    if check_ood and config.OOD_ENABLED:
        verdict = get_detector(model_name).evaluate(img, logits[0], features)

    out = {
        "image": str(image_path),
        "model": model_name,
        "top": results[0],
        "topk": results,
        "verdict": verdict.to_dict() if verdict else None,
        "severity": None,
        "treatment": None,
    }

    # No point costing the user a spray programme for an image we just refused.
    if with_treatment and (verdict is None or verdict.answerable):
        top_class = results[0]["class"]
        sev = estimate_severity(img, top_class)
        out["severity"] = sev
        out["treatment"] = treatments.get_treatment(
            top_class,
            severity=sev["fraction"] if sev["reliable"] else None,
            confidence=results[0]["confidence"],
        )

    return out


# ==============================================================================
# Terminal rendering
# ==============================================================================

_RULE = "  " + "-" * 62

_DECISION_MARK = {ACCEPT: "[OK]", REVIEW: "[?]", REJECT: "[X]"}


def print_result(res: dict, with_treatment: bool = True):
    print(f"\n  Image  : {res['image']}")
    print(f"  Model  : {res['model']}")

    verdict = res.get("verdict")
    if verdict:
        mark = _DECISION_MARK.get(verdict["decision"], "[?]")
        print(_RULE)
        print(f"  {mark} {verdict['headline']}")
        for reason in verdict["reasons"]:
            print(f"       - {reason}")
        if not verdict["calibrated"]:
            print("       - (detector uncalibrated - run calibrate_ood.py)")

        if verdict["decision"] == REJECT:
            print(_RULE)
            print("  No diagnosis given. Retake the photo: fill the frame with a")
            print("  single tomato leaf, in even light, held steady.")
            print()
            return

    print(_RULE)
    for i, r in enumerate(res["topk"]):
        bar = "#" * int(r["confidence"] * 30)
        print(f"  #{i+1}  {r['display']:<30} {r['confidence']*100:6.2f}%  {bar}")

    sev = res.get("severity")
    if sev and sev["fraction"] > 0 and sev["reliable"]:
        band = treatments.severity_band(sev["fraction"])
        print(_RULE)
        print(f"  Severity: {band['band']} - {band['percent']:.1f}% of the leaf "
              f"is symptomatic")
        if not sev["area_based"]:
            print(f"            (under-reads for this virus - see note)")

    if with_treatment and res.get("treatment"):
        _print_treatment(res["treatment"])

    print()


def _print_treatment(t: dict):
    print(_RULE)
    print(f"  {t['headline']}")
    print()
    print(f"  Pathogen  : {t['pathogen']}  ({t['type']})")
    print(f"  Urgency   : {t['urgency']}/5 ({t['urgency_label']})   "
          f"Contagion: {t['contagion']}")
    lo, hi = t["yield_loss_pct"]
    if hi > 0:
        print(f"  Untreated : {lo}-{hi}% yield loss")

    def block(title, items):
        if not items:
            return
        print(f"\n  {title}")
        for item in items:
            print(f"    - {item}")

    block("ORGANIC / LOW-INPUT", t["organic"])
    block("CHEMICAL", t["chemical"])
    block("CULTURAL / PREVENTIVE", t["cultural"])
    block("AVOID", t["do_not"])

    print(f"\n  {t['disclaimer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--no-treatment", action="store_true",
                        help="Print the label only, no advisory")
    parser.add_argument("--no-ood", action="store_true",
                        help="Skip the abstention layer and always answer")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of formatted text")
    args = parser.parse_args()

    result = predict(args.image, args.model, args.topk,
                     with_treatment=not args.no_treatment,
                     check_ood=not args.no_ood)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result, with_treatment=not args.no_treatment)
