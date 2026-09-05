"""
smoke_test.py  —  End-to-end pipeline test on a tiny slice of the dataset.

Proves that train → evaluate → predict → Grad-CAM all work, in a couple of
minutes on CPU, before you commit to a multi-hour GPU run.

    python smoke_test.py              # all 5 architectures, 1 epoch, tiny subset
    python smoke_test.py --model CustomCNN

Nothing is written to saved_models/ or results/ — everything goes to
smoke_test_out/, which you can delete afterwards.
"""

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

# ── Shrink the problem BEFORE any project module reads the config ─────────────
os.environ.setdefault("TOMATO_MAX_PER_CLASS", "8")   # 8 images/class → 80 total
os.environ.setdefault("TOMATO_NUM_WORKERS", "0")     # no worker processes

import config  # noqa: E402

SMOKE_DIR = config.BASE_DIR / "smoke_test_out"

config.IMAGE_SIZE  = 64      # tiny images → fast forward/backward
config.BATCH_SIZE  = 4
config.NUM_EPOCHS  = 1
config.PATIENCE    = 1
config.MODEL_BATCH_SIZE = {}
config.MODELS_DIR  = SMOKE_DIR / "models"
config.RESULTS_DIR = SMOKE_DIR / "results"
config.LOGS_DIR    = SMOKE_DIR / "logs"
for d in (config.MODELS_DIR, config.RESULTS_DIR, config.LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Imported after the config is patched, so module-level transforms pick up IMAGE_SIZE.
import torch  # noqa: E402
import train  # noqa: E402
import evaluate as evaluate_mod  # noqa: E402
from models import model_builder  # noqa: E402
from utils.dataset import build_samples  # noqa: E402
from utils.gradcam import GradCAM, get_target_layer  # noqa: E402
from utils.dataset import inference_transform  # noqa: E402
from PIL import Image  # noqa: E402

# Train from scratch — the smoke test must not depend on a 670 MB weight download.
_original_get_model = model_builder.get_model


def _get_model_no_download(name, num_classes=10, freeze_backbone=False, pretrained=True):
    return _original_get_model(name, num_classes, freeze_backbone, pretrained=False)


model_builder.get_model = _get_model_no_download
train.get_model = _get_model_no_download
evaluate_mod.get_model = _get_model_no_download

PASSED, FAILED = [], []


def step(title, fn):
    print(f"\n{'━' * 62}\n  ▶  {title}\n{'━' * 62}")
    try:
        fn()
        print(f"  ✅  PASS — {title}")
        PASSED.append(title)
    except Exception as e:
        print(f"  ❌  FAIL — {title}: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILED.append(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Only test this architecture (default: all enabled)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep smoke_test_out/ instead of deleting it")
    args = parser.parse_args()

    models_to_test = ([args.model] if args.model else
                      [m for m, on in config.MODELS_TO_TRAIN.items() if on])

    print("\n" + "═" * 62)
    print("  🍅  SMOKE TEST — full pipeline on a tiny subset")
    print("═" * 62)
    print(f"  Dataset : {config.DATA_DIR}")
    print(f"  Device  : {config.DEVICE}")
    print(f"  Models  : {', '.join(models_to_test)}")
    print(f"  Config  : {config.IMAGE_SIZE}px, batch {config.BATCH_SIZE}, "
          f"1 epoch, {os.environ['TOMATO_MAX_PER_CLASS']} imgs/class")

    # ── 1. Dataset loads ──
    sample_holder = {}

    def _data():
        samples = build_samples(config.DATA_DIR)
        assert samples, "no samples found"
        sample_holder["path"] = samples[0][0]
        print(f"  Loaded {len(samples)} images across {config.NUM_CLASSES} classes")

    step("Dataset scan + stratified split", _data)

    if FAILED:
        _finish(args)
        return

    # ── 2. Train each architecture for 1 epoch ──
    for name in models_to_test:
        step(f"Train 1 epoch — {name}",
             lambda n=name: train.train_model(n, epochs=1, lr=1e-3))

    # ── 3. Evaluate every checkpoint that got written ──
    for name in models_to_test:
        if (config.MODELS_DIR / f"{name}_best.pth").exists():
            step(f"Evaluate on test split — {name}",
                 lambda n=name: evaluate_mod.evaluate_model(n))

    # ── 4. Single-image prediction ──
    trained = [m for m in models_to_test
               if (config.MODELS_DIR / f"{m}_best.pth").exists()]

    def _predict():
        import predict as predict_mod
        predict_mod.get_model = _get_model_no_download
        res = predict_mod.predict(sample_holder["path"], trained[0], topk=3)

        assert len(res["topk"]) == 3
        total = sum(r["confidence"] for r in res["topk"])
        assert 0 < total <= 1.001, f"bad probabilities: {total}"

        # The abstention layer must always render a verdict...
        verdict = res["verdict"]
        assert verdict is not None, "no OOD verdict returned"
        assert verdict["decision"] in {"accept", "review", "reject"}, verdict["decision"]

        # ...and advice must be present exactly when the verdict allows an answer.
        if verdict["answerable"]:
            assert res["treatment"] is not None, "answerable but no treatment plan"
            assert res["treatment"]["urgency"] in range(6)
            assert res["severity"] is not None
        else:
            assert res["treatment"] is None, "refused the image but still advised a spray"

    if trained:
        step("Single-image prediction", _predict)

    # ── 5. Grad-CAM ──
    def _gradcam():
        name  = trained[0]
        ckpt  = torch.load(config.MODELS_DIR / f"{name}_best.pth",
                           map_location=config.DEVICE, weights_only=False)
        model = _get_model_no_download(name, config.NUM_CLASSES)
        model.load_state_dict(ckpt["model_state"])
        model.to(config.DEVICE)

        gcam   = GradCAM(model, get_target_layer(model, name))
        img    = Image.open(sample_holder["path"]).convert("RGB")
        tensor = inference_transform(img).unsqueeze(0)
        heatmap, overlay, idx, conf = gcam(tensor)
        assert heatmap.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE), heatmap.shape
        assert 0 <= idx < config.NUM_CLASSES
        print(f"  Grad-CAM → {config.DISPLAY_NAMES[idx]} ({conf*100:.1f}%), "
              f"heatmap {heatmap.shape}")

    if trained:
        step("Grad-CAM heatmap", _gradcam)

    # ── 6. Flask app imports and registers routes ──
    def _flask():
        import app as flask_app
        routes = {r.rule for r in flask_app.app.url_map.iter_rules()}
        # /stream and /camera/* were removed: a server-opened webcam is the
        # HOST's camera, which is meaningless once the app is deployed. The
        # browser captures frames and POSTs them to /predict instead.
        for required in ("/", "/predict", "/models", "/gradcam",
                         "/treatment/<path:class_name>", "/severity", "/ood/status"):
            assert required in routes, f"missing route {required}"
        print(f"  {len(routes)} routes registered")

    step("Flask app imports + routes", _flask)

    _finish(args)


def _finish(args):
    print("\n" + "═" * 62)
    print("  SMOKE TEST SUMMARY")
    print("═" * 62)
    for t in PASSED:
        print(f"  ✅  {t}")
    for t in FAILED:
        print(f"  ❌  {t}")
    print("═" * 62)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")

    if not args.keep and SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR, ignore_errors=True)
        print(f"  Cleaned up {SMOKE_DIR.name}/  (use --keep to inspect outputs)")

    if FAILED:
        print("\n  ❌  Pipeline is NOT ready.\n")
        sys.exit(1)
    print("\n  ✅  Pipeline verified end to end. Run:  python train.py\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
