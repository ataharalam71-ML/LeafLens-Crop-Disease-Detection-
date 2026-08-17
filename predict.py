"""
predict.py  —  Predict disease from a single image file.

Usage:
    python predict.py --image leaf.jpg
    python predict.py --image leaf.jpg --model ResNet50
    python predict.py --image leaf.jpg --topk 3
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import config
from models.model_builder import get_model
from utils.dataset import inference_transform


def load_model(model_name: str):
    ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE).eval()
    return model


@torch.no_grad()
def predict(image_path: str, model_name: str = None, topk: int = 3):
    model_name = model_name or config.INFERENCE_MODEL
    model      = load_model(model_name)

    img    = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(config.DEVICE)

    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(topk)
    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append({
            "class":       config.CLASS_NAMES[idx],
            "display":     config.DISPLAY_NAMES[idx],
            "confidence":  round(float(prob), 4),
        })

    print(f"\n  Image  : {image_path}")
    print(f"  Model  : {model_name}")
    print(f"  {'─'*40}")
    for i, r in enumerate(results):
        bar = "█" * int(r["confidence"] * 30)
        print(f"  #{i+1}  {r['display']:<30} {r['confidence']*100:6.2f}%  {bar}")
    print()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--model", default=None,  help="Model name")
    parser.add_argument("--topk",  type=int, default=3)
    args = parser.parse_args()
    predict(args.image, args.model, args.topk)
