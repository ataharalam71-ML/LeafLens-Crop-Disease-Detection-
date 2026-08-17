"""
utils/gradcam.py  —  Gradient-weighted Class Activation Mapping (Grad-CAM)
Highlights which leaf regions the model focuses on when making a prediction.

Usage (standalone):
    python utils/gradcam.py --image leaf.jpg --model EfficientNetB0
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

import config
from models.model_builder import get_model
from utils.dataset import inference_transform


# ─── Hook-based Grad-CAM ──────────────────────────────────────────────────────

class GradCAM:
    """
    Works with any model that has a named convolutional layer.

    Example
    -------
    model  = build_efficientnet_b0(10)
    gcam   = GradCAM(model, target_layer=model.features[-1])
    heatmap, overlay = gcam(tensor, class_idx=None)   # None → argmax
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model.eval()
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            input_tensor : (1, 3, H, W) normalised tensor on CPU/GPU
            class_idx    : target class index; None → argmax prediction

        Returns:
            heatmap  – (H, W) float32 in [0, 1]  (spatial importance map)
            overlay  – (H, W, 3) uint8 BGR        (heatmap superimposed on img)
            class_idx – int, the class that was visualised
            confidence – float, softmax probability for that class
        """
        input_tensor = input_tensor.clone().to(config.DEVICE)
        input_tensor.requires_grad_(True)

        # ── Forward ──
        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))

        confidence = float(probs[0, class_idx])

        # ── Backward w.r.t. chosen class ──
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # ── Pool gradients → channel weights ──
        gradients  = self._gradients          # (1, C, h, w)
        activations = self._activations       # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * activations).sum(dim=1).squeeze(0)  # (h, w)
        cam     = F.relu(cam)

        # ── Normalise to [0, 1] ──
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        heatmap = cam.cpu().numpy()  # (h, w)  float32

        # ── Resize to input resolution ──
        h_in = input_tensor.shape[2]
        w_in = input_tensor.shape[3]
        heatmap_resized = cv2.resize(heatmap, (w_in, h_in))

        # ── Colourmap overlay ──
        heatmap_uint8  = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Reconstruct original image from normalised tensor
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = input_tensor.squeeze(0).detach().cpu().numpy()  # (3, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))                 # (H, W, 3)
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap_color, 0.45, 0)

        return heatmap_resized, overlay, class_idx, confidence


# ─── Auto-select a good target layer per architecture ────────────────────────

def get_target_layer(model, model_name: str):
    """Return the last conv / feature block suitable for Grad-CAM."""
    name = model_name.lower()
    if "efficientnet" in name:
        return model.features[-1]
    elif "resnet" in name:
        return model.layer4[-1]
    elif "mobilenet" in name:
        return model.features[-1]
    elif "vgg" in name:
        return model.features[-1]
    elif "alexnet" in name:
        return model.features[-1]
    elif "densenet" in name:
        # NOT features[-1] (norm5): DenseNet applies an in-place ReLU to its
        # output, which the full backward hook rejects. The last dense block
        # feeds a plain BatchNorm, so it hooks cleanly.
        return model.features.denseblock4
    elif "unet" in name:
        # Deepest spatial map in the U-Net (14x14) — the decoder output is
        # full-resolution but too shallow to give a meaningful CAM.
        return model.bottleneck
    elif "customcnn" in name:
        # Last conv block in our CustomCNN
        return model.features[-1]
    else:
        raise ValueError(f"No target-layer mapping for '{model_name}'. "
                         "Pass target_layer manually.")


# ─── Full pipeline: image file → saved heatmap PNG ───────────────────────────

def run_gradcam(image_path: str,
                model_name: str = None,
                out_path: str = None,
                class_idx: int = None) -> str:
    """
    Load model + image, compute Grad-CAM, save overlay PNG.

    Returns path to the saved overlay image.
    """
    model_name = model_name or config.INFERENCE_MODEL
    ckpt_path  = config.MODELS_DIR / f"{model_name}_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run train.py first."
        )

    # Load model
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE)

    target_layer = get_target_layer(model, model_name)
    gcam         = GradCAM(model, target_layer)

    # Preprocess image
    img    = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0)

    # Compute
    heatmap, overlay, pred_idx, confidence = gcam(tensor, class_idx)

    pred_label = config.DISPLAY_NAMES[pred_idx]
    print(f"  Grad-CAM  →  {pred_label}  ({confidence*100:.1f}%)")

    # Save
    if out_path is None:
        stem     = Path(image_path).stem
        out_path = str(config.RESULTS_DIR / f"{stem}_gradcam_{model_name}.png")

    # Add text banner to overlay
    banner_h = 36
    banner   = np.zeros((banner_h, overlay.shape[1], 3), dtype=np.uint8)
    text     = f"{pred_label}  {confidence*100:.1f}%  [{model_name}]"
    cv2.putText(banner, text, (8, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    combined = np.vstack([banner, overlay])

    cv2.imwrite(out_path, combined)
    print(f"  Saved  →  {out_path}")
    return out_path


# ─── Batch: run Grad-CAM on every test image and save a grid ─────────────────

def gradcam_grid(image_paths: list,
                 model_name: str = None,
                 grid_cols: int = 4,
                 out_path: str = None) -> str:
    """
    Generate a grid image with Grad-CAM overlays for multiple leaf images.
    """
    import math

    model_name = model_name or config.INFERENCE_MODEL
    ckpt_path  = config.MODELS_DIR / f"{model_name}_best.pth"
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE)

    target_layer = get_target_layer(model, model_name)
    gcam         = GradCAM(model, target_layer)

    cell_h, cell_w = 224, 224
    overlays = []

    for img_path in image_paths:
        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = inference_transform(img).unsqueeze(0)
            _, overlay, pred_idx, conf = gcam(tensor)
            overlay_r = cv2.resize(overlay, (cell_w, cell_h))

            # Label strip
            label = config.DISPLAY_NAMES[pred_idx]
            strip = np.zeros((28, cell_w, 3), dtype=np.uint8)
            cv2.putText(strip, f"{label} {conf*100:.0f}%",
                        (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cell = np.vstack([overlay_r, strip])
            overlays.append(cell)
        except Exception as e:
            print(f"  ⚠️  Skipped {img_path}: {e}")

    if not overlays:
        return ""

    n_cols  = min(grid_cols, len(overlays))
    n_rows  = math.ceil(len(overlays) / n_cols)
    cell_total_h = cell_h + 28

    # Pad to fill grid
    while len(overlays) < n_rows * n_cols:
        overlays.append(np.zeros((cell_total_h, cell_w, 3), dtype=np.uint8))

    rows = []
    for r in range(n_rows):
        row_cells = overlays[r * n_cols : (r + 1) * n_cols]
        rows.append(np.hstack(row_cells))
    grid = np.vstack(rows)

    if out_path is None:
        out_path = str(config.RESULTS_DIR / f"gradcam_grid_{model_name}.png")

    cv2.imwrite(out_path, grid)
    print(f"  Grid saved  →  {out_path}  ({len(image_paths)} images)")
    return out_path


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualisation")
    parser.add_argument("--image",  required=True, help="Path to leaf image")
    parser.add_argument("--model",  default=None,  help="Model name")
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Class index to visualise (default: top prediction)")
    parser.add_argument("--out",    default=None,  help="Output PNG path")
    args = parser.parse_args()

    run_gradcam(args.image, args.model, args.out, args.class_idx)
