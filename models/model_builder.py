"""
models/model_builder.py
Build different CNN architectures for tomato disease detection.
Includes: CustomCNN, ResNet50, EfficientNetB0, MobileNetV3, VGG16,
          AlexNet, UNet, DenseNet121
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Custom CNN (trained from scratch — good baseline)
# ══════════════════════════════════════════════════════════════════════════════
class CustomCNN(nn.Module):
    """
    5-block CNN with BatchNorm, Dropout, and a 2-layer classifier head.
    Input : (B, 3, 224, 224)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   64),    # → 112×112
            conv_block(64,  128),   # → 56×56
            conv_block(128, 256),   # → 28×28
            conv_block(256, 512),   # → 14×14
            conv_block(512, 512),   # → 7×7
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → 512×1×1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ResNet-50  (transfer learning)
# ══════════════════════════════════════════════════════════════════════════════
def build_resnet50(num_classes: int = 10, freeze_backbone: bool = False,
                   pretrained: bool = True) -> nn.Module:
    """
    Pretrained ResNet-50 with a replaced final FC layer.
    freeze_backbone=True → train only the head (faster convergence).
    pretrained=False     → random init, no weight download (used by checks/tests).
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EfficientNet-B0  (transfer learning — recommended default)
# ══════════════════════════════════════════════════════════════════════════════
def build_efficientnet_b0(num_classes: int = 10, freeze_backbone: bool = False,
                          pretrained: bool = True) -> nn.Module:
    """
    Pretrained EfficientNet-B0.  Excellent accuracy / parameter trade-off.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MobileNet-V3 Large  (transfer learning — edge / real-time friendly)
# ══════════════════════════════════════════════════════════════════════════════
def build_mobilenetv3(num_classes: int = 10, freeze_backbone: bool = False,
                      pretrained: bool = True) -> nn.Module:
    """
    MobileNetV3-Large.  Best choice if you need fast camera inference.
    """
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[0].in_features   # 960
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VGG-16  (transfer learning — heavier, optional)
# ══════════════════════════════════════════════════════════════════════════════
def build_vgg16(num_classes: int = 10, freeze_backbone: bool = False,
                pretrained: bool = True) -> nn.Module:
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6.  AlexNet  (transfer learning — classic 2012 baseline, very fast)
# ══════════════════════════════════════════════════════════════════════════════
def build_alexnet(num_classes: int = 10, freeze_backbone: bool = False,
                  pretrained: bool = True) -> nn.Module:
    """
    AlexNet with a replaced final FC layer.
    Small and quick to train — useful as a "classic architecture" comparison
    point against the modern backbones above.
    Keeps `.features` (5 conv layers), so Grad-CAM works the same way.
    """
    weights = models.AlexNet_Weights.DEFAULT if pretrained else None
    model = models.alexnet(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[6].in_features    # 4096
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7.  U-Net  (built from scratch — encoder/decoder adapted for classification)
# ══════════════════════════════════════════════════════════════════════════════
class UNetClassifier(nn.Module):
    """
    Classic U-Net (contracting path → bottleneck → expanding path with skip
    connections), adapted from segmentation to whole-image classification.

    The head pools BOTH the bottleneck (deep, semantic) and the final decoder
    map (shallow, high-resolution) and concatenates them, so the decoder and the
    skip connections actually receive gradient instead of being dead weight.

    Input : (B, 3, 224, 224)
    Output: (B, num_classes)

    base_ch=32 (instead of the paper's 64) keeps activation memory reasonable at
    224x224 — it still trains comfortably on a laptop GPU or CPU.
    """

    def __init__(self, num_classes: int = 10, base_ch: int = 32):
        super().__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        c5 = base_ch * 16

        # ── Contracting path ──
        self.enc1 = double_conv(3,  c1)     # 224x224
        self.enc2 = double_conv(c1, c2)     # 112x112
        self.enc3 = double_conv(c2, c3)     # 56x56
        self.enc4 = double_conv(c3, c4)     # 28x28
        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ──
        self.bottleneck = double_conv(c4, c5)   # 14x14

        # ── Expanding path (transpose conv + skip concat) ──
        self.up4  = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = double_conv(c4 * 2, c4)

        self.up3  = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = double_conv(c3 * 2, c3)

        self.up2  = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = double_conv(c2 * 2, c2)

        self.up1  = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = double_conv(c1 * 2, c1)

        # ── Classification head ──
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c5 + c1, 512),        # bottleneck feats + decoder feats
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Contracting
        e1 = self.enc1(x)                   # (B, c1, 224, 224)
        e2 = self.enc2(self.pool(e1))       # (B, c2, 112, 112)
        e3 = self.enc3(self.pool(e2))       # (B, c3,  56,  56)
        e4 = self.enc4(self.pool(e3))       # (B, c4,  28,  28)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, c5,  14,  14)

        # Expanding, with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Pool both depths and classify
        feats = torch.cat([self.gap(b), self.gap(d1)], dim=1)
        return self.classifier(feats)


def build_unet(num_classes: int = 10, freeze_backbone: bool = False,
               pretrained: bool = True) -> nn.Module:
    """
    U-Net classifier.  Trained from scratch, so `freeze_backbone` / `pretrained`
    are accepted (and ignored) purely to keep the factory signature uniform.
    """
    return UNetClassifier(num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DenseNet-121  (transfer learning — strong on leaf-disease datasets)
# ══════════════════════════════════════════════════════════════════════════════
def build_densenet121(num_classes: int = 10, freeze_backbone: bool = False,
                      pretrained: bool = True) -> nn.Module:
    """
    DenseNet-121 with a replaced classifier.  Dense connectivity reuses features
    aggressively, which tends to help on fine-grained texture differences like
    early blight vs. septoria spotting.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier.in_features      # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Factory — one function to get any model by name
# ══════════════════════════════════════════════════════════════════════════════
def get_model(name: str, num_classes: int = 10, freeze_backbone: bool = False,
              pretrained: bool = True) -> nn.Module:
    """
    name options:
        "CustomCNN"       – from-scratch 5-block CNN
        "ResNet50"        – transfer learning
        "EfficientNetB0"  – transfer learning (recommended)
        "MobileNetV3"     – transfer learning (fastest inference)
        "VGG16"           – transfer learning (heavy)
        "AlexNet"         – transfer learning (classic, very fast)
        "UNet"            – from-scratch encoder/decoder with skip connections
        "DenseNet121"     – transfer learning (strong on leaf textures)

    pretrained=False skips the ImageNet weight download — use it when you only
    need the architecture (loading a checkpoint, architecture checks, tests).
    """
    name = name.strip()
    builders = {
        "CustomCNN":      lambda: CustomCNN(num_classes),
        "ResNet50":       lambda: build_resnet50(num_classes, freeze_backbone, pretrained),
        "EfficientNetB0": lambda: build_efficientnet_b0(num_classes, freeze_backbone, pretrained),
        "MobileNetV3":    lambda: build_mobilenetv3(num_classes, freeze_backbone, pretrained),
        "VGG16":          lambda: build_vgg16(num_classes, freeze_backbone, pretrained),
        "AlexNet":        lambda: build_alexnet(num_classes, freeze_backbone, pretrained),
        "UNet":           lambda: build_unet(num_classes, freeze_backbone, pretrained),
        "DenseNet121":    lambda: build_densenet121(num_classes, freeze_backbone, pretrained),
    }
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(builders)}")
    return builders[name]()


def count_parameters(model: nn.Module) -> str:
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"
