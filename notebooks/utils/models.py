"""Model builders for the first three pipelines + fine-tuning helpers.

All models output a SINGLE logit (no sigmoid in the model — applied at eval).
Design choice for the from-scratch nets: a stride-1 stem with NO early max-pool,
so the fine high-frequency texture that separates real from generated images is
preserved into the first stages (an aggressive ResNet stem would discard it).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    ps = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in ps)


# ---- cnn-scratch -----------------------------------------------------------

class _ConvBNAct(nn.Sequential):
    def __init__(self, c_in, c_out, pool=False):
        layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                  nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class CNNScratch(nn.Module):
    """Small baseline: stride-1 stem (no pool) → 4 conv blocks w/ pool → GAP → FC."""

    def __init__(self, in_ch=3, widths=(32, 64, 128, 256, 256), p_drop=0.3):
        super().__init__()
        self.stem = _ConvBNAct(in_ch, widths[0], pool=False)          # keep full res
        blocks = [_ConvBNAct(widths[i], widths[i + 1], pool=True) for i in range(len(widths) - 1)]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(widths[-1], 1)

    def forward(self, x):
        x = self.features(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x)).squeeze(1)                        # (B,)


def build_cnn_scratch(in_ch=3, widths=(32, 64, 128, 256, 256), p_drop=0.3) -> nn.Module:
    return CNNScratch(in_ch, widths, p_drop)


# ---- cnn-residual (custom, pre-act resblocks + SE attention) ---------------

class _SE(nn.Module):
    """Squeeze-and-Excitation channel attention (cheaper than CBAM, preserves spatial high-freq)."""

    def __init__(self, c, reduction=16):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.fc = nn.Sequential(nn.Linear(c, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, c), nn.Sigmoid())

    def forward(self, x):
        s = x.mean((2, 3))                  # GAP -> (B,C)
        s = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * s


class _PreActBlock(nn.Module):
    """Pre-activation BasicBlock (BN-ReLU-Conv x2) + SE, with projection shortcut on change."""

    def __init__(self, c_in, c_out, stride=1, attention="se", se_reduction=16):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.att = _SE(c_out, se_reduction) if attention == "se" else nn.Identity()
        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.proj(out) if self.proj is not None else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out = self.att(out)
        return out + shortcut


class CNNResidual(nn.Module):
    """Stride-1 stem (no early pool) → stage1 @ full res → stages 2/3 stride-2 → SE → GAP → FC."""

    def __init__(self, in_ch=3, stem_width=64, stage_widths=(64, 128, 256),
                 blocks_per_stage=(2, 2, 2), attention="se", se_reduction=16, p_drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, stem_width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(stem_width), nn.ReLU(inplace=True))
        stages, c_in = [], stem_width
        for si, (c_out, n_blocks) in enumerate(zip(stage_widths, blocks_per_stage)):
            for bi in range(n_blocks):
                stride = 2 if (bi == 0 and si > 0) else 1      # stage1 stays at full res
                stages.append(_PreActBlock(c_in, c_out, stride, attention, se_reduction))
                c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.post = nn.Sequential(nn.BatchNorm2d(c_in), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(c_in, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # zero-init the last BN (bn2) in each residual block -> identity at start (stabilizes)
        for m in self.modules():
            if isinstance(m, _PreActBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stages(self.stem(x))
        x = self.post(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x)).squeeze(1)                        # (B,)


def build_cnn_residual(in_ch=3, stem_width=64, stage_widths=(64, 128, 256),
                       blocks_per_stage=(2, 2, 2), attention="se", se_reduction=16,
                       p_drop=0.3) -> nn.Module:
    return CNNResidual(in_ch, stem_width, stage_widths, blocks_per_stage,
                       attention, se_reduction, p_drop)


# ---- cnn-finetune (timm backbone + 1-logit head) ---------------------------

def build_cnn_finetune(backbone: str = "resnet50", pretrained: bool = True, p_drop: float = 0.3):
    """timm backbone with num_classes=1 (GAP -> dropout -> Linear head). forward -> (B,1)."""
    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=1, drop_rate=p_drop)


def imagenet_cfg(model) -> dict:
    """Resolve a timm model's expected input size + normalization."""
    import timm
    return timm.data.resolve_model_data_config(model)


def freeze_backbone(model) -> None:
    """Freeze everything except the classifier head (timm `get_classifier`)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = True


# Name prefixes mapping backbone modules to LR groups (group 0 = head, larger LR).
_GROUP_PREFIXES = {
    "resnet": {1: ("layer3", "layer4"), 2: ("conv1", "bn1", "layer1", "layer2")},
    "efficientnet": {1: ("blocks.4", "blocks.5", "blocks.6", "conv_head", "bn2"),
                     2: ("conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3")},
}


def _family(backbone: str) -> str:
    return "efficientnet" if "efficientnet" in backbone else "resnet"


# ---- vit-lora (ViT-Base + LoRA via peft) -----------------------------------

def build_vit_lora(model_name: str = "vit_base_patch16_224.augreg_in21k",
                   pretrained: bool = True, r: int = 16, lora_alpha: int = 16,
                   lora_dropout: float = 0.05, p_drop: float = 0.1):
    """Frozen ViT-Base with LoRA on the attention `qkv` projections + a trainable head.

    Returns a peft model whose `forward(x) -> (B,1)` logits. Only LoRA + head train.
    `merge_and_unload()` later gives a plain timm ViT (for eval/attention-rollout).
    """
    import timm
    from peft import LoraConfig, get_peft_model

    base = timm.create_model(model_name, pretrained=pretrained, num_classes=1, drop_rate=p_drop)
    cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                     target_modules=["qkv"], modules_to_save=["head"])
    return get_peft_model(base, cfg)


# ---- clip-probe (trained neural head on frozen embeddings) -----------------

class _MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=256, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(p_drop), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_mlp_head(in_dim: int, hidden: int = 256, p_drop: float = 0.3) -> nn.Module:
    """Small MLP classifier head for frozen CLIP/DINOv2 embeddings. forward -> (B,)."""
    return _MLPHead(in_dim, hidden, p_drop)


# ---- two-stream (RGB + frequency fusion) -----------------------------------

class _FeatCNN(nn.Module):
    """Compact conv feature extractor -> (B, feat). Stride-1 stem, then pooled blocks."""

    def __init__(self, in_ch, widths=(32, 64, 128, 256), feat=256):
        super().__init__()
        layers = [_ConvBNAct(in_ch, widths[0], pool=False)]
        for i in range(len(widths) - 1):
            layers.append(_ConvBNAct(widths[i], widths[i + 1], pool=True))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class TwoStream(nn.Module):
    """Spatial RGB CNN + frequency (FFT log-magnitude) CNN -> fusion. Multi-component.

    `forward(x)` returns the fused logit (B,); `forward_all(x)` returns
    (fused, spatial, frequency) logits so the app can report per-component p_fake.
    The frequency branch input (luminance log-magnitude FFT) is computed on the fly.
    """

    LUMA = (0.299, 0.587, 0.114)

    def __init__(self, feat=256, p_drop=0.3):
        super().__init__()
        self.spatial = _FeatCNN(3, feat=feat)
        self.freq = _FeatCNN(1, feat=feat)
        self.spatial_head = nn.Linear(self.spatial.out_dim, 1)
        self.freq_head = nn.Linear(self.freq.out_dim, 1)
        self.drop = nn.Dropout(p_drop)
        self.fusion_head = nn.Linear(self.spatial.out_dim + self.freq.out_dim, 1)

    def _freq_input(self, x):
        # Compute per-sample log-magnitude FFT of luminance, in float32 (FFT isn't autocast-safe).
        with torch.autocast(device_type=x.device.type, enabled=False):
            w = torch.tensor(self.LUMA, device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            gray = (x.float() * w).sum(1, keepdim=True)
            f = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
            mag = torch.log1p(f.abs())
            mu = mag.mean(dim=(-2, -1), keepdim=True)
            sd = mag.std(dim=(-2, -1), keepdim=True) + 1e-5
            return ((mag - mu) / sd)

    def forward_all(self, x):
        s = self.spatial(x)
        fr = self.freq(self._freq_input(x))
        s_logit = self.spatial_head(s).squeeze(1)
        f_logit = self.freq_head(fr).squeeze(1)
        fused = self.fusion_head(self.drop(torch.cat([s, fr], dim=1))).squeeze(1)
        return fused, s_logit, f_logit

    def forward(self, x):
        return self.forward_all(x)[0]


def build_two_stream(feat: int = 256, p_drop: float = 0.3) -> nn.Module:
    return TwoStream(feat=feat, p_drop=p_drop)


def build_discriminative_param_groups(model, backbone: str, head_lr: float = 1e-3,
                                      decay: float = 0.3, weight_decay: float = 1e-4) -> list[dict]:
    """3 LR groups (head / late / early), each split into decay & no-decay (BN+bias) params.

    head gets head_lr; group g gets head_lr * decay**g. No weight decay on BN/bias.
    """
    prefixes = _GROUP_PREFIXES[_family(backbone)]

    def group_of(name: str) -> int:
        for g, prefs in prefixes.items():
            if any(name.startswith(p) for p in prefs):
                return g
        return 2  # default: deepest/earliest backbone layers

    buckets: dict[tuple[int, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith(("fc", "classifier"))   # timm head: resnet=fc, effnet=classifier
        g = 0 if is_head else group_of(name)
        no_decay = p.ndim <= 1 or name.endswith(".bias")   # BN/bias -> no weight decay
        buckets.setdefault((g, no_decay), []).append(p)

    groups = []
    for (g, no_decay), params in sorted(buckets.items()):
        groups.append({"params": params, "lr": head_lr * (decay ** g),
                       "weight_decay": 0.0 if no_decay else weight_decay})
    return groups
