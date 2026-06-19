"""cnn-finetune — two-stage transfer learning (224px, ImageNet norm).

One selectable key per backbone (efficientnet_b0 / resnet50), but both write to
the same 'cnn-finetune' artifact + predictions folder. The committed file is the
slim EMA weights (full fine-tuned model), so we build with pretrained=False (no
ImageNet download needed) and load via the slim save_weights format.
"""
from __future__ import annotations

from .base import BasePipeline, M, T


class CNNFinetunePipeline(BasePipeline):
    pipeline = "cnn-finetune"
    working_size = 224
    norm = "imagenet"

    def __init__(self, device=None, backbone: str = "efficientnet_b0"):
        self.backbone = backbone
        self.key = f"cnn-finetune-{backbone}"
        self.label = f"cnn-finetune ({backbone})"
        self.weights_filename = f"best_{backbone}.pt"
        super().__init__(device)

    def build(self):
        # Full fine-tuned weights are saved, so the frozen ImageNet init is unnecessary.
        return M.build_cnn_finetune(self.backbone, pretrained=False, p_drop=0.3)

    def load_weights(self, model, ckpt_path):
        T.load_weights(ckpt_path, model, map_location=self.device)
        return model

    def target_layers(self):
        # ResNet50 -> last bottleneck; EfficientNet-B0 -> final conv head.
        if "resnet" in self.backbone:
            return [self.model.layer4[-1]]
        return [self.model.conv_head]
