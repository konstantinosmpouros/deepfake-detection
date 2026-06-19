"""Tests for utils/models.py — builder output shapes + the single-logit contract.

Everything runs on CPU with tiny batches. The from-scratch / hand-built nets
(scratch, residual, two-stream, freqcross, srm) need no downloads and run
unmarked. The timm/peft-backed builders are marked `slow` (pretrained=False, so
random weights — still no network) and the discriminative-LR helper rides along.
"""
from __future__ import annotations

import pytest
import torch

from utils import models as M


@torch.no_grad()
def _logits(model, x):
    model.eval()
    return model(x)


# ---- from-scratch nets (no download) --------------------------------------
def test_count_params_positive():
    n = M.count_params(M.build_cnn_scratch())
    assert n > 0
    # trainable_only on a fresh model (all grads on) equals the full count.
    assert M.count_params(M.build_cnn_scratch(), trainable_only=True) == n


def test_cnn_scratch_outputs_one_logit_per_sample():
    out = _logits(M.build_cnn_scratch(), torch.randn(2, 3, 64, 64))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_cnn_residual_outputs_one_logit_per_sample():
    out = _logits(M.build_cnn_residual(), torch.randn(2, 3, 64, 64))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_mlp_head_shapes():
    head = M.build_mlp_head(in_dim=512, hidden=128, n_layers=2)
    out = _logits(head, torch.randn(4, 512))
    assert out.shape == (4,)


def test_two_stream_forward_and_components():
    model = M.build_two_stream(feat=64)
    x = torch.randn(2, 3, 64, 64)
    fused = _logits(model, x)
    assert fused.shape == (2,)
    model.eval()
    with torch.no_grad():
        all_out = model.forward_all(x)
    assert len(all_out) == 3                          # fused, spatial, frequency
    for logit in all_out:
        assert logit.shape == (2,)
    # forward() must equal the fused component of forward_all().
    assert torch.allclose(fused, all_out[0])


def test_freqcross_four_components_and_attention_weights():
    model = M.build_freqcross(size=64, feat=64, n_radial=32)
    x = torch.randn(2, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model.forward_all(x)
        attn = model.attention_weights(x)
    assert len(out) == 4                              # fused, spatial, freq, radial
    for logit in out:
        assert logit.shape == (2,)
    assert attn.shape == (2, 3)                       # softmax over the 3 branches
    assert torch.allclose(attn.sum(dim=1), torch.ones(2), atol=1e-5)


def test_srm_cnn_forward_and_residual_channels():
    model = M.build_srm_cnn(feat=64, bayar_ch=3)
    x = torch.randn(2, 3, 48, 48)
    out = _logits(model, x)
    assert out.shape == (2,)
    model.eval()
    with torch.no_grad():
        res = model.residual(x)
    assert res.shape[1] == 3 + 3                       # SRM(3) ++ Bayar(bayar_ch)


def test_srm_bayar_constraint_is_enforced():
    model = M.build_srm_cnn(feat=32, bayar_ch=3)
    with torch.no_grad():
        model.bayar(torch.randn(1, 3, 16, 16))         # triggers _constrain()
        w = model.bayar.conv.weight.data
        c = model.bayar.k // 2
        # Center taps are pinned to -1; the rest of each filter sums to +1.
        assert torch.allclose(w[:, :, c, c], torch.full_like(w[:, :, c, c], -1.0))
        off_center_sum = w.sum(dim=(2, 3)) - w[:, :, c, c]
        assert torch.allclose(off_center_sum, torch.ones_like(off_center_sum), atol=1e-4)


# ---- timm / peft backbones (random weights, no download; heavier) ---------
@pytest.mark.slow
def test_cnn_finetune_efficientnet_outputs_logit():
    model = M.build_cnn_finetune("efficientnet_b0", pretrained=False)
    out = _logits(model, torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)                         # timm head -> (B,1), squeezed in loop


@pytest.mark.slow
def test_freeze_backbone_reduces_trainable_params():
    model = M.build_cnn_finetune("efficientnet_b0", pretrained=False)
    total = M.count_params(model)
    M.freeze_backbone(model)
    trainable = M.count_params(model, trainable_only=True)
    assert 0 < trainable < total                       # only the head trains
    M.unfreeze_all(model)
    assert M.count_params(model, trainable_only=True) == total


@pytest.mark.slow
def test_discriminative_param_groups_lrs_and_weight_decay():
    model = M.build_cnn_finetune("resnet50", pretrained=False)
    head_lr, decay, wd = 1e-3, 0.3, 1e-4
    groups = M.build_discriminative_param_groups(
        model, "resnet50", head_lr=head_lr, decay=decay, weight_decay=wd)
    assert len(groups) >= 2
    lrs = {round(g["lr"], 10) for g in groups}
    allowed = {round(head_lr * decay ** g, 10) for g in (0, 1, 2)}
    assert lrs <= allowed
    assert max(g["lr"] for g in groups) == pytest.approx(head_lr)   # head = largest LR
    wds = {g["weight_decay"] for g in groups}
    assert 0.0 in wds                                  # BN/bias bucket -> no decay
    assert wd in wds                                   # weight params -> decay
