"""Tests for the JEPA training integration (#13) in ``training/train.py``.

Verifies ``compute_step_loss``: with the World Model on it adds the JEPA
feature-reconstruction loss to the trajectory loss and surfaces the
``prediction_error`` introspection signal; with it off the behaviour is the
plain trajectory loss. Uses a mock camera backbone — no module edits, so this
stays independent of the World Action Model internals (#93).
"""

from unittest.mock import patch

import torch
import torch.nn as nn

from model_components.losses import TrajectoryImitationLoss
from training.train import compute_step_loss

T, S = 8, 2          # num_timesteps, num_signals -> trajectory dim 16
V = 2                # cameras (small for speed)


class _MockBackbone(nn.Module):
    """4-stage mock matching the real backbone (last map = 768 channels)."""

    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone_channels = 1440
        self._st = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, 3, 1, 1), nn.AdaptiveAvgPool2d(64)),
            nn.Sequential(nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32)),
            nn.Sequential(nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16)),
            nn.Sequential(nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8)),
        ])

    def forward(self, x):
        outs, h = [], x
        for s in self._st:
            h = s(h)
            outs.append(h)
        return outs


def _model(device, enable_world_model):
    from model_components.auto_e2e import AutoE2E
    kw = {}
    if enable_world_model:
        kw = {"enable_world_model": True,
              "world_model_kwargs": {"feature_channels": 768}}
    with patch("model_components.reactive_e2e.Backbone", _MockBackbone):
        return AutoE2E(num_views=V, view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                       num_timesteps=T, num_signals=S, **kw).to(device)


def _batch(device, with_world_model):
    B = 1
    batch = {
        "visual_tiles": torch.randn(B, V, 3, 256, 256, device=device),
        "map_input": torch.randn(B, 3, 256, 256, device=device),
        "visual_history": torch.randn(B, 896, device=device),
        "egomotion_history": torch.randn(B, 256, device=device),
        "trajectory_target": torch.randn(B, T * S, device=device),
    }
    if with_world_model:
        batch["history_frames"] = torch.randn(B, 4, V, 3, 256, 256, device=device)
        batch["future_frames"] = torch.randn(B, 4, V, 3, 256, 256, device=device)
    return batch


def _loss_fn(device):
    return TrajectoryImitationLoss(num_timesteps=T, num_signals=S).to(device)


def test_jepa_term_added_and_prediction_error_surfaced(device):
    model = _model(device, enable_world_model=True)
    batch = _batch(device, with_world_model=True)
    total, traj, pred_err = compute_step_loss(
        model, batch, _loss_fn(device), jepa_weight=0.5)

    assert pred_err is not None and pred_err.ndim == 0 and torch.isfinite(pred_err)
    # total = traj + 0.5 * jepa -> differs from the plain trajectory loss
    assert not torch.isclose(total.detach(), traj)
    assert total.requires_grad


def test_jepa_backward_reaches_world_model(device):
    model = _model(device, enable_world_model=True)
    batch = _batch(device, with_world_model=True)
    total, _traj, _pe = compute_step_loss(model, batch, _loss_fn(device))
    total.backward()
    wam = model.World_Action_Model_E2E
    assert any(p.grad is not None for p in wam.future_predictor.parameters()), \
        "JEPA loss must backprop into the World Model predictor"


def test_world_model_off_is_plain_trajectory_loss(device):
    model = _model(device, enable_world_model=False)
    batch = _batch(device, with_world_model=False)
    total, traj, pred_err = compute_step_loss(model, batch, _loss_fn(device))
    assert pred_err is None                      # no introspection signal
    assert torch.isclose(total.detach(), traj)   # exactly the trajectory loss
    assert model.World_Action_Model_E2E is None


def test_jepa_weight_scales_the_term(device):
    """A larger lambda yields a larger total (same inputs/seed)."""
    model = _model(device, enable_world_model=True)
    batch = _batch(device, with_world_model=True)
    fn = _loss_fn(device)
    torch.manual_seed(0)
    small, traj_s, pe_s = compute_step_loss(model, batch, fn, jepa_weight=0.0)
    large, traj_l, pe_l = compute_step_loss(model, batch, fn, jepa_weight=10.0)
    # jepa_weight=0 -> total == trajectory loss; weight=10 -> strictly larger
    assert torch.isclose(small.detach(), traj_s)
    assert large.detach() > small.detach()
