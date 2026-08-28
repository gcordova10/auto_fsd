"""Tests for the scene-invariance gate (evaluation/bev_scene_invariance.py)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from evaluation.bev_scene_invariance import (
    POINTS,
    FusionProbe,
    relative_distance,
    scene_invariance,
)


def test_relative_distance_is_zero_for_identical_and_scale_free():
    a = torch.randn(64, 8)
    mask = torch.ones(64, 8, dtype=torch.bool)
    assert relative_distance(a, a.clone(), mask) == 0.0
    d = relative_distance(a, a + 1.0, mask)
    assert d > 0.0
    assert math.isclose(relative_distance(3.0 * a, 3.0 * (a + 1.0), mask), d, rel_tol=1e-6)


def test_relative_distance_uses_only_masked_cells_and_nan_when_too_few():
    a = torch.zeros(10)
    b = torch.zeros(10)
    b[:5] = 100.0                      # differs only OUTSIDE the mask
    mask = torch.zeros(10, dtype=torch.bool)
    mask[5:] = True
    a[5:] = 1.0
    b[5:] = 1.0
    assert relative_distance(a, b, mask) == 0.0
    assert math.isnan(relative_distance(a, b, torch.zeros(10, dtype=torch.bool)))


class _ViewFusion(nn.Module):
    def __init__(self, c: int = 4):
        super().__init__()
        self.output_proj = nn.Linear(c, c)

    def forward(self, x):
        return self.output_proj(x) * 2.0


class _FakeReactive(nn.Module):
    """Minimal stand-in with the three hook points the probe expects."""

    def __init__(self, c: int = 4):
        super().__init__()
        self.FeatureFusion = nn.Module()
        self.FeatureFusion.view_fusion = _ViewFusion(c)
        self.FusedFeaturePooling = nn.Linear(c, 2)

    def forward(self, x):
        return self.FusedFeaturePooling(self.FeatureFusion.view_fusion(x))


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Reactive_E2E = _FakeReactive()

    def forward(self, camera_tiles, **_ignored):
        return self.Reactive_E2E(camera_tiles)


def _sample(x: torch.Tensor):
    batch = {"visual_tiles": x, "map_context": None, "visual_history": None,
             "egomotion_history": None, "route_mask": None, "map_valid": None,
             "route_valid": None}
    return batch, None, None


def test_probe_captures_the_three_points_and_removes_its_hooks():
    reactive = _FakeReactive()
    x = torch.randn(1, 3, 4)
    probe = FusionProbe(reactive)
    reactive(x)
    assert set(probe.data) == set(POINTS)
    assert torch.equal(probe.data["pre_residual"], x)
    assert probe.data["image_bev"].shape == (1, 3, 4)
    assert probe.data["planner_input"].shape == (1, 3, 2)
    probe.remove()
    assert not reactive.FeatureFusion.view_fusion.output_proj._forward_hooks
    assert not reactive.FeatureFusion.view_fusion._forward_hooks
    assert not reactive.FusedFeaturePooling._forward_hooks


def test_scene_invariance_reads_zero_for_identical_scenes_and_positive_otherwise():
    torch.manual_seed(0)
    model = _FakeModel().eval()
    x = torch.randn(1, 3, 4)
    same = scene_invariance(model, [_sample(x), _sample(x.clone())])
    assert all(same[p] == 0.0 for p in POINTS)
    other = scene_invariance(model, [_sample(x), _sample(torch.randn(1, 3, 4))])
    assert all(other[p] > 0.0 for p in POINTS)
