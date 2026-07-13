"""Spatial BEV context for the reasoning head (@riita10069's #121 root cause).

The reasoning coupling trains to a learned no-op (`alpha≈0.004`) because the
head's inputs (visual_history + ego) are a strict SUBSET of the planner's, so by
the data-processing inequality its latent carries no trajectory information the
planner lacks.  The one signal the planner throws away is the BEV's spatial
structure (`bev_features.mean(dim=(2,3))` in the bezier planner).

These tests pin the fix and, crucially, the property that makes it worth doing:
a coarse spatial pool preserves *where* a hazard is, whereas the planner's mean
pool does not.  They also pin the two independent switches (see vs reshape).
"""

from __future__ import annotations

import torch

from model_components.reasoning.reasoning_band import BevContextTokenizer, ReasoningBand

B = 2
VH_DIM = 896
D = 64          # small model dim (divisible by num_heads=4)
BEV_C = 32      # BEV channels
BEV_H, BEV_W = 45, 30   # small stand-in for the real 450x300 grid


def _bev(hot_row: int | None = None, hot_col: int | None = None) -> torch.Tensor:
    """A BEV that is empty except for one 'hazard' blob at (hot_row, hot_col)."""
    bev = torch.zeros(B, BEV_C, BEV_H, BEV_W)
    if hot_row is not None and hot_col is not None:
        bev[:, :, hot_row, hot_col] = 5.0
    return bev


class TestBevTokenizer:
    def test_token_shape_is_the_coarse_grid(self):
        tok = BevContextTokenizer(BEV_C, D, grid=(16, 16))
        out = tok(_bev())
        assert out.shape == (B, 16 * 16, D)  # 256 tokens, not 45*30 cells

    def test_grid_is_an_ablation_knob(self):
        for g in [(4, 4), (8, 8), (16, 16)]:
            tok = BevContextTokenizer(BEV_C, D, grid=g)
            assert tok(_bev()).shape == (B, g[0] * g[1], D)

    def test_coarse_pool_preserves_WHERE_the_hazard_is(self):
        """The whole point of #121: a hazard top-left and the same hazard
        bottom-right must produce DIFFERENT context — which the planner's
        mean-pool cannot do, since it collapses both to the same vector."""
        tok = BevContextTokenizer(BEV_C, D, grid=(16, 16)).eval()
        top_left = _bev(hot_row=2, hot_col=2)
        bottom_right = _bev(hot_row=BEV_H - 3, hot_col=BEV_W - 3)

        # What the planner sees (mean over H,W): identical — location is lost.
        assert torch.allclose(
            top_left.mean(dim=(2, 3)), bottom_right.mean(dim=(2, 3)), atol=1e-6
        )
        # What the reasoning head sees: different — location survives.
        assert not torch.allclose(tok(top_left), tok(bottom_right), atol=1e-4)

    def test_output_is_finite(self):
        tok = BevContextTokenizer(BEV_C, D)
        assert torch.isfinite(tok(_bev(5, 5))).all()


class TestBandWithBevContext:
    def test_band_ignores_bev_when_disabled(self):
        band = ReasoningBand(visual_history_dim=VH_DIM, hidden_dim=D)  # bev_channels=None
        assert band.bev_tokenizer is None
        pred = band(torch.randn(B, VH_DIM), mode="train", bev_features=_bev(5, 5))
        assert pred.horizon_tokens.shape == (B, 5, D)  # runs, BEV simply unused

    def test_bev_changes_the_reasoning_latent(self):
        band = ReasoningBand(
            visual_history_dim=VH_DIM, hidden_dim=D, bev_channels=BEV_C
        ).eval()
        vh = torch.randn(B, VH_DIM)
        left = band(vh, mode="infer", bev_features=_bev(2, 2)).reasoning_latent
        right = band(vh, mode="infer", bev_features=_bev(BEV_H - 3, BEV_W - 3)).reasoning_latent
        # Same visual history, hazard in a different place -> different latent.
        # This is exactly the non-redundancy the coupling needs to have a reason
        # to open alpha.
        assert not torch.allclose(left, right, atol=1e-4)

    def test_switch_1_see_but_do_not_reshape_is_the_default(self):
        """Default bev_detach=True: the head SEES the BEV but the reasoning loss
        must NOT reshape the shared BEV/backbone."""
        band = ReasoningBand(
            visual_history_dim=VH_DIM, hidden_dim=D, bev_channels=BEV_C
        )
        assert band.bev_detach is True
        bev = _bev(5, 5).requires_grad_(True)
        band(torch.randn(B, VH_DIM), mode="train", bev_features=bev).reasoning_latent.sum().backward()
        assert bev.grad is None, "default must not backprop into the shared BEV"

    def test_switch_2_gradient_path_is_opt_in(self):
        """bev_detach=False opts into the representation-learning path: the
        reasoning loss now reaches the shared BEV/backbone."""
        band = ReasoningBand(
            visual_history_dim=VH_DIM, hidden_dim=D, bev_channels=BEV_C, bev_detach=False
        )
        bev = _bev(5, 5).requires_grad_(True)
        band(torch.randn(B, VH_DIM), mode="train", bev_features=bev).reasoning_latent.sum().backward()
        assert bev.grad is not None and torch.isfinite(bev.grad).all()

    def test_bev_tokens_extend_the_context_not_the_interface(self):
        # The decoder contract is unchanged: only the context set grows.
        band = ReasoningBand(
            visual_history_dim=VH_DIM, hidden_dim=D, bev_channels=BEV_C, bev_grid=(4, 4)
        )
        pred = band(
            torch.randn(B, VH_DIM),
            mode="train",
            ego_context=torch.randn(B, 256),
            bev_features=_bev(5, 5),
        )
        assert pred.horizon_tokens.shape == (B, 5, D)
        assert pred.reasoning_latent.shape == (B, D)
