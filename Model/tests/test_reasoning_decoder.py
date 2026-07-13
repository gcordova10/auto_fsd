"""Tests for the cross-attention decoder band (@riita10069's #98 §2-§7 spec).

The band migrated from a flat trunk-MLP (#108) to per-source context tokens (§2),
horizon queries (§3) + a cross-attention decoder (§4), per-horizon structured
heads (§5), a training-only teacher-alignment head (§6), and a pooled
``reasoning_latent = MLP(AttentionPool(horizon_tokens))`` (§7).  These tests pin
that surface; the output contract shared with #108 (per-group per-horizon logits,
confidence, the zero-init gate no-op) is exercised by ``test_reasoning_band.py``.
"""

from __future__ import annotations

import torch

from model_components.reasoning.reasoning_band import ReasoningBand, ReasoningPrediction

B = 2
VH_DIM = 896
EGO_DIM = 256
D = 64  # small model dim for CPU tests (divisible by num_heads=4)


def _band(**kw) -> ReasoningBand:
    return ReasoningBand(visual_history_dim=VH_DIM, hidden_dim=D, **kw)


class TestContextTokens:
    def test_visual_only_context_when_no_ego(self):
        # §2: v1 minimum is the visual token alone when ego_context is absent.
        band = _band()
        pred = band(torch.randn(B, VH_DIM), mode="train")
        assert pred.horizon_tokens.shape == (B, 5, D)

    def test_ego_context_is_accepted(self):
        # §2: ego_context becomes a second context token.
        band = _band(ego_context_dim=EGO_DIM)
        pred = band(torch.randn(B, VH_DIM), mode="train", ego_context=torch.randn(B, EGO_DIM))
        assert pred.horizon_tokens.shape == (B, 5, D)

    def test_horizon_tokens_are_horizon_specific(self):
        # Distinct learned queries should give distinct horizon tokens.
        band = _band().eval()
        pred = band(torch.randn(B, VH_DIM), mode="train")
        assert not torch.allclose(pred.horizon_tokens[:, 0], pred.horizon_tokens[:, 1])


class TestReasoningLatent:
    def test_latent_shape_train_and_infer(self):
        band = _band()
        vh = torch.randn(B, VH_DIM)
        assert band(vh, mode="train").reasoning_latent.shape == (B, D)
        assert band(vh, mode="infer").reasoning_latent.shape == (B, D)

    def test_latent_is_finite(self):
        band = _band()
        assert torch.isfinite(band(torch.randn(B, VH_DIM), mode="train").reasoning_latent).all()

    def test_latent_is_the_planner_interface(self):
        # §7/§8: the band's planner-facing output is reasoning_latent; the
        # zero-init coupling itself lives in Reactive_E2E (tested at the AutoE2E
        # level), so the band no longer modulates the visual history directly.
        band = _band()
        pred = band(torch.randn(B, VH_DIM), mode="infer")
        assert not hasattr(pred, "modulated_visual_history")
        assert pred.reasoning_latent.shape == (B, D)

    def test_latent_receives_gradient(self):
        band = _band()
        pred = band(torch.randn(B, VH_DIM), mode="train")
        pred.reasoning_latent.sum().backward()
        # Gradient reaches the decoder queries through the pool -> latent path.
        assert band.horizon_queries.grad is not None
        assert torch.isfinite(band.horizon_queries.grad).all()


class TestHorizonQueries:
    def test_five_learned_queries(self):
        band = _band(num_future_horizons=4)
        assert band.horizon_queries.shape == (5, D)  # 1 current + 4 future

    def test_custom_horizon_count_slices_queries(self):
        band = _band(num_future_horizons=2)
        logits = band(torch.randn(B, VH_DIM), mode="train").logits
        for group in band.taxonomy.groups:
            assert len(logits[group.name]) == 3  # 1 + 2


class TestAlignmentHeadTrainingOnly:
    def test_absent_by_default(self):
        band = _band()
        assert band.alignment_head is None
        pred = band(torch.randn(B, VH_DIM), mode="train")
        assert pred.student_reasoning_embedding is None

    def test_per_horizon_projection_when_teacher_dim_set(self):
        # §6: alignment head projects EACH horizon token -> [B, H, D_teacher].
        band = _band(teacher_embed_dim=128)
        emb_train = band(torch.randn(B, VH_DIM), mode="train").student_reasoning_embedding
        emb_infer = band(torch.randn(B, VH_DIM), mode="infer").student_reasoning_embedding
        assert emb_train is not None and emb_train.shape == (B, 5, 128)
        assert emb_infer is not None and emb_infer.shape == (B, 1, 128)

    def test_embedding_derives_from_horizon_tokens(self):
        # §6: the embedding is a pure function of the horizon tokens (no separate
        # planner control path).
        band = _band(teacher_embed_dim=16).eval()
        pred = band(torch.randn(B, VH_DIM), mode="infer")
        expected = band.alignment_head(pred.horizon_tokens)
        assert torch.allclose(pred.student_reasoning_embedding, expected, atol=1e-6)


def test_returns_typed_prediction_with_new_fields():
    band = _band(teacher_embed_dim=32)
    pred = band(torch.randn(B, VH_DIM), mode="train")
    assert isinstance(pred, ReasoningPrediction)
    assert pred.reasoning_latent.shape == (B, D)
    assert pred.horizon_tokens.shape == (B, 5, D)
    assert pred.student_reasoning_embedding.shape == (B, 5, 32)
