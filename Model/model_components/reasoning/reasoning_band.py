"""Reasoning Band — multi-label, multi-horizon scenario classification (issue #98).

The band consumes the 896-dim Encoded Visual History (the same vector the World
Action Model produces via :meth:`WorldActionModel.aggregate_history`) and
decodes it into per-group sigmoid classification heads for the current scenario
and — in training — four future horizons at +1 … +4 s (@1 Hz).

Its scenario prediction is supervised by the Video-Language-Model loss
(student/teacher, see ``Model/training/losses/reasoning_loss.py``).  In
addition, the band emits a pooled ``reasoning_latent`` that feeds the trajectory
planner through a **zero-init residual inside** ``Reactive_E2E`` (§8, agreed in
issues #98/#103): the residual's scale starts at zero, so the reactive baseline
is byte-identical at initialisation and the coupling only takes effect as
training moves it.  A per-horizon **confidence head** (issue #103, temporal-first)
accompanies the class logits.

Architecture (cross-attention decoder — @riita10069's #98 §2-§7 spec; migrated
from the flat trunk-MLP of the core band in #108):

    per-source context tokens (§2)          horizon queries [H, d]  (§3)
      visual_history [B,896] → visual_token         │
      ego_context   [B,256]  → ego_token            │
      (route/map optional, v2)                      │
              └──────── context_tokens [B, N, d] ───┘
                                │
                    cross-attention decoder (§4: 2 layers, 4 heads, GELU, dropout 0.1)
                                │
                        horizon_tokens [B, H, d]
                          ├── per-group heads → multi-label logits per horizon (§5)
                          ├── confidence head → [B, H]                          (§5.6)
                          ├── alignment head  → student_reasoning_embedding [B,H,D_teacher]
                          │                     (§6: training-only aux; read-only at infer)
                          └── MLP(AttentionPool(·)) → reasoning_latent [B, d]    (§7)
                                        └── Reactive_E2E: visual_ctx += alpha·reason_proj(z) → planner (§8)

Each horizon gets its own learned query, so the future-horizon predictions are
horizon-specific tokens produced by the decoder — not one shared pooled vector.
Context tokens are projected per input source (§2), so visual, ego (and later
route/map) semantics stay separate rather than being fused too early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .scenario_taxonomy import ScenarioTaxonomy, DEFAULT_TAXONOMY


# ---------------------------------------------------------------------------
# Typed output containers
# ---------------------------------------------------------------------------

# ReasoningOutput[group_name][horizon_index] = raw logits [B, num_classes]
ReasoningOutput = Dict[str, List[torch.Tensor]]


@dataclass
class ReasoningPrediction:
    """Typed result of one reasoning-band forward pass (1 Hz tick).

    Fields:
        logits: dict mapping each taxonomy group to a list of per-horizon raw
            logits ``[B, num_classes]`` (apply ``torch.sigmoid`` for
            probabilities).  Train mode: ``1 + num_future_horizons`` entries;
            other modes: 1 (current scenario only).
        confidence: ``[B, num_horizons]`` raw logits for the per-horizon
            confidence (issue #103, temporal-first).  Same horizon count as
            ``logits``.  NOTE: not supervised in the core PR — a trainable
            placeholder.  Supervise it with
            :func:`~training.losses.reasoning_loss.confidence_brier_loss`
            against a target (e.g. the cross-teacher agreement fraction); the
            full supervise-and-consume-by-the-planner loop is tracked in #110.
        reasoning_latent: ``[B, hidden_dim]`` — ``MLP(AttentionPool(horizon_tokens))``
            (§7).  This is the **planner-facing interface**: ``Reactive_E2E``
            injects it into the planner's visual context through a zero-init
            residual (§8), so the reactive baseline is unchanged until training
            moves the coupling.
        horizon_tokens: ``[B, num_horizons, hidden_dim]`` — the per-horizon
            decoder tokens (§9), exposed for metrics/debug and as the surface the
            structured heads and the alignment head read from.
        student_reasoning_embedding: ``[B, num_horizons, teacher_embed_dim]`` or
            ``None`` — a teacher-aligned embedding from a **training-only**
            auxiliary head, projected **per horizon** (§6; present only when
            ``teacher_embed_dim`` is set).  Per @riita10069's #98 decision it is
            NOT fed to the planner by default; it is shaped by an alignment loss
            in training and exposed **read-only** at inference for debug / OOD
            drift checks against cached teacher-embedding prototypes (never by
            calling the teacher at runtime).
    """

    logits: ReasoningOutput
    confidence: torch.Tensor
    reasoning_latent: torch.Tensor
    horizon_tokens: torch.Tensor
    student_reasoning_embedding: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# BEV spatial context (the #121 root-cause fix)
# ---------------------------------------------------------------------------


class BevContextTokenizer(nn.Module):
    """Turn the unpooled BEV ``[B, C, H, W]`` into spatial context tokens.

    Why this exists (@riita10069's #121 root cause): the reasoning head's inputs
    (``visual_history`` + ``ego_context``) are a strict subset of the planner's,
    so by the data-processing inequality its latent cannot carry trajectory
    information the planner lacks — and the coupling correctly learns ``alpha≈0``.
    The one signal the planner *discards* is the BEV's spatial structure: the
    bezier planner reduces it with ``bev_features.mean(dim=(2, 3))``.  Feeding the
    reasoning head the **unpooled** BEV gives it *where* a hazard is, which the
    mean-pooled planner context provably cannot reconstruct.

    Attending the BEV cell-per-token is not viable (the default grid is 450x300,
    i.e. 135k cells), so the grid is adaptively pooled to a coarse ``grid``
    (default 16x16 = 256 tokens): coarse enough to be cheap at 1 Hz, but still
    *spatial* — unlike the mean pool, it preserves location.  ``grid`` is the
    natural ablation knob.

    Args:
        in_channels: BEV feature channels (``embed_dim``, default 256).
        hidden_dim: decoder model dimension the tokens are projected to.
        grid: coarse grid the BEV is pooled to, ``(h, w)`` (default 16x16).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        grid: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # Learned 2-D position, flattened in the same row-major order as the
        # tokens, so the decoder can tell cells apart.
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid[0] * grid[1], hidden_dim) * 0.02
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, grid_h * grid_w, hidden_dim]``."""
        pooled = self.pool(bev)                       # [B, C, gh, gw]
        tokens = pooled.flatten(2).transpose(1, 2)    # [B, gh*gw, C]
        return self.norm(self.proj(tokens) + self.pos_embed)


# ---------------------------------------------------------------------------
# Attention pool (horizon tokens -> single reasoning latent)
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """Pool a token set ``[B, N, d] -> [B, d]`` with a single learned query.

    Same idiom as ``ViewAttentionPool`` in the World Action Model: one learned
    query attends over the tokens, weighting them by relevance, followed by a
    LayerNorm.
    """

    def __init__(self, dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, d]
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)
        return self.norm(pooled.squeeze(1))


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ReasoningBand(nn.Module):
    """Multi-label, multi-horizon scenario classification band for AutoE2E.

    Consumes the 896-dim Encoded Visual History produced by the World Action
    Model and outputs, per taxonomy group, sigmoid classification logits for
    the current scene and (in training) four future horizons at +1 … +4 s,
    plus a per-horizon confidence (issue #103).  A pooled ``reasoning_latent``
    feeds the trajectory planner through a zero-init gate.

    Each input source is projected into its own context token (§2), and one
    learned query per horizon attends to those tokens through a cross-attention
    decoder (§4); each horizon therefore has its own token rather than sharing a
    single pooled feature.  The v1 minimum context is ``[visual_token]`` (plus
    ``ego_token`` when ``ego_context`` is supplied); route/map tokens are a v2
    extension.

    Args:
        visual_history_dim: input dimensionality (must match the Encoded Visual
            History dimension, default 896).
        hidden_dim: decoder model dimension ``d`` (default 256, per §2).
        num_future_horizons: future horizons predicted in training (default 4,
            for h=+1..+4 s at 1 Hz).
        taxonomy: scenario label registry (defaults to :data:`DEFAULT_TAXONOMY`).
        ego_context_dim: dimensionality of the optional ``ego_context`` input
            (default 256, per §1).
        num_decoder_layers: cross-attention decoder depth (default 2, per §4).
        num_heads: attention heads in the decoder / pool (default 4; ``hidden_dim``
            must be divisible by it).
        dropout: decoder dropout (default 0.1, per §4).
        teacher_embed_dim: if set, adds a training-only alignment head that
            projects each horizon token to a teacher-aligned
            ``student_reasoning_embedding`` (§6; @riita10069's #98 decision).
            ``None`` (default) omits the head.

    Example::

        band = ReasoningBand()
        pred = band(visual_history, mode="train")   # visual_history: [B, 896]
        pred.logits["cause"]             # list of 5 tensors [B, n_cause] (h=0..4)
        pred.confidence                  # [B, 5]
        pred.reasoning_latent            # [B, 256]  (planner-facing)
        pred.modulated_visual_history    # [B, 896]  (== input at init)
    """

    def __init__(
        self,
        visual_history_dim: int = 896,
        hidden_dim: int = 256,
        num_future_horizons: int = 4,
        taxonomy: Optional[ScenarioTaxonomy] = None,
        ego_context_dim: int = 256,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        teacher_embed_dim: Optional[int] = None,
        bev_channels: Optional[int] = None,
        bev_grid: tuple[int, int] = (16, 16),
        bev_detach: bool = True,
    ) -> None:
        super().__init__()
        self.visual_history_dim = visual_history_dim
        self.hidden_dim = hidden_dim
        self.num_future_horizons = num_future_horizons
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        total_horizons = 1 + num_future_horizons

        # Per-source context-token projections (§2): each input source is
        # projected separately so their semantics stay distinct.
        self.visual_proj = self._source_projection(visual_history_dim, hidden_dim)
        self.ego_proj = self._source_projection(ego_context_dim, hidden_dim)

        # Spatial BEV context (#121).  TWO INDEPENDENT SWITCHES, deliberately:
        #   * bev_channels != None → the head SEES the BEV (spatial input).
        #   * bev_detach          → whether the reasoning loss RESHAPES the shared
        #     BEV/backbone.  Default True (see-but-don't-reshape): this repo has
        #     twice found auxiliary gradient into the shared trunk hurts at low
        #     data (the JEPA loss floor; the non-zero-init visual_history_proj), so
        #     the gradient path is opt-in and can be ablated independently of the
        #     input.  Conflating the two would make any result unattributable.
        self.bev_detach = bev_detach
        self.bev_tokenizer: Optional[BevContextTokenizer] = (
            BevContextTokenizer(bev_channels, hidden_dim, bev_grid)
            if bev_channels is not None
            else None
        )

        # One learned query per horizon (§3): h=0 current, h=1..N future.
        self.horizon_queries = nn.Parameter(
            torch.randn(total_horizons, hidden_dim) * 0.02
        )

        # Cross-attention decoder (§4): horizon queries attend to context tokens.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # One classification head per group, applied to each horizon token (§5).
        self.heads = nn.ModuleDict(
            {group.name: nn.Linear(hidden_dim, len(group)) for group in self.taxonomy.groups}
        )

        # Per-horizon confidence (raw logit per horizon token; §5.6 / issue #103).
        self.confidence_head = nn.Linear(hidden_dim, 1)

        # Planner-facing latent (§7): MLP(AttentionPool(horizon_tokens)).
        self.pool = AttentionPool(hidden_dim, num_heads=num_heads)
        self.latent_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Training-only teacher-alignment head (§6): per-horizon projection +
        # LayerNorm.  Optional (off unless teacher_embed_dim is set).
        self.alignment_head: Optional[nn.Module] = (
            nn.Sequential(
                nn.Linear(hidden_dim, teacher_embed_dim),
                nn.LayerNorm(teacher_embed_dim),
            )
            if teacher_embed_dim is not None
            else None
        )

    @staticmethod
    def _source_projection(in_dim: int, hidden_dim: int) -> nn.Sequential:
        """Per-source context-token projection (§2): LN → Linear → GELU → Linear."""
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        visual_history: torch.Tensor,
        mode: str = "infer",
        ego_context: Optional[torch.Tensor] = None,
        bev_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ReasoningPrediction:
        """Run the reasoning band.

        Args:
            visual_history: ``[B, visual_history_dim]`` — the Encoded Visual
                History from :meth:`WorldActionModel.aggregate_history`.
            mode: ``"train"`` produces all ``1 + num_future_horizons`` horizons;
                any other value produces only the current horizon.
            ego_context: optional ``[B, ego_context_dim]`` ego-motion context
                (§1/§2).  When supplied it becomes a second context token; when
                ``None`` the decoder attends to the visual token alone.
            images: unused by this variant (kept so the frozen-VLM variant can
                share the same call signature in ``AutoE2E``).

        Returns:
            A :class:`ReasoningPrediction`.
        """
        del images  # only the frozen-VLM variant consumes raw frames
        num_horizons = 1 + self.num_future_horizons if mode == "train" else 1

        # Per-source context tokens (§2): visual always present, ego if supplied.
        tokens = [self.visual_proj(visual_history)]  # each [B, d]
        if ego_context is not None:
            tokens.append(self.ego_proj(ego_context))
        context = torch.stack(tokens, dim=1)  # [B, N, d]

        # Spatial BEV tokens (#121): the one signal the planner discards.
        if bev_features is not None and self.bev_tokenizer is not None:
            bev = bev_features.detach() if self.bev_detach else bev_features
            context = torch.cat([context, self.bev_tokenizer(bev)], dim=1)

        # One query per active horizon -> horizon-specific tokens (§3/§4).
        queries = self.horizon_queries[:num_horizons].unsqueeze(0).expand(
            visual_history.shape[0], -1, -1
        )
        horizon_tokens = self.decoder(queries, context)  # [B, H, d]

        logits: ReasoningOutput = {
            group.name: [
                self.heads[group.name](horizon_tokens[:, h, :])
                for h in range(num_horizons)
            ]
            for group in self.taxonomy.groups
        }

        confidence = self.confidence_head(horizon_tokens).squeeze(-1)  # [B, H]

        # Planner-facing latent (§7) and training-only alignment embedding (§6).
        reasoning_latent = self.latent_mlp(self.pool(horizon_tokens))  # [B, d]
        student_reasoning_embedding = (
            self.alignment_head(horizon_tokens)  # [B, H, D_teacher]
            if self.alignment_head is not None
            else None
        )

        return ReasoningPrediction(
            logits=logits,
            confidence=confidence,
            reasoning_latent=reasoning_latent,
            horizon_tokens=horizon_tokens,
            student_reasoning_embedding=student_reasoning_embedding,
        )
