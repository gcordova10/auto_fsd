"""Frozen Moondream2 reasoning branch (issue #98, proposed by @m-zain-khawaja).

A pretrained, **frozen** tiny VLM (Moondream2, Apache-2.0) reads the front
camera directly at 1 Hz — independent from the World Model — and its
caption/pointing output is mapped onto the scenario taxonomy and fed to the
trajectory planner through the same zero-init gate as the trained band, so the
two variants are compared behind one interface.

Pipeline (per 1 Hz tick)::

    front-cam frame [B, 3, H, W]
        └── frozen Moondream2  → caption text (+ optional object points)
                └── keyword mapping onto the taxonomy → per-class scores
                        └── zero-init gate → modulated visual_history → planner

Design notes:
* **Single-image, single-horizon.** Moondream2 has no temporal input, so this
  variant predicts the *current* scenario only (one horizon) — the trained
  band keeps the current + 4 future horizons of the 1 July design.
* **No heavy deps at import time.** The captioner is injectable
  (``caption_fn``); the default lazily loads ``transformers`` and the
  Moondream checkpoint on first use.  Tests/CI inject a stub captioner.
* **Trainable parameters = the gate only** (zero-init), so enabling this
  branch leaves the reactive baseline byte-identical at initialisation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..reasoning_band import ReasoningOutput, ReasoningPrediction, ZeroInitGate
from ..scenario_taxonomy import ScenarioTaxonomy, DEFAULT_TAXONOMY

# Caption phrases per taxonomy label.  A label scores 1.0 when any of its
# phrases appears in the (lower-cased) caption; labels without an entry match
# their own name with underscores replaced by spaces.  This is deliberately a
# simple, deterministic v1 mapping — the trained band is the learned variant.
LABEL_PHRASES: Dict[str, List[str]] = {
    "continue_straight": ["continue straight", "driving straight", "straight ahead"],
    "curve_left": ["curve left", "curving left", "bend to the left"],
    "curve_right": ["curve right", "curving right", "bend to the right"],
    "change_lane_left": ["change lane left", "changing lane to the left", "merging left"],
    "change_lane_right": ["change lane right", "changing lane to the right", "merging right"],
    "turn_left": ["turn left", "turning left", "left turn"],
    "turn_right": ["turn right", "turning right", "right turn"],
    "nudge_out": ["nudge", "nudging out", "edging out"],
    "give_way": ["give way", "giving way", "yield", "yielding"],
    "stop_for_object_in_path": ["object in path", "object on the road", "obstacle", "debris"],
    "close_to_vru": ["pedestrian", "cyclist", "bicycle", "vulnerable road user", "person crossing"],
    "avoid_roadworks": ["roadwork", "road work", "construction", "traffic cone", "barrier"],
    "stop_for_emergency_vehicle": ["emergency vehicle", "ambulance", "police car", "fire truck", "siren"],
    "fair_day": ["clear day", "sunny", "fair weather"],
    "fair_night": ["clear night"],
    "rain_day": ["rain", "raining", "wet road"],
    "rain_night": ["rain at night", "rainy night"],
    "snow_day": ["snow", "snowy", "snow-covered"],
    "snow_night": ["snow at night", "snowy night"],
    "fog_day": ["fog", "foggy", "mist"],
    "fog_night": ["fog at night", "foggy night"],
}

# Point-query labels: when a ``point_fn`` is provided, a positive detection
# count for these queries raises the corresponding edge-case score to 1.0.
POINT_QUERIES: Dict[str, str] = {
    "close_to_vru": "person",
    "stop_for_object_in_path": "object on the road",
}


def _phrases_for(label: str) -> List[str]:
    return LABEL_PHRASES.get(label, [label.replace("_", " ")])


class MoondreamReasoningBranch(nn.Module):
    """Frozen tiny-VLM reasoning branch (Moondream2) behind the band interface.

    Args:
        visual_history_dim: dimensionality of the visual history the gate
            modulates (default 896).
        taxonomy: scenario label registry (defaults to
            :data:`DEFAULT_TAXONOMY`).
        caption_fn: callable mapping a ``[B, 3, H, W]`` frame batch to one
            caption string per sample.  Injectable for tests; when ``None``
            the default lazily loads the Moondream2 checkpoint on first call.
        point_fn: optional callable mapping (frames, query) to a per-sample
            detection count, used to boost the edge-case classes in
            :data:`POINT_QUERIES`.
        model_id: HuggingFace checkpoint for the default captioner.
        revision: checkpoint revision to pin.  Pinned by default because
            the loader uses ``trust_remote_code=True`` — an unpinned
            revision would execute whatever code the upstream repo's HEAD
            ships at load time (supply-chain risk).

    Example (test/CI usage, no downloads)::

        branch = MoondreamReasoningBranch(
            caption_fn=lambda imgs: ["turning left in the rain"] * imgs.shape[0]
        )
        pred = branch(visual_history, mode="infer", images=frames)
    """

    def __init__(
        self,
        visual_history_dim: int = 896,
        taxonomy: Optional[ScenarioTaxonomy] = None,
        caption_fn: Optional[Callable[[torch.Tensor], Sequence[str]]] = None,
        point_fn: Optional[Callable[[torch.Tensor, str], Sequence[int]]] = None,
        model_id: str = "vikhyatk/moondream2",
        revision: Optional[str] = "2025-06-21",
    ) -> None:
        super().__init__()
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        # Horizon contract shared with ReasoningBand: training code reads
        # this to build teacher targets with the matching horizon count
        # (a single-image backbone cannot anticipate future scenes).
        self.num_future_horizons = 0
        self._caption_fn = caption_fn
        self._point_fn = point_fn
        self._model_id = model_id
        self._revision = revision
        # Lazily-loaded default captioner (a transformers model; typed as Any
        # because transformers is imported only inside _default_captions).
        self._model: Optional[Any] = None

        # The only trainable parameters: the zero-init planner gate.
        self.gate = ZeroInitGate(
            scenario_dim=self.taxonomy.total_classes(),
            visual_history_dim=visual_history_dim,
        )

    # ------------------------------------------------------------------
    # Frozen captioner (lazy default)
    # ------------------------------------------------------------------

    def _default_captions(self, images: torch.Tensor) -> List[str]:
        """Caption a frame batch with the frozen Moondream2 checkpoint."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "The default Moondream2 captioner needs `transformers`. "
                    "Install it, or pass a custom `caption_fn=` (tests inject "
                    "a stub)."
                ) from exc
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id, revision=self._revision, trust_remote_code=True
            )
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            # Store OUTSIDE nn.Module's registry (object.__setattr__): the
            # frozen 0.5B captioner must not end up in state_dict/checkpoints
            # nor be moved by .to(device) calls on the branch.
            object.__setattr__(self, "_model", model)

        captioner = self._model
        assert captioner is not None  # loaded above; narrows Optional for mypy

        from torchvision.transforms.functional import to_pil_image

        captions: List[str] = []
        with torch.no_grad():
            for frame in images:
                result = captioner.caption(to_pil_image(frame.cpu()), length="short")
                captions.append(str(result["caption"]))
        return captions

    # ------------------------------------------------------------------
    # Caption/points -> taxonomy scores
    # ------------------------------------------------------------------

    def _scores(
        self, captions: Sequence[str], images: torch.Tensor,
        device: torch.device,
    ) -> ReasoningOutput:
        """Map captions (+ optional point detections) to per-group scores.

        Returns raw logits per group (single horizon), consistent with the
        trained band's output contract.
        """
        lowered = [c.lower() for c in captions]

        point_hits: Dict[str, Sequence[int]] = {}
        if self._point_fn is not None:
            for label, query in POINT_QUERIES.items():
                point_hits[label] = self._point_fn(images, query)

        eps = 1e-4
        out: ReasoningOutput = {}
        for group in self.taxonomy.groups:
            scores = torch.zeros(len(lowered), len(group), device=device)
            for j, label in enumerate(group.labels):
                phrases = _phrases_for(label)
                for i, caption in enumerate(lowered):
                    # Day/night precedence: generic weather words ("rain") are
                    # substrings of night captions; a caption that mentions
                    # night must not also activate the *_day variant.
                    if label.endswith("_day") and "night" in caption:
                        continue
                    if any(p in caption for p in phrases):
                        scores[i, j] = 1.0
                counts = point_hits.get(label)
                if counts is not None:
                    for i, n in enumerate(counts):
                        if n > 0:
                            scores[i, j] = 1.0
            # Raw logits, matching the trained band's contract.
            out[group.name] = [torch.logit(scores.clamp(eps, 1.0 - eps))]
        return out

    # ------------------------------------------------------------------
    # Band interface
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_history: torch.Tensor,
        mode: str = "infer",
        images: Optional[torch.Tensor] = None,
    ) -> ReasoningPrediction:
        """Run the frozen branch.

        Args:
            visual_history: ``[B, visual_history_dim]`` — modulated by the
                gate and handed to the planner (the frozen VLM itself does not
                read it, keeping the branch independent from the world model).
            mode: accepted for interface parity; this variant always produces
                a single (current) horizon — Moondream2 is single-image.
            images: ``[B, 3, H, W]`` front-camera frames.  Required.

        Returns:
            A :class:`ReasoningPrediction` with one horizon.
        """
        del mode  # single-image backbone: current horizon only
        if images is None:
            raise ValueError(
                "MoondreamReasoningBranch needs the front-camera frames "
                "(images=...); it does not consume the visual history."
            )

        captions = list(
            self._caption_fn(images)
            if self._caption_fn is not None
            else self._default_captions(images)
        )
        if len(captions) != images.shape[0]:
            raise ValueError(
                f"caption_fn returned {len(captions)} captions for a batch "
                f"of {images.shape[0]} frames."
            )

        logits = self._scores(captions, images, device=visual_history.device)

        probs_per_group = [torch.sigmoid(logits[g.name][0]) for g in self.taxonomy.groups]
        current_probs = torch.cat(probs_per_group, dim=1)

        # Confidence (single horizon): did the captioner give us any signal
        # at all for this frame?  Max class probability, as a raw logit.
        eps = 1e-4
        max_prob = current_probs.max(dim=1, keepdim=True).values
        confidence = torch.logit(max_prob.clamp(eps, 1.0 - eps))

        modulated = self.gate(visual_history, current_probs)
        return ReasoningPrediction(
            logits=logits,
            confidence=confidence,
            modulated_visual_history=modulated,
        )
