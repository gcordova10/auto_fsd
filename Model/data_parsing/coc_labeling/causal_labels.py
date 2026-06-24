"""Causal pseudo-label pipeline for the System-2 head (T4 — issues #17 / #3).

The merged causal head (#81) trains via
``causal_consistency_loss(decision_logits, labels)`` where ``labels`` are
integer class indices in ``CAUSAL_CLASSES`` — described in its docstring as
"pseudo-labels produced by a VLM prompted over the KITScenes LongTail dataset".
This module builds those labels (the missing supervision side).

Design follows NVIDIA's Chain-of-Causation autolabeler
(NVlabs/alpamayo-coc-autolabeler; Alpamayo-R1, arXiv:2511.00088): a VLM teacher
labels keyframes and emits a free-form ``effect_on_ego_behavior`` plus a
meta-action. Here we:

* map that free-form / meta-action output into the 5 ``CAUSAL_CLASSES``,
* expose a pluggable :class:`VLMTeacher` (real Qwen3-VL / GPT backends plug in
  by subclassing) plus a deterministic :class:`KeywordVLMTeacher` for offline
  use and CI (no network / no GPU),
* parse the autolabeler's ``cot_*.yaml`` schema into a class index,
* build the ``[B]`` label tensor for ``causal_consistency_loss`` (with an
  optional abstain index for low-confidence samples).

The taxonomy is imported from the head so the class list stays a single source
of truth (its order is part of the loss contract).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch

from model_components.causal_reasoning import CAUSAL_CLASSES

CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CAUSAL_CLASSES)}
INDEX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_INDEX.items()}
_CLEAR_INDEX = CLASS_TO_INDEX["clear"]

# Keyword → class, in priority order (most safety-critical first) so that a
# scene mentioning several factors is labelled by the dominant one — e.g.
# "pedestrian crossing at the traffic light" → pedestrian.
_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pedestrian", ("pedestrian", "person", "cyclist", "vru", "jaywalk",
                    "stroller", "child")),
    ("traffic_light", ("traffic light", "red light", "green light",
                       "yellow light", "stop line", "signal")),
    ("intersection", ("intersection", "junction", "roundabout", "turn",
                      "merge", "crossing")),
    ("obstacle", ("obstacle", "construction", "cone", "blocked", "debris",
                  "stopped vehicle", "lead vehicle", "parked", "barrier")),
)


def text_to_class_index(text: str) -> int:
    """Map a free-form CoC / caption string to a ``CAUSAL_CLASSES`` index.

    Returns the index of the first (highest-priority) class whose keywords
    appear in ``text``; falls back to ``"clear"`` when none match.
    """
    t = (text or "").lower()
    for cls, keywords in _KEYWORD_RULES:
        if any(k in t for k in keywords):
            return CLASS_TO_INDEX[cls]
    return _CLEAR_INDEX


class VLMTeacher(ABC):
    """Pluggable teacher that assigns a causal class (+ confidence) to a sample.

    Real backends (Qwen3-VL, GPT-5, NVIDIA-hosted) subclass this and call the
    VLM on the keyframe; here ``sample`` can be any object the backend
    understands. Returns ``(class_index, confidence)`` with confidence in
    ``[0, 1]`` so the caller can abstain on low-confidence labels.
    """

    @abstractmethod
    def label(self, sample) -> tuple[int, float]:
        ...


class KeywordVLMTeacher(VLMTeacher):
    """Deterministic teacher: classify the sample's text via keyword rules.

    Offline / CI-friendly stand-in for a real VLM, and also the natural way to
    convert a real VLM's free-form ``effect_on_ego_behavior`` into the
    taxonomy. ``sample`` is a ``str`` or a mapping with a ``text_key`` field.
    Confidence is ``0.0`` for an empty caption (abstain) and ``1.0`` otherwise.
    """

    def __init__(self, text_key: str = "caption"):
        self.text_key = text_key

    def label(self, sample) -> tuple[int, float]:
        if isinstance(sample, str):
            text = sample
        elif isinstance(sample, dict):
            text = str(sample.get(self.text_key, ""))
        else:
            text = str(sample)
        confidence = 1.0 if text.strip() else 0.0
        return text_to_class_index(text), confidence


def parse_coc_yaml(yaml_text: str) -> int:
    """Parse an ``alpamayo-coc-autolabeler`` ``cot_*.yaml`` into a class index.

    Schema (from the autolabeler):
    ``ego_behavior_schema.effect_on_ego_behavior`` — free-form CoC text.
    """
    import yaml  # lazy: keep the module importable without pyyaml

    data = yaml.safe_load(yaml_text) or {}
    behavior = data.get("ego_behavior_schema") or {}
    effect = behavior.get("effect_on_ego_behavior", "")
    return text_to_class_index(effect)


def build_causal_labels(samples: Iterable, teacher: VLMTeacher | None = None,
                        *, device=None,
                        abstain_index: int | None = None) -> torch.Tensor:
    """Build the ``[B]`` integer label tensor for ``causal_consistency_loss``.

    Args:
        samples: iterable of teacher inputs (captions / dicts / frames).
        teacher: a :class:`VLMTeacher` (default :class:`KeywordVLMTeacher`).
        device: device for the output tensor.
        abstain_index: if set, samples the teacher returns with confidence 0
            are assigned this index (e.g. an ``ignore_index`` for the loss),
            instead of the keyword fallback.

    Returns:
        ``torch.long`` tensor of shape ``[B]`` with class indices.
    """
    teacher = teacher or KeywordVLMTeacher()
    indices: list[int] = []
    for sample in samples:
        idx, confidence = teacher.label(sample)
        if confidence == 0.0 and abstain_index is not None:
            idx = abstain_index
        indices.append(idx)
    return torch.tensor(indices, dtype=torch.long, device=device)
