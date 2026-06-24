"""Tests for the CoC causal pseudo-label pipeline (T4 — #17/#3).

Verifies the text→class mapping, the teacher interface, the autolabeler YAML
parser, the [B] label builder, and — crucially — that the labels drive the
already-merged ``causal_consistency_loss`` end-to-end (the head's missing
supervision side).
"""

import pytest
import torch

from data_parsing.coc_labeling import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    KeywordVLMTeacher,
    VLMTeacher,
    build_causal_labels,
    parse_coc_yaml,
    text_to_class_index,
)
from model_components.causal_reasoning import (
    CAUSAL_CLASSES,
    NUM_CAUSAL_CLASSES,
    CausalReasoningModule,
    causal_consistency_loss,
)


def test_taxonomy_matches_head_contract():
    # Single source of truth: indices follow CAUSAL_CLASSES order exactly.
    assert list(CLASS_TO_INDEX) == list(CAUSAL_CLASSES)
    assert all(INDEX_TO_CLASS[i] == c for i, c in enumerate(CAUSAL_CLASSES))
    assert "clear" in CLASS_TO_INDEX


@pytest.mark.parametrize("text,expected", [
    ("A pedestrian is crossing the road", "pedestrian"),
    ("Stopped at a red light", "traffic_light"),
    ("Approaching a roundabout", "intersection"),
    ("A construction cone blocks the lane", "obstacle"),
    ("Clear road ahead, cruising", "clear"),
    ("", "clear"),
])
def test_text_to_class_index(text, expected):
    assert text_to_class_index(text) == CLASS_TO_INDEX[expected]


def test_priority_most_safety_critical_wins():
    # pedestrian outranks traffic_light when both are mentioned.
    idx = text_to_class_index("pedestrian crossing at the traffic light")
    assert idx == CLASS_TO_INDEX["pedestrian"]


def test_keyword_teacher_str_dict_and_abstain():
    teacher = KeywordVLMTeacher()
    idx, conf = teacher.label("a pedestrian ahead")
    assert idx == CLASS_TO_INDEX["pedestrian"] and conf == 1.0
    idx, conf = teacher.label({"caption": "red light"})
    assert idx == CLASS_TO_INDEX["traffic_light"] and conf == 1.0
    idx, conf = teacher.label("   ")  # empty → abstain
    assert conf == 0.0
    assert isinstance(teacher, VLMTeacher)


def test_parse_coc_yaml():
    yaml = pytest.importorskip("yaml")  # noqa: F841
    doc = (
        "ego_behavior_schema:\n"
        "  effect_on_ego_behavior: 'Yield to the pedestrian stepping off the curb.'\n"
    )
    assert parse_coc_yaml(doc) == CLASS_TO_INDEX["pedestrian"]


def test_build_causal_labels_shape_dtype_values(device):
    samples = ["pedestrian ahead", "red light", "open road"]
    labels = build_causal_labels(samples, device=device)
    assert labels.shape == (3,)
    assert labels.dtype == torch.long
    assert labels.tolist() == [
        CLASS_TO_INDEX["pedestrian"],
        CLASS_TO_INDEX["traffic_light"],
        CLASS_TO_INDEX["clear"],
    ]
    assert labels.device.type == device.type


def test_build_causal_labels_abstain_index():
    labels = build_causal_labels(["", "pedestrian"], abstain_index=-100)
    assert labels[0].item() == -100  # empty caption abstains
    assert labels[1].item() == CLASS_TO_INDEX["pedestrian"]


def test_end_to_end_activates_causal_consistency_loss(device):
    """Labels from the pipeline drive the merged causal head's loss."""
    torch.manual_seed(0)
    samples = ["pedestrian crossing", "red light", "roundabout ahead",
               "cone blocking lane", "clear highway"]
    labels = build_causal_labels(samples, device=device)
    assert labels.min() >= 0 and labels.max() < NUM_CAUSAL_CLASSES

    head = CausalReasoningModule(embed_dim=256).to(device)
    context = torch.randn(len(samples), 256, device=device)
    _reasoning_latent, decision_logits = head(context)
    assert decision_logits.shape == (len(samples), NUM_CAUSAL_CLASSES)

    loss = causal_consistency_loss(decision_logits, labels, label_smoothing=0.1)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())


def test_class_weights_upweight_rare_longtail(device):
    """class_weights path (upweighting rare long-tail classes) stays finite."""
    labels = build_causal_labels(["pedestrian", "clear"], device=device)
    head = CausalReasoningModule(embed_dim=256).to(device)
    logits = head(torch.randn(2, 256, device=device))[1]
    weights = torch.ones(NUM_CAUSAL_CLASSES, device=device)
    weights[CLASS_TO_INDEX["pedestrian"]] = 5.0
    loss = causal_consistency_loss(logits, labels, class_weights=weights)
    assert torch.isfinite(loss)
