"""Tests for the action-relevant ontology axes (#98 decoder migration).

The scenario taxonomy grows append-only into @riita10069's action-relevant
ontology (cause / hazard / relation / response).  The band, loss and teachers
all iterate ``taxonomy.groups``, so adding axes must not need any interface
change — this checks that end to end.
"""

from __future__ import annotations

import torch

from model_components.reasoning.reasoning_band import ReasoningBand
from model_components.reasoning.scenario_taxonomy import DEFAULT_TAXONOMY, ScenarioTaxonomy
from model_components.reasoning.teachers.deterministic import DeterministicTeacher
from training.losses.reasoning_loss import ReasoningLoss

B = 2
VH_DIM = 896


class TestOntologyAxes:
    def test_new_axes_registered_with_key_labels(self):
        t = ScenarioTaxonomy()
        assert "vru_conflict" in t["cause"].labels
        assert "red_light" in t["cause"].labels
        assert "vru_collision_risk" in t["hazard_event"].labels
        assert "crossing_path" in t["relation_to_ego"].labels
        assert "prepare_stop" in t["longitudinal_response"].labels

    def test_legacy_indices_unchanged_after_migration(self):
        # append-only: the legacy axes keep their exact indices (loss contract).
        t = ScenarioTaxonomy()
        assert t["maneuver"].index("continue_straight") == 0
        assert t["maneuver"].index("turn_right") == 6
        assert t["weather_env"].index("fog_night") == 7

    def test_cause_axis_index_is_stable(self):
        t = ScenarioTaxonomy()
        # sentinels first/last so the index contract is pinned.
        assert t["cause"].index("lead_vehicle") == 0
        assert t["cause"].index("unknown_cause") == len(t["cause"]) - 1

    def test_band_emits_heads_for_ontology_axes(self):
        band = ReasoningBand(visual_history_dim=VH_DIM, hidden_dim=32)
        pred = band(torch.zeros(B, VH_DIM), mode="train")
        for axis in ("cause", "hazard_event", "relation_to_ego", "longitudinal_response"):
            assert axis in pred.logits
            assert len(pred.logits[axis]) == 5  # current + 4 horizons
            assert pred.logits[axis][0].shape == (B, DEFAULT_TAXONOMY.num_classes(axis))

    def test_loss_covers_ontology_axes(self):
        band = ReasoningBand(visual_history_dim=VH_DIM, hidden_dim=32)
        pred = band(torch.zeros(B, VH_DIM), mode="train")
        teacher = DeterministicTeacher(
            active_labels={"cause": ["red_light"], "longitudinal_response": ["stop"]}
        )
        targets = teacher.label([torch.zeros(B, 3, 8, 8) for _ in range(5)],
                                num_future_horizons=4)
        loss = ReasoningLoss()(pred.logits, targets)
        assert torch.isfinite(loss) and loss.item() > 0.0
