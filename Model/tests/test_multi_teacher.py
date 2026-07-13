"""Cross-teacher agreement as the confidence signal (#98, #110).

Today a horizon's `confidence` is the single teacher's opinion of itself — which
is exactly what @riita10069 argued against in #98 ("token probability is not
enough for confidence"), proposing teacher agreement instead. These tests pin
that: the fused confidence rises with agreement and falls with disagreement, and
a field nobody agrees on abstains rather than being guessed.
"""

from __future__ import annotations

import pytest

from data_processing.reasoning_label_generation.multi_teacher import MultiTeacher
from data_processing.reasoning_label_generation.schema import (
    NUM_HORIZONS,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)
from data_processing.reasoning_label_generation.teacher_client import (
    TeacherClient,
    TeacherRequest,
)


class FakeTeacher(TeacherClient):
    """Returns whatever labels the test asks for, on every horizon."""

    def __init__(self, name: str, labels: dict, *, abstain: bool = False, confidence: float = 0.9):
        super().__init__(provider="fake", model=name, strict=False)
        self._labels = labels
        self._abstains = abstain
        self._confidence = confidence

    def label(self, request: TeacherRequest) -> ReasoningLabelRecord:
        if self._abstains:
            return self._abstain(request, "test")  # reuse the base-class helper
        return ReasoningLabelRecord(
            schema_version="reasoning_label_v1",
            sample_id=request.sample_id,
            timestamp=request.timestamp,
            dataset_name=request.dataset_name,
            teacher_provider=self.provider,
            teacher_model=self.model,
            prompt_version=self.prompt_version,
            request_mode=self.request_mode,
            horizons=[
                ReasoningHorizonLabel(
                    horizon_sec=float(i),
                    confidence=self._confidence,  # self-reported: must be IGNORED
                    **self._labels,
                )
                for i in range(NUM_HORIZONS)
            ],
        )



def _req() -> TeacherRequest:
    return TeacherRequest(sample_id="s0", dataset_name="test")


class TestConstruction:
    def test_needs_at_least_two_teachers(self):
        with pytest.raises(ValueError, match="at least two teachers"):
            MultiTeacher([FakeTeacher("a", {})])

    def test_rejects_bad_min_agreement(self):
        with pytest.raises(ValueError, match="min_agreement"):
            MultiTeacher([FakeTeacher("a", {}), FakeTeacher("b", {})], min_agreement=1.5)


class TestAgreementIsTheConfidence:
    def test_full_agreement_gives_confidence_one(self):
        labels = {"cause": ["red_light"], "longitudinal_response": "stop"}
        mt = MultiTeacher([FakeTeacher("a", labels), FakeTeacher("b", labels)])
        rec = mt.label(_req())
        assert rec.horizons[0].confidence == pytest.approx(1.0)
        assert rec.horizons[0].cause == ["red_light"]
        assert rec.horizons[0].longitudinal_response == "stop"

    def test_the_self_reported_confidence_is_ignored(self):
        """The whole point: confidence comes from agreement, not from what the
        teacher says about itself. Two teachers that disagree cannot yield 0.9
        just because both claimed 0.9."""
        mt = MultiTeacher([
            FakeTeacher("a", {"cause": ["red_light"]}, confidence=0.9),
            FakeTeacher("b", {"cause": ["lead_vehicle"]}, confidence=0.9),
        ])
        rec = mt.label(_req())
        assert rec.horizons[0].confidence < 0.9

    def test_disagreement_lowers_confidence(self):
        agree = MultiTeacher([
            FakeTeacher("a", {"cause": ["red_light"]}),
            FakeTeacher("b", {"cause": ["red_light"]}),
        ]).label(_req())
        disagree = MultiTeacher([
            FakeTeacher("a", {"cause": ["red_light", "cut_in"]}),
            FakeTeacher("b", {"cause": ["red_light"]}),
        ], min_agreement=0.4).label(_req())
        assert disagree.horizons[0].confidence < agree.horizons[0].confidence

    def test_total_disagreement_gives_zero_confidence(self):
        # Nothing clears the bar -> nothing kept -> confidence 0 -> the loss
        # weights this horizon to zero rather than training on a coin flip.
        mt = MultiTeacher([
            FakeTeacher("a", {"longitudinal_response": "stop"}),
            FakeTeacher("b", {"longitudinal_response": "keep_speed"}),
            FakeTeacher("c", {"longitudinal_response": "accelerate"}),
        ], min_agreement=0.5)
        rec = mt.label(_req())
        assert rec.horizons[0].confidence == 0.0
        assert rec.horizons[0].longitudinal_response is None


class TestFusionRules:
    def test_multi_label_keeps_what_clears_the_bar(self):
        mt = MultiTeacher([
            FakeTeacher("a", {"hazard_event": ["vru_collision_risk", "cut_in_risk"]}),
            FakeTeacher("b", {"hazard_event": ["vru_collision_risk"]}),
        ], min_agreement=0.6)
        h = mt.label(_req()).horizons[0]
        assert h.hazard_event == ["vru_collision_risk"]   # 2/2 kept, 1/2 dropped

    def test_single_label_takes_the_plurality(self):
        mt = MultiTeacher([
            FakeTeacher("a", {"relation_to_ego": "crossing_path"}),
            FakeTeacher("b", {"relation_to_ego": "crossing_path"}),
            FakeTeacher("c", {"relation_to_ego": "adjacent"}),
        ], min_agreement=0.5)
        h = mt.label(_req()).horizons[0]
        assert h.relation_to_ego == "crossing_path"
        assert h.confidence == pytest.approx(2 / 3)

    def test_single_label_abstains_rather_than_guessing(self):
        # A wrong single label trains the head against noise; an abstained one is
        # masked out. Abstaining is strictly safer.
        mt = MultiTeacher([
            FakeTeacher("a", {"tactical_response": "wait_for_gap"}),
            FakeTeacher("b", {"tactical_response": "proceed"}),
        ], min_agreement=0.75)
        assert mt.label(_req()).horizons[0].tactical_response is None


class TestAbstention:
    def test_abstained_members_are_dropped_not_counted(self):
        mt = MultiTeacher([
            FakeTeacher("a", {"cause": ["red_light"]}),
            FakeTeacher("b", {}, abstain=True),
        ], strict=False)
        h = mt.label(_req()).horizons[0]
        # 'b' never voted, so 'a' is 1/1 — an abstention must not be read as a
        # dissenting vote that halves the confidence.
        assert h.cause == ["red_light"]
        assert h.confidence == pytest.approx(1.0)

    def test_all_abstained_abstains(self):
        mt = MultiTeacher([
            FakeTeacher("a", {}, abstain=True),
            FakeTeacher("b", {}, abstain=True),
        ], strict=False)
        assert mt.label(_req()).abstained is True

    def test_all_abstained_raises_when_strict(self):
        mt = MultiTeacher([
            FakeTeacher("a", {}, abstain=True),
            FakeTeacher("b", {}, abstain=True),
        ], strict=True)
        with pytest.raises(RuntimeError, match="every teacher abstained"):
            mt.label(_req())


def test_all_five_horizons_are_produced():
    labels = {"cause": ["red_light"]}
    mt = MultiTeacher([FakeTeacher("a", labels), FakeTeacher("b", labels)])
    assert len(mt.label(_req()).horizons) == NUM_HORIZONS
