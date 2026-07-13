"""Cross-teacher agreement as a label-free confidence signal (#98, #110).

Today a horizon's ``confidence`` is whatever the single teacher reports about
itself.  @riita10069 argued against exactly that in #98: *"Token probability is
not enough for confidence... a high-probability text answer does not necessarily
mean the driving concept is correct"*, and proposed deriving it from **teacher
agreement**, self-consistency, schema validity and temporal consistency instead.

This module supplies the agreement half.  :class:`MultiTeacher` runs N teachers
over the same sample, fuses their labels per field, and sets each horizon's
``confidence`` to the **fraction of teachers that agreed** on what it kept — a
calibrated, label-free signal that costs no new annotation.  Where the teachers
disagree, the confidence falls; where they converge, it rises.

That is also the supervision target @ZaynabEM's #110 needs: it asks for a
confidence that is *supervised* rather than free-running, without inventing new
labels.

Fusion rules (they follow the schema's own multi/single split):
    * multi-label (``hazard_event``, ``cause``): keep a label when at least
      ``min_agreement`` of the teachers listed it.
    * single-label (``relation_to_ego``, the four response axes): plurality vote;
      abstain (``None``) when no option clears ``min_agreement``, rather than
      guessing — an abstained field is masked out of the loss, a wrong one is not.
    * a horizon's confidence is the mean agreement of the fields it kept.
    * if every teacher abstained on a sample, the fused record abstains too (R9).

Offline only, like every other teacher: it never runs inside the training loop.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import (
    NUM_HORIZONS,
    HORIZON_SECONDS,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)
from .teacher_client import TeacherClient, TeacherRequest

# Which core groups hold a list of active labels, and which hold exactly one.
_MULTI_LABEL_GROUPS: Tuple[str, ...] = ("hazard_event", "cause")
_SINGLE_LABEL_GROUPS: Tuple[str, ...] = (
    "relation_to_ego",
    "longitudinal_response",
    "lateral_response",
    "tactical_response",
    "rule_response",
)


class MultiTeacher(TeacherClient):
    """Fuse several teachers; ``confidence`` becomes the inter-teacher agreement.

    Args:
        teachers: two or more teacher clients to run over the same sample.
        min_agreement: fraction of teachers (in ``(0, 1]``) that must agree
            before a label is kept.  Default 0.5 (a simple majority).
        provider / model / prompt_version: provenance recorded on the fused
            record.  ``model`` defaults to a joined list of the members.
        strict: as in :class:`TeacherClient` — with ``strict=False`` a sample on
            which every member abstained yields an abstained record instead of
            raising.

    Raises:
        ValueError: fewer than two teachers, an out-of-range ``min_agreement``,
            or members whose taxonomies disagree (a label index must mean the
            same class in every member before their votes can be compared).
    """

    def __init__(
        self,
        teachers: Sequence[TeacherClient],
        *,
        min_agreement: float = 0.5,
        provider: str = "multi_teacher",
        model: Optional[str] = None,
        prompt_version: str = "action_relevant_reasoning_v2",
        request_mode: str = "clip_horizons",
        strict: bool = True,
    ) -> None:
        if len(teachers) < 2:
            raise ValueError(
                f"MultiTeacher needs at least two teachers to have an agreement "
                f"signal; got {len(teachers)}."
            )
        if not 0.0 < min_agreement <= 1.0:
            raise ValueError(
                f"min_agreement must be in (0, 1]; got {min_agreement}."
            )

        # A vote only means something if every member speaks the same label space.
        reference = teachers[0].taxonomy
        for other in teachers[1:]:
            if other.taxonomy is not reference and not _same_label_space(
                reference, other.taxonomy
            ):
                raise ValueError(
                    "MultiTeacher members must share the same taxonomy: their label "
                    "sets differ, so their votes are not comparable."
                )

        super().__init__(
            provider=provider,
            model=model or "+".join(t.model for t in teachers),
            prompt_version=prompt_version,
            request_mode=request_mode,
            taxonomy=reference,
            strict=strict,
        )
        self.teachers = list(teachers)
        self.min_agreement = min_agreement

    def label(self, request: TeacherRequest) -> ReasoningLabelRecord:
        """Run every member and fuse their records by agreement."""
        records = [t.label(request) for t in self.teachers]
        usable = [r for r in records if not r.abstained]

        if not usable:
            if self.strict:
                raise RuntimeError(
                    f"every teacher abstained on sample {request.sample_id!r}; "
                    "no agreement signal is available."
                )
            return self._abstain(request, "all_teachers_abstained")

        horizons = [
            self._fuse_horizon(usable, h_idx) for h_idx in range(NUM_HORIZONS)
        ]
        return ReasoningLabelRecord(
            schema_version=usable[0].schema_version,
            sample_id=request.sample_id,
            timestamp=request.timestamp,
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            teacher_provider=self.provider,
            teacher_model=self.model,
            prompt_version=self.prompt_version,
            request_mode=self.request_mode,
            horizons=horizons,
            provenance="teacher_gt",
        )

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_horizon(
        self, records: Sequence[ReasoningLabelRecord], h_idx: int
    ) -> ReasoningHorizonLabel:
        """Fuse one horizon across teachers; confidence = mean agreement kept."""
        votes = [
            r.horizons[h_idx] for r in records if h_idx < len(r.horizons)
        ]
        n = len(votes)
        fused: Dict[str, object] = {}
        agreements: List[float] = []

        for group in _MULTI_LABEL_GROUPS:
            kept: List[str] = []
            counts = Counter(
                label for v in votes for label in (getattr(v, group) or [])
            )
            for label, count in counts.items():
                share = count / n
                if share >= self.min_agreement:
                    kept.append(label)
                    agreements.append(share)
            fused[group] = sorted(kept)

        for group in _SINGLE_LABEL_GROUPS:
            counts = Counter(
                getattr(v, group) for v in votes if getattr(v, group) is not None
            )
            if counts:
                label, count = counts.most_common(1)[0]
                share = count / n
                if share >= self.min_agreement:
                    fused[group] = label
                    agreements.append(share)
                    continue
            # No option cleared the bar: abstain rather than guess. An abstained
            # field is masked out of the loss; a wrong one trains against noise.
            fused[group] = None

        # The horizon's confidence IS the agreement — not a teacher's opinion of
        # itself. Nothing kept (total disagreement) means zero confidence, which
        # zeroes this horizon's weight in the loss.
        confidence = sum(agreements) / len(agreements) if agreements else 0.0

        return ReasoningHorizonLabel(
            horizon_sec=HORIZON_SECONDS[h_idx],
            confidence=confidence,
            provenance="teacher_gt",
            **fused,  # type: ignore[arg-type]
        )


def _same_label_space(a: object, b: object) -> bool:
    """True when two taxonomies expose the same groups with the same labels."""
    try:
        groups_a = {g.name: tuple(g.labels) for g in a.groups}  # type: ignore[attr-defined]
        groups_b = {g.name: tuple(g.labels) for g in b.groups}  # type: ignore[attr-defined]
    except AttributeError:
        return False
    return groups_a == groups_b
