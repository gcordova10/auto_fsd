"""Single source of truth for the scenario label space (issue #98).

Three independent axes, each MULTI-LABEL (several labels can be active
simultaneously across and within axes):

* **maneuver** — normal driving manoeuvres (7 classes).
* **edge_case** — edge-case / long-tail manoeuvres (6 classes).
* **weather_env** — weather × time-of-day combinations (8 classes).

The registry is extensible: call :meth:`ScenarioTaxonomy.register_group` to
add a new axis (e.g. KIT high-level labels) without changing existing group
indices or breaking the loss contract.  Index ordering within each group is
stable and part of the loss contract — do NOT reorder entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class TaxonomyGroup:
    """One axis in the scenario taxonomy.

    Args:
        name: unique identifier (e.g. ``"maneuver"``).
        labels: ordered tuple of class names.  Index order is part of the
            loss contract — append only, never insert or reorder.
    """

    name: str
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"TaxonomyGroup '{self.name}' contains duplicate labels: "
                f"{self.labels}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        """Return the stable index of *label* (raises ``KeyError`` if absent)."""
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(
                f"Label '{label}' not found in group '{self.name}'. "
                f"Known labels: {self.labels}"
            )


# ---------------------------------------------------------------------------
# Canonical label sets (do NOT reorder — index is part of the loss contract)
# ---------------------------------------------------------------------------

_MANEUVER_LABELS: tuple[str, ...] = (
    "continue_straight",
    "curve_left",
    "curve_right",
    "change_lane_left",
    "change_lane_right",
    "turn_left",
    "turn_right",
)

_EDGE_CASE_LABELS: tuple[str, ...] = (
    "nudge_out",
    "give_way",
    "stop_for_object_in_path",
    "close_to_vru",
    "avoid_roadworks",
    "stop_for_emergency_vehicle",
)

_WEATHER_ENV_LABELS: tuple[str, ...] = (
    "fair_day",
    "fair_night",
    "rain_day",
    "rain_night",
    "snow_day",
    "snow_night",
    "fog_day",
    "fog_night",
)

# ---------------------------------------------------------------------------
# Action-relevant ontology axes (@riita10069's #98 spec §5.3 / §5.4 / §11).
# These are the planner-relevant "why / relation / response" axes — the value
# the #98 thread converged on (the *cause* is what the trajectory doesn't
# already encode). Added append-only alongside the legacy axes; do NOT reorder.
# ---------------------------------------------------------------------------

_CAUSE_LABELS: tuple[str, ...] = (
    "lead_vehicle",
    "cut_in",
    "cross_traffic",
    "oncoming_vehicle",
    "pedestrian_crossing",
    "vru_conflict",
    "red_light",
    "stop_sign",
    "yield_sign",
    "human_direction",
    "route_turn",
    "route_merge",
    "lane_ending",
    "object_blocking_path",
    "occlusion",
    "poor_visibility",
    "slippery_road",
    "uncertainty_high",
    "unknown_cause",
)

_HAZARD_LABELS: tuple[str, ...] = (
    "collision_risk",
    "vru_collision_risk",
    "cut_in_risk",
    "merge_conflict",
    "red_light_violation_risk",
    "right_of_way_violation_risk",
    "offroad_risk",
    "occlusion_risk",
    "low_friction_risk",
    "unknown_hazard",
)

_RELATION_TO_EGO_LABELS: tuple[str, ...] = (
    "same_lane_ahead",
    "adjacent",
    "crossing_path",
    "about_to_cross_path",
    "merging_into_ego_path",
    "cutting_into_ego_path",
    "oncoming_conflict",
    "blocking_current_lane",
    "blocking_target_lane",
    "blocking_route",
    "intersection_conflict",
    "occluded_near_path",
    "unknown_relation",
)

_LONGITUDINAL_RESPONSE_LABELS: tuple[str, ...] = (
    "keep_speed",
    "accelerate",
    "coast",
    "slow_down",
    "prepare_stop",
    "stop",
    "stay_stopped",
    "creep",
    "yield",
    "follow_lead_vehicle",
    "increase_gap",
    "emergency_brake",
)

# response is split into longitudinal / lateral / tactical / rule (riita #98
# §5.4), because a single "response" class is too ambiguous (a "slow_down" can be
# a lead vehicle, a red light, or a route turn). Labels are riita's §12 / §5.4.
_LATERAL_RESPONSE_LABELS: tuple[str, ...] = (
    "keep_lane",
    "nudge_left",
    "nudge_right",
    "lane_change_left",
    "lane_change_right",
    "avoid_left",
    "avoid_right",
    "return_to_lane",
    "pull_over",
    "reverse",
)

_TACTICAL_RESPONSE_LABELS: tuple[str, ...] = (
    "proceed",
    "proceed_with_caution",
    "wait_for_gap",
    "wait_for_actor",
    "wait_for_signal",
    "creep_for_visibility",
    "negotiate_merge",
    "negotiate_unprotected_turn",
    "reroute_or_wait",
)

_RULE_RESPONSE_LABELS: tuple[str, ...] = (
    "wait_for_green",
    "stop_at_stop_line",
    "yield_to_vru",
    "yield_to_oncoming",
    "obey_human_direction",
    "respect_speed_limit",
)


class ScenarioTaxonomy:
    """Registry of scenario label groups.

    The three canonical groups are registered at construction time.
    Additional groups (e.g. KIT high-level labels) can be added later via
    :meth:`register_group` or :meth:`extend` without breaking any existing
    group's index contract.

    Example::

        taxonomy = ScenarioTaxonomy()
        g = taxonomy["maneuver"]
        idx = g.index("turn_left")   # → 6  (stable)

        # Extend with KIT labels (append-only within the new group):
        taxonomy.register_group("kit_context", ["intersection", "construction_zone"])

    """

    def __init__(self) -> None:
        self._groups: Dict[str, TaxonomyGroup] = {}

        # Legacy placeholder axes (kept first so their indices never move).
        # maneuver is the "action" label the #98 thread found least additive
        # (the trajectory already encodes it) and weather_env is not
        # action-relevant; both stay for back-compat and will be deprecated
        # once the ontology axes below are trained.
        self.register_group("maneuver", list(_MANEUVER_LABELS))
        self.register_group("edge_case", list(_EDGE_CASE_LABELS))
        self.register_group("weather_env", list(_WEATHER_ENV_LABELS))

        # Action-relevant ontology axes (@riita10069's #98 spec) — the
        # planner-relevant "why / relation / response" the design converged on.
        # Migrated append-only: the band, loss and teachers all iterate
        # ``groups`` so they pick these up automatically with no interface change.
        self.register_group("cause", list(_CAUSE_LABELS))
        self.register_group("hazard_event", list(_HAZARD_LABELS))
        self.register_group("relation_to_ego", list(_RELATION_TO_EGO_LABELS))
        self.register_group("lateral_response", list(_LATERAL_RESPONSE_LABELS))
        self.register_group("tactical_response", list(_TACTICAL_RESPONSE_LABELS))
        self.register_group("rule_response", list(_RULE_RESPONSE_LABELS))
        self.register_group(
            "longitudinal_response", list(_LONGITUDINAL_RESPONSE_LABELS)
        )

    # ------------------------------------------------------------------
    # Extension API
    # ------------------------------------------------------------------

    def register_group(self, name: str, labels: Sequence[str]) -> TaxonomyGroup:
        """Register a new label group.

        Args:
            name: unique group identifier.
            labels: ordered list of class names.  Index order is stable once
                registered — append-only semantics are the caller's
                responsibility for subsequent :meth:`extend` calls.

        Returns:
            The newly created :class:`TaxonomyGroup`.

        Raises:
            ValueError: if *name* is already registered.
        """
        if name in self._groups:
            raise ValueError(
                f"Group '{name}' is already registered. "
                "Use extend() to append labels to an existing group."
            )
        group = TaxonomyGroup(name=name, labels=tuple(labels))
        self._groups[name] = group
        return group

    def extend(self, name: str, new_labels: Sequence[str]) -> TaxonomyGroup:
        """Append *new_labels* to an existing group (append-only).

        This is the only sanctioned way to grow a group's label set so that
        existing indices remain stable.

        Args:
            name: group to extend.
            new_labels: labels to append (must not overlap with current set).

        Returns:
            The updated :class:`TaxonomyGroup`.
        """
        if name not in self._groups:
            raise KeyError(
                f"Group '{name}' is not registered. "
                "Call register_group() first."
            )
        existing = self._groups[name]
        overlap = set(new_labels) & set(existing.labels)
        if overlap:
            raise ValueError(
                f"Labels {sorted(overlap)} already exist in group '{name}'."
            )
        updated_labels = existing.labels + tuple(new_labels)
        self._groups[name] = TaxonomyGroup(name=name, labels=updated_labels)
        return self._groups[name]

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> TaxonomyGroup:
        try:
            return self._groups[name]
        except KeyError:
            raise KeyError(
                f"Group '{name}' not found. "
                f"Registered groups: {list(self._groups)}"
            )

    def __contains__(self, name: object) -> bool:
        return name in self._groups

    @property
    def groups(self) -> List[TaxonomyGroup]:
        """All registered groups in insertion order."""
        return list(self._groups.values())

    @property
    def group_names(self) -> List[str]:
        """Names of all registered groups in insertion order."""
        return list(self._groups.keys())

    def num_classes(self, name: str) -> int:
        """Number of classes in group *name*."""
        return len(self[name])

    def total_classes(self) -> int:
        """Total number of classes across all groups.

        Used as the conditioning dimension of the planner gate (the
        current-scenario probabilities of every group, concatenated).
        """
        return sum(len(g) for g in self._groups.values())


# Module-level default taxonomy instance — shared across the package.
DEFAULT_TAXONOMY: ScenarioTaxonomy = ScenarioTaxonomy()
