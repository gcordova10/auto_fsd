"""Weather/environment targets straight from L2D episode metadata (issue #98).

L2D episodes ship environment annotations out of the box — ``Conditions``
(Snow, Clear, Rain) and ``Lighting`` (Dawn, Day, Dusk, Night) per the dataset
card (https://huggingface.co/blog/lerobot-goes-to-driving-school) — so the
``weather_env`` taxonomy axis needs **no VLM at all**: the ground truth is a
deterministic mapping from metadata strings.  This keeps the (noisier,
costlier) VLM teachers for the axes that genuinely need visual understanding
(maneuver, edge_case).

The mapping is a pure function over caller-supplied strings; reading the
metadata out of the L2D parquet/dataset is the dataloader's job.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..scenario_taxonomy import ScenarioTaxonomy, DEFAULT_TAXONOMY

# L2D "Conditions" -> weather prefix of the weather_env labels.
_CONDITIONS_TO_WEATHER = {
    "clear": "fair",
    "fair": "fair",
    "sunny": "fair",
    "rain": "rain",
    "rainy": "rain",
    "snow": "snow",
    "snowy": "snow",
    "fog": "fog",
    "foggy": "fog",
}

# L2D "Lighting" -> day/night suffix.  Dawn and dusk are low-sun daylight; we
# fold them into "day" (the taxonomy is day/night only — extending it to a
# dawn/dusk bucket is a taxonomy decision, not a mapping one).
_LIGHTING_TO_TIME = {
    "day": "day",
    "daylight": "day",
    "dawn": "day",
    "dusk": "day",
    "night": "night",
}


def weather_env_targets(
    conditions: Sequence[str],
    lighting: Sequence[str],
    taxonomy: Optional[ScenarioTaxonomy] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Map L2D metadata strings to ``weather_env`` multi-label targets.

    Args:
        conditions: per-sample L2D ``Conditions`` values (case-insensitive).
        lighting: per-sample L2D ``Lighting`` values (case-insensitive),
            index-aligned with ``conditions``.
        taxonomy: label registry (defaults to :data:`DEFAULT_TAXONOMY`).
        device: optional device for the returned tensor.

    Returns:
        ``[B, num_weather_env_classes]`` float tensor with the matching label
        set to 1.0.  Samples whose metadata doesn't map onto the taxonomy
        (unknown strings) stay all-zero — an explicit abstain, so downstream
        losses can mask them rather than learn from a guess.

    Raises:
        ValueError: if the two sequences have different lengths.
    """
    if len(conditions) != len(lighting):
        raise ValueError(
            f"conditions ({len(conditions)}) and lighting ({len(lighting)}) "
            "must be index-aligned."
        )
    tax = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
    group = tax["weather_env"]

    targets = torch.zeros(len(conditions), len(group), device=device)
    for i, (cond, light) in enumerate(zip(conditions, lighting)):
        weather = _CONDITIONS_TO_WEATHER.get(cond.strip().lower())
        time = _LIGHTING_TO_TIME.get(light.strip().lower())
        if weather is None or time is None:
            continue  # abstain: unmapped metadata stays all-zero
        label = f"{weather}_{time}"
        if label in group.labels:
            targets[i, group.index(label)] = 1.0
    return targets
