"""Reasoning-band faithfulness check via intervention (#98/#103).

Recent VLA benchmarks show that a model's stated reasoning can be *decorative*
rather than causal — high observational alignment while interventions on the
reasoning leave the trajectory unchanged (VLADriveBench, arXiv:2606.12706).
This module measures the opposite, causal notion directly in our stack: run
the same batch **with and without the reasoning band's planner coupling** and
report how much the trajectory actually moves.

Because the §8 reasoning residual is zero-initialised (alpha=0, no-op), the
delta is exactly 0.0 at initialisation and only becomes positive once training
pushes the coupling away from zero — so this doubles as a regression check that
enabling the band does not perturb the reactive baseline before training.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def reasoning_intervention_delta(
    model: torch.nn.Module,
    camera_tiles: torch.Tensor,
    map_input: torch.Tensor,
    visual_history: torch.Tensor,
    egomotion_history: torch.Tensor,
    projection: Optional[Any] = None,
    geometry_type: Optional[str] = None,
    image_transform: Optional[Any] = None,
) -> dict[str, float]:
    """Measure how much the reasoning band's coupling moves the trajectory.

    Runs ``model`` twice in ``mode="infer"`` on the same inputs: once as-is
    (reasoning band active) and once with the band bypassed (intervention),
    then compares the predicted trajectories.

    Args:
        model: an ``AutoE2E`` instance with ``enable_reasoning_band=True``.
        camera_tiles / map_input / visual_history / egomotion_history: one
            evaluation batch, as in ``AutoE2E.forward``.
        projection / geometry_type / image_transform: the current geometry ABI
            forwarded to ``AutoE2E.forward`` (replaces the old ``camera_params``
            argument).

    Returns:
        dict with:
        * ``trajectory_l2``: mean L2 distance between the coupled and
          intervened trajectories (0.0 while the coupling is untrained).
        * ``coupling_shift``: mean L2 norm of the §8 reasoning residual
          ``alpha * reason_proj(reasoning_latent)`` that Reactive_E2E adds to the
          planner's visual context — how hard the reasoning is steering the
          planner (0.0 at init, since alpha starts at zero).

    Raises:
        ValueError: if the model has no reasoning band to intervene on.
    """
    band = getattr(model, "Reasoning_Band", None)
    if band is None:
        raise ValueError(
            "reasoning_intervention_delta needs a model built with "
            "enable_reasoning_band=True."
        )

    was_training = model.training
    model.eval()

    # The World Model's rolling buffer is per-sequence state that every
    # forward PUSHES to — without snapshot/restore the coupled and intervened
    # runs would see different histories (non-zero delta even with an
    # untrained gate) and the caller's rollout state would be advanced.
    buffer = getattr(model, "visual_history_buffer", None)
    saved_frames = list(buffer._buf) if buffer is not None else None

    def _restore_buffer() -> None:
        if buffer is not None and saved_frames is not None:
            buffer._buf = list(saved_frames)

    fwd_kwargs = dict(
        projection=projection,
        geometry_type=geometry_type,
        image_transform=image_transform,
        mode="infer",
    )

    # Capture the reasoning latent the band emits, so we can measure the §8
    # residual Reactive_E2E adds to the planner's visual context.
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
        captured["reasoning_latent"] = output.reasoning_latent.detach()

    handle = band.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            coupled = model(
                camera_tiles, map_input, visual_history, egomotion_history,
                **fwd_kwargs,
            )
            _restore_buffer()
    finally:
        handle.remove()

    try:
        with torch.no_grad():
            # Intervention: bypass the band entirely (planner sees the effective
            # visual history unmodulated), then restore it.  setattr keeps mypy
            # happy about temporarily nulling an nn.Module attribute.
            setattr(model, "Reasoning_Band", None)
            intervened = model(
                camera_tiles, map_input, visual_history, egomotion_history,
                **fwd_kwargs,
            )
    finally:
        model.Reasoning_Band = band
        _restore_buffer()
        if was_training:
            model.train()

    if "reasoning_latent" not in captured:
        raise RuntimeError(
            "the reasoning band did not run during the coupled forward; "
            "cannot compute coupling_shift."
        )

    coupled_traj = coupled[0] if isinstance(coupled, tuple) else coupled
    intervened_traj = intervened[0] if isinstance(intervened, tuple) else intervened

    trajectory_l2 = torch.linalg.vector_norm(
        coupled_traj - intervened_traj, dim=-1
    ).mean()

    # The §8 residual Reactive_E2E adds to the planner's visual context:
    # alpha * reason_proj(reasoning_latent).  Zero at init (alpha starts at 0).
    reactive: Any = model.Reactive_E2E
    with torch.no_grad():
        residual = reactive.reason_alpha * reactive.reason_proj(
            captured["reasoning_latent"]
        )
    coupling_shift = torch.linalg.vector_norm(residual, dim=-1).mean()

    return {
        "trajectory_l2": float(trajectory_l2),
        "coupling_shift": float(coupling_shift),
    }
