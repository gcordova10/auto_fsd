"""Scene-invariance gate for the BEV fusion chain (no retraining, repo-only deps).

Given a trained checkpoint and pairs of different scenes, report at three points of
the fusion chain how much the features DIFFER between the two scenes:

    pre_residual   the sampled camera signal before the learned BEV-query template
                   is added to it (input of ``view_fusion.output_proj``)
    image_bev      the camera BEV that leaves ``view_fusion`` (before map fusion)
    planner_input  the pooled features the planner consumes

A value near 0 at ``image_bev`` means the camera BEV is scene-invariant: whatever
scene information reaches the planner enters through the map fusion afterwards, not
through the cameras. After an encoder swap the reading is direct: if ``image_bev`` is
still ~0 the encoder was not the blocker and the loss is downstream, in the fusion.

Metric: ``||a - b|| / mean(||a||, ||b||)`` in float64, computed only over cells where
both scenes have camera observations (``BEVViewFusion`` zeroes unobserved cells, and
shared zeros make plain correlation read ~1.000 regardless of content; the relative
distance does not saturate).

Usage (from ``Model/``)::

    python -m evaluation.bev_scene_invariance --checkpoint ckpt.pt \\
        --shards shards_index.json [--pairs 4]

For a different fusion module, the three hook points in :class:`FusionProbe` are the
only thing to repoint.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import torch

POINTS = ("pre_residual", "image_bev", "planner_input")


def relative_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """``||a-b|| / mean(||a||, ||b||)`` over ``mask`` (float64); ``nan`` if < 2 cells."""
    x, y = a.double().reshape(-1), b.double().reshape(-1)
    m = mask.reshape(-1).bool()
    if int(m.sum()) < 2:
        return float("nan")
    x, y = x[m], y[m]
    scale = 0.5 * (x.norm() + y.norm())
    return float((x - y).norm() / scale) if scale > 1e-12 else float("nan")


class FusionProbe:
    """Forward hooks on the fusion chain of a ``Reactive_E2E`` module.

    Captures the three :data:`POINTS` of one forward pass in :attr:`data`. Call
    :meth:`remove` when done.
    """

    def __init__(self, reactive: torch.nn.Module) -> None:
        r: Any = reactive          # attribute access on nn.Module is typed Tensor | Module
        view_fusion = r.FeatureFusion.view_fusion
        self.data: dict[str, torch.Tensor] = {}

        def keep(name: str):
            def hook(_module, _inputs, output):
                t = output[0] if isinstance(output, (tuple, list)) else output
                self.data[name] = t.detach().clone()

            return hook

        def keep_input(name: str):
            def hook(_module, inputs, _output):
                self.data[name] = inputs[0].detach().clone()

            return hook

        self._handles = [
            view_fusion.output_proj.register_forward_hook(keep_input("pre_residual")),
            view_fusion.register_forward_hook(keep("image_bev")),
        ]
        # The flow-matching planner consumes the full BEV grid and has no
        # FusedFeaturePooling; planner_input is then reported as nan.
        if getattr(r, "FusedFeaturePooling", None) is not None:
            self._handles.append(
                r.FusedFeaturePooling.register_forward_hook(keep("planner_input")))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def _run_sample(model: torch.nn.Module, batch: dict[str, Any], projection: Any,
                geometry_type: Any) -> dict[str, torch.Tensor]:
    m: Any = model
    probe = FusionProbe(m.Reactive_E2E)
    try:
        with torch.no_grad():
            model(camera_tiles=batch["visual_tiles"],
                  map_context=batch["map_context"],
                  visual_history=batch["visual_history"],
                  egomotion_history=batch["egomotion_history"],
                  route_mask=batch["route_mask"],
                  map_valid=batch["map_valid"],
                  route_valid=batch["route_valid"],
                  projection=projection, geometry_type=geometry_type, mode="infer")
    finally:
        probe.remove()
    return dict(probe.data)


def scene_invariance(model: torch.nn.Module, samples: list[tuple[dict[str, Any], Any, Any]],
                     ) -> dict[str, float]:
    """Mean relative distance between consecutive sample pairs at each point.

    ``samples`` is a list of ``(batch, projection, geometry_type)`` with batch size 1;
    samples ``0-1``, ``2-3``, ... form the scene pairs.
    """
    acc: dict[str, list[float]] = {p: [] for p in POINTS}
    for i in range(0, len(samples) - 1, 2):
        caps = [_run_sample(model, *samples[j]) for j in (i, i + 1)]
        for p in POINTS:
            if p not in caps[0] or p not in caps[1]:
                acc[p].append(float("nan"))
                continue
            mask = (caps[0][p] != 0) & (caps[1][p] != 0)
            acc[p].append(relative_distance(caps[0][p], caps[1][p], mask))
    return {p: float(torch.tensor(v).nanmean()) if v else float("nan")
            for p, v in acc.items()}


def load_model(checkpoint: str) -> torch.nn.Module:
    from model_components.auto_e2e import AutoE2E

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("checkpoint_config") or ck.get("config") or {})
    state = ck.get("model_state_dict") or ck.get("state_dict") or ck
    model = AutoE2E(**{k: v for k, v in cfg.items()
                       if k in AutoE2E.__init__.__code__.co_varnames})
    missing, _ = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if "num_batches_tracked" not in k]
    if critical:
        raise SystemExit(f"checkpoint does not match this architecture: {critical[:3]}")
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", required=True, help="shards_index.json")
    ap.add_argument("--pairs", type=int, default=4)
    args = ap.parse_args()

    from data_parsing.pre_extracted import make_multi_dataset_loader

    model = load_model(args.checkpoint)
    idx = json.loads(pathlib.Path(args.shards).read_text())
    dirs = idx["packed"] if isinstance(idx, dict) and "packed" in idx else idx
    dirs = [d for d in dirs if list(pathlib.Path(d).glob("*.tar"))]
    loader = make_multi_dataset_loader(dirs[:8], batch_size=1, num_workers=0,
                                       split="all", shuffle=0, max_active_loaders=1)
    samples: list[tuple[dict[str, Any], Any, Any]] = []
    for batch, projection, geometry_type in loader:
        samples.append((batch, projection, geometry_type))
        if len(samples) >= 2 * args.pairs:
            break

    result = scene_invariance(model, samples)
    m: Any = model
    view_fusion = m.Reactive_E2E.FeatureFusion.view_fusion
    print(f"bev_queries std {float(view_fusion.bev_queries.weight.detach().std()):.4f} "
          f"(learned additive template)")
    print(f"{'stage':<16}{'rel. distance between scenes (mean over pairs)':>48}")
    for p in POINTS:
        print(f"{p:<16}{result[p]:>48.6f}")
    print("~0 at image_bev = the camera BEV is scene-invariant: any scene information "
          "reaching the planner enters through the map fusion afterwards, not through "
          "the cameras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
