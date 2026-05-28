"""Configuration helpers for autoencoder training.

This module keeps config parsing and derived runtime switches out of the CLI
entrypoint. Defaults and key names intentionally mirror the training script.
"""

from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class DebugOptions:
    """Runtime debug switches, with defaults matching the configured training path."""

    norm_kind: str
    color_act: str
    fixed_chunk: bool
    no_augment: bool
    voxel_dedup: str
    grad_clip: float
    log_grad_norm_every: int


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def debug_options(cfg: dict) -> DebugOptions:
    """Read debug options without changing their configured defaults."""

    debug_cfg = cfg.get("debug", {}) or {}
    data_debug_cfg = cfg.get("data", {}) or {}
    return DebugOptions(
        norm_kind=str(debug_cfg.get("norm", "bn")).lower(),
        color_act=str(debug_cfg.get("color_activation", "clamp")).lower(),
        fixed_chunk=bool(
            data_debug_cfg.get("fixed_chunk", debug_cfg.get("fixed_chunk", False))
        ),
        no_augment=bool(debug_cfg.get("no_augment", False)),
        voxel_dedup=str(debug_cfg.get("voxel_dedup", "random")).lower(),
        grad_clip=float(debug_cfg.get("grad_clip", 1.0)),
        log_grad_norm_every=int(debug_cfg.get("log_grad_norm_every", 0)),
    )


def log_debug_options(debug: DebugOptions) -> None:
    if any([
        debug.norm_kind != "bn",
        debug.color_act != "clamp",
        debug.fixed_chunk,
        debug.no_augment,
        debug.voxel_dedup != "random",
        debug.grad_clip != 1.0,
        debug.log_grad_norm_every > 0,
    ]):
        print(
            "[debug] norm=" + debug.norm_kind
            + f" color_activation={debug.color_act}"
            + f" fixed_chunk={debug.fixed_chunk} no_augment={debug.no_augment}"
            + f" voxel_dedup={debug.voxel_dedup} grad_clip={debug.grad_clip}"
            + f" log_grad_norm_every={debug.log_grad_norm_every}"
        )


def canonical_camera_score_key(score_key: str) -> str:
    key = str(score_key).lower()
    aliases = {
        "frustum_image_coverage": "frustum_image_coverage",
        "chunk_coverage": "frustum_image_coverage",
        "image_coverage": "frustum_image_coverage",
    }
    if key not in aliases:
        raise ValueError(
            "camera_sampling.score_key must be 'frustum_image_coverage' "
            "('chunk_coverage' is accepted as a legacy alias); "
            f"got {score_key!r}."
        )
    return aliases[key]


def camera_sampling_config(cfg: dict) -> tuple:
    camera_cfg = cfg.get("camera_sampling", {}) or {}
    mode = str(camera_cfg.get("mode", "orbit")).lower()
    if mode in {"default", "orbit", "around_bbox", "bbox"}:
        mode = "orbit"
    elif mode in {"score", "scored", "scoring", "gaussiangpt"}:
        mode = "scoring"
    else:
        raise ValueError(
            "camera_sampling.mode must be 'orbit' or 'scoring', "
            f"got {mode!r}"
        )
    return camera_cfg, mode


def camera_dataset_type(cfg: dict) -> str:
    camera_cfg, _mode = camera_sampling_config(cfg)
    dataset_type = str(camera_cfg.get("dataset_type", "ase")).lower()
    aliases = {
        "ase": "ase",
        "3dfront": "3dfront",
        "3d-front": "3dfront",
        "3d_front": "3dfront",
        "front": "3dfront",
    }
    if dataset_type not in aliases:
        raise ValueError(
            "camera_sampling.dataset_type must be 'ase' or '3dfront', "
            f"got {dataset_type!r}"
        )
    return aliases[dataset_type]


def validate_camera_sampling_config(cfg: dict) -> None:
    camera_cfg, mode = camera_sampling_config(cfg)
    if mode != "scoring":
        return

    dataset_type = camera_dataset_type(cfg)
    if dataset_type == "ase":
        canonical_camera_score_key(
            camera_cfg.get("score_key", "frustum_image_coverage")
        )
        temperature = float(camera_cfg.get("temperature", 1.0))
        if temperature <= 0.0:
            raise ValueError(
                "camera_sampling.temperature must be > 0 for scoring; "
                f"got {temperature}."
            )
        return

    raise NotImplementedError(
        "camera_sampling.dataset_type='3dfront' requires the GaussianGPT "
        "3D-FRONT path with pre-rendered images and depth maps to compute "
        "visible chunk area. This repository path currently trains from ASE "
        "camera caches only, so 3D-FRONT depth-map scoring is intentionally "
        "not emulated."
    )


def effective_render_view_count(cfg: dict, loss_cfg: Optional[dict] = None) -> int:
    loss_cfg = cfg.get("loss", {}) if loss_cfg is None else loss_cfg
    n_views = int(loss_cfg.get("n_images", 0))
    camera_cfg, mode = camera_sampling_config(cfg)
    if mode == "scoring" and camera_cfg.get("num_views") is not None:
        n_views = int(camera_cfg["num_views"])
    return n_views


def validation_pruning_config(cfg: dict) -> dict:
    validation_cfg = cfg.get("validation", {}) or {}
    occ_threshold = validation_cfg.get(
        "occ_threshold",
        validation_cfg.get("occupancy_threshold", 0.5),
    )
    prune_min_keep = validation_cfg.get(
        "prune_min_keep",
        validation_cfg.get("min_keep", 1),
    )
    return {
        "prune": bool(validation_cfg.get("prune", False)),
        "occ_threshold": float(occ_threshold),
        "prune_min_keep": int(prune_min_keep),
    }


def camera_score_key(cfg: dict) -> str:
    camera_cfg, _mode = camera_sampling_config(cfg)
    return canonical_camera_score_key(
        camera_cfg.get("score_key", "frustum_image_coverage")
    )


def camera_preferred_coverage(cfg: dict) -> float:
    camera_cfg, _mode = camera_sampling_config(cfg)
    return float(camera_cfg.get("preferred_coverage", 0.4))
