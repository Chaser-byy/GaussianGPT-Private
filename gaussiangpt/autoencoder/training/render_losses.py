"""Render supervision and camera sampling helpers for AE training."""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from gaussiangpt.autoencoder.training.config import (
    camera_dataset_type,
    camera_preferred_coverage,
    camera_sampling_config,
    camera_score_key,
    effective_render_view_count,
    validate_camera_sampling_config,
)
from gaussiangpt.autoencoder.training.gaussian_features import build_world_positions
from gaussiangpt.utils.rendering import (
    HAS_RASTERIZER,
    make_camera_from_world_to_camera,
    sample_cameras_around_bbox,
    render_gaussians,
)


def _camera_candidate_count(cfg: dict, n_views: int) -> Optional[int]:
    camera_cfg, mode = camera_sampling_config(cfg)
    if mode != "scoring":
        return None
    if camera_dataset_type(cfg) == "ase":
        return None
    if camera_cfg.get("num_candidates") is not None:
        value = int(camera_cfg["num_candidates"])
        return value if value > 0 else None
    return max(1, 8 * max(1, int(n_views)))


def _camera_candidates_from_sample(sample: dict) -> list:
    meta = sample.get("metadata", {}) or {}
    candidates = meta.get("top_cameras")
    if candidates is None:
        candidates = (meta.get("camera_debug", {}) or {}).get("top_cameras", [])
    return list(candidates or [])


def _camera_int_field(item: dict, primary: str, fallback: str) -> int:
    value = item.get(primary)
    if value is None:
        value = item.get(fallback)
    if value is None:
        return -1
    return int(value)


def _camera_candidate_signature(candidates: list, score_key: str) -> tuple:
    signature = []
    for item in candidates:
        score = item.get(score_key, item.get("frustum_image_coverage", 0.0))
        try:
            score_value = round(float(score), 6)
        except (TypeError, ValueError):
            score_value = 0.0
        signature.append(
            (
                str(item.get("camera_id", "")),
                _camera_int_field(item, "frame_index", "frame_id"),
                score_value,
            )
        )
    return tuple(signature)


def _scored_camera_to_minicam(
    camera_info: dict,
    image_size: int,
    fallback_fov_deg: float,
    device: torch.device,
):
    world_to_camera = torch.as_tensor(
        np.asarray(camera_info["w2c"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    width = float(camera_info.get("width", image_size))
    height = float(camera_info.get("height", image_size))
    fx = float(camera_info.get("fx", 0.0))
    fy = float(camera_info.get("fy", 0.0))
    fallback_fov = math.radians(float(fallback_fov_deg))
    fovx = 2.0 * math.atan(width / max(2.0 * fx, 1e-8)) if fx > 0 else fallback_fov
    fovy = 2.0 * math.atan(height / max(2.0 * fy, 1e-8)) if fy > 0 else fallback_fov
    return make_camera_from_world_to_camera(
        world_to_camera=world_to_camera,
        fovx=fovx,
        fovy=fovy,
        image_height=image_size,
        image_width=image_size,
    )


def _camera_sampling_generator(
    cfg: dict,
    sample: dict,
    device: torch.device,
    rng: Optional[torch.Generator],
    global_step: Optional[int],
) -> Optional[torch.Generator]:
    camera_cfg, _mode = camera_sampling_config(cfg)
    seed_value = camera_cfg.get("seed")
    if seed_value is None:
        return rng

    meta = sample.get("metadata", {}) or {}
    seed = int(seed_value) + int(global_step or 0) * 1_000_003
    for char in str(meta.get("scene_id", "")):
        seed = (seed * 33 + ord(char)) % (2**63 - 1)
    chunk = meta.get("chunk_min_voxel", sample.get("chunk_origin", [0, 0, 0]))
    if torch.is_tensor(chunk):
        chunk = chunk.detach().cpu().tolist()
    for idx, value in enumerate(chunk):
        seed = (seed * 1009 + (idx + 1) * int(value)) % (2**63 - 1)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sample_scored_cameras(
    sample: dict,
    cfg: dict,
    n_views: int,
    image_size: int,
    fallback_fov_deg: float,
    device: torch.device,
    rng: Optional[torch.Generator],
    global_step: Optional[int],
) -> tuple:
    """Sample cameras with the dataset-specific GaussianGPT Appendix C rule."""

    validate_camera_sampling_config(cfg)
    camera_cfg, _mode = camera_sampling_config(cfg)
    dataset_type = camera_dataset_type(cfg)
    score_key = camera_score_key(cfg)
    temperature = float(camera_cfg.get("temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError(
            "camera_sampling.temperature must be > 0 for scoring; "
            f"got {temperature}."
        )
    preferred_coverage = camera_preferred_coverage(cfg)
    raw_candidates = _camera_candidates_from_sample(sample)
    candidates = [item for item in raw_candidates if "w2c" in item]
    debug = {
        "source": "scoring",
        "dataset_type": dataset_type,
        "score_key": score_key,
        "candidate_count": len(raw_candidates),
        "usable_candidate_count": len(candidates),
        "selection_mode": None,
        "sampled_count": 0,
        "fallback": None,
        "preferred_coverage": preferred_coverage,
    }
    if not candidates:
        raise RuntimeError(
            "camera_sampling.mode=scoring,dataset_type=ase requires scored ASE "
            "camera candidates with poses in sample metadata. Rebuild the ASE "
            "camera cache or run the dataset with include_camera_matrices=True."
        )

    score_values = []
    for item in candidates:
        score = item.get(score_key, item.get("frustum_image_coverage", 0.0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score) or score < 0.0:
            score = 0.0
        score_values.append(score)

    scores = torch.tensor(score_values, dtype=torch.float32, device=device)
    score_min = float(scores.min().detach().item())
    score_max = float(scores.max().detach().item())
    score_mean = float(scores.mean().detach().item())
    score_sum = float(scores.sum().detach().item())

    positive_mask = scores > 0.0
    preferred_mask = scores >= float(preferred_coverage)
    positive_probability_count = int(torch.count_nonzero(positive_mask).detach().item())
    preferred_count = int(torch.count_nonzero(preferred_mask).detach().item())

    if preferred_count > 0:
        pool_mask = preferred_mask
        debug["selection_mode"] = "preferred"
        debug["fallback"] = None
    elif positive_probability_count > 0:
        pool_mask = positive_mask
        debug["selection_mode"] = "fallback_any_overlap"
        debug["fallback"] = "no_preferred_use_any_overlap"
    else:
        pool_mask = torch.ones_like(scores, dtype=torch.bool)
        debug["selection_mode"] = "fallback_uniform_all_zero"
        debug["fallback"] = "all_zero_scores_uniform"
        meta = sample.get("metadata", {}) or {}
        print(
            "WARNING: camera_sampling.mode=scoring found no positive "
            "frustum_image_coverage cameras; sampling uniformly from all ASE "
            f"cameras for scene={meta.get('scene_id', '')} "
            f"chunk_min={meta.get('chunk_min_voxel', sample.get('chunk_origin', None))}."
        )

    pool_indices = torch.nonzero(pool_mask, as_tuple=False).flatten()
    pool_scores = scores[pool_indices]
    if debug["selection_mode"] == "fallback_uniform_all_zero":
        pool_probs = torch.full_like(pool_scores, 1.0 / float(pool_scores.numel()))
    else:
        weights = pool_scores.pow(1.0 / temperature)
        pool_probs = weights / weights.sum().clamp_min(1e-12)

    replacement = int(n_views) > int(pool_indices.numel())
    if replacement:
        debug["fallback"] = (
            f"{debug['fallback']}+replacement"
            if debug["fallback"]
            else "candidate_count_lt_num_views_replacement"
        )
    generator = _camera_sampling_generator(cfg, sample, device, rng, global_step)
    sampled_pool_indices = torch.multinomial(
        pool_probs,
        int(n_views),
        replacement=replacement,
        generator=generator,
    )
    sampled = pool_indices[sampled_pool_indices]
    sampled_indices = [int(value) for value in sampled.detach().cpu().tolist()]
    selected = [candidates[index] for index in sampled_indices]
    cameras = [
        _scored_camera_to_minicam(item, image_size, fallback_fov_deg, device)
        for item in selected
    ]

    full_probs = torch.zeros_like(scores)
    full_probs[pool_indices] = pool_probs
    prob_min = float(pool_probs.min().detach().item())
    prob_max = float(pool_probs.max().detach().item())
    entropy = float(
        (-(pool_probs * torch.log(pool_probs.clamp_min(1e-12))).sum()).detach().item()
    )
    full_prob_values = full_probs.detach().cpu().tolist()
    top_cameras = []
    for index, item in enumerate(candidates[: min(len(candidates), 8)]):
        score = float(score_values[index])
        if preferred_count > 0:
            selected_by = "preferred" if score >= preferred_coverage else "excluded"
        elif positive_probability_count > 0:
            selected_by = "fallback_any_overlap" if score > 0.0 else "excluded"
        else:
            selected_by = "fallback_uniform_all_zero"
        top_cameras.append(
            {
                "camera_id": str(item.get("camera_id", "")),
                "frame_index": _camera_int_field(item, "frame_index", "frame_id"),
                "frame_id": _camera_int_field(item, "frame_id", "frame_index"),
                "file_path": str(item.get("file_path", "")),
                "score": score,
                "frustum_image_coverage": score,
                "projected_area": float(item.get("projected_area", 0.0) or 0.0),
                "image_area": float(item.get("image_area", 0.0) or 0.0),
                "num_intersection_vertices": int(
                    item.get("num_intersection_vertices", 0) or 0
                ),
                "selected_by": selected_by,
                "sampling_probability": float(full_prob_values[index]),
            }
        )
    debug.update(
        {
            "sampled_count": len(cameras),
            "sampled_indices": sampled_indices,
            "sampled_frame_indices": [
                _camera_int_field(item, "frame_index", "frame_id")
                for item in selected
            ],
            "sampled_camera_ids": [str(item.get("camera_id", "")) for item in selected],
            "score_min": score_min,
            "score_max": score_max,
            "score_mean": score_mean,
            "score_sum": score_sum,
            "positive_probability_count": positive_probability_count,
            "preferred_count": preferred_count,
            "selection_pool_count": int(pool_indices.numel()),
            "prob_min": prob_min,
            "prob_max": prob_max,
            "prob_entropy": entropy,
            "temperature": temperature,
            "top_cameras": top_cameras,
        }
    )
    return cameras, debug


def _should_log_camera_sampling(cfg: dict, global_step: Optional[int]) -> bool:
    camera_cfg, mode = camera_sampling_config(cfg)
    if mode != "scoring":
        return False
    if global_step is None:
        return False
    every = int(camera_cfg.get("log_every", 100))
    return int(global_step) == 1 or (every > 0 and int(global_step) % every == 0)


def _log_camera_sampling(sample: dict, global_step: int, debug: dict) -> None:
    meta = sample.get("metadata", {}) or {}
    chunk = meta.get("chunk_min_voxel", sample.get("chunk_origin", None))
    if torch.is_tensor(chunk):
        chunk = chunk.detach().cpu().tolist()
    print(
        "  [camera sampling] "
        f"step={global_step} mode=scoring "
        f"dataset={debug.get('dataset_type', 'ase')} "
        f"scene={meta.get('scene_id', '')} "
        f"chunk_min={chunk} source={debug.get('source')} "
        f"candidates={debug.get('usable_candidate_count', 0)}/"
        f"{debug.get('candidate_count', 0)} "
        f"sampled={debug.get('sampled_count', 0)} "
        f"score_key={debug.get('score_key')} "
        f"selection={debug.get('selection_mode', 'unknown')} "
        f"preferred={debug.get('preferred_count', 0)} "
        f"positive={debug.get('positive_probability_count', 0)} "
        f"pool={debug.get('selection_pool_count', 0)} "
        f"score={debug.get('score_min', 0.0):.4g}/"
        f"{debug.get('score_mean', 0.0):.4g}/"
        f"{debug.get('score_max', 0.0):.4g} "
        f"prob={debug.get('prob_min', 0.0):.4g}/"
        f"{debug.get('prob_max', 0.0):.4g} "
        f"entropy={debug.get('prob_entropy', 0.0):.4g} "
        f"indices={debug.get('sampled_indices', [])} "
        f"frames={debug.get('sampled_frame_indices', [])} "
        f"fallback={debug.get('fallback')}"
    )
    for rank, item in enumerate(debug.get("top_cameras", [])[:8]):
        print(
            "    [camera candidate] "
            f"rank={rank} frame_index={item.get('frame_index')} "
            f"frame_id={item.get('frame_id')} "
            f"file={item.get('file_path')} "
            f"score={item.get('score', 0.0):.6g} "
            f"frustum_image_coverage={item.get('frustum_image_coverage', 0.0):.6g} "
            f"projected_area={item.get('projected_area', 0.0):.3f} "
            f"image_area={item.get('image_area', 0.0):.3f} "
            f"num_intersection_vertices={item.get('num_intersection_vertices', 0)} "
            f"selected_by={item.get('selected_by')} "
            f"prob={item.get('sampling_probability', 0.0):.6g}"
        )


def render_loss_for_sample(
    sample: dict,
    pred_gaussians: dict,
    gt_gaussians: dict,
    cfg: dict,
    device: torch.device,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
    gt_render_cache: Optional[dict] = None,
    global_step: Optional[int] = None,
) -> tuple:
    """Render N views of GT and predicted Gaussians; return (l_rgb, l_perc).

    The GT renderings are detached so the renderer is only used as a fixed
    photo-realistic supervision signal. Returns scalar tensors on `device`.
    """
    if not HAS_RASTERIZER or device.type != "cuda":
        zero = torch.zeros((), device=device)
        return zero, zero

    loss_cfg = cfg.get("loss", {})
    camera_cfg, camera_mode = camera_sampling_config(cfg)
    n_views = effective_render_view_count(cfg, loss_cfg)
    img_size = int(loss_cfg.get("render_size", 128))
    if n_views <= 0:
        zero = torch.zeros((), device=device)
        return zero, zero

    base_voxel_size = float(cfg["data"]["base_voxel_size"])

    # Build absolute world positions for both GT and reconstruction.
    # GaussianGPT predicts offsets as unbounded world-space values
    # (Appendix C), so we always size the camera sphere from the GT bbox
    # rather than relying on any prediction-side bound.
    gt_position = build_world_positions(
        sample, gt_gaussians["offset"].detach(), base_voxel_size, device
    )
    pred_sample = sample
    if "pred_voxel_coords" in sample:
        pred_sample = dict(sample)
        pred_sample["voxel_coords"] = sample["pred_voxel_coords"]
    pred_position = build_world_positions(
        pred_sample, pred_gaussians["offset"], base_voxel_size, device
    )

    bg = torch.zeros(3, device=device)
    gt_pack = {
        "position": gt_position,
        "scale": gt_gaussians["scale"].detach(),
        "rotation": gt_gaussians["rotation"].detach(),
        "opacity": gt_gaussians["opacity"].detach(),
        "color": gt_gaussians["color"].detach(),
    }
    pred_pack = {
        "position": pred_position,
        "scale": pred_gaussians["scale"],
        "rotation": pred_gaussians["rotation"],
        "opacity": pred_gaussians["opacity"],
        "color": pred_gaussians["color"],
    }

    fov_deg = float(loss_cfg.get("fov_deg", 60.0))
    radius_factor = float(loss_cfg.get("radius_factor", 1.5))
    upper_hemisphere_only = bool(loss_cfg.get("upper_hemisphere_only", False))
    camera_jitter = float(loss_cfg.get("camera_jitter", 0.0))
    score_key = camera_score_key(cfg)
    candidates = _camera_candidates_from_sample(sample)
    camera_debug = None
    cache_key = None
    cameras = None
    cached_gt_imgs = None
    cacheable_cameras = camera_mode == "scoring" or camera_jitter == 0.0
    if gt_render_cache is not None and cacheable_cameras:
        meta = sample.get("metadata", {})
        chunk_origin_value = meta.get("chunk_min_voxel", sample.get("chunk_origin"))
        if torch.is_tensor(chunk_origin_value):
            chunk_origin_value = chunk_origin_value.detach().cpu().tolist()
        chunk_origin_key = tuple(int(v) for v in chunk_origin_value)
        candidate_count_for_cache = _camera_candidate_count(cfg, n_views)
        cache_key = (
            str(meta.get("scene_id", "")),
            chunk_origin_key,
            int(sample["voxel_coords"].shape[0]),
            str(device),
            int(n_views),
            int(img_size),
            float(fov_deg),
            float(radius_factor),
            bool(upper_hemisphere_only),
            str(camera_mode),
            (candidate_count_for_cache if candidate_count_for_cache is not None else "all")
            if camera_mode == "scoring" else 0,
            float(camera_cfg.get("temperature", 1.0)) if camera_mode == "scoring" else 0.0,
            float(camera_preferred_coverage(cfg)) if camera_mode == "scoring" else 0.0,
            str(score_key) if camera_mode == "scoring" else "",
            _camera_candidate_signature(candidates, score_key)
            if camera_mode == "scoring"
            else (),
        )
        cached = gt_render_cache.get(cache_key)
        if cached is not None:
            cameras = cached["cameras"]
            cached_gt_imgs = cached["gt_imgs"]
            camera_debug = dict(cached.get("camera_debug", {}))
            if camera_debug:
                camera_debug["source"] = "gt_render_cache"

    if cameras is None:
        bbox_min = gt_position.min(dim=0).values.detach()
        bbox_max = gt_position.max(dim=0).values.detach()
        if camera_mode == "scoring":
            cameras, camera_debug = sample_scored_cameras(
                sample=sample,
                cfg=cfg,
                n_views=n_views,
                image_size=img_size,
                fallback_fov_deg=fov_deg,
                device=device,
                rng=rng,
                global_step=global_step,
            )
        if not cameras:
            if camera_mode == "scoring":
                if camera_debug is None:
                    camera_debug = {}
                camera_debug["source"] = "orbit_fallback"
            # Camera sphere is sized from the GT bbox so the views naturally
            # cover the scene as it gets reconstructed. This is the original
            # orbit path and remains the default behavior.
            cameras = sample_cameras_around_bbox(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                n_views=n_views,
                image_height=img_size,
                image_width=img_size,
                fov_deg=fov_deg,
                radius_factor=radius_factor,
                upper_hemisphere_only=upper_hemisphere_only,
                jitter=camera_jitter,
                rng=rng,
            )

    if (
        camera_mode == "scoring"
        and camera_debug is not None
        and _should_log_camera_sampling(cfg, global_step)
    ):
        _log_camera_sampling(sample, int(global_step), camera_debug)

    l_rgb = torch.zeros((), device=device)
    l_perc = torch.zeros((), device=device)
    perc_buf_pred, perc_buf_gt = [], []
    perc_batch_size = max(1, int(loss_cfg.get("perceptual_batch_size", 1)))
    gt_imgs_to_cache = [] if cache_key is not None and cached_gt_imgs is None else None
    for view_idx, cam in enumerate(cameras):
        if cached_gt_imgs is not None:
            img_gt = cached_gt_imgs[view_idx]
        else:
            with torch.no_grad():
                img_gt = render_gaussians(gt_pack, cam, bg_color=bg).detach()
            if gt_imgs_to_cache is not None:
                gt_imgs_to_cache.append(img_gt)
        img_pred = render_gaussians(pred_pack, cam, bg_color=bg)
        l_rgb = l_rgb + torch.nn.functional.l1_loss(img_pred, img_gt)
        if perceptual is not None and float(loss_cfg.get("lambda_perc", 0.0)) > 0:
            perc_buf_pred.append(img_pred)
            perc_buf_gt.append(img_gt)
    if gt_imgs_to_cache is not None:
        gt_render_cache[cache_key] = {
            "cameras": cameras,
            "gt_imgs": tuple(gt_imgs_to_cache),
            "camera_debug": camera_debug,
        }

    l_rgb = l_rgb / float(n_views)
    if perc_buf_pred:
        # VGG at 512px is memory-heavy. Run it in small chunks so the peak
        # activation footprint is bounded by perceptual_batch_size rather than
        # n_views, while preserving the same average-over-views loss.
        for start in range(0, len(perc_buf_pred), perc_batch_size):
            end = start + perc_batch_size
            pred_chunk = torch.stack(perc_buf_pred[start:end], dim=0)
            gt_chunk = torch.stack(perc_buf_gt[start:end], dim=0)
            chunk_loss = perceptual(pred_chunk, gt_chunk)
            l_perc = l_perc + chunk_loss * float(pred_chunk.shape[0])
        l_perc = l_perc / float(len(perc_buf_pred))
    return l_rgb, l_perc


def render_losses_for_batch(
    batch: dict,
    coords: torch.Tensor,
    pred_coords: torch.Tensor,
    pred_gaussians: dict,
    gt_gaussians: dict,
    cfg: dict,
    device: torch.device,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
    gt_render_cache: Optional[dict] = None,
    global_step: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render per sample in a sparse batch and average over rendered samples."""

    l_rgb = torch.tensor(0.0, device=device)
    l_perc = torch.tensor(0.0, device=device)
    gt_batch_indices = coords[:, 0]
    pred_batch_indices = pred_coords[:, 0]

    render_count = 0
    for meta_idx, _meta in enumerate(batch["metas"]):
        gt_mask = gt_batch_indices == meta_idx
        pred_mask = pred_batch_indices == meta_idx
        if not bool(gt_mask.any()) or not bool(pred_mask.any()):
            continue
        render_count += 1

        sample_pred = {k: v[pred_mask] for k, v in pred_gaussians.items()}
        sample_gt = {k: v[gt_mask] for k, v in gt_gaussians.items()}
        meta_sample = {
            "voxel_coords": coords[gt_mask, 1:4],
            "pred_voxel_coords": pred_coords[pred_mask, 1:4],
            "chunk_origin": torch.tensor(
                batch["metas"][meta_idx]["chunk_min_voxel"], device=device
            ),
            "metadata": batch["metas"][meta_idx],
        }

        single_rgb, single_perc = render_loss_for_sample(
            meta_sample,
            sample_pred,
            sample_gt,
            cfg,
            device,
            perceptual=perceptual,
            rng=rng,
            gt_render_cache=gt_render_cache,
            global_step=global_step,
        )
        l_rgb = l_rgb + single_rgb
        l_perc = l_perc + single_perc

    if render_count > 0:
        l_rgb = l_rgb / render_count
        l_perc = l_perc / render_count
    return l_rgb, l_perc


def build_perceptual_loss_if_needed(
    loss_cfg: dict,
    render_n_views: int,
    device: torch.device,
) -> tuple[bool, Optional[nn.Module]]:
    """Lazily construct VGG perceptual loss only when render supervision uses it."""

    use_render = (
        float(loss_cfg.get("lambda_rgb", 0.0)) > 0.0
        or float(loss_cfg.get("lambda_perc", 0.0)) > 0.0
    ) and render_n_views > 0
    perceptual = None
    if use_render:
        if not HAS_RASTERIZER:
            print("WARNING: lambda_rgb/perc > 0 but diff-gaussian-rasterization "
                  "is not importable; rendering losses will be skipped.")
        elif device.type != "cuda":
            print("WARNING: rendering losses require CUDA; skipping on CPU.")
        else:
            print(
                f"Rendering supervision enabled: "
                f"n_views={render_n_views} "
                f"img={loss_cfg.get('render_size', 128)}^2 "
                f"lambda_rgb={loss_cfg.get('lambda_rgb', 0.0)} "
                f"lambda_perc={loss_cfg.get('lambda_perc', 0.0)}"
            )
            if float(loss_cfg.get("lambda_perc", 0.0)) > 0:
                from gaussiangpt.utils.perceptual import VGGPerceptualLoss
                perceptual = VGGPerceptualLoss(
                    device=device,
                    weights_path=loss_cfg.get("vgg19_weights_path"),
                )
    return use_render, perceptual


def fixed_gt_render_cache(
    fixed_chunk: bool,
    use_render: bool,
    loss_cfg: dict,
    camera_mode: str,
    device: torch.device,
) -> Optional[dict]:
    """Create the fixed-chunk GT render cache when the existing conditions allow it."""

    cache_fixed_gt_render = (
        fixed_chunk
        and use_render
        and HAS_RASTERIZER
        and device.type == "cuda"
        and (
            camera_mode == "scoring"
            or float(loss_cfg.get("camera_jitter", 0.0)) == 0.0
        )
        and bool(loss_cfg.get("cache_fixed_gt_render", True))
    )
    if cache_fixed_gt_render:
        print(f"[fixed_chunk] GT render cache enabled (camera_sampling={camera_mode})")
        return {}
    if fixed_chunk and use_render:
        print("[fixed_chunk] GT render cache disabled")
    return None
