"""Validation, reconstruction export, and visual check helpers for AE training."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement

from gaussiangpt.autoencoder.training.config import (
    camera_sampling_config,
    effective_render_view_count,
    validation_pruning_config,
)
from gaussiangpt.autoencoder.training.gaussian_features import (
    ase_batch_sample_to_legacy_sample,
    build_world_positions,
)
from gaussiangpt.autoencoder.training.losses import compute_batch_loss
from gaussiangpt.autoencoder.training.losses import make_gt_prune_mask_fn
from gaussiangpt.autoencoder.training.render_losses import (
    HAS_RASTERIZER,
    render_gaussians,
    sample_cameras_around_bbox,
    sample_scored_cameras,
)


def save_gaussians_as_ply(gaussians: dict, path: str):
    """Write reconstructed Gaussians as a 3DGS-compatible Binary PLY file.

    The autoencoder decoder already returns attributes in their natural
    representation (see gaussian_heads.GaussianAttributeDecoder._postprocess):
      * color   in (0, 1)    -- linear RGB
      * opacity in [-10, 10] -- already in logit space (3DGS stores it this way)
      * scale   in R+        -- linear positive scale (3DGS stores log-scale)
      * rotation             -- unit quaternion
    """
    positions = gaussians["position"].detach().cpu().numpy()

    n_pts = positions.shape[0]

    # 1) Colour / SH -> 3DGS f_dc (degree 0) + f_rest (higher orders).
    #    3DGS rendering convention: rendered_dc = f_dc * C0 + 0.5,  C0 = 1/(2*sqrt(pi)).
    #    Loader (data._load_ply_scene) stores SH as [f_dc(3), f_rest[:9]] = 12 dims,
    #    so we round-trip the same layout here when "sh" is present.
    C0 = 0.28209479177387814
    f_rest = np.zeros((n_pts, 45), dtype=np.float32)
    if "sh" in gaussians and gaussians["sh"] is not None:
        sh = gaussians["sh"].detach().cpu().numpy()
        f_dc = sh[:, :3].astype(np.float32, copy=False)
        n_rest = min(sh.shape[1] - 3, 45)
        if n_rest > 0:
            f_rest[:, :n_rest] = sh[:, 3:3 + n_rest]
    else:
        colors = gaussians.get("color", torch.full((n_pts, 3), 0.5))
        colors = colors.detach().cpu().clamp(0.001, 0.999)
        f_dc = ((colors - 0.5) / C0).numpy()

    # 2) Opacity is ALREADY a logit out of the decoder; the 3DGS PLY field
    #    is also a logit (renderer applies sigmoid). Just write it through.
    #    The previous code re-applied `logit(...)`, which treats the stored
    #    logit as a probability and produced near +/- inf values (so the
    #    reconstructed scene rendered as either fully transparent or fully
    #    opaque garbage).
    opacity = gaussians.get(
        "opacity", torch.full((n_pts, 1), 2.2)  # ~sigmoid(2.2) ~= 0.9
    ).detach().cpu().clamp(-10.0, 10.0).numpy()

    # 3) Scale -> log-scale (3DGS PLY stores log-space scale).
    scale = gaussians.get("scale", torch.ones(n_pts, 3)).detach().cpu().clamp_min(1e-8)
    log_scale = torch.log(scale).numpy()

    # 4) Rotation -> unit quaternion (w, x, y, z); 3DGS renderer normalises again at use.
    if "rotation" in gaussians and gaussians["rotation"].numel() > 0:
        rotation = torch.nn.functional.normalize(
            gaussians["rotation"].detach().cpu(), dim=-1
        ).numpy()
    else:
        # Fallback: identity quaternion (w=1) so the renderer doesn't blow up.
        rotation = np.zeros((n_pts, 4), dtype=np.float32)
        rotation[:, 0] = 1.0

    # 3DGS PLYs include normal fields, but Gaussian splats do not use them here.
    normals = np.zeros_like(positions)

    # Build the structured vertex array expected by plyfile.
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

    for i in range(3):
        dtype_full.append((f'f_dc_{i}', 'f4'))
    for i in range(45):
        dtype_full.append((f'f_rest_{i}', 'f4'))

    dtype_full.append(('opacity', 'f4'))

    for i in range(3):
        dtype_full.append((f'scale_{i}', 'f4'))
    for i in range(4):
        dtype_full.append((f'rot_{i}', 'f4'))

    elements = np.empty(positions.shape[0], dtype=dtype_full)

    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]

    for i in range(3):
        elements[f'f_dc_{i}'] = f_dc[:, i]
    for i in range(45):
        elements[f'f_rest_{i}'] = f_rest[:, i]

    elements['opacity'] = opacity[:, 0]

    for i in range(3):
        elements[f'scale_{i}'] = log_scale[:, i]
    for i in range(4):
        elements[f'rot_{i}'] = rotation[:, i]

    # Write a binary little-endian PLY.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text=False).write(path)


def save_validation_reconstruction(
    raw_model,
    sample: dict,
    cfg: dict,
    device: torch.device,
    ply_path: str,
    image_path: Optional[str] = None,
    prune: bool = False,
    occupancy_threshold: float = 0.5,
    prune_min_keep: int = 1,
    gt_prune_with_gt_logits: bool = False,
):
    """Save a validation reconstruction.

    Always writes the predicted Gaussians as a 3DGS-compatible .ply at
    `ply_path`. If `image_path` is given, additionally renders N views of
    GT and predicted Gaussians and saves them as a 2-row PNG (top row =
    GT, bottom row = predicted) for quick eyeballing.
    """
    voxel_coords = sample["voxel_coords"].to(device)
    gaussians = {k: v.to(device) for k, v in sample.items()
                 if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}
    prune_mask_fn = None
    if gt_prune_with_gt_logits:
        prune_mask_fn, _occ_target_cache, _gt_prune_debug = make_gt_prune_mask_fn(
            voxel_coords, device
        )
    decoder_prune = bool(prune or gt_prune_with_gt_logits)

    pred_gaussians, _, _, _ = raw_model(
        gaussians,
        voxel_coords,
        prune=decoder_prune,
        occupancy_threshold=occupancy_threshold,
        min_keep=prune_min_keep,
        prune_mask_fn=prune_mask_fn,
    )
    pred_coords = pred_gaussians.pop("_coords", None)
    if pred_coords is not None:
        pred_voxel_coords = pred_coords[:, 1:] if pred_coords.shape[1] == 4 else pred_coords
    else:
        pred_voxel_coords = voxel_coords
    base_voxel_size = float(cfg["data"]["base_voxel_size"])
    pred_sample = dict(sample)
    pred_sample["voxel_coords"] = pred_voxel_coords
    pred_gaussians["position"] = build_world_positions(
        pred_sample, pred_gaussians["offset"], base_voxel_size, device
    )
    save_gaussians_as_ply(pred_gaussians, ply_path)

    # ---- Optional: render GT vs. Pred views and save as a side-by-side PNG ----
    if image_path is None or not HAS_RASTERIZER or device.type != "cuda":
        return

    loss_cfg = cfg.get("loss", {})
    camera_cfg, camera_mode = camera_sampling_config(cfg)
    n_views = effective_render_view_count(cfg, loss_cfg)
    img_size = int(loss_cfg.get("render_size", 128))
    if n_views <= 0:
        return

    gt_position = build_world_positions(
        sample, gaussians["offset"], base_voxel_size, device
    )
    bbox_min = gt_position.min(dim=0).values
    bbox_max = gt_position.max(dim=0).values
    cameras = []
    if camera_mode == "scoring":
        cameras, _camera_debug = sample_scored_cameras(
            sample=sample,
            cfg=cfg,
            n_views=n_views,
            image_size=img_size,
            fallback_fov_deg=float(loss_cfg.get("fov_deg", 60.0)),
            device=device,
            rng=None,
            global_step=None,
        )
    if not cameras:
        # Validation orbit views are deterministic (no jitter) so renderings
        # are directly comparable across val runs.
        cameras = sample_cameras_around_bbox(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            n_views=n_views,
            image_height=img_size,
            image_width=img_size,
            fov_deg=float(loss_cfg.get("fov_deg", 60.0)),
            radius_factor=float(loss_cfg.get("radius_factor", 1.5)),
            upper_hemisphere_only=bool(loss_cfg.get("upper_hemisphere_only", False)),
            jitter=0.0,
        )

    bg = torch.zeros(3, device=device)
    gt_pack = {
        "position": gt_position,
        "scale": gaussians["scale"],
        "rotation": gaussians["rotation"],
        "opacity": gaussians["opacity"],
        "color": gaussians["color"],
    }
    pred_pack = {
        "position": pred_gaussians["position"],
        "scale": pred_gaussians["scale"],
        "rotation": pred_gaussians["rotation"],
        "opacity": pred_gaussians["opacity"],
        "color": pred_gaussians["color"],
    }

    gt_imgs, pred_imgs = [], []
    for cam in cameras:
        gt_imgs.append(render_gaussians(gt_pack, cam, bg_color=bg))
        pred_imgs.append(render_gaussians(pred_pack, cam, bg_color=bg))

    # 2-row grid: first row GT, second row Pred (each row has n_views images).
    from torchvision.utils import make_grid, save_image
    stacked = torch.stack(gt_imgs + pred_imgs, dim=0)  # (2N, 3, H, W)
    grid = make_grid(stacked, nrow=n_views, pad_value=1.0)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    save_image(grid, image_path)


def validate(
    raw_model,
    val_loader,
    cfg: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    output_dir: str,
    perceptual: nn.Module = None,
    gt_render_cache: Optional[dict] = None,
):
    """Run validation over the full validation loader and print average losses."""
    if len(val_loader) == 0:
        print(f"Validation skipped at step {global_step}: val_loader is empty")
        return None

    raw_model.eval()
    total_loss = total_occ = total_lfq = 0.0
    total_rgb = total_perc = 0.0
    val_prune_cfg = validation_pruning_config(cfg)
    val_prune = bool(val_prune_cfg["prune"])
    val_gt_prune = bool(val_prune_cfg["prune_with_gt_logits"])
    val_occ_threshold = float(val_prune_cfg["occ_threshold"])
    val_prune_min_keep = int(val_prune_cfg["prune_min_keep"])
    val_decoder_prune = bool(val_prune or val_gt_prune)
    if val_gt_prune:
        prediction_mode = "gt-pruned"
    elif val_prune:
        prediction_mode = "occ-head-pruned"
    else:
        prediction_mode = "unpruned"
    print(
        f"[validation pruning] enabled={val_decoder_prune} "
        f"gt_prune={val_gt_prune} "
        f"occ_head_prune={bool(val_prune and not val_gt_prune)} "
        f"occ_threshold={val_occ_threshold} prune_min_keep={val_prune_min_keep} "
        f"render_prediction={prediction_mode} saved_reconstruction={prediction_mode}"
    )
    recon_tag = f"epoch_{epoch:04d}_step_{global_step:08d}"
    recon_path = os.path.join(output_dir, "val_reconstructions", f"{recon_tag}.ply")
    image_path = os.path.join(output_dir, "val_renderings", f"{recon_tag}.png")
    saved_reconstruction = False
    # Validation uses a deterministic camera-jitter RNG so reconstruction
    # quality is comparable across val runs. The generator must live on
    # the same device as the tensors it samples (camera jitter is CUDA
    # when rendering happens on GPU).
    val_rng = torch.Generator(device=device)
    val_rng.manual_seed(int(global_step))
    with torch.no_grad():
        for val_batch_idx, batch in enumerate(val_loader):
            (batch_loss, batch_occ, batch_lfq,
             batch_rgb, batch_perc) = compute_batch_loss(
                raw_model, batch, cfg, device, backward=False,
                perceptual=perceptual, rng=val_rng,
                gt_render_cache=gt_render_cache,
                global_step=global_step,
                decoder_prune=val_prune,
                gt_prune_with_gt_logits=val_gt_prune,
                occupancy_threshold=val_occ_threshold,
                prune_min_keep=val_prune_min_keep,
                log_pruning=True,
                log_prefix=f"validation batch={val_batch_idx}",
            )
            total_loss += batch_loss.item()
            total_occ += batch_occ
            total_lfq += batch_lfq
            total_rgb += batch_rgb
            total_perc += batch_perc
            if not saved_reconstruction and batch["coords"].numel() > 0:
                save_validation_reconstruction(
                    raw_model, ase_batch_sample_to_legacy_sample(batch, 0), cfg, device,
                    ply_path=recon_path, image_path=image_path,
                    prune=val_prune,
                    occupancy_threshold=val_occ_threshold,
                    prune_min_keep=val_prune_min_keep,
                    gt_prune_with_gt_logits=val_gt_prune,
                )
                saved_reconstruction = True

    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    print(
        f"Validation Epoch {epoch} Step {global_step} "
        f"Loss: {avg_loss:.4f} "
        f"rgb: {total_rgb / n_batches:.4f} perc: {total_perc / n_batches:.4f} "
        f"occ: {total_occ / n_batches:.4f} lfq: {total_lfq / n_batches:.4f}"
    )
    if saved_reconstruction:
        print(f"Saved validation reconstruction ({prediction_mode}): {recon_path}")
        if (
            HAS_RASTERIZER
            and device.type == "cuda"
            and effective_render_view_count(cfg, cfg.get("loss", {})) > 0
        ):
            print(f"Saved validation renderings ({prediction_mode}):     {image_path}")
    return avg_loss
