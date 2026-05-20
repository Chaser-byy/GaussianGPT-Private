"""Train the GaussianGPT autoencoder.

Usage:
    python scripts/train_autoencoder.py --config configs/autoencoder_scene.yaml

Paper training details:
  - Adam optimizer, lr=1e-4, cosine decay to 10%
  - 4 RTX A6000 GPUs, effective batch size 8 (scenes) / 24 (objects)
  - ~4 days for scenes, ~2 days for PhotoShape
"""
import os
import argparse
import yaml
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from plyfile import PlyData, PlyElement

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiangpt.autoencoder import GaussianAutoencoder
from gaussiangpt.autoencoder.diagnostics import ColorClampDiagnostics
from gaussiangpt.data import GaussianSceneDataset
from gaussiangpt.utils.rendering import (
    HAS_RASTERIZER,
    sample_cameras_around_bbox,
    render_gaussians,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder_scene.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/autoencoder")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_every_steps", type=int, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    return parser.parse_args()


def sparse_collate(batch):
    """Collate function for variable-size sparse Gaussian data.

    Returns a list of per-sample dicts rather than stacking tensors,
    since the number of voxels N differs across samples.
    """
    return batch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_world_positions(
    sample: dict,
    pred_offset: torch.Tensor,
    base_voxel_size: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert per-voxel offsets into absolute world-space positions.

    `voxel_coords` are chunk-local; `chunk_origin` (if present) shifts them
    back into the scene's global voxel frame so that GT and reconstruction
    live in the same coordinate system.
    """
    voxel_coords = sample["voxel_coords"].to(device)
    if "chunk_origin" in sample:
        abs_voxel_coords = voxel_coords + sample["chunk_origin"].to(device)
    else:
        abs_voxel_coords = voxel_coords
    voxel_centers = (abs_voxel_coords.to(pred_offset.dtype) + 0.5) * base_voxel_size
    return voxel_centers + pred_offset


def _render_loss_for_sample(
    sample: dict,
    pred_gaussians: dict,
    gt_gaussians: dict,
    cfg: dict,
    device: torch.device,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
) -> tuple:
    """Render N views of GT and predicted Gaussians; return (l_rgb, l_perc).

    The GT renderings are detached so the renderer is only used as a fixed
    photo-realistic supervision signal. Returns scalar tensors on `device`.
    """
    if not HAS_RASTERIZER or device.type != "cuda":
        zero = torch.zeros((), device=device)
        return zero, zero

    loss_cfg = cfg.get("loss", {})
    n_views = int(loss_cfg.get("n_images", 0))
    img_size = int(loss_cfg.get("render_size", 128))
    if n_views <= 0:
        zero = torch.zeros((), device=device)
        return zero, zero

    base_voxel_size = float(cfg["data"]["base_voxel_size"])

    # Build absolute world positions for both GT and reconstruction.
    # GaussianGPT predicts offsets as unbounded world-space values
    # (Appendix C), so we always size the camera sphere from the GT bbox
    # rather than relying on any prediction-side bound.
    gt_position = _build_world_positions(
        sample, gt_gaussians["offset"].detach(), base_voxel_size, device
    )
    pred_sample = sample
    if "pred_voxel_coords" in sample:
        pred_sample = dict(sample)
        pred_sample["voxel_coords"] = sample["pred_voxel_coords"]
    pred_position = _build_world_positions(
        pred_sample, pred_gaussians["offset"], base_voxel_size, device
    )

    # Camera sphere is sized from the GT bbox so the views naturally cover
    # the scene as it gets reconstructed.
    bbox_min = gt_position.min(dim=0).values.detach()
    bbox_max = gt_position.max(dim=0).values.detach()
    cameras = sample_cameras_around_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_views=n_views,
        image_height=img_size,
        image_width=img_size,
        fov_deg=float(loss_cfg.get("fov_deg", 60.0)),
        radius_factor=float(loss_cfg.get("radius_factor", 1.5)),
        upper_hemisphere_only=bool(loss_cfg.get("upper_hemisphere_only", False)),
        jitter=float(loss_cfg.get("camera_jitter", 0.0)),
        rng=rng,
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

    l_rgb = torch.zeros((), device=device)
    l_perc = torch.zeros((), device=device)
    perc_buf_pred, perc_buf_gt = [], []
    for cam in cameras:
        with torch.no_grad():
            img_gt = render_gaussians(gt_pack, cam, bg_color=bg)
        img_pred = render_gaussians(pred_pack, cam, bg_color=bg)
        l_rgb = l_rgb + torch.nn.functional.l1_loss(img_pred, img_gt)
        if perceptual is not None and float(loss_cfg.get("lambda_perc", 0.0)) > 0:
            perc_buf_pred.append(img_pred)
            perc_buf_gt.append(img_gt)

    l_rgb = l_rgb / float(n_views)
    if perc_buf_pred:
        # Stack views into a batch for one VGG call (fewer kernel launches).
        l_perc = perceptual(
            torch.stack(perc_buf_pred, dim=0),
            torch.stack(perc_buf_gt, dim=0),
        )
    return l_rgb, l_perc


# def compute_batch_loss(
#     raw_model,
#     batch_list,
#     cfg: dict,
#     device: torch.device,
#     backward: bool = False,
#     perceptual: nn.Module = None,
#     rng: torch.Generator = None,
# ):
#     """Compute one sparse batch loss, optionally backpropagating per-sample losses.

#     GaussianGPT Eq. (1):
#         L = lambda_rgb  * L1(renderings)         # L3DG L_RGB
#           + lambda_perc * VGG19(renderings)      # L3DG L_perc
#           + lambda_occ  * BCE(occupancy)         # L3DG L_occ
#           + lambda_lfq  * softplus(L_LFQ + 5)    # LFQ codebook entropy
#     """
#     loss_cfg = cfg.get("loss", {})
#     lambda_rgb = float(loss_cfg.get("lambda_rgb", 0.0))
#     lambda_perc = float(loss_cfg.get("lambda_perc", 0.0))
#     lambda_occ = float(loss_cfg.get("lambda_occ", 0.0))
#     lambda_lfq = float(loss_cfg.get("lambda_lfq", 0.0))

#     batch_loss = torch.tensor(0.0, device=device)
#     batch_occ = batch_lfq = 0.0
#     batch_rgb = batch_perc = 0.0
#     n = max(len(batch_list), 1)

#     for sample in batch_list:
#         voxel_coords = sample["voxel_coords"].to(device)
#         gaussians = {k: v.to(device) for k, v in sample.items()
#                      if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

#         pred_gaussians, occ_list, lfq_loss, indices = raw_model(gaussians, voxel_coords)

#         # ---- L_occ: BCE on the per-stage occupancy logits ----
#         from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI
#         l_occ = torch.tensor(0.0, device=device)
#         if occ_list:
#             for stage_idx, occ in enumerate(occ_list):
#                 if HAS_MINKOWSKI:
#                     occ_feat = occ.F  # (M, 1)
#                     targets = torch.ones(occ_feat.shape[0], device=device)
#                 else:
#                     # Dense: occ is (1, 1, X', Y', Z')
#                     scale = 2 ** (len(occ_list) - stage_idx)
#                     occ_flat = occ.view(-1)
#                     gt_grid = torch.zeros_like(occ_flat)
#                     stage_vc = (voxel_coords // scale).clamp(0, occ.shape[2] - 1)
#                     flat_idx = (
#                         stage_vc[:, 0] * occ.shape[3] * occ.shape[4]
#                         + stage_vc[:, 1] * occ.shape[4]
#                         + stage_vc[:, 2]
#                     )
#                     gt_grid.scatter_(0, flat_idx, 1.0)
#                     occ_feat = occ_flat.unsqueeze(-1)
#                     targets = gt_grid
#                 l_occ = l_occ + torch.nn.functional.binary_cross_entropy_with_logits(
#                     occ_feat.squeeze(-1), targets
#                 )
#             l_occ = l_occ / len(occ_list)

#         # ---- L_RGB / L_perc: rendering supervision (L3DG-style) ----
#         if (lambda_rgb > 0 or lambda_perc > 0) and "position" not in pred_gaussians:
#             # The decoder doesn't output absolute positions; we attach them
#             # here so the renderer can use them. The same is done in
#             # `_render_loss_for_sample` for the GT.
#             pred_gaussians["position"] = _build_world_positions(
#                 sample, pred_gaussians["offset"], cfg["data"]["base_voxel_size"], device,
#             )
#         if lambda_rgb > 0 or lambda_perc > 0:
#             l_rgb, l_perc = _render_loss_for_sample(
#                 sample, pred_gaussians, gaussians, cfg, device,
#                 perceptual=perceptual, rng=rng,
#             )
#         else:
#             l_rgb = torch.zeros((), device=device)
#             l_perc = torch.zeros((), device=device)

#         # GaussianGPT Eq. (1): the LFQ entropy term is wrapped in
#         #     λ_LFQ · softplus(L_LFQ + 5)
#         # The purpose of the offset + softplus is purely cosmetic --- it
#         # keeps the displayed loss positive (since L_LFQ ∈ [−log 2, +log 2]
#         # can dip below zero) without meaningfully changing the gradient
#         # (sigmoid(L_LFQ + 5) ≈ 1 throughout the operating range).
#         l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)

#         sample_loss = (
#             lambda_rgb * l_rgb
#             + lambda_perc * l_perc
#             + lambda_occ * l_occ
#             + lambda_lfq * l_lfq
#         ) / n

#         if backward:
#             sample_loss.backward()

#         batch_loss = batch_loss + sample_loss.detach()
#         batch_occ += l_occ.item() / n
#         # Log the actual term that enters the loss (softplus-wrapped), so
#         # the printed value matches the gradient that backprops.
#         batch_lfq += l_lfq.item() / n
#         batch_rgb += float(l_rgb.detach().item()) / n
#         batch_perc += float(l_perc.detach().item()) / n

#     return batch_loss, batch_occ, batch_lfq, batch_rgb, batch_perc

def compute_batch_loss(
    raw_model,
    batch,             # 关键：此时传入的是由 ase_sparse_collate 打包好的大字典，不再是原版的 list
    cfg: dict,
    device: torch.device,
    backward: bool = False,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
):
    from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI

    loss_cfg = cfg.get("loss", {})
    lambda_rgb = float(loss_cfg.get("lambda_rgb", 0.0))
    lambda_perc = float(loss_cfg.get("lambda_perc", 0.0))
    lambda_occ = float(loss_cfg.get("lambda_occ", 0.0))
    lambda_lfq = float(loss_cfg.get("lambda_lfq", 0.0))

    # 1. 提取全批次拼接后的 4D 坐标与 14D 原始高斯特征
    coords = batch["coords"].to(device)       # 形状: (N_total, 4) -> [b, x, y, z]
    feats = batch["feats"].to(device)         # 形状: (N_total, 14)
    if not HAS_MINKOWSKI and coords.shape[1] == 4:
        raise RuntimeError(
            "The ASE dataloader produces batched sparse coordinates and requires "
            "MinkowskiEngine. Dense fallback only supports the legacy single-sample "
            "3D-coordinate path."
        )
    
    # 2. 核心衔接机制：将 14 维扁平特征解包切片，对齐师兄原模型 heads 期待的输入字典
    gaussians = {
        "offset": feats[:, 0:3],
        "color": feats[:, 3:6],
        "opacity": feats[:, 6:7],
        "scale": feats[:, 7:10],
        "rotation": feats[:, 10:14]
    }

    # 3. 彻底告别 for 循环！整个 Batch 放入 Sparse CNN 一把梭完成前向推理
    pred_gaussians, occ_list, lfq_loss, indices = raw_model(gaussians, coords)
    pred_coords = pred_gaussians.pop("_coords", None)
    if pred_coords is None:
        pred_coords = coords

    # ---- L_occ 占位损失计算 ----
    l_occ = torch.tensor(0.0, device=device)
    if occ_list:
        for stage_idx, occ in enumerate(occ_list):
            occ_feat = occ.F  # (M, 1)
            # 此时真值可以直接对应当前激活特征覆盖范围（全 1 监督掩码）
            targets = torch.ones(occ_feat.shape[0], device=device)
            l_occ = l_occ + torch.nn.functional.binary_cross_entropy_with_logits(
                occ_feat.squeeze(-1), targets
            )
        l_occ = l_occ / len(occ_list)

    # ---- L_LFQ 量化离散损失 ----
    l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)

    # ---- L_RGB / L_perc 渲染损失（前向并行，渲染串行） ----
    l_rgb = torch.tensor(0.0, device=device)
    l_perc = torch.tensor(0.0, device=device)
    
    if lambda_rgb > 0 or lambda_perc > 0:
        # 由于每个场景块有自己独立的相机视角，渲染必须通过 batch_index 拆开分别渲染
        gt_batch_indices = coords[:, 0]  # 提取第 0 列的 batch 标签
        pred_batch_indices = pred_coords[:, 0]

        render_count = 0
        for meta_idx, _meta in enumerate(batch["metas"]):
            gt_mask = gt_batch_indices == meta_idx
            pred_mask = pred_batch_indices == meta_idx
            if not bool(gt_mask.any()) or not bool(pred_mask.any()):
                continue
            render_count += 1
            
            # 剥离出当前单样本的预测值与真值
            sample_pred = {k: v[pred_mask] for k, v in pred_gaussians.items()}
            sample_gt = {k: v[gt_mask] for k, v in gaussians.items()}
            
            # 重新组装元数据供渲染器定位
            meta_sample = {
                "voxel_coords": coords[gt_mask, 1:4],  # 去掉开头的 batch 维，还原成 3D 坐标
                "pred_voxel_coords": pred_coords[pred_mask, 1:4],
                "chunk_origin": torch.tensor(batch["metas"][meta_idx]["chunk_min_voxel"], device=device)
            }
            
            # 复用原版的单视图渲染损失函数
            # 注意：传入单样本的相机或场景参数，可以从 batch["metas"][meta_idx] 中灵活读取
            single_rgb, single_perc = _render_loss_for_sample(
                meta_sample, sample_pred, sample_gt, cfg, device,
                perceptual=perceptual, rng=rng
            )
            l_rgb = l_rgb + single_rgb
            l_perc = l_perc + single_perc
        if render_count > 0:
            l_rgb = l_rgb / render_count
            l_perc = l_perc / render_count

    # 4. 综合总 Loss
    total_loss = (
        lambda_rgb * l_rgb
        + lambda_perc * l_perc
        + lambda_occ * l_occ
        + lambda_lfq * l_lfq
    )

    if backward:
        total_loss.backward()

    return (
        total_loss, 
        l_occ.item(), 
        l_lfq.item(), 
        float(l_rgb.detach().item()), 
        float(l_perc.detach().item())
    )


def ase_batch_sample_to_legacy_sample(batch: dict, sample_index: int = 0) -> dict:
    """Extract one collated ASE sample in the legacy per-sample format.

    Validation reconstruction still runs through the older single-sample helper,
    so this keeps that path isolated from the batched training representation.
    """
    coords = batch["coords"]
    feats = batch.get("target_feats", batch["feats"])
    mask = coords[:, 0] == int(sample_index)
    if not bool(mask.any()):
        raise ValueError(f"ASE batch does not contain sample_index={sample_index}")

    sample_feats = feats[mask]
    meta = batch["metas"][sample_index]
    return {
        "voxel_coords": coords[mask, 1:4],
        "chunk_origin": torch.as_tensor(meta["chunk_min_voxel"], dtype=torch.long),
        "offset": sample_feats[:, 0:3],
        "color": sample_feats[:, 3:6],
        "opacity": sample_feats[:, 6:7],
        "scale": sample_feats[:, 7:10],
        "rotation": sample_feats[:, 10:14],
        "metadata": meta,
    }

def save_gaussians_as_ply(gaussians: dict, path: str):
    """Write reconstructed Gaussians as a 3DGS-compatible Binary PLY file.

    The autoencoder decoder already returns attributes in their natural
    representation (see gaussian_heads.GaussianAttributeDecoder._postprocess):
      * color   ∈ (0, 1)    -- linear RGB
      * opacity ∈ [-10, 10] -- already in logit space (3DGS stores it this way)
      * scale   ∈ R+        -- linear positive scale (3DGS stores log-scale)
      * rotation            -- unit quaternion
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
    #    logit as a probability and produced near-±inf values (so the
    #    reconstructed scene rendered as either fully transparent or fully
    #    opaque garbage).
    opacity = gaussians.get(
        "opacity", torch.full((n_pts, 1), 2.2)  # ~sigmoid(2.2)≈0.9
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

    # 法线补零
    normals = np.zeros_like(positions)

    # 6. 构建 NumPy 结构化数组 (定义各个属性的数据类型)
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
    
    # 7. 填入数据
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

    # 8. 保存为 Binary Little Endian PLY
    os.makedirs(os.path.dirname(path), exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    # text=False 保证输出的是 binary 格式
    PlyData([el], text=False).write(path)

def save_validation_reconstruction(
    raw_model,
    sample: dict,
    cfg: dict,
    device: torch.device,
    ply_path: str,
    image_path: Optional[str] = None,
):
    """Save a validation reconstruction.

    Always writes the predicted Gaussians as a 3DGS-compatible .ply at
    `ply_path`. If `image_path` is given, additionally renders N views of
    GT and predicted Gaussians and saves them as a 2-row PNG (top row =
    GT, bottom row = predicted) for quick eyeballing.
    """
    voxel_coords = sample["voxel_coords"].to(device)
    chunk_origin = sample["chunk_origin"].to(device)
    gaussians = {k: v.to(device) for k, v in sample.items()
                 if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

    pred_gaussians, _, _, _ = raw_model(gaussians, voxel_coords)
    pred_coords = pred_gaussians.pop("_coords", None)
    if pred_coords is not None:
        pred_voxel_coords = pred_coords[:, 1:] if pred_coords.shape[1] == 4 else pred_coords
    else:
        pred_voxel_coords = voxel_coords
    base_voxel_size = float(cfg["data"]["base_voxel_size"])
    pred_abs_voxel_coords = pred_voxel_coords + chunk_origin
    pred_voxel_centers = (pred_abs_voxel_coords.to(pred_gaussians["offset"].dtype) + 0.5) * base_voxel_size
    pred_gaussians["position"] = pred_voxel_centers + pred_gaussians["offset"]
    save_gaussians_as_ply(pred_gaussians, ply_path)

    # ---- Optional: render GT vs. Pred views and save as a side-by-side PNG ----
    if image_path is None or not HAS_RASTERIZER or device.type != "cuda":
        return

    loss_cfg = cfg.get("loss", {})
    n_views = int(loss_cfg.get("n_images", 0))
    img_size = int(loss_cfg.get("render_size", 128))
    if n_views <= 0:
        return

    abs_voxel_coords = voxel_coords + chunk_origin
    voxel_centers = (abs_voxel_coords.to(gaussians["offset"].dtype) + 0.5) * base_voxel_size
    gt_position = voxel_centers + gaussians["offset"]
    bbox_min = gt_position.min(dim=0).values
    bbox_max = gt_position.max(dim=0).values
    # Validation views are deterministic (no jitter) so renderings are
    # directly comparable across val runs.
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
):
    """Run validation over the full validation loader and print average losses."""
    if len(val_loader) == 0:
        print(f"Validation skipped at step {global_step}: val_loader is empty")
        return None

    raw_model.eval()
    total_loss = total_occ = total_lfq = 0.0
    total_rgb = total_perc = 0.0
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
        for batch in val_loader:
            (batch_loss, batch_occ, batch_lfq,
             batch_rgb, batch_perc) = compute_batch_loss(
                raw_model, batch, cfg, device, backward=False,
                perceptual=perceptual, rng=val_rng,
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
        print(f"Saved validation reconstruction: {recon_path}")
        if HAS_RASTERIZER and device.type == "cuda" and int(cfg.get("loss", {}).get("n_images", 0)) > 0:
            print(f"Saved validation renderings:     {image_path}")
    return avg_loss


def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPU(s), device: {device}")

    # ---- Debug toggles (all OFF by default = paper-faithful) ----
    # See README / configs/autoencoder_scene.yaml for the full list.
    debug_cfg = cfg.get("debug", {}) or {}
    data_debug_cfg = cfg.get("data", {}) or {}
    norm_kind = str(debug_cfg.get("norm", "bn")).lower()
    color_act = str(debug_cfg.get("color_activation", "clamp")).lower()
    fixed_chunk = bool(data_debug_cfg.get("fixed_chunk", debug_cfg.get("fixed_chunk", False)))
    no_augment = bool(debug_cfg.get("no_augment", False))
    voxel_dedup = str(debug_cfg.get("voxel_dedup", "random")).lower()
    grad_clip = float(debug_cfg.get("grad_clip", 1.0))
    log_grad_norm_every = int(debug_cfg.get("log_grad_norm_every", 0))
    if any([
        norm_kind != "bn",
        color_act != "clamp",
        fixed_chunk,
        no_augment,
        voxel_dedup != "random",
        grad_clip != 1.0,
        log_grad_norm_every > 0,
    ]):
        print(
            "[debug] norm=" + norm_kind
            + f" color_activation={color_act}"
            + f" fixed_chunk={fixed_chunk} no_augment={no_augment}"
            + f" voxel_dedup={voxel_dedup} grad_clip={grad_clip}"
            + f" log_grad_norm_every={log_grad_norm_every}"
        )

    # Model
    model = GaussianAutoencoder(
        base_ch=cfg["model"]["base_ch"],
        n_down=cfg["model"]["n_down"],
        codebook_size=cfg["model"]["codebook_size"],
        use_sh=cfg["model"].get("use_sh", False),
        voxel_size=cfg["data"]["base_voxel_size"],
        norm=norm_kind,
        color_activation=color_act,
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)

    # Dataset
    from gaussiangpt.autoencoder.data_preprocess.dataset import ASEChunkDataset
    from gaussiangpt.autoencoder.data_preprocess.collate import ase_sparse_collate
    data_cfg = cfg["data"]
    cache_root = args.cache_root or data_cfg.get("cache_root") or args.data_dir
    if not cache_root:
        raise ValueError(
            "ASE dataloader requires data.cache_root or --cache_root "
            "(--data_dir is also accepted as a compatibility fallback)."
        )
    base_voxel_size = float(data_cfg["base_voxel_size"])
    chunk_size_voxels = tuple(data_cfg.get("chunk_size", [160, 160, 160]))
    chunk_size_world = float(
        data_cfg.get("ase_chunk_size", chunk_size_voxels[0] * base_voxel_size)
    )
    train_samples_per_epoch = int(data_cfg.get("train_samples_per_epoch", 1000))
    val_samples_per_epoch = int(data_cfg.get("val_samples_per_epoch", 200))
    occupancy_threshold = float(data_cfg.get("min_occupancy_ae", 0.2))
    max_candidate_chunks = int(data_cfg.get("max_candidate_chunks", 10))
    top_k_cameras = int(data_cfg.get("top_k_cameras", 12))
    z_mode = str(data_cfg.get("z_mode", "fixed_160"))
    preferred_coverage = float(data_cfg.get("preferred_coverage", 0.4))
    train_scene_ids = data_cfg.get("train_scene_ids")
    val_scene_ids = data_cfg.get("val_scene_ids")

    train_dataset = ASEChunkDataset(
        cache_root=cache_root,
        num_samples_per_epoch=train_samples_per_epoch,
        chunk_size=chunk_size_world,
        occupancy_threshold=occupancy_threshold,
        max_candidate_chunks=max_candidate_chunks,
        top_k_cameras=top_k_cameras,
        seed=int(data_cfg.get("seed", 42)),
        z_mode=z_mode,
        preferred_coverage=preferred_coverage,
        scene_ids=train_scene_ids,
        fixed_chunk=fixed_chunk,
    )
    fixed_sample = train_dataset.fixed_sample() if fixed_chunk else None
    
    # 替换验证集实例化（可以共享同一个类，指定不同的场景 id 列表）
    val_dataset = ASEChunkDataset(
        cache_root=cache_root,
        num_samples_per_epoch=val_samples_per_epoch,
        chunk_size=chunk_size_world,
        occupancy_threshold=occupancy_threshold,
        max_candidate_chunks=max_candidate_chunks,
        top_k_cameras=top_k_cameras,
        seed=int(data_cfg.get("val_seed", 4242)),
        z_mode=z_mode,
        preferred_coverage=preferred_coverage,
        scene_ids=val_scene_ids,
        fixed_chunk=fixed_chunk,
        fixed_sample=fixed_sample,
    )
    if fixed_chunk:
        print("[fixed_chunk] enabled")
        summary = train_dataset.fixed_chunk_summary()
        if summary is not None:
            print(f"[fixed_chunk] using {summary}")
        print("[fixed_chunk] validation dataset reuses the training fixed chunk")
    else:
        print("[fixed_chunk] disabled; using online chunk sampling")

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    
    # 关键：将 collate_fn 替换为你写的真正的并行化组装函数 ase_sparse_collate
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=True,
        collate_fn=ase_sparse_collate,  # 替换这里！
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(data_cfg.get("val_num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=ase_sparse_collate,  # 替换这里！
    )
    # data_dir = args.data_dir or cfg["data"]["data_dir"]
    # train_dataset = GaussianSceneDataset(
    #     data_dir=data_dir,
    #     base_voxel_size=cfg["data"]["base_voxel_size"],
    #     n_down=cfg["model"]["n_down"],
    #     chunk_size=tuple(cfg["data"]["chunk_size"]),
    #     min_occupancy=cfg["data"].get("min_occupancy_ae", 0.2),
    #     augment=(not no_augment),
    #     split="train",
    #     fixed_chunk=fixed_chunk,
    #     voxel_dedup=voxel_dedup,
    # )
    # val_dataset = GaussianSceneDataset(
    #     data_dir=data_dir,
    #     base_voxel_size=cfg["data"]["base_voxel_size"],
    #     n_down=cfg["model"]["n_down"],
    #     chunk_size=tuple(cfg["data"]["chunk_size"]),
    #     min_occupancy=cfg["data"].get("min_occupancy_ae", 0.2),
    #     augment=False,
    #     split="val",
    #     # Validation always uses deterministic chunks; voxel dedup tracks
    #     # the train setting so the GT representation stays consistent.
    #     fixed_chunk=fixed_chunk,
    #     voxel_dedup=voxel_dedup,
    # )

    # batch_size = args.batch_size or cfg["training"]["batch_size"]
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True,
    #     num_workers=4, pin_memory=False, drop_last=True,
    #     collate_fn=sparse_collate,
    # )
    # val_loader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=False,
    #     num_workers=2, pin_memory=False,
    #     collate_fn=sparse_collate,
    # )

    # Optimizer
    lr = args.lr or cfg["training"]["lr"]
    optimizer = Adam(model.parameters(), lr=lr)
    epochs = args.epochs or cfg["training"]["epochs"]
    val_every_steps = args.val_every_steps
    if val_every_steps is None:
        val_every_steps = cfg["training"].get("val_every_steps", 0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Resume: 从断点继续训练
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)
    config_save_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # Note: MinkowskiEngine is incompatible with DataParallel's scatter mechanism.
    # Use the unwrapped model for forward; DataParallel is only safe for dense fallback.
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    global_step = 0

    # ---- Per-head decoder diagnostics ----
    # Multi-line readout per `color_check_every` steps:
    #   * colour head pre-activation distribution (clamp/sigmoid aware)
    #   * opacity distribution + decisive fraction (alpha near 0 or 1) --
    #     a low decisive fraction is the classic "soft averaging"
    #     failure mode that turns vivid colours into grey mush.
    #   * scale distribution vs voxel size -- "ballooning" Gaussians
    #     also cause desaturation and detail loss.
    # Set `diagnostics.color_check_every <= 0` in the config to disable.
    diag_cfg = cfg.get("diagnostics", {})
    color_diag = ColorClampDiagnostics(
        raw_model.attr_decoder,
        every=int(diag_cfg.get("color_check_every", 200)),
        voxel_size=float(cfg["data"]["base_voxel_size"]),
    )

    # ---- Optional rendering / perceptual supervision ----
    # Lazily build the VGG perceptual loss only when actually requested,
    # so that disabling it (lambda_perc=0 or n_images=0) avoids loading
    # ~80MB of weights and an extra GPU-side module.
    loss_cfg = cfg.get("loss", {})
    use_render = (
        float(loss_cfg.get("lambda_rgb", 0.0)) > 0.0
        or float(loss_cfg.get("lambda_perc", 0.0)) > 0.0
    ) and int(loss_cfg.get("n_images", 0)) > 0
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
                f"n_views={loss_cfg.get('n_images')} "
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

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, batch_list in enumerate(train_loader):
            # batch_list is a list of per-sample dicts (sparse_collate)
            # Accumulate gradients over the batch manually
            optimizer.zero_grad()
            (batch_loss, batch_occ, batch_lfq,
             batch_rgb, batch_perc) = compute_batch_loss(
                raw_model, batch_list, cfg, device, backward=True,
                perceptual=perceptual,
            )

            # ``clip_grad_norm_`` returns the *pre-clip* total gradient
            # norm; logging it occasionally is one of the cheapest ways
            # to spot the "everything is being clipped" pathology.
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip,
            )
            optimizer.step()
            global_step += 1

            total_loss += batch_loss.item()
            if step % 100 == 0:
            # if True:
                print(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} "
                    f"Loss: {batch_loss.item():.4f} "
                    f"rgb: {batch_rgb:.4f} perc: {batch_perc:.4f} "
                    f"occ: {batch_occ:.4f} lfq: {batch_lfq:.4f}"
                )
            if log_grad_norm_every > 0 and global_step % log_grad_norm_every == 0:
                clipped = float(pre_clip_norm) > grad_clip
                print(
                    f"  [grad] step={global_step} "
                    f"pre_clip_norm={float(pre_clip_norm):.3f} "
                    f"clip={grad_clip:.2f} "
                    f"{'(clipped)' if clipped else ''}"
                )

            # Colour-clamp dead-gradient check. Reads .grad (still alive
            # after optimizer.step), then drops the captured features so
            # the next step doesn't see stale data.
            color_diag.maybe_log(global_step)
            color_diag.clear()

            if val_every_steps > 0 and global_step % val_every_steps == 0:
            # if True:
                validate(
                    raw_model, val_loader, cfg, device, epoch,
                    global_step, args.output_dir, perceptual=perceptual,
                )
                model.train()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % cfg["training"].get("save_every", 10) == 0:
        # if True:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final
    torch.save({
        "epoch": epochs - 1,
        "model": model.state_dict(),
        "config": cfg,
    }, os.path.join(args.output_dir, "final.pt"))
    color_diag.close()
    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)
