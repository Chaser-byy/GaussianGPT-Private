"""Batch loss computation for AE training."""

from typing import Optional

import torch
import torch.nn as nn

from gaussiangpt.autoencoder.training.gaussian_features import gaussian_features_to_attrs
from gaussiangpt.autoencoder.training.render_losses import render_losses_for_batch


def _count_coords_by_sample(coords: Optional[torch.Tensor], n_samples: int) -> list:
    if coords is None or coords.numel() == 0:
        return [0 for _ in range(n_samples)]
    if coords.dim() == 2 and coords.shape[1] >= 4:
        batch_index = coords[:, 0].detach().long().cpu()
        counts = torch.bincount(batch_index, minlength=n_samples)
        return [int(v) for v in counts[:n_samples].tolist()]
    return [int(coords.shape[0])] + [0 for _ in range(max(n_samples - 1, 0))]


def _decoder_pre_prune_counts(occ_list: list, fallback_counts: list, n_samples: int) -> list:
    if not occ_list:
        return fallback_counts
    last_occ = occ_list[-1]
    if hasattr(last_occ, "C"):
        return _count_coords_by_sample(last_occ.C, n_samples)
    if torch.is_tensor(last_occ) and last_occ.dim() >= 5:
        batch_count = min(int(last_occ.shape[0]), n_samples)
        counts = [int(last_occ[i].numel()) for i in range(batch_count)]
        counts.extend([0 for _ in range(n_samples - batch_count)])
        return counts
    return fallback_counts


def _log_validation_pruning_counts(
    batch: dict,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    occ_list: list,
    prune: bool,
    log_prefix: str,
) -> None:
    n_samples = len(batch.get("metas", [])) or 1
    gt_counts = _count_coords_by_sample(gt_coords, n_samples)
    post_counts = _count_coords_by_sample(pred_coords, n_samples)
    pre_counts = _decoder_pre_prune_counts(occ_list, post_counts, n_samples) if prune else post_counts
    mode = "pruned" if prune else "unpruned"
    for sample_idx in range(n_samples):
        print(
            f"  [{log_prefix} pruning] sample={sample_idx} "
            f"mode={mode} gt_voxels={gt_counts[sample_idx]} "
            f"pre_voxels={pre_counts[sample_idx]} post_voxels={post_counts[sample_idx]} "
            f"pre_gaussians={pre_counts[sample_idx]} post_gaussians={post_counts[sample_idx]}"
        )


def _coord_hash(coords: torch.Tensor) -> torch.Tensor:
    """Build collision-free integer hashes for a small set of 4D sparse coords."""

    coords = coords.to(torch.long)
    mins = coords.min(dim=0).values
    shifted = coords - mins
    dims = shifted.max(dim=0).values + 1
    key = shifted[:, 0]
    for dim in range(1, shifted.shape[1]):
        key = key * dims[dim] + shifted[:, dim]
    return key


def _sparse_occupancy_targets(
    occ,
    gt_coords: torch.Tensor,
    stage_idx: int,
    n_stages: int,
    device: torch.device,
) -> tuple:
    """Align a decoder occupancy SparseTensor with the current chunk occupancy.

    ``gt_coords`` are batched base-voxel coordinates ``[b, x, y, z]`` from the
    dataloader. Each decoder occupancy head lives at its own tensor stride, so
    GT occupied voxels are downsampled to that stage before matching against
    ``occ.C``. The returned target has exactly one value per occupancy logit.
    """

    occ_coords = occ.C.to(device=device, dtype=torch.long)
    occ_logits = occ.F.squeeze(-1)
    tensor_stride = getattr(occ, "tensor_stride", None)
    if tensor_stride is None:
        stride = 2 ** max(n_stages - stage_idx - 1, 0)
    elif isinstance(tensor_stride, (list, tuple)):
        stride = int(tensor_stride[0])
    else:
        stride = int(tensor_stride)
    stride = max(stride, 1)

    if occ_logits.numel() == 0:
        return occ_logits, torch.empty_like(occ_logits), stride

    gt_stage_coords = gt_coords.to(device=device, dtype=torch.long).clone()
    gt_stage_coords[:, 1:] = torch.div(
        gt_stage_coords[:, 1:], stride, rounding_mode="floor"
    )
    gt_stage_coords = torch.unique(gt_stage_coords, dim=0)

    all_coords = torch.cat([occ_coords, gt_stage_coords], dim=0)
    all_hash = _coord_hash(all_coords)
    occ_hash = all_hash[: occ_coords.shape[0]]
    gt_hash = torch.unique(all_hash[occ_coords.shape[0]:])
    targets = torch.isin(occ_hash, gt_hash).to(dtype=occ_logits.dtype)
    return occ_logits, targets, stride


def _loss_weights(cfg: dict) -> tuple[float, float, float, float]:
    """Return the four scalar weights used in GaussianGPT Eq. (1)."""

    loss_cfg = cfg.get("loss", {})
    return (
        float(loss_cfg.get("lambda_rgb", 0.0)),
        float(loss_cfg.get("lambda_perc", 0.0)),
        float(loss_cfg.get("lambda_occ", 0.0)),
        float(loss_cfg.get("lambda_lfq", 0.0)),
    )


def compute_batch_loss(
    raw_model,
    batch,
    cfg: dict,
    device: torch.device,
    backward: bool = False,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
    gt_render_cache: Optional[dict] = None,
    global_step: Optional[int] = None,
    decoder_prune: bool = False,
    occupancy_threshold: float = 0.5,
    prune_min_keep: int = 1,
    log_pruning: bool = False,
    log_prefix: str = "validation",
):
    """Compute one loss step for an ASE batch collated by ase_sparse_collate."""

    from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI

    lambda_rgb, lambda_perc, lambda_occ, lambda_lfq = _loss_weights(cfg)

    coords = batch["coords"].to(device)       # (N_total, 4): [b, x, y, z]
    feats = batch["feats"].to(device)         # (N_total, 14)
    if not HAS_MINKOWSKI and coords.shape[1] == 4:
        raise RuntimeError(
            "The ASE dataloader produces batched sparse coordinates and requires "
            "MinkowskiEngine. Dense fallback only supports the legacy single-sample "
            "3D-coordinate path."
        )

    gaussians = gaussian_features_to_attrs(feats)

    # Forward the full sparse batch through the model.
    pred_gaussians, occ_list, lfq_loss, indices = raw_model(
        gaussians,
        coords,
        prune=decoder_prune,
        occupancy_threshold=occupancy_threshold,
        min_keep=prune_min_keep,
    )
    pred_coords = pred_gaussians.pop("_coords", None)
    if pred_coords is None:
        pred_coords = coords
    if log_pruning:
        _log_validation_pruning_counts(
            batch,
            coords,
            pred_coords,
            occ_list,
            decoder_prune,
            log_prefix,
        )

    # ---- L_occ: BCE on sparse occupancy logits aligned to GT occupancy ----
    l_occ = torch.tensor(0.0, device=device)
    occ_debug = []
    occ_stage_count = 0
    if occ_list:
        for stage_idx, occ in enumerate(occ_list):
            occ_logits, targets, stride = _sparse_occupancy_targets(
                occ, coords, stage_idx, len(occ_list), device
            )
            if occ_logits.numel() == 0:
                continue
            stage_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                occ_logits, targets
            )
            l_occ = l_occ + stage_loss
            occ_stage_count += 1
            positives = int(targets.sum().detach().item())
            total = int(targets.numel())
            occ_debug.append(
                {
                    "stage": stage_idx,
                    "stride": stride,
                    "logits": total,
                    "positives": positives,
                    "ratio": float(positives / max(total, 1)),
                    "loss": float(stage_loss.detach().item()),
                }
            )
        if occ_stage_count > 0:
            l_occ = l_occ / occ_stage_count
    occ_log_every = int(cfg.get("debug", {}).get("occ_check_every", 0))
    if occ_log_every > 0 and global_step is not None and occ_debug:
        if global_step == 1 or global_step % occ_log_every == 0:
            pieces = [
                (
                    f"s{item['stage']}:stride={item['stride']} "
                    f"logits={item['logits']} pos={item['positives']} "
                    f"ratio={item['ratio']:.3f} loss={item['loss']:.4f}"
                )
                for item in occ_debug
            ]
            print(
                f"  [occ target] step={global_step} "
                f"gt_voxels={int(coords.shape[0])} "
                + " | ".join(pieces)
            )

    # ---- L_LFQ: quantization entropy term ----
    l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)

    # ---- L_RGB / L_perc: sparse forward is batched, rendering stays per sample ----
    l_rgb = torch.tensor(0.0, device=device)
    l_perc = torch.tensor(0.0, device=device)

    if lambda_rgb > 0 or lambda_perc > 0:
        l_rgb, l_perc = render_losses_for_batch(
            batch,
            coords,
            pred_coords,
            pred_gaussians,
            gaussians,
            cfg,
            device,
            perceptual=perceptual,
            rng=rng,
            gt_render_cache=gt_render_cache,
            global_step=global_step,
        )

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
        float(l_perc.detach().item()),
    )
