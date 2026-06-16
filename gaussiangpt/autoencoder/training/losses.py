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


def _loss_weights(cfg: dict) -> tuple[float, float, float, float]:
    """Return the four scalar weights used in GaussianGPT Eq. (1)."""

    loss_cfg = cfg.get("loss", {})
    return (
        float(loss_cfg.get("lambda_rgb", 0.0)),
        float(loss_cfg.get("lambda_perc", 0.0)),
        float(loss_cfg.get("lambda_occ", 0.0)),
        float(loss_cfg.get("lambda_lfq", 0.0)),
    )


def _should_log_gt_prune(cfg: dict, global_step: Optional[int]) -> bool:
    if global_step is None:
        return False
    if int(global_step) == 1:
        return True
    debug_cfg = cfg.get("debug", {}) or {}
    every = int(debug_cfg.get("gt_prune_check_every", debug_cfg.get("occ_check_every", 0)))
    return every > 0 and int(global_step) % every == 0


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
    gt_prune: bool = False,
    occ_head_prune: bool = False,
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

    gt_prune_enabled = bool(gt_prune)
    occ_head_prune_enabled = bool(occ_head_prune)
    if gt_prune_enabled and not HAS_MINKOWSKI:
        raise RuntimeError(
            "GT decoder pruning requires MinkowskiEngine sparse decoder coordinates "
            "so GT occupancy targets can be aligned per stage."
        )

    occ_target_cache = {} if HAS_MINKOWSKI else None

    pruning_enabled = bool(gt_prune_enabled or occ_head_prune_enabled)

    # Forward the full sparse batch through the model.
    pred_gaussians, occ_list, lfq_loss, indices = raw_model(
        gaussians,
        coords,
        occupancy_threshold=occupancy_threshold,
        min_keep=prune_min_keep,
        gt_prune=gt_prune_enabled,
        occ_head_prune=occ_head_prune_enabled,
        occ_target_cache=occ_target_cache,
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
            pruning_enabled,
            log_prefix,
        )
    if (
        gt_prune_enabled
        and occ_target_cache
        and _should_log_gt_prune(cfg, global_step)
    ):
        pieces = [
            (
                f"s{stage_idx}:stride={stride} "
                f"pre_voxels={int(targets.numel())} "
                f"post_voxels={int(targets.sum().detach().item())}"
            )
            for stage_idx, (_occ_logits, targets, stride)
            in sorted(occ_target_cache.items())
        ]
        print(
            f"  [gt prune] step={global_step} "
            f"enabled=True source=gt_occ_targets "
            f"gt_voxels={int(coords.shape[0])} "
            + " | ".join(pieces)
        )

    # ---- L_occ: BCE on sparse occupancy logits aligned to GT occupancy ----
    l_occ = torch.tensor(0.0, device=device)
    occ_debug = []
    occ_stage_count = 0
    if occ_list:
        for stage_idx, occ in enumerate(occ_list):
            cached_targets = (
                occ_target_cache.get(stage_idx)
                if occ_target_cache is not None
                else None
            )
            if cached_targets is None:
                raise RuntimeError(
                    "Decoder did not cache sparse occupancy targets for "
                    f"stage {stage_idx}. L_occ requires the raw input "
                    "coordinate_map_key from the current model forward."
                )
            occ_logits, targets, stride = cached_targets
            if targets.numel() != occ_logits.numel():
                raise ValueError(
                    "Cached occupancy targets length must match occupancy logits: "
                    f"got {targets.numel()} targets for {occ_logits.numel()} logits "
                    f"at stage {stage_idx}."
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
