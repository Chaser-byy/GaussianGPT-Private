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


def _as_batched_coords(coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    coords = coords.to(device=device)
    if coords.dim() == 2 and coords.shape[1] == 3:
        try:
            import MinkowskiEngine as ME
        except ImportError:
            batch_idx = torch.zeros(
                coords.shape[0], 1, dtype=torch.long, device=device
            )
            return torch.cat([batch_idx, coords.to(dtype=torch.long)], dim=1)
        return ME.utils.batched_coordinates(
            [coords.to(dtype=torch.int32)],
            dtype=torch.int32,
            device=device,
        ).to(dtype=torch.long)
    return coords.to(dtype=torch.long)


def _require_minkowski_engine():
    try:
        import MinkowskiEngine as ME
    except ImportError as exc:
        raise RuntimeError(
            "Sparse occupancy target alignment requires MinkowskiEngine."
        ) from exc
    return ME


def _kernel_map_out_indices(mapping, device: torch.device) -> torch.Tensor:
    """Return output row indices from a MinkowskiEngine kernel map."""

    def _out_from_pair(in_out):
        if isinstance(in_out, (list, tuple)) and len(in_out) == 2:
            return in_out[1]
        if torch.is_tensor(in_out) and in_out.dim() == 2:
            if in_out.shape[0] == 2:
                return in_out[1]
            if in_out.shape[1] == 2:
                return in_out[:, 1]
        raise RuntimeError(
            "Unexpected MinkowskiEngine kernel_map return format while "
            "aligning sparse occupancy targets."
        )

    if isinstance(mapping, dict):
        pieces = []
        for in_out in mapping.values():
            out_indices = _out_from_pair(in_out)
            if out_indices.numel() > 0:
                pieces.append(out_indices.to(device=device, dtype=torch.long))
        if not pieces:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(pieces, dim=0).unique()

    if isinstance(mapping, (list, tuple)) and len(mapping) == 2:
        out_indices = mapping[1]
        if torch.is_tensor(out_indices):
            return out_indices.to(device=device, dtype=torch.long)

    raise RuntimeError(
        "Unexpected MinkowskiEngine kernel_map return format while aligning "
        "sparse occupancy targets."
    )


def _coordinate_map_out_indices(
    coord_manager,
    in_key,
    out_key,
    device: torch.device,
) -> torch.Tensor:
    kernel_map = getattr(coord_manager, "kernel_map", None)
    if kernel_map is None:
        kernel_map = getattr(coord_manager, "get_kernel_map", None)
    if kernel_map is None:
        raise RuntimeError(
            "Sparse occupancy target alignment requires "
            "MinkowskiEngine CoordinateManager.kernel_map."
        )
    mapping = kernel_map(in_key, out_key, stride=1, kernel_size=1, dilation=1)
    return _kernel_map_out_indices(mapping, device)


def _sparse_occupancy_targets(
    occ,
    gt_coords: torch.Tensor,
    stage_idx: int,
    n_stages: int,
    device: torch.device,
) -> tuple:
    """Align a decoder occupancy SparseTensor with the current chunk occupancy.

    ``gt_coords`` are base-voxel coordinates from the dataloader, either
    batched ``[b, x, y, z]`` or a single-sample ``[x, y, z]`` tensor. Each
    decoder occupancy head lives at its own tensor stride, so GT occupied
    voxels are downsampled to that stage before matching against ``occ.C``.
    The returned target has exactly one value per occupancy logit.
    """

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

    _require_minkowski_engine()

    gt_stage_coords = _as_batched_coords(gt_coords, device).clone()
    gt_stage_coords[:, 1:] = torch.div(
        gt_stage_coords[:, 1:], stride, rounding_mode="floor"
    )
    targets = torch.zeros_like(occ_logits)
    if gt_stage_coords.numel() == 0:
        return occ_logits, targets, stride

    coord_manager = getattr(occ, "coordinate_manager", None)
    if coord_manager is None or getattr(occ, "coordinate_map_key", None) is None:
        raise RuntimeError(
            "Sparse occupancy target alignment requires MinkowskiEngine "
            "SparseTensor coordinate manager state."
        )

    tensor_stride = [stride] * (gt_stage_coords.shape[1] - 1)
    # Let ME's CoordinateManager own deduplication and coordinate hashing.
    gt_key, (unique_map, _) = coord_manager.insert_and_map(
        gt_stage_coords.to(dtype=torch.int32),
        tensor_stride,
        f"gt_occ_targets_s{stage_idx}_{id(occ)}_{id(gt_stage_coords)}",
    )
    if unique_map.numel() == 0:
        return occ_logits, targets, stride

    occ_indices = _coordinate_map_out_indices(
        coord_manager,
        gt_key,
        occ.coordinate_map_key,
        device,
    )
    if occ_indices.numel() > 0:
        targets[occ_indices.to(device=device, dtype=torch.long)] = 1
    return occ_logits, targets, stride


def make_gt_prune_mask_fn(
    gt_coords: torch.Tensor,
    device: torch.device,
) -> tuple:
    """Build a decoder-stage keep-mask callback from GT occupancy coordinates."""

    gt_coords = _as_batched_coords(gt_coords, device)
    occ_target_cache = {}
    gt_prune_debug = []

    def gt_prune_mask_fn(occ, stage_idx: int, n_stages: int) -> torch.Tensor:
        occ_logits, targets, stride = _sparse_occupancy_targets(
            occ, gt_coords, stage_idx, n_stages, device
        )
        # Prune from GT occupancy targets only; occ_logits stay cached for L_occ.
        keep = targets.to(dtype=torch.bool)
        occ_target_cache[stage_idx] = (occ_logits, targets, stride)
        gt_prune_debug.append(
            {
                "stage": stage_idx,
                "stride": stride,
                "pre": int(targets.numel()),
                "post": int(keep.sum().detach().item()),
            }
        )
        return keep

    return gt_prune_mask_fn, occ_target_cache, gt_prune_debug


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
    decoder_prune: bool = False,
    occupancy_threshold: float = 0.5,
    prune_min_keep: int = 1,
    log_pruning: bool = False,
    log_prefix: str = "validation",
    train_prune_with_gt_logits: bool = False,
    gt_prune_with_gt_logits: bool = False,
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

    gt_prune_enabled = bool(train_prune_with_gt_logits or gt_prune_with_gt_logits)
    if gt_prune_enabled and not HAS_MINKOWSKI:
        raise RuntimeError(
            "GT decoder pruning requires MinkowskiEngine sparse decoder coordinates "
            "so GT occupancy targets can be aligned per stage."
        )

    if gt_prune_enabled:
        gt_prune_mask_fn, occ_target_cache, gt_prune_debug = make_gt_prune_mask_fn(
            coords, device
        )
    else:
        gt_prune_mask_fn = None
        occ_target_cache = {}
        gt_prune_debug = []

    decoder_prune_enabled = bool(decoder_prune or gt_prune_enabled)

    # Forward the full sparse batch through the model.
    pred_gaussians, occ_list, lfq_loss, indices = raw_model(
        gaussians,
        coords,
        prune=decoder_prune_enabled,
        occupancy_threshold=occupancy_threshold,
        min_keep=prune_min_keep,
        prune_mask_fn=gt_prune_mask_fn,
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
            decoder_prune_enabled,
            log_prefix,
        )
    if gt_prune_enabled and gt_prune_debug and _should_log_gt_prune(cfg, global_step):
        pieces = [
            (
                f"s{item['stage']}:stride={item['stride']} "
                f"pre_voxels={item['pre']} post_voxels={item['post']}"
            )
            for item in gt_prune_debug
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
            cached_targets = occ_target_cache.get(stage_idx)
            if cached_targets is None:
                occ_logits, targets, stride = _sparse_occupancy_targets(
                    occ, coords, stage_idx, len(occ_list), device
                )
            else:
                occ_logits, targets, stride = cached_targets
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
