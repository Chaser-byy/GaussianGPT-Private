"""ASE dataset and dataloader construction for AE training."""

from dataclasses import dataclass

from torch.utils.data import DataLoader

from gaussiangpt.autoencoder.training.config import (
    DebugOptions,
    camera_dataset_type,
    camera_preferred_coverage,
    camera_sampling_config,
    camera_score_key,
    effective_render_view_count,
    validate_camera_sampling_config,
)


@dataclass
class DataLoaders:
    """ASE dataloaders plus the render/camera values derived while building them."""

    train_loader: DataLoader
    val_loader: DataLoader
    loss_cfg: dict
    camera_mode: str
    render_n_views: int
    fixed_chunk: bool


def build_ase_dataloaders(cfg: dict, args, debug: DebugOptions) -> DataLoaders:
    """Build ASE train/validation datasets and dataloaders from config values."""

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
    loss_cfg = cfg.get("loss", {})
    camera_cfg, camera_mode = camera_sampling_config(cfg)
    validate_camera_sampling_config(cfg)
    camera_dataset = camera_dataset_type(cfg) if camera_mode == "scoring" else "n/a"
    render_n_views = effective_render_view_count(cfg, loss_cfg)
    include_camera_matrices = camera_mode == "scoring"
    # ASE scoring evaluates every real camera; top_k/num_candidates remain only
    # for legacy orbit-era configs and do not restrict scoring selection.
    sampler_top_k_cameras = 0 if camera_mode == "scoring" else top_k_cameras
    candidate_label = "all" if sampler_top_k_cameras <= 0 else str(sampler_top_k_cameras)
    print(
        "[camera sampling] "
        f"mode={camera_mode} dataset_type={camera_dataset} "
        f"render_views={render_n_views} "
        f"candidate_cameras={candidate_label} "
        f"score_key={camera_score_key(cfg)} "
        f"temperature={camera_cfg.get('temperature', 1.0)} "
        f"preferred_coverage={camera_preferred_coverage(cfg)} "
        f"selection=preferred_then_any_overlap_then_uniform"
    )

    train_dataset = ASEChunkDataset(
        cache_root=cache_root,
        num_samples_per_epoch=train_samples_per_epoch,
        chunk_size=chunk_size_world,
        occupancy_threshold=occupancy_threshold,
        max_candidate_chunks=max_candidate_chunks,
        top_k_cameras=sampler_top_k_cameras,
        seed=int(data_cfg.get("seed", 42)),
        z_mode=z_mode,
        preferred_coverage=preferred_coverage,
        include_camera_matrices=include_camera_matrices,
        scene_ids=train_scene_ids,
        fixed_chunk=debug.fixed_chunk,
    )
    fixed_sample = train_dataset.fixed_sample() if debug.fixed_chunk else None

    # Validation uses the same ASE chunk dataset, optionally with a distinct scene list.
    val_dataset = ASEChunkDataset(
        cache_root=cache_root,
        num_samples_per_epoch=val_samples_per_epoch,
        chunk_size=chunk_size_world,
        occupancy_threshold=occupancy_threshold,
        max_candidate_chunks=max_candidate_chunks,
        top_k_cameras=sampler_top_k_cameras,
        seed=int(data_cfg.get("val_seed", 4242)),
        z_mode=z_mode,
        preferred_coverage=preferred_coverage,
        include_camera_matrices=include_camera_matrices,
        scene_ids=val_scene_ids,
        fixed_chunk=debug.fixed_chunk,
        fixed_sample=fixed_sample,
    )
    if debug.fixed_chunk:
        print("[fixed_chunk] enabled")
        summary = train_dataset.fixed_chunk_summary()
        if summary is not None:
            print(f"[fixed_chunk] using {summary}")
        print("[fixed_chunk] validation dataset reuses the training fixed chunk")
    else:
        print("[fixed_chunk] disabled; using online chunk sampling")

    batch_size = args.batch_size or cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=True,
        collate_fn=ase_sparse_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(data_cfg.get("val_num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=ase_sparse_collate,
    )
    return DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_cfg=loss_cfg,
        camera_mode=camera_mode,
        render_n_views=render_n_views,
        fixed_chunk=debug.fixed_chunk,
    )
