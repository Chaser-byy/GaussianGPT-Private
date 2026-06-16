"""Train the GaussianGPT autoencoder.

Usage:
    python scripts/train_autoencoder.py --config configs/autoencoder_scene.yaml

Paper training details:
  - Adam optimizer, lr=1e-4, cosine decay to 10%
  - 4 RTX A6000 GPUs, effective batch size 8 (scenes) / 24 (objects)
  - ~4 days for scenes, ~2 days for PhotoShape
"""
import argparse
import os
import sys

import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiangpt.autoencoder import GaussianAutoencoder
from gaussiangpt.autoencoder.diagnostics import ColorClampDiagnostics
from gaussiangpt.autoencoder.training.config import (
    DebugOptions,
    debug_options,
    load_config,
    log_debug_options,
    model_use_generative_transpose,
)
from gaussiangpt.autoencoder.training.data import build_ase_dataloaders
from gaussiangpt.autoencoder.training.losses import compute_batch_loss
from gaussiangpt.autoencoder.training.render_losses import (
    build_perceptual_loss_if_needed,
    fixed_gt_render_cache,
)
from gaussiangpt.autoencoder.training.validation import validate


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


def build_model(
    cfg: dict,
    debug: DebugOptions,
    device: torch.device,
    n_gpus: int,
) -> nn.Module:
    """Construct the autoencoder and preserve the existing DataParallel behavior."""

    model_cfg = cfg["model"]
    model = GaussianAutoencoder(
        base_ch=model_cfg["base_ch"],
        n_down=model_cfg["n_down"],
        codebook_size=model_cfg["codebook_size"],
        use_sh=model_cfg.get("use_sh", False),
        voxel_size=cfg["data"]["base_voxel_size"],
        norm=debug.norm_kind,
        color_activation=debug.color_act,
        use_generative_transpose=model_use_generative_transpose(cfg),
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
    return model


def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPU(s), device: {device}")

    debug = debug_options(cfg)
    log_debug_options(debug)

    model = build_model(cfg, debug, device, n_gpus)
    loaders = build_ase_dataloaders(cfg, args, debug)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    loss_cfg = loaders.loss_cfg

    lr = args.lr or cfg["training"]["lr"]
    optimizer = Adam(model.parameters(), lr=lr)
    epochs = args.epochs or cfg["training"]["epochs"]
    val_every_steps = args.val_every_steps
    if val_every_steps is None:
        val_every_steps = cfg["training"].get("val_every_steps", 0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Resume from a saved training checkpoint.
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
    train_auto_prune = model_use_generative_transpose(cfg)
    print(
        "[training pruning] "
        f"generative_transpose={train_auto_prune} "
        f"auto_prune={train_auto_prune} "
        f"source={'occ_logits_or_gt_targets' if train_auto_prune else 'disabled'}"
    )

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

    use_render, perceptual = build_perceptual_loss_if_needed(
        loss_cfg, loaders.render_n_views, device
    )
    gt_render_cache = fixed_gt_render_cache(
        loaders.fixed_chunk, use_render, loss_cfg, loaders.camera_mode, device
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, batch_list in enumerate(train_loader):
            optimizer.zero_grad()
            (batch_loss, batch_occ, batch_lfq,
             batch_rgb, batch_perc) = compute_batch_loss(
                raw_model, batch_list, cfg, device, backward=True,
                perceptual=perceptual, gt_render_cache=gt_render_cache,
                global_step=global_step + 1,
            )

            # ``clip_grad_norm_`` returns the *pre-clip* total gradient
            # norm; logging it occasionally is one of the cheapest ways
            # to spot the "everything is being clipped" pathology.
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), debug.grad_clip,
            )
            optimizer.step()
            global_step += 1

            total_loss += batch_loss.item()
            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} "
                    f"Loss: {batch_loss.item():.4f} "
                    f"rgb: {batch_rgb:.4f} perc: {batch_perc:.4f} "
                    f"occ: {batch_occ:.4f} lfq: {batch_lfq:.4f}"
                )
            if debug.log_grad_norm_every > 0 and global_step % debug.log_grad_norm_every == 0:
                clipped = float(pre_clip_norm) > debug.grad_clip
                print(
                    f"  [grad] step={global_step} "
                    f"pre_clip_norm={float(pre_clip_norm):.3f} "
                    f"clip={debug.grad_clip:.2f} "
                    f"{'(clipped)' if clipped else ''}"
                )

            # Colour-clamp dead-gradient check. Reads .grad (still alive
            # after optimizer.step), then drops the captured features so
            # the next step doesn't see stale data.
            color_diag.maybe_log(global_step)
            color_diag.clear()

            if val_every_steps > 0 and global_step % val_every_steps == 0:
                validate(
                    raw_model, val_loader, cfg, device, epoch,
                    global_step, args.output_dir, perceptual=perceptual,
                    gt_render_cache=gt_render_cache,
                )
                model.train()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % cfg["training"].get("save_every", 10) == 0:
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
