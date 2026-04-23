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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiangpt.autoencoder import GaussianAutoencoder
from gaussiangpt.data import GaussianSceneDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder_scene.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/autoencoder")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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


def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPU(s), device: {device}")

    # Model
    model = GaussianAutoencoder(
        base_ch=cfg["model"]["base_ch"],
        n_down=cfg["model"]["n_down"],
        codebook_size=cfg["model"]["codebook_size"],
        use_sh=cfg["model"].get("use_sh", False),
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)

    # Dataset
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    train_dataset = GaussianSceneDataset(
        data_dir=data_dir,
        base_voxel_size=cfg["data"]["base_voxel_size"],
        n_down=cfg["model"]["n_down"],
        chunk_size=tuple(cfg["data"]["chunk_size"]),
        min_occupancy=cfg["data"].get("min_occupancy_ae", 0.2),
        augment=True,
        split="train",
    )
    val_dataset = GaussianSceneDataset(
        data_dir=data_dir,
        base_voxel_size=cfg["data"]["base_voxel_size"],
        n_down=cfg["model"]["n_down"],
        chunk_size=tuple(cfg["data"]["chunk_size"]),
        min_occupancy=cfg["data"].get("min_occupancy_ae", 0.2),
        augment=False,
        split="val",
    )

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True,
        collate_fn=sparse_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False,
        collate_fn=sparse_collate,
    )

    # Optimizer
    lr = args.lr or cfg["training"]["lr"]
    optimizer = Adam(model.parameters(), lr=lr)
    epochs = args.epochs or cfg["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Note: MinkowskiEngine is incompatible with DataParallel's scatter mechanism.
    # Use the unwrapped model for forward; DataParallel is only safe for dense fallback.
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, batch_list in enumerate(train_loader):
            # batch_list is a list of per-sample dicts (sparse_collate)
            # Accumulate gradients over the batch manually
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            batch_attr = batch_occ = batch_lfq = 0.0

            for sample in batch_list:
                voxel_coords = sample["voxel_coords"].to(device)
                gaussians = {k: v.to(device) for k, v in sample.items()
                             if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

                # Forward pass
                pred_gaussians, occ_list, lfq_loss, indices = raw_model(gaussians, voxel_coords)

                # Attribute reconstruction loss (L1 per attribute)
                attr_keys = [k for k in pred_gaussians if k in gaussians]
                if attr_keys:
                    l_attr = sum(
                        torch.nn.functional.l1_loss(pred_gaussians[k], gaussians[k])
                        for k in attr_keys
                    ) / len(attr_keys)
                else:
                    l_attr = torch.tensor(0.0, device=device)

                # Occupancy loss
                from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI
                l_occ = torch.tensor(0.0, device=device)
                if occ_list:
                    for stage_idx, occ in enumerate(occ_list):
                        if HAS_MINKOWSKI:
                            occ_feat = occ.F  # (M, 1)
                            targets = torch.ones(occ_feat.shape[0], device=device)
                        else:
                            # Dense: occ is (1, 1, X', Y', Z')
                            # Remaining upsampling factor at this stage
                            scale = 2 ** (len(occ_list) - stage_idx)
                            occ_flat = occ.view(-1)
                            gt_grid = torch.zeros_like(occ_flat)
                            stage_vc = (voxel_coords // scale).clamp(0, occ.shape[2] - 1)
                            flat_idx = (
                                stage_vc[:, 0] * occ.shape[3] * occ.shape[4]
                                + stage_vc[:, 1] * occ.shape[4]
                                + stage_vc[:, 2]
                            )
                            gt_grid.scatter_(0, flat_idx, 1.0)
                            occ_feat = occ_flat.unsqueeze(-1)
                            targets = gt_grid
                        l_occ = l_occ + torch.nn.functional.binary_cross_entropy_with_logits(
                            occ_feat.squeeze(-1), targets
                        )
                    l_occ = l_occ / len(occ_list)

                # LFQ entropy loss with softplus offset (paper: λ_LFQ * softplus(L_LFQ + 5))
                l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)

                sample_loss = (
                    cfg["loss"]["lambda_rgb"] * l_attr
                    + cfg["loss"]["lambda_occ"] * l_occ
                    + cfg["loss"]["lambda_lfq"] * l_lfq
                ) / len(batch_list)

                sample_loss.backward()
                batch_loss = batch_loss + sample_loss.detach()
                batch_attr += l_attr.item() / len(batch_list)
                batch_occ += l_occ.item() / len(batch_list)
                batch_lfq += lfq_loss.item() / len(batch_list)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} "
                      f"Loss: {batch_loss.item():.4f} attr: {batch_attr:.4f} "
                      f"occ: {batch_occ:.4f} lfq: {batch_lfq:.4f}")

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
    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)
