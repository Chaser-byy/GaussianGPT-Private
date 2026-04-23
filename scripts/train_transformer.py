"""Train the GaussianGPT transformer (GPT stage).

Usage:
    python scripts/train_transformer.py --config configs/transformer_scene.yaml

Paper training details:
  - AdamW + Muon optimizer with per-module learning rates
  - 4 GH200 GPUs, effective batch size 64
  - ~1 day for scenes, ~4.5 hours for PhotoShape
  - Context window: 16384 (scenes), 8192 (objects)
  - Temperature 0.9, Nucleus sampling p=0.9 at inference
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiangpt.autoencoder import GaussianAutoencoder
from gaussiangpt.transformer import GaussianGPT
from gaussiangpt.data import GaussianSceneDataset, TokenizedSceneDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer_scene.yaml")
    parser.add_argument("--ae_checkpoint", type=str, required=True,
                        help="Path to trained autoencoder checkpoint")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/transformer")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer_groups(model: GaussianGPT, cfg: dict):
    """Build per-module optimizer groups following nanochat/paper config.

    Paper Table 3:
      Output Heads:    AdamW, lr=0.004 * scale
      Embeddings:      AdamW, lr=0.2 * scale
      Residual Weights: AdamW, lr=0.005
      Input Skip Weights: AdamW, lr=0.5
      Others (Muon):   Muon, lr=0.02
    """
    lr_scale = cfg.get("lr_scale", 1.0)

    output_head_params = []
    embedding_params = []
    residual_params = []
    skip_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "pos_head" in name or "feat_head" in name:
            output_head_params.append(param)
        elif "pos_emb" in name or "feat_emb" in name:
            embedding_params.append(param)
        elif "res_scale" in name:
            residual_params.append(param)
        elif "skip_" in name:
            skip_params.append(param)
        else:
            other_params.append(param)

    adamw_groups = [
        {"params": output_head_params, "lr": 0.004 * lr_scale},
        {"params": embedding_params, "lr": 0.2 * lr_scale},
        {"params": residual_params, "lr": 0.005},
        {"params": skip_params, "lr": 0.5},
    ]

    # Note: Muon optimizer requires external implementation
    # Fallback to AdamW for "other" params
    adamw_groups.append({"params": other_params, "lr": 0.02})

    optimizer = AdamW(
        adamw_groups,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.025,
    )
    return optimizer


def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPU(s)")

    # Load autoencoder (frozen)
    ae_cfg = cfg["autoencoder"]
    autoencoder = GaussianAutoencoder(
        base_ch=ae_cfg["base_ch"],
        n_down=ae_cfg["n_down"],
        codebook_size=ae_cfg["codebook_size"],
        use_sh=ae_cfg.get("use_sh", False),
    ).to(device)
    ckpt = torch.load(args.ae_checkpoint, map_location=device)
    autoencoder.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)
    print(f"Loaded autoencoder from {args.ae_checkpoint}")

    # GPT model — operates on latent-space tokens
    # chunk_size in config is in base voxels; latent chunk = chunk_size / 2^n_down
    gpt_cfg = cfg["model"]
    chunk_size = tuple(cfg["data"]["chunk_size"])
    n_down = ae_cfg["n_down"]
    latent_chunk_size = tuple(c // (2 ** n_down) for c in chunk_size)
    model = GaussianGPT(
        size=gpt_cfg["size"],
        context_len=gpt_cfg["context_len"],
        chunk_size=latent_chunk_size,
        codebook_size=ae_cfg["codebook_size"],
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)

    # Dataset
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    train_dataset = TokenizedSceneDataset(
        data_dir=data_dir,
        autoencoder=autoencoder,
        base_voxel_size=cfg["data"]["base_voxel_size"],
        n_down=ae_cfg["n_down"],
        chunk_size=chunk_size,
        min_occupancy=cfg["data"].get("min_occupancy_gpt", 0.3),
        augment=True,
        split="train",
        device=device,
    )
    val_dataset = TokenizedSceneDataset(
        data_dir=data_dir,
        autoencoder=autoencoder,
        base_voxel_size=cfg["data"]["base_voxel_size"],
        n_down=ae_cfg["n_down"],
        chunk_size=chunk_size,
        min_occupancy=cfg["data"].get("min_occupancy_gpt", 0.3),
        augment=False,
        split="val",
        device=device,
    )

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Optimizer
    optimizer = build_optimizer_groups(
        model.module if hasattr(model, "module") else model,
        cfg["training"],
    )
    epochs = args.epochs or cfg["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

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

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            tokens = batch["tokens"].to(device)       # (B, T)
            coords = batch["coords"].to(device)       # (B, T, 3)
            token_type = batch["token_type"].to(device)  # (B, T)

            # Forward: predict next token (forward returns pos_logits, feat_logits, loss)
            logits_pos, logits_feat, _ = model(tokens[:, :-1], coords[:, :-1], token_type[:, :-1])

            # Compute cross-entropy loss with vocabulary masking
            m = model.module if hasattr(model, "module") else model
            loss = m.compute_loss(logits_pos, logits_feat, tokens[:, 1:], token_type[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                coords = batch["coords"].to(device)
                token_type = batch["token_type"].to(device)
                logits_pos, logits_feat, _ = model(tokens[:, :-1], coords[:, :-1], token_type[:, :-1])
                m = model.module if hasattr(model, "module") else model
                loss = m.compute_loss(logits_pos, logits_feat, tokens[:, 1:], token_type[:, 1:])
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch}: train_CE={avg_loss:.4f}, val_CE={val_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
        }, save_path)

        # Keep latest symlink
        latest = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(save_path), latest)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)
