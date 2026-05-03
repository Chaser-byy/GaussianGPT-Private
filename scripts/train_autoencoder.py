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

import numpy as np
from plyfile import PlyData, PlyElement

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
    parser.add_argument("--val_every_steps", type=int, default=None)
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


def compute_batch_loss(raw_model, batch_list, cfg: dict, device: torch.device, backward: bool = False):
    """Compute one sparse batch loss, optionally backpropagating per-sample losses."""
    batch_loss = torch.tensor(0.0, device=device)
    batch_attr = batch_occ = batch_lfq = 0.0

    for sample in batch_list:
        voxel_coords = sample["voxel_coords"].to(device)
        gaussians = {k: v.to(device) for k, v in sample.items()
                     if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

        pred_gaussians, occ_list, lfq_loss, indices = raw_model(gaussians, voxel_coords)

        attr_keys = [k for k in pred_gaussians if k in gaussians]
        if attr_keys:
            l_attr = sum(
                torch.nn.functional.l1_loss(pred_gaussians[k], gaussians[k])
                for k in attr_keys
            ) / len(attr_keys)
        else:
            l_attr = torch.tensor(0.0, device=device)

        from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI
        l_occ = torch.tensor(0.0, device=device)
        if occ_list:
            for stage_idx, occ in enumerate(occ_list):
                if HAS_MINKOWSKI:
                    occ_feat = occ.F  # (M, 1)
                    targets = torch.ones(occ_feat.shape[0], device=device)
                else:
                    # Dense: occ is (1, 1, X', Y', Z')
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

        l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)
        sample_loss = (
            cfg["loss"]["lambda_rgb"] * l_attr
            + cfg["loss"]["lambda_occ"] * l_occ
            + cfg["loss"]["lambda_lfq"] * l_lfq
        ) / len(batch_list)

        if backward:
            sample_loss.backward()

        batch_loss = batch_loss + sample_loss.detach()
        batch_attr += l_attr.item() / len(batch_list)
        batch_occ += l_occ.item() / len(batch_list)
        batch_lfq += lfq_loss.item() / len(batch_list)

    return batch_loss, batch_attr, batch_occ, batch_lfq


# def save_gaussians_as_ply(gaussians: dict, path: str):
#     """Write reconstructed Gaussians as a 3DGS-style ASCII PLY file."""
#     positions = gaussians["position"].detach().cpu()
#     colors = gaussians.get("color", torch.full_like(positions, 0.5)).detach().cpu().clamp(0, 1)
#     opacity = gaussians.get("opacity", torch.zeros(len(positions), 1)).detach().cpu()
#     scale = gaussians.get("scale", torch.ones(len(positions), 3)).detach().cpu().clamp_min(1e-8)
#     rotation = gaussians.get("rotation", torch.zeros(len(positions), 4)).detach().cpu()
#     if rotation.numel() > 0:
#         rotation = torch.nn.functional.normalize(rotation, dim=-1)

#     f_dc = (colors - 0.5) / 0.28209479177387814
#     log_scale = torch.log(scale)
#     zeros3 = torch.zeros_like(positions)
#     f_rest = torch.zeros(len(positions), 9)

#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w") as f:
#         f.write("ply\n")
#         f.write("format ascii 1.0\n")
#         f.write(f"element vertex {len(positions)}\n")
#         for name in ("x", "y", "z", "nx", "ny", "nz"):
#             f.write(f"property float {name}\n")
#         for i in range(3):
#             f.write(f"property float f_dc_{i}\n")
#         for i in range(9):
#             f.write(f"property float f_rest_{i}\n")
#         f.write("property float opacity\n")
#         for i in range(3):
#             f.write(f"property float scale_{i}\n")
#         for i in range(4):
#             f.write(f"property float rot_{i}\n")
#         f.write("end_header\n")

#         rows = torch.cat([positions, zeros3, f_dc, f_rest, opacity, log_scale, rotation], dim=1)
#         for row in rows.tolist():
#             f.write(" ".join(f"{v:.8f}" for v in row) + "\n")

def save_gaussians_as_ply(gaussians: dict, path: str):
    """Write reconstructed Gaussians as a 3DGS-compatible Binary PLY file."""
    positions = gaussians["position"].detach().cpu().numpy()
    
    # 1. 处理颜色 (转为 SH0 系数)
    colors = gaussians.get("color", torch.full_like(gaussians["position"], 0.5)).detach().cpu().clamp(0.001, 0.999)
    f_dc = (colors - 0.5) / 0.28209479177387814
    f_dc = f_dc.numpy()

    # 2. 处理 f_rest (补齐标准 3DGS 所需的 45 个 0)
    f_rest = np.zeros((positions.shape[0], 45), dtype=np.float32)

    # 3. 处理不透明度 (需要存入激活前的值，即 Inverse Sigmoid)
    # 假设你的 opacity 目前在 [0, 1] 之间
    opacity = gaussians.get("opacity", torch.full((len(positions), 1), 0.9)).detach().cpu()
    opacity = torch.clamp(opacity, 1e-5, 1 - 1e-5) # 防止 logit 计算出现 inf
    opacity = torch.logit(opacity).numpy() 

    # 4. 处理缩放 (取对数)
    scale = gaussians.get("scale", torch.ones(len(positions), 3)).detach().cpu().clamp_min(1e-8)
    log_scale = torch.log(scale).numpy()

    # 5. 处理旋转 (四元数归一化)
    rotation = gaussians.get("rotation", torch.zeros(len(positions), 4)).detach().cpu()
    # 如果全 0 会导致渲染器出错，默认单位四元数应当是 [1, 0, 0, 0]
    if rotation.numel() > 0:
        rotation = torch.nn.functional.normalize(rotation, dim=-1).numpy()
    else:
        rotation = np.zeros((positions.shape[0], 4), dtype=np.float32)
        rotation[:, 0] = 1.0 # default w=1

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

def save_validation_reconstruction(raw_model, sample: dict, cfg: dict, device: torch.device, path: str):
    """Run a validation sample through the full autoencoder and save the reconstructed 3DGS."""
    voxel_coords = sample["voxel_coords"].to(device)
    chunk_origin = sample["chunk_origin"].to(device)
    gaussians = {k: v.to(device) for k, v in sample.items()
                 if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

    pred_gaussians, _, _, _ = raw_model(gaussians, voxel_coords)
    abs_voxel_coords = voxel_coords + chunk_origin
    voxel_centers = (abs_voxel_coords.to(pred_gaussians["offset"].dtype) + 0.5) * cfg["data"]["base_voxel_size"]
    pred_gaussians["position"] = voxel_centers + pred_gaussians["offset"]
    save_gaussians_as_ply(pred_gaussians, path)


def validate(
    raw_model,
    val_loader,
    cfg: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    output_dir: str,
):
    """Run validation over the full validation loader and print average losses."""
    if len(val_loader) == 0:
        print(f"Validation skipped at step {global_step}: val_loader is empty")
        return None

    raw_model.eval()
    total_loss = total_attr = total_occ = total_lfq = 0.0
    recon_path = os.path.join(
        output_dir,
        "val_reconstructions",
        f"epoch_{epoch:04d}_step_{global_step:08d}.ply",
    )
    saved_reconstruction = False
    with torch.no_grad():
        for batch_list in val_loader:
            batch_loss, batch_attr, batch_occ, batch_lfq = compute_batch_loss(
                raw_model, batch_list, cfg, device, backward=False
            )
            total_loss += batch_loss.item()
            total_attr += batch_attr
            total_occ += batch_occ
            total_lfq += batch_lfq
            if not saved_reconstruction and batch_list:
                save_validation_reconstruction(raw_model, batch_list[0], cfg, device, recon_path)
                saved_reconstruction = True

    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    print(f"Validation Epoch {epoch} Step {global_step} "
          f"Loss: {avg_loss:.4f} attr: {total_attr / n_batches:.4f} "
          f"occ: {total_occ / n_batches:.4f} lfq: {total_lfq / n_batches:.4f}")
    if saved_reconstruction:
        print(f"Saved validation reconstruction: {recon_path}")
    return avg_loss


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

    # Note: MinkowskiEngine is incompatible with DataParallel's scatter mechanism.
    # Use the unwrapped model for forward; DataParallel is only safe for dense fallback.
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    global_step = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, batch_list in enumerate(train_loader):
            # batch_list is a list of per-sample dicts (sparse_collate)
            # Accumulate gradients over the batch manually
            optimizer.zero_grad()
            batch_loss, batch_attr, batch_occ, batch_lfq = compute_batch_loss(
                raw_model, batch_list, cfg, device, backward=True
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            total_loss += batch_loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} "
                      f"Loss: {batch_loss.item():.4f} attr: {batch_attr:.4f} "
                      f"occ: {batch_occ:.4f} lfq: {batch_lfq:.4f}")

            if val_every_steps > 0 and global_step % val_every_steps == 0:
                validate(raw_model, val_loader, cfg, device, epoch, global_step, args.output_dir)
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
    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)
