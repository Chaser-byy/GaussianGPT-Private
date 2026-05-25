#!/usr/bin/env python3
"""Render every ASE camera for scenes under data/cache/ASE_overfit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gaussiangpt.autoencoder.data_preprocess.ase import (  # noqa: E402
    ASESceneRecord,
    discover_ase_scenes,
    load_ase_scene_gaussians,
    read_ase_cameras,
)
from gaussiangpt.utils.rendering import (  # noqa: E402
    make_camera_from_world_to_camera,
    render_gaussians,
)


SH_C0 = 0.28209479177387814


def _scene_records(root: Path) -> list[ASESceneRecord]:
    direct_records = discover_ase_scenes(root)
    if direct_records:
        return direct_records

    # Also support passing one scene directory directly.
    if (root / "transforms_train.json").is_file() and (root / "ckpts").is_dir():
        parent_records = discover_ase_scenes(root.parent)
        return [record for record in parent_records if record.scene_dir == root]
    return []


def _iter_limited(items: Iterable, limit: int | None):
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            break
        yield item


def _camera_fovs(width: int, height: int, fx: float, fy: float) -> tuple[float, float]:
    fovx = 2.0 * math.atan(float(width) / (2.0 * float(fx)))
    fovy = 2.0 * math.atan(float(height) / (2.0 * float(fy)))
    return fovx, fovy


def _gaussian_pack(
    record: ASESceneRecord,
    device: torch.device,
    *,
    color_mode: str,
    scale_mode: str,
    opacity_mode: str,
) -> dict[str, torch.Tensor]:
    scene = load_ase_scene_gaussians(record)

    color_np = scene.color
    if color_mode == "sh-dc":
        color_np = np.clip(color_np * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
    elif color_mode == "raw-rgb":
        color_np = np.clip(color_np, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"unknown color_mode={color_mode!r}")

    scale_np = scene.scale
    if scale_mode == "log":
        scale_np = np.exp(scale_np)
    elif scale_mode != "linear":
        raise ValueError(f"unknown scale_mode={scale_mode!r}")
    scale_np = np.maximum(scale_np, 1e-8).astype(np.float32)

    opacity_np = scene.opacity
    if opacity_mode == "activated":
        opacity_np = np.log(
            np.clip(opacity_np, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - opacity_np, 1e-6, 1.0)
        )
    elif opacity_mode != "logit":
        raise ValueError(f"unknown opacity_mode={opacity_mode!r}")
    opacity_np = opacity_np.astype(np.float32)

    rotation_np = scene.rotation.astype(np.float32)
    rotation_np = rotation_np / (
        np.linalg.norm(rotation_np, axis=1, keepdims=True) + 1e-8
    )

    pack = {
        "position": torch.from_numpy(scene.xyz.astype(np.float32)).to(device),
        "scale": torch.from_numpy(scale_np).to(device),
        "rotation": torch.from_numpy(rotation_np).to(device),
        "opacity": torch.from_numpy(opacity_np).to(device),
        "color": torch.from_numpy(color_np).to(device),
    }
    return pack


def _save_image(image: torch.Tensor, path: Path) -> None:
    image = image.detach().clamp(0.0, 1.0).cpu()
    array = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def render_scene(
    record: ASESceneRecord,
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    if not record.valid:
        return {
            "scene_id": record.scene_id,
            "status": "skipped",
            "warnings": record.warnings,
        }

    cameras = read_ase_cameras(record.transforms_path, scene_id=record.scene_id)
    if len(cameras.frames) == 0:
        return {
            "scene_id": record.scene_id,
            "status": "skipped",
            "warnings": ["no camera frames"],
        }

    gaussians = _gaussian_pack(
        record,
        device,
        color_mode=args.color_mode,
        scale_mode=args.scale_mode,
        opacity_mode=args.opacity_mode,
    )

    width = int(args.width or cameras.width)
    height = int(args.height or cameras.height)
    fovx, fovy = _camera_fovs(cameras.width, cameras.height, cameras.fx, cameras.fy)
    bg = torch.tensor(args.background, dtype=torch.float32, device=device)

    scene_output = output_root / record.scene_id
    rendered = 0
    with torch.no_grad():
        for frame in _iter_limited(cameras.frames, args.max_frames):
            frame_index = int(frame["frame_index"])
            camera_id = str(frame["camera_id"])
            w2c = torch.from_numpy(frame["w2c"]).to(device=device, dtype=torch.float32)
            cam = make_camera_from_world_to_camera(
                w2c,
                fovx=fovx,
                fovy=fovy,
                image_height=height,
                image_width=width,
                znear=args.znear,
                zfar=args.zfar,
            )
            image = render_gaussians(gaussians, cam, bg_color=bg)
            _save_image(image, scene_output / f"{frame_index:06d}_{camera_id}.png")
            rendered += 1

    return {
        "scene_id": record.scene_id,
        "status": "ok",
        "scene_dir": str(record.scene_dir),
        "ply_path": str(record.ply_path),
        "transforms_path": str(record.transforms_path),
        "output_dir": str(scene_output),
        "num_gaussians": int(gaussians["position"].shape[0]),
        "num_cameras": len(cameras.frames),
        "num_rendered": rendered,
        "image_width": width,
        "image_height": height,
        "color_mode": args.color_mode,
        "scale_mode": args.scale_mode,
        "opacity_mode": args.opacity_mode,
        "warnings": list(cameras.warnings),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all ASE overfit cameras with the CUDA 3DGS rasterizer."
    )
    parser.add_argument("--root", type=Path, default=Path("./data/cache/ASE_overfit"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--color-mode", choices=("sh-dc", "raw-rgb"), default="sh-dc")
    parser.add_argument("--scale-mode", choices=("log", "linear"), default="log")
    parser.add_argument(
        "--opacity-mode",
        choices=("logit", "activated"),
        default="logit",
        help="Standard 3DGS PLY stores opacity as logits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ASE root does not exist: {root}")

    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (root / "rendered_cameras").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    records = _scene_records(root)
    if not records:
        raise RuntimeError(
            f"No ASE scenes found under {root}. Expected scene/transforms_train.json "
            "and scene/ckpts/point_cloud_*.ply."
        )

    summary = {
        "root": str(root),
        "output_root": str(output_root),
        "device": str(device),
        "num_scenes_found": len(records),
        "scenes": [],
    }

    for record in _iter_limited(records, args.max_scenes):
        print(f"[render] {record.scene_id}")
        scene_summary = render_scene(record, output_root, args, device)
        summary["scenes"].append(scene_summary)
        print(
            f"  {scene_summary['status']}: "
            f"{scene_summary.get('num_rendered', 0)} frames"
        )

    summary_path = output_root / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[render] wrote {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
