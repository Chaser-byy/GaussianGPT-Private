# scripts/debug_ase_pose_3dgs_render.py
#
# Python 3.9 compatible.
#
# Full-scene 3DGS render debug for ASE / Aria Synthetic Environments.
#
# This script directly uses diff_gaussian_rasterization.
# It does NOT depend on Scaffold-GS GaussianModel or Camera classes.
#
# Goal:
#   Verify how transforms_train.json camera poses should be interpreted.

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


SH_C0 = 0.28209479177387814


PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_quat_np(q):
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return q / norm


def focal2fov(focal, pixels):
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def parse_frame_id(file_path):
    m = re.search(r"frame(\d+)", str(file_path))
    if m is None:
        return -1
    return int(m.group(1))


def read_ply_header(f):
    first = f.readline().decode("utf-8").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")

    fmt = None
    vertex_count = None
    vertex_properties = []
    current_element = None

    while True:
        line = f.readline().decode("utf-8").strip()

        if line.startswith("format "):
            fmt = line.split()[1]

        elif line.startswith("element "):
            parts = line.split()
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])

        elif line.startswith("property ") and current_element == "vertex":
            parts = line.split()
            if parts[1] == "list":
                raise NotImplementedError("PLY list property is not supported")
            prop_type = parts[1]
            prop_name = parts[2]
            vertex_properties.append((prop_name, prop_type))

        elif line == "end_header":
            break

    if fmt is None:
        raise ValueError("Missing PLY format")
    if vertex_count is None:
        raise ValueError("Missing vertex count")
    if len(vertex_properties) == 0:
        raise ValueError("Missing vertex properties")

    return vertex_count, vertex_properties, fmt


def load_3dgs_ply(ply_path, max_points=None, seed=0):
    """
    Load vanilla 3DGS PLY fields:
        x y z
        f_dc_0 f_dc_1 f_dc_2
        opacity
        scale_0 scale_1 scale_2
        rot_0 rot_1 rot_2 rot_3

    Important:
        For original 3DGS rendering, scale uses exp(raw_scale),
        opacity uses sigmoid(raw_opacity),
        rotation uses normalized quaternion.

    This is different from your AE feature representation decision,
    where scale will later use softplus.
    """
    ply_path = Path(ply_path)

    with open(ply_path, "rb") as f:
        vertex_count, vertex_properties, fmt = read_ply_header(f)

        if fmt != "binary_little_endian":
            raise ValueError("This debug script currently expects binary_little_endian PLY")

        dtype_fields = []
        for name, typ in vertex_properties:
            if typ not in PLY_DTYPE_MAP:
                raise ValueError("Unsupported PLY type: {}".format(typ))
            dtype_fields.append((name, "<" + PLY_DTYPE_MAP[typ]))

        dtype = np.dtype(dtype_fields)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    names = data.dtype.fields.keys()

    required = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    for name in required:
        if name not in names:
            raise ValueError("PLY missing required field: {}".format(name))

    xyz = np.stack(
        [
            data["x"].astype(np.float32),
            data["y"].astype(np.float32),
            data["z"].astype(np.float32),
        ],
        axis=1,
    )

    f_dc = np.stack(
        [
            data["f_dc_0"].astype(np.float32),
            data["f_dc_1"].astype(np.float32),
            data["f_dc_2"].astype(np.float32),
        ],
        axis=1,
    )

    raw_opacity = data["opacity"].astype(np.float32)
    opacity = sigmoid_np(raw_opacity)[:, None]

    raw_scale = np.stack(
        [
            data["scale_0"].astype(np.float32),
            data["scale_1"].astype(np.float32),
            data["scale_2"].astype(np.float32),
        ],
        axis=1,
    )
    scales = np.exp(raw_scale).astype(np.float32)

    rotations = np.stack(
        [
            data["rot_0"].astype(np.float32),
            data["rot_1"].astype(np.float32),
            data["rot_2"].astype(np.float32),
            data["rot_3"].astype(np.float32),
        ],
        axis=1,
    )
    rotations = normalize_quat_np(rotations)

    if max_points is not None and max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)

        xyz = xyz[idx]
        f_dc = f_dc[idx]
        opacity = opacity[idx]
        scales = scales[idx]
        rotations = rotations[idx]

    # Rasterizer expects SH features.
    # For degree 0:
    #   shape should be [N, 1, 3]
    shs = f_dc[:, None, :].astype(np.float32)

    return {
        "xyz": xyz.astype(np.float32),
        "shs": shs.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": rotations.astype(np.float32),
    }


def load_transforms(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fx = float(meta["fx"])
    fy = float(meta["fy"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    width = int(float(meta["width"]))
    height = int(float(meta["height"]))

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    T_device_camera = np.asarray(meta["transform_device_camera"], dtype=np.float32)

    return {
        "meta": meta,
        "frames": meta["frames"],
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "T_device_camera": T_device_camera,
        "bbox_min": np.asarray(meta["bbox_min"], dtype=np.float32),
        "bbox_max": np.asarray(meta["bbox_max"], dtype=np.float32),
        "FoVx": focal2fov(fx, width),
        "FoVy": focal2fov(fy, height),
    }


def make_pose_candidates(frame_T, T_device_camera):
    """
    We keep only the candidates worth testing after your point-projection result.

    D = inv(frame_T) has been basically ruled out:
      - D_raw: all black
      - D_opengl_to_opencv: camera center often outside / below scene

    So here we test:
      A_raw
      B_raw
      C_raw
      A_opengl_to_opencv
      B_opengl_to_opencv
      C_opengl_to_opencv

    Current strongest candidate from point projection:
      C_raw = frame_T @ inv(transform_device_camera)
    """
    frame_T = np.asarray(frame_T, dtype=np.float32)
    T_device_camera = np.asarray(T_device_camera, dtype=np.float32)
    inv_T_device_camera = np.linalg.inv(T_device_camera).astype(np.float32)

    base = {
        "A_frame_as_c2w": frame_T,
        "B_frame_mul_device_camera": frame_T @ T_device_camera,
        "C_frame_mul_inv_device_camera": frame_T @ inv_T_device_camera,
    }

    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    candidates = {}
    for name, c2w in base.items():
        candidates[name + "__raw"] = c2w.astype(np.float32)
        candidates[name + "__opengl_to_opencv"] = (c2w @ opengl_to_opencv).astype(np.float32)

    return candidates


def get_projection_matrix(znear, zfar, fovx, fovy):
    """
    Same style as graphdeco / 3DGS projection matrix.
    z_sign = 1.0 means camera looks along +z in camera coordinates.
    """
    tan_half_fov_y = math.tan(fovy / 2.0)
    tan_half_fov_x = math.tan(fovx / 2.0)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def make_rasterizer_camera(c2w, width, height, fovx, fovy, device):
    """
    Convert candidate c2w to matrices expected by diff_gaussian_rasterization.

    We assume candidate c2w uses OpenCV-style camera coordinates:
      x right
      y down
      z forward

    world_view_transform in graphdeco code is w2c transposed.
    projection_matrix is also transposed.
    full_proj_transform = world_view_transform @ projection_matrix.
    """
    c2w = np.asarray(c2w, dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)

    world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)

    znear = 0.01
    zfar = 100.0
    projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy).to(device).transpose(0, 1)

    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )

    camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)

    return world_view_transform, full_proj_transform, camera_center


def tensorize_gaussians(gaussians, device):
    means3D = torch.tensor(gaussians["xyz"], dtype=torch.float32, device=device)
    shs = torch.tensor(gaussians["shs"], dtype=torch.float32, device=device)
    opacities = torch.tensor(gaussians["opacity"], dtype=torch.float32, device=device)
    scales = torch.tensor(gaussians["scales"], dtype=torch.float32, device=device)
    rotations = torch.tensor(gaussians["rotations"], dtype=torch.float32, device=device)

    means2D = torch.zeros_like(means3D, dtype=torch.float32, device=device)

    return means3D, means2D, shs, opacities, scales, rotations


@torch.no_grad()
def render_one(
    gaussians_torch,
    c2w,
    width,
    height,
    fovx,
    fovy,
    out_path,
    device,
    bg_value=0.0,
    scaling_modifier=1.0,
    debug=False,
):
    means3D, means2D, shs, opacities, scales, rotations = gaussians_torch

    world_view_transform, full_proj_transform, camera_center = make_rasterizer_camera(
        c2w=c2w,
        width=width,
        height=height,
        fovx=fovx,
        fovy=fovy,
        device=device,
    )

    bg = torch.tensor([bg_value, bg_value, bg_value], dtype=torch.float32, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=math.tan(fovx * 0.5),
        tanfovy=math.tan(fovy * 0.5),
        bg=bg,
        scale_modifier=float(scaling_modifier),
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    result = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    if isinstance(result, tuple):
        rendered_image = result[0]
        radii = result[1] if len(result) > 1 else None
    else:
        rendered_image = result
        radii = None

    image = rendered_image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    image_u8 = (image * 255.0).astype(np.uint8)

    Image.fromarray(image_u8).save(out_path)

    nonblack = float((image_u8.sum(axis=2) > 0).mean())

    stats = {
        "image_path": str(out_path),
        "nonblack_ratio": nonblack,
    }

    if radii is not None:
        radii_cpu = radii.detach().cpu()
        stats["num_radii_positive"] = int((radii_cpu > 0).sum().item())
        stats["num_radii_total"] = int(radii_cpu.numel())

    return stats


def make_contact_sheet(image_paths, out_path, tile_w, tile_h):
    if len(image_paths) == 0:
        return

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h))
        images.append(img)

    cols = 3
    rows = int(math.ceil(len(images) / float(cols)))

    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(20, 20, 20))

    for i, img in enumerate(images):
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        sheet.paste(img, (x, y))

    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-indices", nargs="+", type=int, default=[0, 80, 160, 240, 320])
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bg", type=float, default=0.0)
    parser.add_argument("--scaling-modifier", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for diff_gaussian_rasterization")

    device = torch.device("cuda")

    scene_dir = Path(args.scene_dir)
    transforms_path = scene_dir / "transforms_train.json"
    ply_path = scene_dir / "ckpts" / "point_cloud_30000.ply"

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("==== Paths ====")
    print("scene_dir:", scene_dir)
    print("transforms:", transforms_path)
    print("ply:", ply_path)
    print("out_dir:", out_dir)

    cam = load_transforms(transforms_path)

    print("\n==== Camera ====")
    print("width height:", cam["width"], cam["height"])
    print("fx fy cx cy:", cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    print("FoVx FoVy:", cam["FoVx"], cam["FoVy"])
    print("frames:", len(cam["frames"]))
    print("bbox_min:", cam["bbox_min"])
    print("bbox_max:", cam["bbox_max"])

    print("\n==== Load Gaussian PLY ====")
    max_points = args.max_points
    if max_points <= 0:
        max_points = None

    gaussians = load_3dgs_ply(
        ply_path=ply_path,
        max_points=max_points,
        seed=args.seed,
    )

    print("num gaussians:", gaussians["xyz"].shape[0])
    print("xyz min:", gaussians["xyz"].min(axis=0))
    print("xyz max:", gaussians["xyz"].max(axis=0))
    print("opacity range:", gaussians["opacity"].min(), gaussians["opacity"].max())
    print("scale range:", gaussians["scales"].min(), gaussians["scales"].max())

    print("\n==== Move Gaussian tensors to CUDA ====")
    gaussians_torch = tensorize_gaussians(gaussians, device=device)

    metrics = {
        "scene_dir": str(scene_dir),
        "transforms_path": str(transforms_path),
        "ply_path": str(ply_path),
        "width": cam["width"],
        "height": cam["height"],
        "FoVx": cam["FoVx"],
        "FoVy": cam["FoVy"],
        "max_points": args.max_points,
        "frames": {},
    }

    all_images = []

    for frame_index in args.frame_indices:
        if frame_index < 0 or frame_index >= len(cam["frames"]):
            print("Skip invalid frame index:", frame_index)
            continue

        frame = cam["frames"][frame_index]
        file_path = frame.get("file_path", "")
        frame_id = parse_frame_id(file_path)

        frame_T = np.asarray(frame["transform_matrix"], dtype=np.float32)
        candidates = make_pose_candidates(frame_T, cam["T_device_camera"])

        frame_key = "frame_index_{:04d}_id_{:07d}".format(frame_index, frame_id)
        frame_out_dir = out_dir / frame_key
        ensure_dir(frame_out_dir)

        print("\n==== Render {} | {} ====".format(frame_key, file_path))

        metrics["frames"][frame_key] = {
            "frame_index": int(frame_index),
            "frame_id": int(frame_id),
            "file_path": file_path,
            "candidates": {},
        }

        frame_images = []

        # Put C_raw first because it was the strongest candidate in point projection.
        ordered_names = [
            "C_frame_mul_inv_device_camera__raw",
            "A_frame_as_c2w__raw",
            "B_frame_mul_device_camera__raw",
            "C_frame_mul_inv_device_camera__opengl_to_opencv",
            "A_frame_as_c2w__opengl_to_opencv",
            "B_frame_mul_device_camera__opengl_to_opencv",
        ]

        for name in ordered_names:
            if name not in candidates:
                continue

            c2w = candidates[name]
            out_png = frame_out_dir / "{}.png".format(name)

            try:
                stats = render_one(
                    gaussians_torch=gaussians_torch,
                    c2w=c2w,
                    width=cam["width"],
                    height=cam["height"],
                    fovx=cam["FoVx"],
                    fovy=cam["FoVy"],
                    out_path=out_png,
                    device=device,
                    bg_value=args.bg,
                    scaling_modifier=args.scaling_modifier,
                    debug=args.debug,
                )
            except Exception as e:
                print("FAILED:", name, repr(e))
                stats = {
                    "image_path": str(out_png),
                    "error": repr(e),
                }

            metrics["frames"][frame_key]["candidates"][name] = stats

            if "error" not in stats:
                print(
                    "{:<48s} nonblack={:.6f} radii={}/{}".format(
                        name,
                        stats.get("nonblack_ratio", -1.0),
                        stats.get("num_radii_positive", -1),
                        stats.get("num_radii_total", -1),
                    )
                )
                frame_images.append(out_png)
                all_images.append(out_png)
            else:
                print("{:<48s} ERROR {}".format(name, stats["error"]))

        make_contact_sheet(
            frame_images,
            frame_out_dir / "contact_sheet.png",
            cam["width"],
            cam["height"],
        )

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    make_contact_sheet(
        all_images,
        out_dir / "all_contact_sheet.png",
        cam["width"],
        cam["height"],
    )

    print("\n==== Done ====")
    print("metrics:", metrics_path)
    print("all contact sheet:", out_dir / "all_contact_sheet.png")


if __name__ == "__main__":
    main()
