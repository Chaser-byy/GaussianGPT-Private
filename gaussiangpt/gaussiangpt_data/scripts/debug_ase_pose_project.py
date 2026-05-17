# scripts/debug_ase_pose_project.py
#
# Python 3.9 compatible.
#
# This script does NOT use a real 3DGS rasterizer.
# It only projects Gaussian centers from point_cloud_30000.ply
# to debug camera pose conventions in transforms_train.json.

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


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


def focal2fov(focal, pixels):
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def parse_frame_id(file_path):
    """
    Example:
        rgb_undistorted/frame0000066.jpg -> 66
    """
    m = re.search(r"frame(\d+)", str(file_path))
    if m is None:
        return -1
    return int(m.group(1))


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

    T_device_camera = np.asarray(
        meta["transform_device_camera"],
        dtype=np.float32,
    )

    frames = meta["frames"]

    bbox_min = np.asarray(meta["bbox_min"], dtype=np.float32)
    bbox_max = np.asarray(meta["bbox_max"], dtype=np.float32)

    return {
        "meta": meta,
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "T_device_camera": T_device_camera,
        "frames": frames,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "FoVx": focal2fov(fx, width),
        "FoVy": focal2fov(fy, height),
    }


def read_ply_header(f):
    """
    Read PLY header from an opened binary file.

    Returns:
        header_lines: list[str]
        vertex_count: int
        vertex_properties: list[(name, type)]
        fmt: str
    """
    first = f.readline().decode("utf-8").strip()
    if first != "ply":
        raise ValueError("Not a PLY file: missing 'ply' header")

    header_lines = [first]
    fmt = None
    vertex_count = None
    vertex_properties = []
    current_element = None

    while True:
        line = f.readline().decode("utf-8").strip()
        header_lines.append(line)

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
                raise NotImplementedError("List properties in vertex are not supported")

            prop_type = parts[1]
            prop_name = parts[2]
            vertex_properties.append((prop_name, prop_type))

        elif line == "end_header":
            break

    if fmt is None:
        raise ValueError("PLY format is missing")
    if vertex_count is None:
        raise ValueError("PLY vertex count is missing")
    if len(vertex_properties) == 0:
        raise ValueError("PLY vertex properties are missing")

    return header_lines, vertex_count, vertex_properties, fmt


def load_3dgs_ply_xyz_color(ply_path, max_points=None, seed=0):
    """
    Minimal PLY reader for 3DGS point_cloud_30000.ply.

    We read:
        x, y, z
        f_dc_0, f_dc_1, f_dc_2 if available

    Color is only for debug visualization:
        rgb = clip(f_dc * SH_C0 + 0.5, 0, 1)

    This is NOT a faithful 3DGS render.
    """
    ply_path = Path(ply_path)

    with open(ply_path, "rb") as f:
        _, vertex_count, vertex_properties, fmt = read_ply_header(f)

        names = [p[0] for p in vertex_properties]
        types = [p[1] for p in vertex_properties]

        for required in ["x", "y", "z"]:
            if required not in names:
                raise ValueError("PLY missing required property: {}".format(required))

        if fmt == "binary_little_endian":
            dtype_fields = []
            for name, typ in vertex_properties:
                if typ not in PLY_DTYPE_MAP:
                    raise ValueError("Unsupported PLY property type: {}".format(typ))
                dtype_fields.append((name, "<" + PLY_DTYPE_MAP[typ]))

            dtype = np.dtype(dtype_fields)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)

        elif fmt == "ascii":
            # Slow path, but useful for completeness.
            arr = np.loadtxt(f, dtype=np.float32, max_rows=vertex_count)
            data = {}
            for i, name in enumerate(names):
                data[name] = arr[:, i]

        else:
            raise ValueError("Unsupported PLY format: {}".format(fmt))

    if isinstance(data, np.ndarray) and data.dtype.fields is not None:
        xyz = np.stack(
            [
                data["x"].astype(np.float32),
                data["y"].astype(np.float32),
                data["z"].astype(np.float32),
            ],
            axis=1,
        )

        has_dc = (
            "f_dc_0" in data.dtype.fields
            and "f_dc_1" in data.dtype.fields
            and "f_dc_2" in data.dtype.fields
        )
        if has_dc:
            fdc = np.stack(
                [
                    data["f_dc_0"].astype(np.float32),
                    data["f_dc_1"].astype(np.float32),
                    data["f_dc_2"].astype(np.float32),
                ],
                axis=1,
            )
            rgb = np.clip(fdc * SH_C0 + 0.5, 0.0, 1.0)
        else:
            rgb = np.ones_like(xyz, dtype=np.float32) * 0.8

    else:
        xyz = np.stack(
            [
                data["x"].astype(np.float32),
                data["y"].astype(np.float32),
                data["z"].astype(np.float32),
            ],
            axis=1,
        )
        rgb = np.ones_like(xyz, dtype=np.float32) * 0.8

    if max_points is not None and max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz.astype(np.float32), rgb.astype(np.float32)


def make_pose_candidates(frame_T, T_device_camera):
    """
    We produce candidate c2w matrices.

    Base candidates:
        A: c2w = frame_T
        B: c2w = frame_T @ T_device_camera
        C: c2w = frame_T @ inv(T_device_camera)
        D: c2w = inv(frame_T)

    Then each base candidate has two axis versions:
        raw
        opengl_to_opencv

    The projection function below assumes OpenCV camera coordinates:
        x right
        y down
        z forward

    If a pose is OpenGL-style c2w:
        x right
        y up
        z backward
    then:
        c2w_cv = c2w_gl @ diag(1, -1, -1, 1)
    """
    frame_T = np.asarray(frame_T, dtype=np.float32)
    T_device_camera = np.asarray(T_device_camera, dtype=np.float32)

    inv_T_device_camera = np.linalg.inv(T_device_camera).astype(np.float32)

    base = {
        "A_frame_as_c2w": frame_T,
        "B_frame_mul_device_camera": frame_T @ T_device_camera,
        "C_frame_mul_inv_device_camera": frame_T @ inv_T_device_camera,
        "D_inv_frame": np.linalg.inv(frame_T).astype(np.float32),
    }

    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    candidates = {}
    for name, c2w in base.items():
        candidates[name + "__raw"] = c2w.astype(np.float32)
        candidates[name + "__opengl_to_opencv"] = (c2w @ opengl_to_opencv).astype(np.float32)

    return candidates


def bbox_corners(bbox_min, bbox_max):
    corners = []
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            for z in [bbox_min[2], bbox_max[2]]:
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float32)


def transform_world_to_cam(points_world, c2w):
    """
    Input:
        points_world: [N, 3]
        c2w: [4, 4]

    Output:
        points_cam: [N, 3], OpenCV convention assumed after candidate construction.
    """
    w2c = np.linalg.inv(c2w).astype(np.float32)

    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_world, ones], axis=1)

    points_cam = (w2c @ points_h.T).T[:, :3]
    return points_cam


def project_cam_to_pixel(points_cam, K):
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    eps = 1e-8
    u = K[0, 0] * x / np.maximum(z, eps) + K[0, 2]
    v = K[1, 1] * y / np.maximum(z, eps) + K[1, 2]

    uv = np.stack([u, v], axis=1)
    return uv, z


def compute_bbox_metrics(c2w, K, width, height, bbox_min, bbox_max):
    corners = bbox_corners(bbox_min, bbox_max)
    points_cam = transform_world_to_cam(corners, c2w)
    uv, z = project_cam_to_pixel(points_cam, K)

    valid_depth = z > 1e-4
    cam_center = c2w[:3, 3].copy()

    if valid_depth.any():
        uv_valid = uv[valid_depth]

        u_min, v_min = uv_valid.min(axis=0)
        u_max, v_max = uv_valid.max(axis=0)

        inter_u_min = max(0.0, float(u_min))
        inter_v_min = max(0.0, float(v_min))
        inter_u_max = min(float(width), float(u_max))
        inter_v_max = min(float(height), float(v_max))

        inter_w = max(0.0, inter_u_max - inter_u_min)
        inter_h = max(0.0, inter_v_max - inter_v_min)
        image_overlap = inter_w * inter_h / float(width * height)
    else:
        u_min = v_min = u_max = v_max = 0.0
        image_overlap = 0.0

    return {
        "camera_center": cam_center.tolist(),
        "bbox_num_front": int(valid_depth.sum()),
        "bbox_depth_min": float(z.min()),
        "bbox_depth_max": float(z.max()),
        "bbox_projected": [
            float(u_min),
            float(v_min),
            float(u_max),
            float(v_max),
        ],
        "bbox_image_overlap": float(image_overlap),
        "bbox_valid_projection": bool(valid_depth.any() and image_overlap > 0.0),
    }


def render_point_projection(
    xyz,
    rgb,
    c2w,
    K,
    width,
    height,
    out_path,
    background=0,
):
    """
    Crude point projection renderer.

    This only projects Gaussian centers. It ignores:
        opacity
        scale
        rotation
        anisotropic covariance
        alpha blending
        SH view-dependent color

    So it is only for camera pose debugging.
    """
    points_cam = transform_world_to_cam(xyz, c2w)
    uv, z = project_cam_to_pixel(points_cam, K)

    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)

    valid = (
        (z > 1e-4)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    img = np.full((height, width, 3), background, dtype=np.uint8)

    if not valid.any():
        Image.fromarray(img).save(out_path)
        return {
            "point_num_total": int(xyz.shape[0]),
            "point_num_front": int((z > 1e-4).sum()),
            "point_num_in_image": 0,
            "point_visible_ratio": 0.0,
            "depth_min_valid": None,
            "depth_max_valid": None,
        }

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]
    rgb_valid = rgb[valid]

    pix = v_valid * width + u_valid

    # Nearest depth per pixel.
    order = np.argsort(z_valid)
    pix_ordered = pix[order]

    _, first_indices = np.unique(pix_ordered, return_index=True)
    chosen = order[first_indices]

    u_chosen = u_valid[chosen]
    v_chosen = v_valid[chosen]
    rgb_chosen = rgb_valid[chosen]

    img[v_chosen, u_chosen] = np.clip(rgb_chosen * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(out_path)

    return {
        "point_num_total": int(xyz.shape[0]),
        "point_num_front": int((z > 1e-4).sum()),
        "point_num_in_image": int(valid.sum()),
        "point_visible_ratio": float(valid.sum()) / float(xyz.shape[0]),
        "depth_min_valid": float(z_valid.min()),
        "depth_max_valid": float(z_valid.max()),
    }


def make_contact_sheet(image_paths, out_path, tile_w, tile_h):
    if len(image_paths) == 0:
        return

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h))
        images.append(img)

    cols = 4
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
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    transforms_path = scene_dir / "transforms_train.json"
    ply_path = scene_dir / "ckpts" / "point_cloud_30000.ply"

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not transforms_path.exists():
        raise FileNotFoundError("Missing transforms_train.json: {}".format(transforms_path))
    if not ply_path.exists():
        raise FileNotFoundError("Missing PLY: {}".format(ply_path))

    cam = load_transforms(transforms_path)

    print("==== Camera JSON ====")
    print("transforms:", transforms_path)
    print("width height:", cam["width"], cam["height"])
    print("K:\n", cam["K"])
    print("FoVx FoVy:", cam["FoVx"], cam["FoVy"])
    print("frames:", len(cam["frames"]))
    print("bbox_min:", cam["bbox_min"])
    print("bbox_max:", cam["bbox_max"])

    print("\n==== Loading PLY ====")
    print("ply:", ply_path)
    xyz, rgb = load_3dgs_ply_xyz_color(
        ply_path,
        max_points=args.max_points,
        seed=args.seed,
    )
    print("loaded points:", xyz.shape[0])
    print("xyz min:", xyz.min(axis=0))
    print("xyz max:", xyz.max(axis=0))

    all_metrics = {
        "scene_dir": str(scene_dir),
        "transforms_path": str(transforms_path),
        "ply_path": str(ply_path),
        "width": cam["width"],
        "height": cam["height"],
        "K": cam["K"].tolist(),
        "FoVx": cam["FoVx"],
        "FoVy": cam["FoVy"],
        "bbox_min": cam["bbox_min"].tolist(),
        "bbox_max": cam["bbox_max"].tolist(),
        "frames": {},
    }

    all_image_paths = []

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

        all_metrics["frames"][frame_key] = {
            "frame_index": int(frame_index),
            "frame_id": int(frame_id),
            "file_path": file_path,
            "candidates": {},
        }

        print("\n==== {} | {} ====".format(frame_key, file_path))

        frame_image_paths = []

        for cand_name, c2w in candidates.items():
            bbox_m = compute_bbox_metrics(
                c2w=c2w,
                K=cam["K"],
                width=cam["width"],
                height=cam["height"],
                bbox_min=cam["bbox_min"],
                bbox_max=cam["bbox_max"],
            )

            out_png = frame_out_dir / "{}.png".format(cand_name)

            point_m = render_point_projection(
                xyz=xyz,
                rgb=rgb,
                c2w=c2w,
                K=cam["K"],
                width=cam["width"],
                height=cam["height"],
                out_path=out_png,
            )

            metrics = {}
            metrics.update(bbox_m)
            metrics.update(point_m)
            metrics["image_path"] = str(out_png)

            all_metrics["frames"][frame_key]["candidates"][cand_name] = metrics

            print(
                "{:<42s} center={} bbox_front={}/8 bbox_overlap={:.4f} "
                "points_front={} points_in_img={} ratio={:.6f}".format(
                    cand_name,
                    np.round(np.asarray(bbox_m["camera_center"]), 3).tolist(),
                    bbox_m["bbox_num_front"],
                    bbox_m["bbox_image_overlap"],
                    point_m["point_num_front"],
                    point_m["point_num_in_image"],
                    point_m["point_visible_ratio"],
                )
            )

            frame_image_paths.append(out_png)
            all_image_paths.append(out_png)

        make_contact_sheet(
            frame_image_paths,
            frame_out_dir / "contact_sheet.png",
            cam["width"],
            cam["height"],
        )

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    make_contact_sheet(
        all_image_paths,
        out_dir / "all_contact_sheet.png",
        cam["width"],
        cam["height"],
    )

    print("\n==== Done ====")
    print("metrics:", metrics_path)
    print("all contact sheet:", out_dir / "all_contact_sheet.png")


if __name__ == "__main__":
    main()