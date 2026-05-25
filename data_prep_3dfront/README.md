# Minimal 3D-FRONT Render Prep

Standalone utilities for a small first pass on
`huanngzh/3D-Front` rendered RGB/depth/camera data.

This folder is intentionally not integrated into GaussianGPT training,
voxelization, tokenization, or the existing ASE pipeline.

The current Hugging Face render layout looks like:

```text
<scene_uuid>/<room_name>/
├── meta.json
├── render_0000.webp
├── depth_0000.exr
├── normal_0000.webp
└── semantic_0000.png
```

`meta.json` contains `camera_angle_x` and a `locations` list. Each location has
one `transform_matrix` and frame records for render/depth/normal. The scanner
uses this metadata first and only falls back to filename glob pairing when
`meta.json` is unavailable.

The `meta.json` poses are Blender/OpenGL camera-to-world matrices, where the
camera looks along local `-Z`. `build-room` converts them to OpenCV/COLMAP/3DGS
camera-to-world, where the camera looks along local `+Z`, before depth
backprojection and before writing `transforms.json`.

GaussianGPT's 3D-FRONT data path is:

1. render multi-view RGB/depth images from 3D-FRONT scenes;
2. initialize Gaussian primitives from depth maps;
3. optimize the initialized scene with Scaffold-GS;
4. later, during autoencoder training, use rendered RGB/depth to score chunk
   camera views by visible chunk area.

This mini project implements steps 1-2 as data preparation and writes a
room-local intermediate package for step 3. It does not run Scaffold-GS itself
and does not implement GaussianGPT chunk/voxel/token training.

## Dependencies

```bash
python -m pip install huggingface_hub pillow numpy opencv-python
```

If OpenCV cannot read EXR in your environment, install imageio as a fallback:

```bash
python -m pip install imageio
```

## 1. Download Test Render Data

The Hugging Face dataset card says evaluation can use files containing the
`test` keyword. This script only downloads render archives and ignores
scene/surface/glb/npy archives.

```bash
python data_prep_3dfront/download_3dfront_render.py \
  --local-dir /path/to/3dfront_hf \
  --test-only \
  --extract
```

For the full render archive instead of test render only:

```bash
python data_prep_3dfront/download_3dfront_render.py \
  --local-dir /path/to/3dfront_hf \
  --extract
```

## 2. Generate Manifest

Point `--data-root` either at the extraction directory or directly at a
`3D-FRONT-RENDER` / `3D-FRONT-TEST-RENDER` directory.

```bash
python data_prep_3dfront/prepare_3dfront_render.py scan \
  --data-root /path/to/3dfront_hf \
  --output /path/to/3dfront_work/manifest.json \
  --max-rooms 5 \
  --layout auto \
  --image-prefix render
```

The manifest records one room per entry. Each frame records:

- `image_path`
- `depth_path`
- `camera_path`
- embedded `camera` values parsed from `meta.json` when available

Warnings are printed and stored if RGB/depth/camera pairing does not match the
expected layout.

## 3. Debug One Room

```bash
python data_prep_3dfront/prepare_3dfront_render.py debug-room \
  --manifest /path/to/3dfront_work/manifest.json \
  --frame-index 0 \
  --depth-scale 1
```

This prints RGB/depth shape, depth range, whether raw depth looks like
millimeters, valid depth ratio, intrinsics, and the camera-to-world matrix.
EXR depth is treated as meters by default. The cleaner first removes invalid
values (`NaN`, `inf`, `<=0`, `>=1e4`, and the float16 EXR sentinel near
`65504`) before applying `--depth-scale` and `--near/--far`. Millimeter
warnings are based on cleaned depth median/percentiles, not raw max.

Use `--room-id ROOM_ID` to select a specific room from the manifest.

Camera convention defaults:

- 3D-FRONT `meta.json` `transform_matrix` values are treated as Blender c2w
  and converted to OpenCV/COLMAP c2w.
- Labeled `c2w` / `transform_matrix` / `pose` matrices in generic camera files
  are treated as OpenCV/COLMAP camera to world unless overridden.
- Labeled `w2c` / `extrinsic` matrices are inverted.
- Unlabeled numeric 4x4 matrices are treated as `c2w`; override with
  `--matrix-is w2c` if needed.
- Override pose convention with `--camera-convention blender` or
  `--camera-convention opencv` if needed.
- If intrinsics are missing, `--fov-deg` is used as a fallback.

## 4. Generate points3D.ply From Depth

```bash
python data_prep_3dfront/prepare_3dfront_render.py build-room \
  --manifest /path/to/3dfront_work/manifest.json \
  --output-dir /path/to/3dfront_work/rooms \
  --max-views 8 \
  --points-per-image 20000 \
  --max-points 300000 \
  --depth-scale 1000 \
  --near 0.05 \
  --far 20 \
  --voxel-size 0.02
```

The output directory is:

```text
/path/to/3dfront_work/rooms/<room_id>/
├── images/
├── points3D.ply
└── transforms.json
```

`transforms.json` is a minimal intermediate format containing per-frame image
paths, intrinsics, OpenCV/COLMAP-style `c2w`, source depth/camera paths, point
counts, and room bbox.

## 5. Prepare Scaffold-GS / 3DGS Room Package

Create a COLMAP text package from the built room:

```bash
python data_prep_3dfront/prepare_3dfront_render.py prepare-scaffold-room \
  --manifest /path/to/3dfront_work/manifest.json \
  --built-rooms-dir /path/to/3dfront_work/rooms \
  --output-dir /path/to/3dfront_work/scaffold_rooms \
  --room-id '<room_id>'
```

The output is:

```text
/path/to/3dfront_work/scaffold_rooms/<room_id>/
├── images/
├── points3D.ply
├── sparse/0/cameras.txt
├── sparse/0/images.txt
├── sparse/0/points3D.txt
└── summary.json
```

This package contains:

- RGB images kept per room
- colored initialization point cloud at `points3D.ply`
- COLMAP text cameras/images/points3D generated from `transforms.json`

`summary.json` records current GaussianGPT constraints:

- `anchor_aligned_to_voxel_centers`
- `one_gaussian_per_anchor`
- `voxel_size`

For the generic COLMAP package these are marked `false` until a specific
Scaffold-GS code path or post-processing adapter enforces them.

## 6. Train One Room With Scaffold-GS / 3DGS

Debug run:

```bash
python data_prep_3dfront/prepare_3dfront_render.py train-scaffold-room \
  --dataset-dir /path/to/3dfront_work/scaffold_rooms/<room_id> \
  --scaffold-repo /path/to/Scaffold-GS-or-3DGS \
  --output-dir /path/to/3dfront_work/scaffold_runs/<room_id>_2k \
  --iterations 2000
```

GaussianGPT-style full run:

```bash
python data_prep_3dfront/prepare_3dfront_render.py train-scaffold-room \
  --dataset-dir /path/to/3dfront_work/scaffold_rooms/<room_id> \
  --scaffold-repo /path/to/Scaffold-GS-or-3DGS \
  --output-dir /path/to/3dfront_work/scaffold_runs/<room_id>_60k \
  --iterations 60000
```

The command expects the external repo to provide a compatible `train.py` using
the common `-s <dataset> -m <model> --iterations <N>` interface. Extra flags can
be passed with `--extra-args`.

For a GaussianGPT-like 3D-FRONT run, use this package as the depth-initialized
input to Scaffold-GS, then optimize externally for the desired iteration count.
The optimized Scaffold-GS anchors/Gaussians are the input to later GaussianGPT
voxelization/chunk/token stages, which are intentionally out of scope here.

No glb scene files, mesh data, or MIDI-3D surface npy point clouds are used.
