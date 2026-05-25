#!/usr/bin/env python3
"""Idempotently apply the local GaussianGPT Scaffold-GS edits.

This is intentionally patch-like but more tolerant than ``git apply``:
servers may already have some of the Scaffold-GS edits, and file modes or
nearby comments may differ. The script only edits the four files touched by
the 3D-FRONT GT Gaussian preparation path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def ensure_contains(path: Path, needle: str, insert_after: str, block: str) -> bool:
    text = read(path)
    if needle in text:
        return False
    if insert_after not in text:
        raise RuntimeError(f"{path}: could not find insertion point: {insert_after!r}")
    text = text.replace(insert_after, insert_after + block, 1)
    write(path, text)
    return True


def replace_once(path: Path, old: str, new: str) -> bool:
    text = read(path)
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"{path}: could not find text to replace: {old[:120]!r}")
    text = text.replace(old, new, 1)
    write(path, text)
    return True


def patch_arguments(repo: Path) -> list[str]:
    path = repo / "arguments" / "__init__.py"
    changed = []
    if ensure_contains(
        path,
        "self.gaussiangpt_mode = False",
        "        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist\n",
        "        self.gaussiangpt_mode = False\n"
        "        self.anchor_voxel_size = 0.025\n"
        "        self.align_anchor_to_voxel_center = False\n"
        "        self.one_gaussian_per_anchor = False\n",
    ):
        changed.append("arguments: model params")
    if ensure_contains(
        path,
        "self.nan_debug_interval = 0",
        "        self.densify_grad_threshold = 0.0002\n",
        "        self.nan_debug_interval = 0\n",
    ):
        changed.append("arguments: nan_debug_interval")
    return changed


def patch_renderer(repo: Path) -> list[str]:
    path = repo / "gaussian_renderer" / "__init__.py"
    changed = []
    if replace_once(
        path,
        "    ob_dist = ob_view.norm(dim=1, keepdim=True)\n",
        "    ob_dist = ob_view.norm(dim=1, keepdim=True).clamp_min(1e-8)\n",
    ):
        changed.append("renderer: clamp ob_dist")
    text = read(path)
    if '"xyz": xyz' not in text:
        old = (
            '                "selection_mask": mask,\n'
            '                "neural_opacity": neural_opacity,\n'
            '                "scaling": scaling,\n'
        )
        new = (
            '                "selection_mask": mask,\n'
            '                "neural_opacity": neural_opacity,\n'
            '                "xyz": xyz,\n'
            '                "color": color,\n'
            '                "scaling": scaling,\n'
            '                "rotation": rot,\n'
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find render debug return block")
        write(path, text.replace(old, new, 1))
        changed.append("renderer: return debug tensors")
    text = read(path)
    if "antialiasing=False" not in text:
        old = (
            "        campos=viewpoint_camera.camera_center,\n"
            "        prefiltered=False,\n"
            "        debug=pipe.debug\n"
        )
        new = (
            "        campos=viewpoint_camera.camera_center,\n"
            "        prefiltered=False,\n"
            "        antialiasing=False,\n"
            "        debug=pipe.debug\n"
        )
        count = text.count(old)
        if count == 0:
            raise RuntimeError(f"{path}: could not find raster settings blocks")
        text = text.replace(old, new)
        write(path, text)
        changed.append(f"renderer: antialiasing flag ({count} blocks)")
    return changed


def patch_gaussian_model(repo: Path) -> list[str]:
    path = repo / "scene" / "gaussian_model.py"
    changed = []
    text = read(path)
    if "gaussiangpt_mode : bool = False" not in text:
        old = (
            "                 add_opacity_dist : bool = False,\n"
            "                 add_cov_dist : bool = False,\n"
            "                 add_color_dist : bool = False,\n"
            "                 ):\n"
        )
        new = (
            "                 add_opacity_dist : bool = False,\n"
            "                 add_cov_dist : bool = False,\n"
            "                 add_color_dist : bool = False,\n"
            "                 gaussiangpt_mode : bool = False,\n"
            "                 anchor_voxel_size : float = 0.025,\n"
            "                 align_anchor_to_voxel_center : bool = False,\n"
            "                 one_gaussian_per_anchor : bool = False,\n"
            "                 ):\n"
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find __init__ signature block")
        text = text.replace(old, new, 1)
        write(path, text)
        changed.append("gaussian_model: init signature")
    text = read(path)
    if "self.gaussiangpt_mode = gaussiangpt_mode" not in text:
        old = (
            "        self.feat_dim = feat_dim\n"
            "        self.n_offsets = n_offsets\n"
            "        self.voxel_size = voxel_size\n"
        )
        new = (
            "        self.feat_dim = feat_dim\n"
            "        self.gaussiangpt_mode = gaussiangpt_mode\n"
            "        self.anchor_voxel_size = anchor_voxel_size\n"
            "        self.align_anchor_to_voxel_center = align_anchor_to_voxel_center\n"
            "        self.one_gaussian_per_anchor = one_gaussian_per_anchor\n"
            "        self.n_offsets = 1 if (gaussiangpt_mode and one_gaussian_per_anchor) else n_offsets\n"
            "        self.voxel_size = anchor_voxel_size if (gaussiangpt_mode and anchor_voxel_size > 0) else voxel_size\n"
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find n_offsets/voxel_size assignment block")
        text = text.replace(old, new, 1)
        write(path, text)
        changed.append("gaussian_model: mode attrs")
    if replace_once(
        path,
        "            nn.Linear(feat_dim, n_offsets),\n",
        "            nn.Linear(feat_dim, self.n_offsets),\n",
    ):
        changed.append("gaussian_model: opacity output dim")
    text = read(path)
    if "def enforce_anchor_voxel_alignment(self):" not in text:
        insert_after = (
            "        self.mlp_color = nn.Sequential(\n"
            "            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),\n"
            "            nn.ReLU(True),\n"
            "            nn.Linear(feat_dim, 3*self.n_offsets),\n"
            "            nn.Sigmoid()\n"
            "        ).cuda()\n"
        )
        block = (
            "        if self.gaussiangpt_mode:\n"
            "            with torch.no_grad():\n"
            "                last = self.mlp_opacity[-2]\n"
            "                last.weight.zero_()\n"
            "                last.bias.fill_(0.1)\n"
            "\n"
            "    def enforce_anchor_voxel_alignment(self):\n"
            "        if not (self.gaussiangpt_mode and self.align_anchor_to_voxel_center):\n"
            "            return\n"
            "        voxel_size = float(self.anchor_voxel_size if self.anchor_voxel_size > 0 else self.voxel_size)\n"
            "        if voxel_size <= 0 or self._anchor.numel() == 0:\n"
            "            return\n"
            "        with torch.no_grad():\n"
            "            self._anchor.data = torch.round(self._anchor.data / voxel_size) * voxel_size\n"
        )
        if insert_after not in text:
            raise RuntimeError(f"{path}: could not find mlp_color block")
        write(path, text.replace(insert_after, insert_after + block, 1))
        changed.append("gaussian_model: alignment helper")
    text = read(path)
    if "fused_point_cloud = torch.round(fused_point_cloud / voxel_size) * voxel_size" not in text:
        old = "        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()\n"
        new = (
            "        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()\n"
            "        if self.gaussiangpt_mode and self.align_anchor_to_voxel_center:\n"
            "            voxel_size = float(self.anchor_voxel_size if self.anchor_voxel_size > 0 else self.voxel_size)\n"
            "            fused_point_cloud = torch.round(fused_point_cloud / voxel_size) * voxel_size\n"
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find fused_point_cloud assignment")
        write(path, text.replace(old, new, 1))
        changed.append("gaussian_model: initial snap")
    text = read(path)
    if "self.enforce_anchor_voxel_alignment()" not in text:
        old = '        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")\n'
        new = old + "        self.enforce_anchor_voxel_alignment()\n"
        if old not in text:
            raise RuntimeError(f"{path}: could not find max_radii2D init")
        write(path, text.replace(old, new, 1))
        changed.append("gaussian_model: enforce after init")
    return changed


TRAIN_HELPERS = r'''

def _tensor_debug_stats(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        data = value.detach()
    else:
        try:
            data = torch.as_tensor(value)
        except Exception:
            return {"type": type(value).__name__}
    stats = {
        "shape": list(data.shape),
        "numel": int(data.numel()),
        "dtype": str(data.dtype),
        "device": str(data.device),
    }
    if data.numel() == 0:
        stats.update({"finite_count": 0, "nan_count": 0, "inf_count": 0, "min": None, "max": None, "mean": None})
        return stats
    if data.is_floating_point() or data.is_complex():
        finite = torch.isfinite(data)
        stats["finite_count"] = int(finite.sum().item())
        stats["nan_count"] = int(torch.isnan(data).sum().item())
        stats["inf_count"] = int(torch.isinf(data).sum().item())
        if finite.any():
            vals = data[finite].float()
            stats["min"] = float(vals.min().item())
            stats["max"] = float(vals.max().item())
            stats["mean"] = float(vals.mean().item())
        else:
            stats.update({"min": None, "max": None, "mean": None})
    else:
        stats["min"] = int(data.min().item()) if data.numel() else None
        stats["max"] = int(data.max().item()) if data.numel() else None
    return stats


def _is_bad_tensor(value):
    if not isinstance(value, torch.Tensor):
        return False
    if value.numel() == 0:
        return False
    return bool((~torch.isfinite(value)).any().item()) if value.is_floating_point() else False


def _loss_item(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _write_nan_debug(model_path, payload):
    path = os.path.join(model_path, "nan_debug.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _apply_stable_debug_preset(args):
    preset = {
        "offset_lr_init": 0.001,
        "offset_lr_final": 0.00001,
        "feature_lr": 0.00075,
        "opacity_lr": 0.002,
        "scaling_lr": 0.0007,
        "rotation_lr": 0.0002,
        "mlp_opacity_lr_init": 0.0002,
        "mlp_opacity_lr_final": 0.000002,
        "mlp_cov_lr_init": 0.0004,
        "mlp_cov_lr_final": 0.0004,
        "mlp_color_lr_init": 0.0008,
        "mlp_color_lr_final": 0.000005,
        "appearance_lr_init": 0.005,
        "appearance_lr_final": 0.00005,
    }
    for key, value in preset.items():
        setattr(args, key, value)
    args.debug_from = 0
    if not getattr(args, "nan_debug_interval", 0):
        args.nan_debug_interval = 10
    return preset


def _nan_debug_payload(iteration, viewpoint_cam, render_pkg, losses, reason):
    tensors = {
        "render": render_pkg.get("render"),
        "visibility_filter": render_pkg.get("visibility_filter"),
        "selection_mask": render_pkg.get("selection_mask"),
        "neural_opacity": render_pkg.get("neural_opacity"),
        "scaling": render_pkg.get("scaling"),
        "rotation": render_pkg.get("rotation"),
        "color": render_pkg.get("color"),
        "xyz": render_pkg.get("xyz"),
        "radii": render_pkg.get("radii"),
    }
    payload = {
        "iteration": int(iteration),
        "reason": reason,
        "camera": {
            "uid": int(getattr(viewpoint_cam, "uid", -1)),
            "image_name": str(getattr(viewpoint_cam, "image_name", "")),
        },
        "losses": {name: _loss_item(value) for name, value in losses.items()},
        "tensors": {name: _tensor_debug_stats(value) for name, value in tensors.items()},
    }
    selection_mask = render_pkg.get("selection_mask")
    visibility_filter = render_pkg.get("visibility_filter")
    if isinstance(selection_mask, torch.Tensor):
        payload["selection_mask_true_count"] = int(selection_mask.sum().item())
    if isinstance(visibility_filter, torch.Tensor):
        payload["visibility_true_count"] = int(visibility_filter.sum().item())
    return payload
'''


def patch_train(repo: Path) -> list[str]:
    path = repo / "train.py"
    changed = []
    if ensure_contains(
        path,
        "def _tensor_debug_stats(value):",
        "    print('Backup Finished!')\n\n",
        TRAIN_HELPERS + "\n",
    ):
        changed.append("train: debug helpers")
    text = read(path)
    if "dataset.gaussiangpt_mode, dataset.anchor_voxel_size" not in text:
        old = (
            "    tb_writer = prepare_output_and_logger(dataset)\n"
            "    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, \n"
            "                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)\n"
        )
        new = (
            "    tb_writer = prepare_output_and_logger(dataset)\n"
            "    if getattr(dataset, \"gaussiangpt_mode\", False):\n"
            "        if getattr(dataset, \"one_gaussian_per_anchor\", False):\n"
            "            dataset.n_offsets = 1\n"
            "        if getattr(dataset, \"anchor_voxel_size\", 0.0) > 0:\n"
            "            dataset.voxel_size = dataset.anchor_voxel_size\n"
            "        opt.update_until = 0\n"
            "        if logger is not None:\n"
            "            logger.info(\n"
            "                \"GaussianGPT mode active: n_offsets=%s voxel_size=%s align_anchor_to_voxel_center=%s; anchor densification disabled\",\n"
            "                dataset.n_offsets,\n"
            "                dataset.voxel_size,\n"
            "                getattr(dataset, \"align_anchor_to_voxel_center\", False),\n"
            "            )\n"
            "    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, \n"
            "                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,\n"
            "                              dataset.gaussiangpt_mode, dataset.anchor_voxel_size, dataset.align_anchor_to_voxel_center, dataset.one_gaussian_per_anchor)\n"
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find training GaussianModel init block")
        write(path, text.replace(old, new, 1))
        changed.append("train: GaussianGPT model init")
    if ensure_contains(
        path,
        "gaussians.enforce_anchor_voxel_alignment()",
        "    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)\n",
        "    gaussians.enforce_anchor_voxel_alignment()\n",
    ):
        changed.append("train: enforce alignment after scene init")
    if replace_once(
        path,
        "        scaling_reg = scaling.prod(dim=1).mean()\n",
        "        if scaling.numel() == 0:\n"
        "            scaling_reg = torch.zeros((), dtype=image.dtype, device=image.device)\n"
        "        else:\n"
        "            scaling_reg = scaling.prod(dim=1).mean()\n",
    ):
        changed.append("train: empty scaling guard")
    text = read(path)
    if "bad_reasons = []" not in text:
        old = "        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg\n\n        loss.backward()\n"
        new = (
            "        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg\n"
            "\n"
            "        loss_parts = {\n"
            "            \"l1\": Ll1,\n"
            "            \"ssim\": ssim_loss,\n"
            "            \"scaling_reg\": scaling_reg,\n"
            "            \"loss\": loss,\n"
            "        }\n"
            "        bad_reasons = []\n"
            "        for name, value in loss_parts.items():\n"
            "            if _is_bad_tensor(value):\n"
            "                bad_reasons.append(f\"{name}_nonfinite\")\n"
            "        for name in (\"render\", \"neural_opacity\", \"scaling\", \"rotation\", \"color\", \"xyz\"):\n"
            "            if _is_bad_tensor(render_pkg.get(name)):\n"
            "                bad_reasons.append(f\"{name}_nonfinite\")\n"
            "        if isinstance(visibility_filter, torch.Tensor) and visibility_filter.numel() > 0 and not bool(visibility_filter.any().item()):\n"
            "            bad_reasons.append(\"visibility_filter_all_false\")\n"
            "        if isinstance(offset_selection_mask, torch.Tensor) and offset_selection_mask.numel() > 0 and not bool(offset_selection_mask.any().item()):\n"
            "            bad_reasons.append(\"selection_mask_all_false\")\n"
            "        if bad_reasons:\n"
            "            payload = _nan_debug_payload(iteration, viewpoint_cam, render_pkg, loss_parts, \",\".join(bad_reasons))\n"
            "            path = _write_nan_debug(scene.model_path, payload)\n"
            "            if logger is not None:\n"
            "                logger.error(\"Stopping at iteration %s due to %s; wrote %s\", iteration, \",\".join(bad_reasons), path)\n"
            "            progress_bar.close()\n"
            "            raise RuntimeError(f\"NaN/empty render debug stop at iteration {iteration}: {','.join(bad_reasons)}\")\n"
            "\n"
            "        if logger is not None and getattr(opt, \"nan_debug_interval\", 0) and iteration % int(opt.nan_debug_interval) == 0:\n"
            "            logger.info(\n"
            "                \"[ITER %s] loss=%g l1=%g ssim=%g scaling_reg=%g visible=%s selected=%s render_min=%s render_max=%s\",\n"
            "                iteration,\n"
            "                _loss_item(loss),\n"
            "                _loss_item(Ll1),\n"
            "                _loss_item(ssim_loss),\n"
            "                _loss_item(scaling_reg),\n"
            "                int(visibility_filter.sum().item()) if isinstance(visibility_filter, torch.Tensor) else -1,\n"
            "                int(offset_selection_mask.sum().item()) if isinstance(offset_selection_mask, torch.Tensor) else -1,\n"
            "                _tensor_debug_stats(image).get(\"min\"),\n"
            "                _tensor_debug_stats(image).get(\"max\"),\n"
            "            )\n"
            "\n"
            "        loss.backward()\n"
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find loss.backward block")
        write(path, text.replace(old, new, 1))
        changed.append("train: nan debug checks")
    if ensure_contains(
        path,
        "gaussians.enforce_anchor_voxel_alignment()\n                gaussians.optimizer.zero_grad",
        "                gaussians.optimizer.step()\n",
        "                gaussians.enforce_anchor_voxel_alignment()\n",
    ):
        changed.append("train: enforce alignment after optimizer")
    text = read(path)
    if "dataset.gaussiangpt_mode, dataset.anchor_voxel_size" not in text[text.find("def render_sets"):]:
        old = (
            "        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, \n"
            "                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)\n"
        )
        new = (
            "        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, \n"
            "                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,\n"
            "                              dataset.gaussiangpt_mode, dataset.anchor_voxel_size, dataset.align_anchor_to_voxel_center, dataset.one_gaussian_per_anchor)\n"
        )
        if old in text:
            write(path, text.replace(old, new, 1))
            changed.append("train: render_sets model init")
    text = read(path)
    if 'parser.add_argument("--stable-debug"' not in text:
        old = (
            "    parser.add_argument(\"--align-anchor-to-voxel-center\", dest=\"align_anchor_to_voxel_center\", action=\"store_true\")\n"
            "    parser.add_argument(\"--one-gaussian-per-anchor\", dest=\"one_gaussian_per_anchor\", action=\"store_true\")\n"
            "    args = parser.parse_args(sys.argv[1:])\n"
        )
        new = (
            "    parser.add_argument(\"--align-anchor-to-voxel-center\", dest=\"align_anchor_to_voxel_center\", action=\"store_true\")\n"
            "    parser.add_argument(\"--one-gaussian-per-anchor\", dest=\"one_gaussian_per_anchor\", action=\"store_true\")\n"
            "    parser.add_argument(\"--stable-debug\", \"--stable_debug\", dest=\"stable_debug\", action=\"store_true\")\n"
            "    parser.add_argument(\"--nan-debug-interval\", dest=\"nan_debug_interval\", type=int, default=0)\n"
            "    args = parser.parse_args(sys.argv[1:])\n"
            "    stable_debug_preset = _apply_stable_debug_preset(args) if args.stable_debug else None\n"
        )
        if old not in text:
            raise RuntimeError(f"{path}: could not find parser gaussiangpt arg block")
        write(path, text.replace(old, new, 1))
        changed.append("train: stable debug parser")
    if ensure_contains(
        path,
        "logger.info(f'stable_debug_preset: {stable_debug_preset}')",
        "    logger.info(f'args: {args}')\n",
        "    if stable_debug_preset is not None:\n"
        "        logger.info(f'stable_debug_preset: {stable_debug_preset}')\n",
    ):
        changed.append("train: stable debug logging")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-repo", type=Path, default=Path("packages/Scaffold-GS"))
    args = parser.parse_args()
    repo = args.scaffold_repo.resolve()
    if not (repo / "train.py").is_file():
        raise SystemExit(f"not a Scaffold-GS repo: {repo}")
    changed = []
    changed.extend(patch_arguments(repo))
    changed.extend(patch_renderer(repo))
    changed.extend(patch_gaussian_model(repo))
    changed.extend(patch_train(repo))
    if changed:
        print("Applied:")
        for item in changed:
            print(f"  - {item}")
    else:
        print("Scaffold-GS GaussianGPT edits were already present.")


if __name__ == "__main__":
    main()
