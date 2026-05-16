"""Build scene-level voxel caches for a fixed-layout ASE root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaussiangpt_ae.data.voxelize import build_ase_voxel_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-id", action="append", default=None)
    args = parser.parse_args()

    summary = build_ase_voxel_cache(
        root=args.root,
        output_root=args.output_root,
        voxel_size=args.voxel_size,
        overwrite=args.overwrite,
        seed=args.seed,
        scene_ids=args.scene_id,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "voxel_cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for key, value in summary.items():
        if key != "build_results":
            print(f"{key}: {value}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
