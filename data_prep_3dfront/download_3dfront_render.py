#!/usr/bin/env python3
"""Download the rendered 3D-FRONT archives from Hugging Face.

Install dependency:
    python -m pip install huggingface_hub

Examples:
    python data_prep_3dfront/download_3dfront_render.py \
        --local-dir /data/3dfront_hf --test-only --extract

    python data_prep_3dfront/download_3dfront_render.py \
        --local-dir /data/3dfront_hf --render-only
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


REPO_ID = "huanngzh/3D-Front"


def _safe_extract_tar(archive: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(output_dir)):
                raise RuntimeError(f"refusing unsafe tar member: {member.name}")
        tar.extractall(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download rendered RGB/depth/camera archives for huanngzh/3D-Front."
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="Directory where Hugging Face files and optional extraction output are stored.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {REPO_ID}",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Download only test render archive(s), useful for first debugging.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        default=True,
        help="Only download render archives; scene glb and surface npy archives are ignored.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision/commit.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded tar.gz archives into --extract-dir or --local-dir.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Extraction directory. Defaults to --local-dir.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token, if your environment requires one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency huggingface_hub. Install with:\n"
            "  python -m pip install huggingface_hub\n"
        ) from exc

    if args.test_only:
        allow_patterns = ["*TEST*RENDER*.tar.gz", "*test*render*.tar.gz"]
    else:
        allow_patterns = ["*RENDER*.tar.gz", "*render*.tar.gz"]

    ignore_patterns = [
        "*SCENE*",
        "*scene*",
        "*SURFACE*",
        "*surface*",
        "*.glb",
        "*.npy",
        "*.npz",
    ]

    print(f"[download] repo={args.repo_id}")
    print(f"[download] local_dir={args.local_dir}")
    print(f"[download] allow_patterns={allow_patterns}")
    print(f"[download] ignore_patterns={ignore_patterns}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=args.token,
    )

    archives = sorted(args.local_dir.rglob("*.tar.gz"))
    render_archives = [
        path
        for path in archives
        if "render" in path.name.lower() and "scene" not in path.name.lower()
    ]
    if not render_archives:
        raise SystemExit(
            "No render tar.gz archives were downloaded. Check the repo file names, "
            "--test-only, and network/authentication settings."
        )
    print("[download] archives:")
    for archive in render_archives:
        print(f"  - {archive}")

    if args.extract:
        extract_dir = args.extract_dir or args.local_dir
        extract_dir.mkdir(parents=True, exist_ok=True)
        for archive in render_archives:
            print(f"[extract] {archive} -> {extract_dir}")
            _safe_extract_tar(archive, extract_dir)
        print(f"[extract] done: {extract_dir}")


if __name__ == "__main__":
    main()
