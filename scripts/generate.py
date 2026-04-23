"""GaussianGPT inference: unconditional generation, completion, and outpainting.

Usage:
    # Unconditional generation
    python scripts/generate.py --mode unconditional --checkpoint checkpoints/transformer/best.pt

    # Scene completion (given partial scene)
    python scripts/generate.py --mode completion --input scene.pt --checkpoint ...

    # Large scene outpainting
    python scripts/generate.py --mode outpainting --checkpoint ... --target_size 5
"""
import os
import argparse
import torch
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiangpt.autoencoder import GaussianAutoencoder
from gaussiangpt.transformer import GaussianGPT
from gaussiangpt.utils.serialization import (
    serialize_latent_grid, deserialize_token_sequence, flat_idx_to_coord
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unconditional", "completion", "outpainting"],
                        default="unconditional")
    parser.add_argument("--ae_checkpoint", type=str, required=True)
    parser.add_argument("--gpt_checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=None,
                        help="Input scene .pt file for completion mode")
    parser.add_argument("--output", type=str, default="generated_scene.pt")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--target_size", type=int, default=3,
                        help="Number of chunks for outpainting")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument("--ae_base_ch", type=int, default=32)
    parser.add_argument("--ae_n_down", type=int, default=3)
    parser.add_argument("--gpt_size", type=str, default="medium")
    parser.add_argument("--gpt_context", type=int, default=16384)
    return parser.parse_args()


def load_models(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Autoencoder
    ae = GaussianAutoencoder(
        base_ch=args.ae_base_ch,
        n_down=args.ae_n_down,
        codebook_size=args.codebook_size,
    ).to(device)
    ae_ckpt = torch.load(args.ae_checkpoint, map_location=device)
    ae.load_state_dict(ae_ckpt["model"] if "model" in ae_ckpt else ae_ckpt)
    ae.eval()

    # GPT — chunk_size from args is in base voxels; latent chunk = chunk_size / 2^n_down
    chunk_size = tuple(args.chunk_size)
    latent_chunk_size = tuple(c // (2 ** args.ae_n_down) for c in chunk_size)
    gpt = GaussianGPT(
        size=args.gpt_size,
        context_len=args.gpt_context,
        chunk_size=latent_chunk_size,
        codebook_size=args.codebook_size,
    ).to(device)
    gpt_ckpt = torch.load(args.gpt_checkpoint, map_location=device)
    gpt.load_state_dict(gpt_ckpt["model"] if "model" in gpt_ckpt else gpt_ckpt)
    gpt.eval()

    return ae, gpt, device


def decode_tokens_to_gaussians(ae, gpt, tokens, coords, token_type, device):
    """Decode generated token sequence back to 3D Gaussians."""
    from gaussiangpt.utils.serialization import deserialize_token_sequence

    # gpt.chunk_size is already the latent chunk size
    chunk_size_t = gpt.chunk_size
    BOS = gpt.BOS
    EOS = gpt.EOS

    # Extract position and feature tokens
    voxel_coords, voxel_codes = deserialize_token_sequence(
        tokens.squeeze(0), chunk_size_t, BOS, EOS
    )

    if len(voxel_coords) == 0:
        return None

    # Decode LFQ codes to latent features
    z_q = ae.quantizer.decode(voxel_codes.to(device))  # (N, num_bits)

    # Decode through CNN decoder
    from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI
    if HAS_MINKOWSKI:
        import MinkowskiEngine as ME
        batch_idx = torch.zeros(len(voxel_coords), 1, dtype=torch.int, device=device)
        coords_me = torch.cat([batch_idx, voxel_coords.to(device)], dim=1)
        sparse_input = ME.SparseTensor(features=z_q, coordinates=coords_me)
        decoded, _ = ae.decoder(sparse_input)
        feat = decoded.F
    else:
        # Dense fallback: place latent features in latent-size grid, decode to base resolution
        cx, cy, cz = chunk_size_t
        grid = torch.zeros(1, z_q.shape[-1], cx, cy, cz, device=device)
        vc = voxel_coords.to(device)
        grid[0, :, vc[:, 0], vc[:, 1], vc[:, 2]] = z_q.T
        decoded, _ = ae.decoder(grid)
        # decoded is at base voxel resolution; sample at voxel center positions
        # Each latent voxel maps to a 2^n_down block; use center of each block
        n_down = ae.encoder.downs.__len__() if hasattr(ae.encoder, 'downs') else 3
        scale = 2 ** n_down
        # Map latent coords to base voxel coords (center of each block)
        vc_base = vc * scale + scale // 2
        vc_base = vc_base.clamp(0, decoded.shape[2] - 1)
        feat = decoded[0, :, vc_base[:, 0], vc_base[:, 1], vc_base[:, 2]].T

    gaussians = ae.attr_decoder(feat)
    gaussians["voxel_coords"] = voxel_coords
    return gaussians


def generate_unconditional(ae, gpt, args, device):
    """Generate a single scene chunk unconditionally."""
    results = []

    for i in range(args.n_samples):
        print(f"Generating sample {i+1}/{args.n_samples}...")
        with torch.no_grad():
            tokens, coords, token_type = gpt.generate(
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=gpt.context_len,
                device=device,
            )

        gaussians = decode_tokens_to_gaussians(ae, gpt, tokens, coords, token_type, device)
        if gaussians is not None:
            results.append({k: v.cpu() for k, v in gaussians.items()})

    return results


def generate_completion(ae, gpt, args, device):
    """Complete a partial scene."""
    assert args.input is not None, "Need --input for completion mode"
    data = torch.load(args.input, map_location=device)

    print("Scene completion: encoding partial scene...")

    # For simplicity, use the first half of the scene as context
    voxel_coords = data.get("voxel_coords", torch.zeros(0, 3, dtype=torch.long))
    voxel_codes = data.get("voxel_codes", torch.zeros(0, dtype=torch.long))

    BOS = gpt.BOS
    EOS = gpt.EOS

    if len(voxel_coords) > 0:
        # serialize_latent_grid expects latent-space coords; gpt.chunk_size is latent chunk size
        prefix_tokens, prefix_coords, prefix_types = serialize_latent_grid(
            voxel_coords, voxel_codes, gpt.chunk_size, BOS, EOS
        )
        # Use first half as prefix
        half = len(prefix_tokens) // 2
        prefix_tokens = prefix_tokens[:half].unsqueeze(0).to(device)
        prefix_coords = prefix_coords[:half].unsqueeze(0).to(device)
        prefix_types = prefix_types[:half].unsqueeze(0).to(device)
    else:
        prefix_tokens = torch.tensor([[BOS]], device=device)
        prefix_coords = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
        prefix_types = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        tokens, coords, token_type = gpt.generate(
            prefix_tokens=prefix_tokens,
            prefix_coords=prefix_coords,
            prefix_token_type=prefix_types,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=gpt.context_len,
            device=device,
        )

    gaussians = decode_tokens_to_gaussians(ae, gpt, tokens, coords, token_type, device)
    return [gaussians] if gaussians is not None else []


def generate_outpainting(ae, gpt, args, device):
    """Generate a large scene via sliding-window outpainting."""
    # Latent chunk dimensions (GPT operates on latent voxels)
    lcx, lcy, lcz = gpt.chunk_size
    target_chunks = args.target_size

    print(f"Generating large scene ({target_chunks}x{target_chunks} chunks)...")

    all_gaussians = []

    # Step 1: Generate initial seed chunk
    with torch.no_grad():
        tokens, coords, token_type = gpt.generate(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=gpt.context_len,
            device=device,
        )

    seed_gaussians = decode_tokens_to_gaussians(ae, gpt, tokens, coords, token_type, device)
    if seed_gaussians is not None:
        all_gaussians.append((seed_gaussians, torch.zeros(3, dtype=torch.long)))

    # Step 2: Outpaint along x and y directions
    for chunk_y in range(target_chunks):
        for chunk_x in range(target_chunks):
            if chunk_x == 0 and chunk_y == 0:
                continue  # Already generated seed

            offset = torch.tensor([chunk_x * lcx, chunk_y * lcy, 0], dtype=torch.long)

            # Resample if empty (up to 5 retries)
            gaussians = None
            for retry in range(5):
                with torch.no_grad():
                    tokens, coords, token_type = gpt.generate(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=gpt.context_len,
                        device=device,
                    )
                gaussians = decode_tokens_to_gaussians(
                    ae, gpt, tokens, coords, token_type, device
                )
                if gaussians is not None and len(gaussians.get("voxel_coords", [])) > 0:
                    break

            if gaussians is not None:
                all_gaussians.append((gaussians, offset))

    return all_gaussians


def main():
    args = parse_args()
    ae, gpt, device = load_models(args)

    if args.mode == "unconditional":
        results = generate_unconditional(ae, gpt, args, device)
    elif args.mode == "completion":
        results = generate_completion(ae, gpt, args, device)
    elif args.mode == "outpainting":
        results = generate_outpainting(ae, gpt, args, device)

    # Save results
    torch.save(results, args.output)
    print(f"Saved {len(results)} generated scene(s) to {args.output}")


if __name__ == "__main__":
    main()
