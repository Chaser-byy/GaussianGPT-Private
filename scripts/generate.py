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
    # ae_ckpt = torch.load(args.ae_checkpoint, map_location=device)
    # ae.load_state_dict(ae_ckpt["model"] if "model" in ae_ckpt else ae_ckpt)
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
    # gpt_ckpt = torch.load(args.gpt_checkpoint, map_location=device)
    # gpt.load_state_dict(gpt_ckpt["model"] if "model" in gpt_ckpt else gpt_ckpt)
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
        latent_stride = 2**ae.n_down   # 8 = 2 ** 3暂时写死, 后面可以改
        sparse_input = ME.SparseTensor(features=z_q, coordinates=coords_me, tensor_stride=latent_stride)
        decoded, _ = ae.decoder(sparse_input)
        feat = decoded.F
        voxel_coords = decoded.C[:, 1:]
    else:
        assert False, "Dense fallback not implemented"

    gaussians = ae.attr_decoder(feat)
    gaussians["voxel_coords"] = voxel_coords
    # Convert offset (relative to voxel center) to absolute world-space positions.
    # Each latent voxel covers a block of size 2^n_down base voxels; the voxel
    # center in base-voxel units is (coord + 0.5) * voxel_size.
    n_down = ae.encoder.downs.__len__() if hasattr(ae.encoder, 'downs') else 3
    voxel_size = float(2 ** n_down)  # base voxels per latent voxel
    vc_world = (voxel_coords.float().to(device) + 0.5) * voxel_size  # (N, 3)
    gaussians["position"] = vc_world + gaussians["offset"]  # absolute world coords
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

    # Also export as standard 3DGS .ply (one file per sample)
    base, ext = os.path.splitext(args.output)
    for idx, item in enumerate(results):
        # outpainting returns (gaussians, offset) tuples; others return plain dicts
        if isinstance(item, tuple):
            g, chunk_offset = item
            # shift positions by chunk offset (in base-voxel units)
            n_down = args.ae_n_down
            voxel_size = float(2 ** n_down)
            g = dict(g)
            if "position" in g:
                g["position"] = g["position"] + chunk_offset.float().to(g["position"].device) * voxel_size
        else:
            g = item

        ply_path = f"{base}_{idx}.ply" if len(results) > 1 else f"{base}.ply"
        _save_3dgs_ply(g, ply_path)
        print(f"  -> 3DGS ply: {ply_path}")


def _save_3dgs_ply(gaussians: dict, path: str):
    """Export Gaussians to standard 3DGS .ply format.

    Handles two cases automatically:
      - Scene model (use_sh=False): 'color' RGB -> converted to SH DC term, f_rest zeros
      - Object model (use_sh=True): 'sh' (12-dim, degree-1) -> f_dc + f_rest, remaining zeros
    """
    pos = gaussians["position"].cpu().float()   # (N, 3)
    N = pos.shape[0]

    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

    if "sh" in gaussians:
        # Object model: sh is (N, 12) = 4 coefficients * 3 channels (degree-1 SH)
        # Layout: [c0_r, c0_g, c0_b, c1_r, c1_g, c1_b, c2_r, c2_g, c2_b, c3_r, c3_g, c3_b]
        sh = gaussians["sh"].cpu().float()  # (N, 12)
        f_dc = sh[:, :3]                    # degree-0 coefficients, already in SH space
        f_rest_deg1 = sh[:, 3:]             # degree-1 coefficients (9 dims)
        # Pad to standard 45 f_rest (degree 1-3); degree 2-3 are zeros
        f_rest = torch.zeros(N, 45, dtype=torch.float32)
        f_rest[:, :9] = f_rest_deg1
    else:
        # Scene model: color RGB [0,1] -> SH DC term
        color = gaussians["color"].cpu().float().clamp(0, 1)  # (N, 3)
        f_dc = (color - 0.5) / C0
        f_rest = torch.zeros(N, 45, dtype=torch.float32)

    opacity = gaussians["opacity"].cpu().float()  # (N, 1), already in logit space
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(1)

    # scale: 3DGS stores log(scale)
    scale = gaussians["scale"].cpu().float()  # (N, 3), softplus output
    log_scale = torch.log(scale.clamp(min=1e-6))

    rot = gaussians["rotation"].cpu().float()  # (N, 4), unit quaternion (w x y z)

    # Build .ply header
    props = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    n_props = len(props)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
    )
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"

    normals = torch.zeros(N, 3, dtype=torch.float32)

    # Stack all properties: (N, n_props)
    data = torch.cat([
        pos, normals, f_dc, f_rest, opacity, log_scale, rot
    ], dim=1)  # (N, n_props)

    assert data.shape[1] == n_props, f"Expected {n_props} props, got {data.shape[1]}"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.detach().numpy().astype("float32").tobytes())


if __name__ == "__main__":
    main()
