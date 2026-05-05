"""Full GaussianGPT Autoencoder: Gaussian -> sparse latent grid -> Gaussian."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .gaussian_heads import GaussianAttributeEncoder, GaussianAttributeDecoder
from .sparse_cnn import SparseEncoder, SparseDecoder, HAS_MINKOWSKI
from .quantizer import LookupFreeQuantizer


class GaussianAutoencoder(nn.Module):
    """
    Full autoencoder pipeline:
      1. Encode Gaussian attributes -> per-voxel feature vector (attr_encoder)
      2. Sparse 3D CNN encoder -> latent grid
      3. LFQ vector quantization -> discrete codes
      4. Sparse 3D CNN decoder -> per-voxel feature vector
      5. Decode feature vector -> Gaussian attributes (attr_decoder)

    Paper config (scenes):
      - base voxel size: 2.5 cm, 3 downsampling stages -> 20 cm latent voxels
      - codebook_size: 4096 (12 bits LFQ)
      - base_ch: 128 (per L3DG Sec. 3.4: encoder stem -> 128 channels, then
        each downsample doubles -> [128, 256, 512] for objects (n_down=2),
        [128, 256, 512, 1024] for scenes (n_down=3)).
      - latent_ch: 12 (= num_bits for LFQ)
    """

    def __init__(
        self,
        base_ch: int = 128,
        n_down: int = 3,
        codebook_size: int = 4096,
        use_sh: bool = False,
        encoder_expand: int = 16,
        decoder_hidden: int = 64,
        voxel_size: float = 0.025,
        norm: str = "bn",
        color_activation: str = "clamp",
    ):
        super().__init__()
        num_bits = int(math.log2(codebook_size))
        assert 2 ** num_bits == codebook_size, "codebook_size must be a power of 2"

        # Gaussian attribute encoder/decoder heads. Both consume voxel_size
        # so that input normalisation and the softplus-scale init bias stay
        # consistent across the round-trip.
        self.attr_encoder = GaussianAttributeEncoder(
            use_sh=use_sh, expand=encoder_expand, voxel_size=voxel_size,
        )
        in_ch = self.attr_encoder.out_dim

        # Sparse 3D CNN (latent_ch == num_bits for LFQ). ``norm`` selects
        # between BatchNorm (paper-faithful) / InstanceNorm / Identity.
        self.encoder = SparseEncoder(
            in_ch=in_ch, base_ch=base_ch, latent_ch=num_bits, n_down=n_down, norm=norm,
        )
        self.decoder = SparseDecoder(
            latent_ch=num_bits, base_ch=base_ch, out_ch=in_ch, n_up=n_down, norm=norm,
        )

        # LFQ quantizer
        self.quantizer = LookupFreeQuantizer(codebook_size=codebook_size)
        self.n_down = n_down
        self.voxel_size = float(voxel_size)

        # Gaussian attribute decoder. Per GaussianGPT Appendix C, offsets
        # are predicted as unbounded world-space values (no offset_bound).
        # ``color_activation`` defaults to the paper's hard clamp; setting
        # it to "sigmoid" relieves the clamp's dead-gradient pathology.
        self.attr_decoder = GaussianAttributeDecoder(
            in_dim=in_ch,
            use_sh=use_sh,
            hidden=decoder_hidden,
            voxel_size=voxel_size,
            color_activation=color_activation,
        )

    def _make_sparse_tensor(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor):
        """Wrap dense features + coords into a MinkowskiEngine SparseTensor.

        Args:
            voxel_features: (N, C) float tensor
            voxel_coords: (N, 3) integer coordinates
        Returns:
            ME.SparseTensor with batch dimension prepended (all batch_idx=0)
        """
        import MinkowskiEngine as ME
        batch_idx = torch.zeros(voxel_coords.shape[0], 1, dtype=torch.int, device=voxel_coords.device)
        coords_me = torch.cat([batch_idx, voxel_coords.int()], dim=1)  # (N, 4): [batch, x, y, z]
        return ME.SparseTensor(features=voxel_features, coordinates=coords_me)

    def forward(
        self,
        gaussians: Dict[str, torch.Tensor],
        voxel_coords: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            gaussians: dict of Gaussian attribute tensors, each (N, attr_dim)
            voxel_coords: (N, 3) integer voxel coordinates (relative to chunk)
        Returns:
            pred_gaussians: dict of reconstructed Gaussian attributes
            occ_logits_list: list of occupancy logits per decoder upsampling stage
            lfq_loss: LFQ entropy loss scalar
            indices: (N,) codebook indices
        """
        # Step 1: encode Gaussian attributes to per-voxel features
        voxel_features = self.attr_encoder(gaussians)  # (N, in_ch)

        # Step 2: sparse 3D CNN encoder
        if HAS_MINKOWSKI:
            sparse_in = self._make_sparse_tensor(voxel_features, voxel_coords)
            z_sparse = self.encoder(sparse_in)
            z = z_sparse.F  # (N_latent, num_bits)
        else:
            # Dense fallback: place features in a (1, C, X, Y, Z) grid
            max_coord = voxel_coords.max(0).values  # (3,)
            gx = int(max_coord[0].item()) + 1
            gy = int(max_coord[1].item()) + 1
            gz = int(max_coord[2].item()) + 1
            C = voxel_features.shape[-1]
            grid = torch.zeros(1, C, gx, gy, gz, device=voxel_features.device)
            vc = voxel_coords
            grid[0, :, vc[:, 0], vc[:, 1], vc[:, 2]] = voxel_features.T
            z_grid = self.encoder(grid)  # (1, num_bits, gx', gy', gz')
            # Use stored n_down for exact downsampling factor
            scale = 2 ** self.n_down
            latent_vc = (vc // scale).clamp(0, z_grid.shape[2] - 1)
            z = z_grid[0, :, latent_vc[:, 0], latent_vc[:, 1], latent_vc[:, 2]].T  # (N, num_bits)

        # Step 3: LFQ quantization
        z_q, indices, lfq_loss = self.quantizer(z)  # z_q: (N_latent, num_bits)

        # Step 4: sparse 3D CNN decoder
        if HAS_MINKOWSKI:
            # Reuse the sparse structure from encoder output, replace features with z_q
            import MinkowskiEngine as ME
            z_q_sparse = ME.SparseTensor(
                features=z_q,
                coordinate_map_key=z_sparse.coordinate_map_key,
                coordinate_manager=z_sparse.coordinate_manager,
            )
            decoded_sparse, occ_list = self.decoder(z_q_sparse)
            decoded_feat = decoded_sparse.F  # (N_latent, in_ch)
        else:
            # Dense fallback: place z_q back in grid, decode, read at voxel positions
            num_bits = z_q.shape[-1]
            z_q_grid = torch.zeros(1, num_bits, z_grid.shape[2], z_grid.shape[3], z_grid.shape[4],
                                   device=z_q.device)
            z_q_grid[0, :, latent_vc[:, 0], latent_vc[:, 1], latent_vc[:, 2]] = z_q.T
            decoded_grid, occ_list = self.decoder(z_q_grid)  # (1, in_ch, gx, gy, gz)
            decoded_feat = decoded_grid[0, :, vc[:, 0], vc[:, 1], vc[:, 2]].T  # (N, in_ch)

        # Step 5: decode features back to Gaussian attributes
        pred_gaussians = self.attr_decoder(decoded_feat)
        return pred_gaussians, occ_list, lfq_loss, indices
