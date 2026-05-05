"""Sparse 3D CNN Encoder-Decoder following L3DG architecture.

Normalisation layers are configurable via the ``norm`` argument so that
training-time pathologies of BatchNorm (small batch + heterogeneous
chunks) can be diagnosed by swapping in InstanceNorm or no-op:

  * ``"bn"``       -- MinkowskiBatchNorm (paper-faithful, default).
  * ``"instance"`` -- MinkowskiInstanceNorm; safer for batch_size <= 4.
  * ``"none"``     -- nn.Identity; useful to isolate norm-related issues.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
except ImportError:
    HAS_MINKOWSKI = False
    import warnings
    warnings.warn("MinkowskiEngine not found. Using dense 3D CNN fallback.")


if HAS_MINKOWSKI:

    def _make_norm(kind: str, out_ch: int) -> nn.Module:
        """Build a sparse normalisation layer by name.

        ``kind`` is case-insensitive and accepts ``bn``, ``instance``, ``none``.
        """
        kind = (kind or "bn").lower()
        if kind == "bn":
            return ME.MinkowskiBatchNorm(out_ch)
        if kind in ("instance", "in"):
            return ME.MinkowskiInstanceNorm(out_ch)
        if kind in ("none", "id", "identity"):
            return nn.Identity()
        raise ValueError(f"Unknown norm kind: {kind!r}")

    class SparseResBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
            super().__init__()
            self.conv1 = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, stride=1, dimension=3)
            self.bn1 = _make_norm(norm, out_ch)
            self.conv2 = ME.MinkowskiConvolution(out_ch, out_ch, kernel_size=3, stride=1, dimension=3)
            self.bn2 = _make_norm(norm, out_ch)
            self.relu = ME.MinkowskiReLU(inplace=True)
            self.skip = (
                ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=1, stride=1, dimension=3)
                if in_ch != out_ch else nn.Identity()
            )

        def forward(self, x):
            res = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + res)

    class SparseDownBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
            super().__init__()
            self.conv = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=2, stride=2, dimension=3)
            self.bn = _make_norm(norm, out_ch)
            self.relu = ME.MinkowskiReLU(inplace=True)
            self.res = SparseResBlock(out_ch, out_ch, norm=norm)

        def forward(self, x):
            return self.res(self.relu(self.bn(self.conv(x))))

    class SparseUpBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
            super().__init__()
            self.conv = ME.MinkowskiConvolutionTranspose(in_ch, out_ch, kernel_size=2, stride=2, dimension=3)
            self.bn = _make_norm(norm, out_ch)
            self.relu = ME.MinkowskiReLU(inplace=True)
            self.res = SparseResBlock(out_ch, out_ch, norm=norm)
            self.occ_head = ME.MinkowskiLinear(out_ch, 1)

        def forward(self, x):
            x = self.res(self.relu(self.bn(self.conv(x))))
            return x, self.occ_head(x)

    class SparseEncoder(nn.Module):
        def __init__(
            self,
            in_ch: int,
            base_ch: int = 128,
            latent_ch: int = 12,
            n_down: int = 3,
            norm: str = "bn",
        ):
            super().__init__()
            self.stem = ME.MinkowskiConvolution(in_ch, base_ch, kernel_size=3, stride=1, dimension=3)
            self.stem_bn = _make_norm(norm, base_ch)
            self.stem_relu = ME.MinkowskiReLU(inplace=True)
            chs = [base_ch * (2 ** i) for i in range(n_down + 1)]
            self.downs = nn.ModuleList(
                [SparseDownBlock(chs[i], chs[i + 1], norm=norm) for i in range(n_down)]
            )
            self.proj = ME.MinkowskiConvolution(chs[-1], latent_ch, kernel_size=1, stride=1, dimension=3)

        def forward(self, x):
            x = self.stem_relu(self.stem_bn(self.stem(x)))
            for d in self.downs:
                x = d(x)
            return self.proj(x)

    class SparseDecoder(nn.Module):
        def __init__(
            self,
            latent_ch: int,
            base_ch: int = 128,
            out_ch: Optional[int] = None,
            n_up: int = 3,
            norm: str = "bn",
        ):
            super().__init__()
            chs = list(reversed([base_ch * (2 ** i) for i in range(n_up + 1)]))
            self.proj = ME.MinkowskiConvolution(latent_ch, chs[0], kernel_size=1, stride=1, dimension=3)
            self.ups = nn.ModuleList(
                [SparseUpBlock(chs[i], chs[i + 1], norm=norm) for i in range(n_up)]
            )
            self.out_proj = ME.MinkowskiConvolution(chs[-1], out_ch or base_ch, kernel_size=1, stride=1, dimension=3)

        def forward(self, x):
            x = self.proj(x)
            occ_list = []
            for u in self.ups:
                x, occ = u(x)
                occ_list.append(occ)
            return self.out_proj(x), occ_list

else:

    class DenseResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(ch, ch, 3, padding=1), nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
                nn.Conv3d(ch, ch, 3, padding=1), nn.BatchNorm3d(ch),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.net(x) + x)

    class SparseEncoder(nn.Module):
        def __init__(
            self,
            in_ch: int,
            base_ch: int = 128,
            latent_ch: int = 12,
            n_down: int = 3,
            norm: str = "bn",  # accepted for API parity; dense fallback ignores it
        ):
            super().__init__()
            chs = [base_ch * (2 ** i) for i in range(n_down + 1)]
            layers: List[nn.Module] = [
                nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True),
                DenseResBlock(base_ch),
            ]
            for i in range(n_down):
                layers += [
                    nn.Conv3d(chs[i], chs[i + 1], 2, stride=2), nn.ReLU(inplace=True),
                    DenseResBlock(chs[i + 1]),
                ]
            layers.append(nn.Conv3d(chs[-1], latent_ch, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class SparseDecoder(nn.Module):
        def __init__(
            self,
            latent_ch: int,
            base_ch: int = 128,
            out_ch: Optional[int] = None,
            n_up: int = 3,
            norm: str = "bn",  # accepted for API parity; dense fallback ignores it
        ):
            super().__init__()
            chs = list(reversed([base_ch * (2 ** i) for i in range(n_up + 1)]))
            self.proj = nn.Conv3d(latent_ch, chs[0], 1)
            ups = []
            for i in range(n_up):
                ups += [
                    nn.ConvTranspose3d(chs[i], chs[i + 1], 2, stride=2),
                    nn.ReLU(inplace=True),
                    DenseResBlock(chs[i + 1]),
                ]
            self.ups = nn.Sequential(*ups)
            self.out_proj = nn.Conv3d(chs[-1], out_ch or base_ch, 1)
            # Dummy occ heads for API compatibility
            self.occ_heads = nn.ModuleList([nn.Conv3d(chs[i + 1], 1, 1) for i in range(n_up)])
            self._n_up = n_up
            self._chs = chs

        def forward(self, x: torch.Tensor):
            x = self.proj(x)
            occ_list = []
            idx = 0
            for i in range(self._n_up):
                # ConvTranspose + ReLU + ResBlock
                x = self.ups[idx](x); idx += 1
                x = self.ups[idx](x); idx += 1
                x = self.ups[idx](x); idx += 1
                occ_list.append(self.occ_heads[i](x))
            return self.out_proj(x), occ_list
