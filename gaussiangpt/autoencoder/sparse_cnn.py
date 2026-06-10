"""Sparse 3D CNN Encoder-Decoder following L3DG architecture.

Normalisation layers are configurable via the ``norm`` argument so that
training-time pathologies of BatchNorm (small batch + heterogeneous
chunks) can be diagnosed by swapping in InstanceNorm or no-op:

  * ``"bn"``       -- MinkowskiBatchNorm (paper-faithful, default).
  * ``"instance"`` -- MinkowskiInstanceNorm; safer for batch_size <= 4.
  * ``"none"``     -- nn.Identity; useful to isolate norm-related issues.

Decoder upsampling can optionally use MinkowskiEngine's generative transpose
convolution. The default remains the existing ``MinkowskiConvolutionTranspose``.
"""
import torch
import torch.nn as nn
from typing import List, Optional

STRIDE2_KERNEL_SIZE = 3


def _tensor_stride_int(tensor) -> int:
    tensor_stride = getattr(tensor, "tensor_stride", 1)
    if isinstance(tensor_stride, (list, tuple)):
        return max(int(tensor_stride[0]), 1)
    return max(int(tensor_stride), 1)


def _kernel_map_input_indices(mapping, device: torch.device) -> torch.Tensor:
    """Return input row indices from a MinkowskiEngine kernel map."""

    def _input_from_pair(in_out):
        if isinstance(in_out, (list, tuple)) and len(in_out) == 2:
            return in_out[0]
        if torch.is_tensor(in_out) and in_out.dim() == 2:
            if in_out.shape[0] == 2:
                return in_out[0]
            if in_out.shape[1] == 2:
                return in_out[:, 0]
        raise RuntimeError("Unexpected MinkowskiEngine kernel_map return format.")

    if isinstance(mapping, dict):
        pieces = []
        for in_out in mapping.values():
            input_indices = _input_from_pair(in_out)
            if input_indices.numel() > 0:
                pieces.append(input_indices.to(device=device, dtype=torch.long))
        if not pieces:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(pieces, dim=0)

    if isinstance(mapping, (list, tuple)) and len(mapping) == 2:
        input_indices = mapping[0]
        if torch.is_tensor(input_indices):
            return input_indices.to(device=device, dtype=torch.long)

    raise RuntimeError("Unexpected MinkowskiEngine kernel_map return format.")

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

    def _make_transpose_conv(
        in_ch: int,
        out_ch: int,
        use_generative_transpose: bool = False,
    ) -> nn.Module:
        if use_generative_transpose:
            conv_cls = getattr(ME, "MinkowskiGenerativeConvolutionTranspose", None)
            if conv_cls is None:
                raise RuntimeError(
                    "use_generative_transpose=True requires "
                    "MinkowskiEngine.MinkowskiGenerativeConvolutionTranspose."
                )
        else:
            conv_cls = ME.MinkowskiConvolutionTranspose
        return conv_cls(
            in_ch,
            out_ch,
            kernel_size=STRIDE2_KERNEL_SIZE,
            stride=2,
            dimension=3,
        )

    def _make_conv_block(in_ch: int, out_ch: int, stride: int, norm: str) -> nn.Module:
        return nn.Sequential(
            ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, stride=stride, dimension=3),
            _make_norm(norm, out_ch),
            ME.MinkowskiReLU(inplace=True),
        )

    def _make_transpose_conv_block(
        in_ch: int,
        out_ch: int,
        norm: str,
        use_generative_transpose: bool = False,
    ) -> nn.Module:
        return nn.Sequential(
            _make_transpose_conv(
                in_ch,
                out_ch,
                use_generative_transpose=use_generative_transpose,
            ),
            _make_norm(norm, out_ch),
            ME.MinkowskiReLU(inplace=True),
        )

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
            self.conv = _make_conv_block(out_ch, out_ch, stride=2, norm=norm)
            self.res1 = SparseResBlock(in_ch, out_ch, norm=norm)
            self.res2 = SparseResBlock(out_ch, out_ch, norm=norm)

        def forward(self, x):
            x = self.res1(x)
            x = self.res2(x)
            return self.conv(x)

    class SparseUpBlock(nn.Module):
        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            norm: str = "bn",
            use_generative_transpose: bool = False,
        ):
            super().__init__()
            self.res1 = SparseResBlock(in_ch, out_ch, norm=norm)
            self.res2 = SparseResBlock(out_ch, out_ch, norm=norm)
            self.transpose = _make_transpose_conv_block(
                out_ch,
                out_ch,
                norm=norm,
                use_generative_transpose=use_generative_transpose,
            )
            self.occ_head = ME.MinkowskiLinear(out_ch, 1)

        def forward(self, x):
            x = self.res1(x)
            x = self.res2(x)
            x = self.transpose(x)
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
            self.conv1 = _make_conv_block(in_ch, base_ch, stride=1, norm=norm)

            chs = [base_ch * (2 ** i) for i in range(n_down + 1)]
            self.downs = nn.ModuleList(
                [SparseDownBlock(chs[i], chs[i + 1], norm=norm) for i in range(n_down)]
            )
            self.proj = ME.MinkowskiConvolution(chs[-1], latent_ch, kernel_size=1, stride=1, dimension=3)
            self.res = SparseResBlock(chs[-1], chs[-1], norm=norm)
            self.encoder_target_keys = {}

        def _cache_target_key(self, x) -> None:
            self.encoder_target_keys[_tensor_stride_int(x)] = x.coordinate_map_key

        def forward(self, x):
            self.encoder_target_keys = {}
            x = self.conv1(x)
            self._cache_target_key(x)
            for d in self.downs:
                x = d(x)
                self._cache_target_key(x)
            x = self.res(x)
            return self.proj(x)

    class SparseDecoder(nn.Module):
        def __init__(
            self,
            latent_ch: int,
            base_ch: int = 128,
            out_ch: int = 224,
            n_up: int = 3,
            norm: str = "bn",
            use_generative_transpose: bool = False,
        ):
            super().__init__()
            chs = list(reversed([base_ch * (2 ** i) for i in range(n_up + 1)]))
            self.conv1 = _make_conv_block(latent_ch, chs[0], stride=1, norm=norm)
            self.ups = nn.ModuleList(
                [
                    SparseUpBlock(
                        chs[i],
                        chs[i + 1],
                        norm=norm,
                        use_generative_transpose=use_generative_transpose,
                    )
                    for i in range(n_up)
                ]
            )
            self.res1 = SparseResBlock(chs[-1], chs[-1], norm=norm)
            self.res2 = SparseResBlock(chs[-1], chs[-1], norm=norm)
            self.out_proj = ME.MinkowskiConvolution(chs[-1], out_ch, kernel_size=1, stride=1, dimension=3)
            self.pruning = ME.MinkowskiPruning()

        @staticmethod
        def _occupancy_keep_mask(occ, threshold: float, min_keep: int) -> torch.Tensor:
            logits = occ.F.squeeze(-1)
            keep = torch.sigmoid(logits) >= float(threshold)
            if min_keep > 0 and logits.numel() > 0 and int(keep.sum().item()) < min_keep:
                topk = min(int(min_keep), logits.numel())
                _, idx = torch.topk(logits, k=topk)
                keep = torch.zeros_like(keep)
                keep[idx] = True
            return keep

        @staticmethod
        def _encoder_target_keep_mask(occ, encoder_target_keys: dict) -> tuple:
            occ_logits = occ.F.squeeze(-1)
            stride = _tensor_stride_int(occ)
            targets = torch.zeros_like(occ_logits)
            keep = torch.zeros(
                occ.F.shape[0],
                dtype=torch.bool,
                device=occ.F.device,
            )

            target_key = encoder_target_keys.get(stride)
            if target_key is None:
                available = sorted(int(k) for k in encoder_target_keys.keys())
                raise KeyError(
                    "Missing encoder target coordinate_map_key for decoder "
                    f"stride {stride}; available strides: {available}."
                )

            coord_manager = getattr(occ, "coordinate_manager", None)
            if coord_manager is None or getattr(occ, "coordinate_map_key", None) is None:
                raise RuntimeError(
                    "GT decoder pruning requires SparseTensor coordinate manager state."
                )

            mapping = coord_manager.kernel_map(
                occ.coordinate_map_key,
                target_key,
                kernel_size=1,
                stride=1,
                dilation=1,
            )
            occ_indices = _kernel_map_input_indices(mapping, occ.F.device)
            if occ_indices.numel() > 0:
                if (
                    int(occ_indices.min().item()) < 0
                    or int(occ_indices.max().item()) >= occ.F.shape[0]
                ):
                    raise RuntimeError(
                        "MinkowskiEngine kernel_map returned out-of-range "
                        "decoder occupancy indices."
                    )
                keep[occ_indices] = True
                targets[occ_indices] = 1

            if keep.numel() != occ.F.shape[0]:
                raise ValueError(
                    "Decoder GT prune mask length must match stage occupancy "
                    f"logits: got {keep.numel()} mask values for "
                    f"{occ.F.shape[0]} sparse voxels."
                )
            if targets.numel() != occ_logits.numel():
                raise ValueError(
                    "Decoder occupancy targets length must match occupancy logits: "
                    f"got {targets.numel()} targets for {occ_logits.numel()} logits."
                )
            return keep, occ_logits, targets, stride

        def forward(
            self,
            x,
            prune: bool = False,
            occupancy_threshold: float = 0.5,
            min_keep: int = 1,
            prune_mask_fn=None,
            encoder_target_keys: Optional[dict] = None,
            prune_with_encoder_targets: bool = False,
            occ_target_cache: Optional[dict] = None,
        ):
            x = self.conv1(x)
            occ_list = []
            for stage_idx, u in enumerate(self.ups):
                x, occ = u(x)
                occ_list.append(occ)
                encoder_keep = None
                if encoder_target_keys is not None:
                    encoder_keep, occ_logits, targets, stride = self._encoder_target_keep_mask(
                        occ,
                        encoder_target_keys,
                    )
                    if occ_target_cache is not None:
                        occ_target_cache[stage_idx] = (occ_logits, targets, stride)
                if prune:
                    if prune_with_encoder_targets:
                        if encoder_keep is None:
                            raise RuntimeError(
                                "GT decoder pruning requires encoder target keys."
                            )
                        keep = encoder_keep
                    elif prune_mask_fn is None:
                        keep = self._occupancy_keep_mask(
                            occ, occupancy_threshold, min_keep
                        )
                    else:
                        keep = prune_mask_fn(occ, stage_idx, len(self.ups))
                    keep = keep.to(device=occ.F.device, dtype=torch.bool)
                    if keep.numel() != occ.F.shape[0]:
                        raise ValueError(
                            "Decoder prune mask length must match stage occupancy "
                            f"logits: got {keep.numel()} mask values for "
                            f"{occ.F.shape[0]} sparse voxels at stage {stage_idx}."
                        )
                    x = self.pruning(x, keep)
            x = self.res1(x)
            x = self.res2(x)
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
                    nn.Conv3d(
                        chs[i],
                        chs[i + 1],
                        STRIDE2_KERNEL_SIZE,
                        stride=2,
                        padding=STRIDE2_KERNEL_SIZE // 2,
                    ),
                    nn.ReLU(inplace=True),
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
            use_generative_transpose: bool = False,  # accepted for API parity; dense fallback ignores it
        ):
            super().__init__()
            chs = list(reversed([base_ch * (2 ** i) for i in range(n_up + 1)]))
            self.proj = nn.Conv3d(latent_ch, chs[0], 1)
            ups = []
            for i in range(n_up):
                ups += [
                    nn.ConvTranspose3d(
                        chs[i],
                        chs[i + 1],
                        STRIDE2_KERNEL_SIZE,
                        stride=2,
                        padding=STRIDE2_KERNEL_SIZE // 2,
                        output_padding=1,
                    ),
                    nn.ReLU(inplace=True),
                    DenseResBlock(chs[i + 1]),
                ]
            self.ups = nn.Sequential(*ups)
            self.out_proj = nn.Conv3d(chs[-1], out_ch or base_ch, 1)
            # Dummy occ heads for API compatibility
            self.occ_heads = nn.ModuleList([nn.Conv3d(chs[i + 1], 1, 1) for i in range(n_up)])
            self._n_up = n_up
            self._chs = chs

        @staticmethod
        def _dense_keep_mask(occ: torch.Tensor, threshold: float, min_keep: int) -> torch.Tensor:
            keep = torch.sigmoid(occ) >= float(threshold)
            if min_keep > 0 and occ.numel() > 0 and int(keep.sum().item()) < min_keep:
                flat_logits = occ.reshape(-1)
                topk = min(int(min_keep), flat_logits.numel())
                _, idx = torch.topk(flat_logits, k=topk)
                flat_keep = torch.zeros_like(flat_logits, dtype=torch.bool)
                flat_keep[idx] = True
                keep = flat_keep.reshape_as(occ)
            return keep

        def forward(
            self,
            x: torch.Tensor,
            prune: bool = False,
            occupancy_threshold: float = 0.5,
            min_keep: int = 1,
            prune_mask_fn=None,
        ):
            x = self.proj(x)
            occ_list = []
            idx = 0
            for i in range(self._n_up):
                # ConvTranspose + ReLU + ResBlock
                x = self.ups[idx](x); idx += 1
                x = self.ups[idx](x); idx += 1
                x = self.ups[idx](x); idx += 1
                occ = self.occ_heads[i](x)
                occ_list.append(occ)
                if prune:
                    if prune_mask_fn is None:
                        keep = self._dense_keep_mask(occ, occupancy_threshold, min_keep)
                    else:
                        keep = prune_mask_fn(occ, i, self._n_up)
                    keep = keep.to(device=occ.device, dtype=torch.bool)
                    if keep.shape != occ.shape:
                        raise ValueError(
                            "Dense decoder prune mask shape must match occupancy "
                            f"logits: got {tuple(keep.shape)} for {tuple(occ.shape)} "
                            f"at stage {i}."
                        )
                    x = x * keep.to(dtype=x.dtype)
            return self.out_proj(x), occ_list
