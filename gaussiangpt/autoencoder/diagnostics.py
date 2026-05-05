"""Training-time diagnostics for the Gaussian autoencoder.

The single most useful "is my model healthy?" probe is to inspect the
*pre-activation* outputs of the per-attribute decoder heads:

  * Color head   -> tells us whether the activation (clamp / sigmoid)
                    is killing or saturating gradients.
  * Opacity head -> tells us whether Gaussians are being "decisive"
                    (alpha near 0 or 1) or all clustered in the
                    "softly averaged" middle band, which makes the
                    rasterizer blur vivid colours into mush.
  * Scale head   -> tells us whether Gaussians stay near voxel-size
                    or balloon outwards (also a desaturation cause).

All three live on the same per-voxel feature tensor, so we can collect
them in one forward pass via a forward-pre-hook on the decoder.

Usage in a training loop:

    from gaussiangpt.autoencoder.diagnostics import ColorClampDiagnostics

    diag = ColorClampDiagnostics(raw_model.attr_decoder, every=200,
                                  voxel_size=cfg["data"]["base_voxel_size"])
    ...
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = compute_loss(...)
        loss.backward()
        optimizer.step()
        diag.maybe_log(global_step)
        diag.clear()
    diag.close()
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .gaussian_heads import GaussianAttributeDecoder


# ---------------------------------------------------------------------------
# Per-head statistics (no_grad; called every `every` steps).
# ---------------------------------------------------------------------------


@torch.no_grad()
def color_saturation_stats(
    decoder: GaussianAttributeDecoder,
    decoded_feat: torch.Tensor,
) -> Dict[str, float]:
    """Re-run the colour head WITHOUT post-processing and return stats
    relevant to BOTH activation modes (clamp and sigmoid):

      * ``color_frac_clamp_dead`` : fraction outside [0, 1]; under
        ``color_activation='clamp'`` these contribute zero gradient.
        (Under sigmoid this number is meaningless -- sigmoid happily
        maps any raw value into (0, 1), so a high "frac dead" only
        means the predictions are not centred on 0.5.)
      * ``color_frac_sigmoid_sat`` : fraction with |raw| > 5; under
        ``color_activation='sigmoid'`` these contribute < 1% of the
        max sigmoid gradient (so they are effectively saturated).

    Plus raw min/max/mean/std and the current ``proj_out`` bias.
    """
    head = decoder.heads["color"]
    raw = head(decoded_feat)  # (N, 3)
    return {
        "color_raw_min":           float(raw.min()),
        "color_raw_max":           float(raw.max()),
        "color_raw_mean":          float(raw.mean()),
        "color_raw_std":           float(raw.std()),
        "color_frac_clamp_dead":   float(((raw < 0.0) | (raw > 1.0)).float().mean()),
        "color_frac_sigmoid_sat":  float((raw.abs() > 5.0).float().mean()),
        "color_bias":              head.proj_out.bias.detach().cpu().tolist(),
    }


@torch.no_grad()
def opacity_distribution_stats(
    decoder: GaussianAttributeDecoder,
    decoded_feat: torch.Tensor,
) -> Dict[str, float]:
    """Re-run the opacity head and return distribution stats on
    *post-sigmoid* alpha (which is what the rasterizer actually uses).

    The most diagnostic number is ``opa_frac_decisive`` -- the fraction
    of Gaussians with alpha near 0 *or* near 1. A healthy 3DGS solution
    has many decisive Gaussians (sharp surface boundaries); a "soft
    averaging" solution has most alphas clustered near 0.5, which is
    exactly the failure mode that turns vivid colours into grey mush.
    """
    head = decoder.heads["opacity"]
    raw = head(decoded_feat).clamp(-10.0, 10.0)  # match forward's clamp
    sig = torch.sigmoid(raw).reshape(-1)  # (N,) post-activation alpha

    # `torch.quantile` requires a 1D float tensor on the same device.
    qs = torch.tensor([0.10, 0.50, 0.90, 0.99], device=sig.device, dtype=sig.dtype)
    p10, p50, p90, p99 = torch.quantile(sig, qs).tolist()

    return {
        "opa_raw_min":         float(raw.min()),
        "opa_raw_max":         float(raw.max()),
        "opa_sigma_p10":       float(p10),
        "opa_sigma_med":       float(p50),
        "opa_sigma_p90":       float(p90),
        "opa_sigma_p99":       float(p99),
        "opa_frac_high":       float((sig > 0.9).float().mean()),
        "opa_frac_low":        float((sig < 0.1).float().mean()),
        "opa_frac_decisive":   float(((sig > 0.9) | (sig < 0.1)).float().mean()),
        "opa_frac_mid":        float(((sig >= 0.3) & (sig <= 0.7)).float().mean()),
    }


@torch.no_grad()
def scale_distribution_stats(
    decoder: GaussianAttributeDecoder,
    decoded_feat: torch.Tensor,
    voxel_size: float,
) -> Dict[str, float]:
    """Re-run the scale head and return distribution stats on the
    *post-softplus* world-space scale (per Gaussian, max-axis).

    Healthy 3DGS Gaussians should have most scales <= 1-2x the base
    voxel size. Median ratio > 3x means the model is "smearing" each
    Gaussian over many voxels, which causes desaturation and detail
    loss exactly like underfitting the colour head does.
    """
    head = decoder.heads["scale"]
    raw = head(decoded_feat)
    sp = F.softplus(raw)               # (N, 3) per-axis world-space scale
    sp_max = sp.max(dim=-1).values     # (N,) most-spread axis

    qs = torch.tensor([0.10, 0.50, 0.90, 0.99], device=sp_max.device, dtype=sp_max.dtype)
    p10, p50, p90, p99 = torch.quantile(sp_max, qs).tolist()

    inv_v = 1.0 / max(float(voxel_size), 1e-12)
    return {
        "scale_max_p10":       float(p10),
        "scale_max_med":       float(p50),
        "scale_max_p90":       float(p90),
        "scale_max_p99":       float(p99),
        "scale_max_med_ratio": float(p50 * inv_v),
        "scale_max_p90_ratio": float(p90 * inv_v),
    }


def head_grad_norms(decoder: GaussianAttributeDecoder) -> Dict[str, float]:
    """Gradient norm of each decoder head's `proj_out`, after backward.

    Compare across attributes: a much smaller norm on the colour head
    than on (e.g.) the scale head is a strong indicator that the colour
    activation is killing or saturating gradients.
    """
    out: Dict[str, float] = {}
    for name, head in decoder.heads.items():
        gw = head.proj_out.weight.grad
        gb = head.proj_out.bias.grad
        out[f"{name}_w_gnorm"] = float(gw.norm()) if gw is not None else 0.0
        out[f"{name}_b_gnorm"] = float(gb.norm()) if gb is not None else 0.0
    return out


# ---------------------------------------------------------------------------
# Pretty-printing.
# ---------------------------------------------------------------------------


def format_color_diag(
    sat: Dict[str, float],
    grads: Dict[str, float],
    color_activation: str = "clamp",
) -> str:
    """One-line summary of the colour head's health.

    The displayed "frac" depends on the active colour activation:
    clamp-mode reports fraction outside [0, 1] (truly dead under
    clamp); sigmoid-mode reports fraction with |raw| > 5 (the band
    where sigmoid is in deep saturation and contributes < 1% of the
    max gradient).
    """
    bias = sat["color_bias"]
    color_g = grads.get("color_w_gnorm", 0.0)
    ref_g = grads.get("scale_w_gnorm", 0.0)  # softplus head -> healthy ref
    ratio = color_g / max(ref_g, 1e-12)
    if (color_activation or "clamp").lower() == "sigmoid":
        frac_label = "frac_sat"  # |raw| > 5
        frac_val = sat["color_frac_sigmoid_sat"]
    else:
        frac_label = "frac_dead"  # outside [0, 1]
        frac_val = sat["color_frac_clamp_dead"]
    return (
        f"color: raw=[{sat['color_raw_min']:+.2f},{sat['color_raw_max']:+.2f}] "
        f"mean={sat['color_raw_mean']:+.2f} std={sat['color_raw_std']:.2f} "
        f"{frac_label}={frac_val * 100:5.1f}% "
        f"bias=[{bias[0]:+.2f},{bias[1]:+.2f},{bias[2]:+.2f}] "
        f"gnorm(color/scale)={ratio:.3f}"
    )


def format_opacity_diag(opa: Dict[str, float]) -> str:
    """One-line summary of the opacity distribution."""
    return (
        f"opa:   raw=[{opa['opa_raw_min']:+.2f},{opa['opa_raw_max']:+.2f}] "
        f"sigma p10/med/p90={opa['opa_sigma_p10']:.2f}/{opa['opa_sigma_med']:.2f}/{opa['opa_sigma_p90']:.2f} "
        f"decisive={opa['opa_frac_decisive'] * 100:5.1f}% "
        f"mid[.3,.7]={opa['opa_frac_mid'] * 100:5.1f}%"
    )


def format_scale_diag(sc: Dict[str, float], voxel_size: float) -> str:
    """One-line summary of the scale distribution (in voxels)."""
    return (
        f"scale: med={sc['scale_max_med']:.4f}m ({sc['scale_max_med_ratio']:.2f}x vox) "
        f"p90={sc['scale_max_p90']:.4f}m ({sc['scale_max_p90_ratio']:.2f}x) "
        f"p99={sc['scale_max_p99']:.4f}m"
    )


# ---------------------------------------------------------------------------
# Hook-based diagnostic class.
# ---------------------------------------------------------------------------


class ColorClampDiagnostics:
    """Forward-pre-hook based diagnostic for the decoder heads.

    Captures the per-batch input to ``GaussianAttributeDecoder.forward``
    (which is the per-voxel feature tensor produced by the sparse CNN
    decoder) and, every ``every`` steps, prints a multi-line summary:

      * Pre-activation colour head distribution (clamp- or sigmoid-aware).
      * Post-sigmoid opacity distribution + decisive fraction.
      * Post-softplus scale distribution + ratio to voxel size.
      * Gradient norms per attribute head (after ``backward()``).

    Call ``maybe_log`` after ``loss.backward()`` (and optionally after
    ``optimizer.step()``); ``optimizer.step()`` does not zero ``.grad``,
    only ``optimizer.zero_grad()`` does (which runs at the start of the
    next iteration in ``train_autoencoder.py``).

    Set ``every <= 0`` to disable cheaply (the hook still fires and
    captures the latest tensor reference, but no extra compute is done).
    """

    def __init__(
        self,
        decoder: GaussianAttributeDecoder,
        every: int = 200,
        voxel_size: float = 0.025,
    ):
        self.decoder = decoder
        self.every = int(every)
        self.voxel_size = float(voxel_size)
        self._captured: Optional[torch.Tensor] = None
        self._handle = decoder.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if self.every <= 0:
            return
        # Only keep the latest call within a step (per-sample batches).
        # Detach to avoid polluting the autograd graph; we'll only look
        # at numerical statistics, not propagate through.
        self._captured = args[0].detach()

    def clear(self) -> None:
        """Drop the captured tensor (call after each optimizer step)."""
        self._captured = None

    def maybe_log(self, step: int, prefix: str = "") -> Optional[Dict[str, float]]:
        """Log diagnostics if it's time to. Returns the merged stats dict
        (or ``None`` if skipped) for callers that want to forward them
        to TensorBoard / WandB."""
        if self.every <= 0:
            return None
        if step % self.every != 0:
            return None
        if self._captured is None:
            return None

        sat = color_saturation_stats(self.decoder, self._captured)
        opa = opacity_distribution_stats(self.decoder, self._captured)
        sc = scale_distribution_stats(self.decoder, self._captured, self.voxel_size)
        grads = head_grad_norms(self.decoder)

        color_act = getattr(self.decoder, "color_activation", "clamp")
        color_line = format_color_diag(sat, grads, color_act)
        opa_line = format_opacity_diag(opa)
        scale_line = format_scale_diag(sc, self.voxel_size)
        print(f"{prefix}[diag step {step}]")
        print(f"{prefix}  {color_line}")
        print(f"{prefix}  {opa_line}")
        print(f"{prefix}  {scale_line}")

        merged: Dict[str, float] = {}
        for d in (sat, opa, sc):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    merged[k] = v
        merged.update(grads)
        return merged

    def close(self) -> None:
        """Remove the forward hook. Idempotent."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
