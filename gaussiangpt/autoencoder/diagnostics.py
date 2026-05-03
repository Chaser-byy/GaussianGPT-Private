"""Training-time diagnostics for the Gaussian autoencoder.

Currently focuses on detecting *dead-gradient* pathologies caused by the
hard `clamp(0, 1)` activation in the colour decoder head (paper-faithful
post-processing per GaussianGPT Appendix C).

Usage in a training loop:

    from gaussiangpt.autoencoder.diagnostics import ColorClampDiagnostics

    diag = ColorClampDiagnostics(raw_model.attr_decoder, every=200)
    ...
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = compute_loss(...)
        loss.backward()
        optimizer.step()
        diag.maybe_log(global_step)  # any time after backward works;
                                      # step() does not clear .grad
        diag.clear()
    diag.close()
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .gaussian_heads import GaussianAttributeDecoder


@torch.no_grad()
def color_saturation_stats(
    decoder: GaussianAttributeDecoder,
    decoded_feat: torch.Tensor,
) -> Dict[str, float]:
    """Re-run the colour head WITHOUT post-processing and report how many
    predictions live outside [0, 1] (i.e. would be killed by the clamp).

    Args:
        decoder: the model's `GaussianAttributeDecoder`
        decoded_feat: (N, in_dim) per-voxel features fed into the decoder

    Returns:
        dict with raw min/max/mean/std, fraction of values < 0 / > 1, the
        union "frac_dead", and the current bias of `proj_out`.
    """
    head = decoder.heads["color"]
    raw = head(decoded_feat)  # (N, 3) pre-clamp logits
    return {
        "color_raw_min":      float(raw.min()),
        "color_raw_max":      float(raw.max()),
        "color_raw_mean":     float(raw.mean()),
        "color_raw_std":      float(raw.std()),
        "color_frac_below_0": float((raw < 0.0).float().mean()),
        "color_frac_above_1": float((raw > 1.0).float().mean()),
        "color_frac_dead":    float(((raw < 0.0) | (raw > 1.0)).float().mean()),
        "color_bias":         head.proj_out.bias.detach().cpu().tolist(),
    }


def head_grad_norms(decoder: GaussianAttributeDecoder) -> Dict[str, float]:
    """Gradient norm of each decoder head's `proj_out`, after backward.

    Compare across attributes: a much smaller norm on the colour head
    than on (e.g.) the scale head is a strong indicator that the colour
    clamp is killing gradients for a non-trivial fraction of predictions.
    """
    out: Dict[str, float] = {}
    for name, head in decoder.heads.items():
        gw = head.proj_out.weight.grad
        gb = head.proj_out.bias.grad
        out[f"{name}_w_gnorm"] = float(gw.norm()) if gw is not None else 0.0
        out[f"{name}_b_gnorm"] = float(gb.norm()) if gb is not None else 0.0
    return out


def format_color_diag(
    sat: Dict[str, float],
    grads: Dict[str, float],
) -> str:
    """Compact one-line summary suitable for training logs."""
    bias = sat["color_bias"]
    color_g = grads.get("color_w_gnorm", 0.0)
    ref_g = grads.get("scale_w_gnorm", 0.0)  # scale uses softplus -> healthy ref
    ratio = color_g / max(ref_g, 1e-12)
    return (
        f"color: raw=[{sat['color_raw_min']:+.2f},{sat['color_raw_max']:+.2f}] "
        f"mean={sat['color_raw_mean']:+.2f} "
        f"frac_dead={sat['color_frac_dead']*100:5.1f}% "
        f"bias=[{bias[0]:+.2f},{bias[1]:+.2f},{bias[2]:+.2f}] "
        f"gnorm(color/scale)={ratio:.3f}"
    )


class ColorClampDiagnostics:
    """Forward-pre-hook based diagnostic for the colour clamp.

    Captures the per-batch input to `GaussianAttributeDecoder.forward`
    (which is the per-voxel feature tensor produced by the sparse CNN
    decoder) and, every `every` steps, prints a one-line summary of:

      * pre-clamp colour distribution (min/max/mean/frac outside [0,1])
      * gradient norm of the colour head vs the scale head

    Call `maybe_log` after `loss.backward()` (and optionally after
    `optimizer.step()`): `step()` does not zero `.grad` — only
    `optimizer.zero_grad()` does, which runs at the start of the next
    iteration in `train_autoencoder.py`.

    Set `every <= 0` to disable cheaply (the hook still fires, but nothing
    is printed and no extra compute is performed besides the capture).
    """

    def __init__(self, decoder: GaussianAttributeDecoder, every: int = 200):
        self.decoder = decoder
        self.every = int(every)
        self._captured: Optional[torch.Tensor] = None
        # forward-pre-hook: receives (module, args). args[0] is decoded_feat.
        self._handle = decoder.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if self.every <= 0:
            return
        # Only keep the latest call within a step (per-sample batches).
        # Detach to avoid polluting the autograd graph; we'll only look at
        # numerical statistics, not propagate through.
        self._captured = args[0].detach()

    def clear(self) -> None:
        """Drop the captured tensor (call after each optimizer step)."""
        self._captured = None

    def maybe_log(self, step: int, prefix: str = "") -> Optional[Dict[str, float]]:
        """Log diagnostics if it's time to. Returns the merged stats dict
        (or None if skipped) for callers that want to also send them to
        TensorBoard / WandB."""
        if self.every <= 0:
            return None
        if step % self.every != 0:
            return None
        if self._captured is None:
            return None
        sat = color_saturation_stats(self.decoder, self._captured)
        grads = head_grad_norms(self.decoder)
        line = format_color_diag(sat, grads)
        print(f"{prefix}[diag step {step}] {line}")
        merged: Dict[str, float] = {}
        for k, v in sat.items():
            if isinstance(v, (int, float)):
                merged[k] = v
        merged.update(grads)
        return merged

    def close(self) -> None:
        """Remove the forward hook. Idempotent."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
