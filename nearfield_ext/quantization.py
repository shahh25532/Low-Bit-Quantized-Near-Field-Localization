"""Uniform mid-rise quantization with straight-through training gradients."""
import torch
import torch.nn as nn


class UniformADC(nn.Module):
    """Element-wise symmetric signed ADC on beamformed real/IQ channels.

    bits=None: floating-point baseline (no ADC).
    bits=1: two levels centered at +/-full_scale/2.
    bits=k: 2^k uniformly spaced mid-rise levels in (-full_scale,full_scale).
    Quantizer clipping is reported separately in diagnostics.
    Uses identity straight-through estimator in training; this is an
    approximation to a non-differentiable ADC, not a hardware model.
    """
    def __init__(self, bits: int | None, full_scale: float = 3.0):
        super().__init__()
        if bits is not None and not (1 <= bits <= 16):
            raise ValueError("bits must be None or in range 1..16")
        if full_scale <= 0:
            raise ValueError("full_scale must be positive")
        self.bits = bits
        self.full_scale = float(full_scale)

    def clipping_fraction(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None:
            return x.new_zeros(())
        return (x.detach().abs() > self.full_scale).float().mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None:
            return x
        levels = 2**self.bits
        step = 2*self.full_scale / levels
        indices = torch.floor((x+self.full_scale) / step).clamp(0, levels - 1)
        quantized = -self.full_scale + (indices+0.5)*step
        return x + (quantized-x).detach() if self.training else quantized
