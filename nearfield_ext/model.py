"""Hybrid beamforming localization estimators.

Original motivation and architecture reference: Fabiani et al., ICC 2025,
https://github.com/mattiafabiani/One-Shot-Near-Field-Localization-with-AI-Optimized-Hybrid-Beamformer-Design
This is an independent reimplementation and extension, not their original code.
"""
import math
import torch
import torch.nn as nn
from .quantization import UniformADC


class SubarrayHybridCombiner(nn.Module):
    """Phase-only, sub-connected, trainable analog combining stage.

    Each RF chain serves a distinct block of N/N_RF antennas. We parameterize
    phases directly, so the phase shifters remain constant modulus exactly.
    """
    def __init__(self, antennas: int, rf_chains: int):
        super().__init__()
        if antennas < rf_chains or antennas % rf_chains:
            raise ValueError("antennas must be divisible by rf_chains")
        self.antennas = antennas
        self.rf_chains = rf_chains
        self.block_size = antennas // rf_chains
        self.phases = nn.Parameter(torch.empty(rf_chains, self.block_size).uniform_(-math.pi,math.pi))

    def forward(self, received_iq: torch.Tensor) -> torch.Tensor:
        if received_iq.ndim != 2 or received_iq.shape[1] != 2*self.antennas:
            raise ValueError("expected batch x (2*antennas) input")
        real = received_iq[:, :self.antennas].reshape(-1,self.rf_chains,self.block_size)
        imag = received_iq[:, self.antennas:].reshape(-1,self.rf_chains,self.block_size)
        cos = torch.cos(self.phases).unsqueeze(0)
        sin = torch.sin(self.phases).unsqueeze(0)
        scale = self.block_size**-0.5
        # e^{-j*phase} (real + j imag)
        out_r = ((real*cos + imag*sin).sum(-1))*scale
        out_i = ((imag*cos - real*sin).sum(-1))*scale
        return torch.stack((out_r,out_i),dim=1)  # [batch,2,rf_chains]


class HybridLocalizer(nn.Module):
    def __init__(self, antennas: int, rf_chains: int, architecture: str,
                 bits: int | None, full_scale: float):
        super().__init__()
        if architecture not in ("cnn", "mlp"):
            raise ValueError("architecture must be cnn or mlp")
        self.combiner = SubarrayHybridCombiner(antennas,rf_chains)
        self.adc = UniformADC(bits, full_scale)
        self.architecture = architecture
        if architecture == "cnn":
            self.features = nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=3, padding=1),nn.ReLU(),
                nn.Conv1d(16, 16, kernel_size=3, padding=1),nn.ReLU(),
                nn.Flatten())
            feature_width = 16*rf_chains
        else:
            self.features = nn.Flatten()
            feature_width = 2*rf_chains
        self.regressor = nn.Sequential(nn.Linear(feature_width, 128),nn.ReLU(),
                                       nn.Linear(128,64),nn.ReLU(),
                                       nn.Linear(64,2),nn.Tanh())

    def forward_with_diagnostics(self, x: torch.Tensor):
        analog = self.combiner(x)
        fraction = self.adc.clipping_fraction(analog)
        quantized = self.adc(analog)
        estimate = self.regressor(self.features(quantized))
        return estimate, fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_diagnostics(x)[0]
