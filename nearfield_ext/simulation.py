"""Reproducible near-field narrowband XL-MIMO observation simulator.

This extension models single-user near-field localization; it is NOT a radar
receiver or a reproduction of radar target detection results.
"""
from dataclasses import dataclass
import numpy as np

LIGHT_SPEED = 299_792_458.0


@dataclass(frozen=True)
class SimulationConfig:
    antennas: int = 64
    carrier_hz: float = 25e9
    range_min_m: float = 1.0
    range_max_m: float = 10.0
    snr_db: tuple[float, ...] = (-15, -10, -5, 0, 5, 10, 15, 20)
    samples_per_snr: int = 300
    seed: int = 2026

    def validate(self) -> None:
        if self.antennas < 4 or self.antennas % 2:
            raise ValueError("antennas must be even and >= 4")
        if not (0 < self.range_min_m < self.range_max_m):
            raise ValueError("require 0 < range_min_m < range_max_m")
        if self.samples_per_snr < 10:
            raise ValueError("samples_per_snr must be >= 10 for grouped stratified splits")
        if len(self.snr_db) == 0 or len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError("snr_db must contain distinct values")


def near_field_array_response(theta: np.ndarray, radius: np.ndarray,
                              antennas: int, carrier_hz: float) -> np.ndarray:
    """Spherical-wave array response, output shape [examples, antennas].

    theta is directional cosine x/r (not an angle in radians), r is meters.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1, 1)
    if theta.shape != radius.shape:
        raise ValueError("theta and radius shapes must match")
    if np.any(np.abs(theta) > 1) or np.any(radius <= 0):
        raise ValueError("theta must be in [-1,1] and radius > 0")
    wavelength = LIGHT_SPEED / carrier_hz
    spacing = wavelength / 2
    indexes = np.arange(antennas, dtype=np.float64) - (antennas - 1) / 2
    displacements = indexes * spacing
    distances = np.sqrt(radius**2 + displacements**2 -
                        2 * radius * theta * displacements)
    phase = -2 * np.pi / wavelength * (distances - radius)
    return np.exp(1j * phase)


def make_splits(cfg: SimulationConfig, train_fraction: float = 0.7,
                val_fraction: float = 0.15) -> dict[str, dict[str, np.ndarray]]:
    """Generate paired-SNR observations; split by *position ID* to avoid leakage.

    Same spatial position is reused across SNR regimes for paired evaluation.
    Its ID is kept entirely in train, validation, or test, never across splits.
    """
    cfg.validate()
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction+val_fraction >= 1:
        raise ValueError("invalid split fractions")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.samples_per_snr
    # Uniform area over the upper half-annulus, equivalent to uniform xy area.
    phi = rng.uniform(0, np.pi, size=n)
    radius = np.sqrt(rng.uniform(cfg.range_min_m**2, cfg.range_max_m**2, size=n))
    theta = np.cos(phi)
    radial_norm = 2 * (radius - cfg.range_min_m) / (cfg.range_max_m-cfg.range_min_m) - 1
    targets = np.stack([theta, radial_norm], axis=-1).astype(np.float32)
    clean = near_field_array_response(theta, radius, cfg.antennas, cfg.carrier_hz)
    all_x, all_y, all_snr, all_position_ids = [], [], [], []
    for db in cfg.snr_db:
        # Matches the upstream model's convention: real & imaginary noise
        # components each N(0, 1/sqrt(linear_SNR)). Documented as nominal SNR.
        sigma = 1 / np.sqrt(10 ** (db / 10))
        noise = rng.normal(0, sigma, clean.shape) + 1j*rng.normal(0, sigma, clean.shape)
        noisy = clean + noise
        all_x.append(np.concatenate([noisy.real, noisy.imag], axis=-1).astype(np.float32))
        all_y.append(targets.copy())
        all_snr.append(np.full(n, db, dtype=np.float32))
        all_position_ids.append(np.arange(n, dtype=np.int32))
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    snr = np.concatenate(all_snr)
    position_id = np.concatenate(all_position_ids)
    position_order = rng.permutation(n)
    nt = max(1, int(n * train_fraction))
    nv = max(1, int(n * val_fraction))
    group_ids = {
        "train": position_order[:nt],
        "val": position_order[nt:nt + nv],
        "test": position_order[nt + nv:],
    }
    result = {}
    for split, ids in group_ids.items():
        mask = np.isin(position_id, ids)
        result[split] = {"x": x[mask], "y": y[mask], "snr_db": snr[mask],
                         "position_id": position_id[mask]}
    return result


def denormalize_target(target: np.ndarray, radius_range: tuple[float,float]):
    """Return Cartesian xy in metres from theta=x/r and normalized radius."""
    theta = np.clip(target[..., 0], -1.0, 1.0)
    r0, r1 = radius_range
    radius = r0 + (np.clip(target[..., 1], -1.0, 1.0)+1.0)*(r1-r0)/2
    x = radius * theta
    y = radius * np.sqrt(np.maximum(0.0, 1 - theta**2))
    return np.stack((x,y), axis=-1)
