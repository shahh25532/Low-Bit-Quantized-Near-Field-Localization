# Low-Bit Quantized Near-Field Localization — Reproducible Simulation Extension

**Status:** self-directed computational pilot; **not peer-reviewed**. This project adds a low-resolution ADC study to a near-field XL-MIMO hybrid beamforming localization testbed. All example metrics must come from actual runs—no claimed performance improvement is assumed.

## Origin and research context

Based on the ideas in Mattia Fabiani et al.'s [ICC 2025 research and repository](https://github.com/mattiafabiani/One-Shot-Near-Field-Localization-with-AI-Optimized-Hybrid-Beamformer-Design). It is an **independently implemented extension**, not a fork or identical reproduction. See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).

**Original contribution being explored here:** placing a uniform 1-, 2- or 4-bit ADC *after* the phase-only sub-connected analog hybrid combiner, compared with an unquantized floating-point reference, over fixed SNR levels (including low SNR). This is a **simulation of one-shot single-source localization**, not radar target detection; the motivation is related to low-bit inference and learned signal processing.

## Architecture

`near-field spherical-wave channel → complex AWGN → phase-only subarray analog combiner → [optional ADC] → CNN/MLP regression → estimated position (x, y)`

- Learned analog phases ensure unit modulus and non-overlapping subarrays.
- Digital CNN and MLP are trained *separately* for each bit depth.
- ADC is a symmetric mid-rise uniform quantizer with fixed full scale; training uses a straight-through gradient estimator, and saturation fraction is logged.
- Uniform-over-area sampling in an upper half-annulus and paired SNR observations; splitting is grouped by target position ID to avoid cross-SNR train/test leakage.
- Outputs are compared using *2D position RMSE in metres*, calculated from both predicted normalized radius and direction cosine.

## Quick start

Recommended: Python 3.10+ with PyTorch installed; on large runs a CUDA GPU is helpful.

```bash
python -m pip install -r requirements.txt
python -m pytest -q

# Quick smoke run (2 configs; NOT definitive research results)
python -m nearfield_ext.experiments --antennas 16 --rf-chains 4 --snrs -10 0 10 --samples-per-snr 60 --epochs 2 --architectures cnn --bit-depths float 2 --device cpu --output-dir results/smoke

# Full example (8 configurations, more samples and training)
python -m nearfield_ext.experiments --antennas 128 --rf-chains 16 --samples-per-snr 2000 --epochs 50 --architectures cnn mlp --bit-depths float 1 2 4 --output-dir results/full
```

`results/<run>/` contains `config.json`, `metrics.csv`, `training_history.csv`, `rmse_vs_snr.png`, `clipping_vs_snr.png`, and PyTorch `.pt` checkpoints (not committed by default). **Keep the config and metrics whenever reporting findings.**

## Reproducibility and caveats

1. The nominal SNR definition follows the upstream convention: independently distributed real/imaginary Gaussian noise of standard deviation `1/sqrt(10^(SNR_dB/10))`; there are two noise components per complex sample, so do not silently interpret the SNR as total complex-signal noise power.
2. Independent complex noise is drawn per SNR; underlying spatial positions are paired across SNR to permit comparison. Position identities are group-separated across train/validation/test.
3. The ADC range is fixed to `--full-scale 3.0` unless changed; this introduces overload/clipping at some operating points. Inspect `clip_fraction` before interpreting quantization effects.
4. Phase-only subarray combining is not the same as every beamformer variant in the upstream work. Models here are not intended to reproduce its reported accuracy figures.
5. Single-user, narrowband simulation with AWGN only in this version. **Multipath, near-field radar targets, hardware measurements, real-world validation, multiple snapshots, and paper reproduction remain future work.**
6. Small 2-epoch smoke runs only test functionality; for research conclusions perform more training, multiple seeds, confidence intervals, and rigorous baselines.

## Suggestions for subsequent work

- Add a physically accurate multipath model (random reflectors and path losses).
- Compare learned phase-only combiner with fixed DFT beamforming under the same ADC budgets.
- Report error quantiles, confidence intervals over several seeds, and complexity in MACs.
- Assess calibration mismatch and nonuniform/noisy quantizers, but only claim benefits supported by runs.

## License & academic integrity

MIT. Original ideas and authors explicitly credited: [ATTRIBUTION.md](ATTRIBUTION.md). Any write-up should describe it as *an independently implemented extension of Fabiani et al.'s approach*, and separate their results from new simulation outcomes.