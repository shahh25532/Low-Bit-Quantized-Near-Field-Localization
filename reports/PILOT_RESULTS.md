# Preliminary pilot experiment (functional test only)

**This is not a research validation or evidence of improvement.** These numbers come from an actual local CPU execution, but used a very small dataset and 4 training epochs.

- Model: new independent simulation with a phase-only, sub-connected analog combiner; CNN and MLP; post-combiner uniform ADC.
- Antennas: 32; RF chains: 4; carrier: 25 GHz; nominal SNR: -15,-10,-5,0,5,10 dB.
- Position IDs per SNR: 120; split: 70% train, 15% validation, 15% test, grouped across SNR; only **18 test locations per SNR**.
- Training: 4 epochs per configuration, Adam 0.001, seed 2026, CPU. For a meaningful study use 50+ epochs, 2000+ positions and multiple seeds.

| Architecture | ADC bits | Mean test RMSE across SNRs (m) |
|---|---:|---:|
| CNN | 1 | 5.237 |
| CNN | 2 | 5.242 |
| CNN | 4 | 5.250 |
| CNN | float | 5.348 |
| MLP | 1 | 5.461 |
| MLP | 2 | 5.586 |
| MLP | 4 | 5.650 |
| MLP | float | 5.631 |

Do not rank methods by these pilot values: the models are undertrained and statistical uncertainty is high. Confirm training convergence and uncertainty before any substantive comparative statement.

Recreate this exact pilot run (not recommended for academic conclusions):

```bash
python -m nearfield_ext.experiments --antennas 32 --rf-chains 4 --snrs -15 -10 -5 0 5 10 --samples-per-snr 120 --epochs 4 --architectures cnn mlp --bit-depths float 1 2 4 --device cpu --output-dir results/pilot
```

Source: `results/pilot/config.json`, `metrics.csv`, `training_history.csv`, and generated `rmse_vs_snr.png` / `clipping_vs_snr.png`.
