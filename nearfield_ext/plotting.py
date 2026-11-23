"""Generate RMSE/SNR and clipping/SNR diagnostic plots from measured results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(rows: list[dict], output_dir: Path):
    output_dir = Path(output_dir)
    for metric, filename, ylabel in (
        ("position_rmse_m", "rmse_vs_snr.png", "Position RMSE (m)"),
        ("clip_fraction", "clipping_vs_snr.png", "ADC clipping fraction"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        groups = sorted(set((r["architecture"], r["bits"]) for r in rows))
        for architecture, bits in groups:
            samples = sorted(
                (r for r in rows if r["architecture"] == architecture and r["bits"] == bits),
                key=lambda r: r["snr_db"]
            )
            ax.plot([r["snr_db"] for r in samples],
                    [r[metric] for r in samples],
                    marker="o", label=f"{architecture.upper()} / {bits}")
        ax.set_xlabel("Nominal SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title("Pilot simulation — not a validated paper result")
        ax.grid(True, alpha=.2)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)
