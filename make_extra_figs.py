"""Figure 11: predictive densities on the regime-switching (cascade) twin.

Reads mmdens* arrays saved by extensions.py stage_multimodal and draws, for
two representative scenario-policy pairs, the empirical replication
distribution against generator samples and a moment-matched Gaussian.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figs")
plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300})
OBJ = ["f1: boarding delay (h)", "f2: unserved patients"]

arr = dict(np.load(os.path.join(OUT, "arrays.npz")))
fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.7))
k = 0
for pair in range(2):
    emp = arr[f"mmdens{pair}_emp"]      # [n_emp, 4]
    gen = arr[f"mmdens{pair}_gen"]      # [n_z, 4]
    for j in range(2):                  # objectives f1, f2
        ax = axes[k]; k += 1
        e, g = emp[:, j], gen[:, j]
        lo = min(e.min(), g.min()); hi = max(e.max(), g.max())
        bins = np.linspace(lo, hi, 26)
        ax.hist(e, bins=bins, density=True, color="0.75", alpha=0.9,
                label="digital twin replications")
        ax.hist(g, bins=bins, density=True, histtype="step", lw=1.6,
                color="#b2182b", label="G2M-SK generator")
        xs = np.linspace(lo, hi, 200)
        mu, sd = g.mean(), g.std() + 1e-9
        ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) /
                (sd * np.sqrt(2 * np.pi)), color="#2166ac", lw=1.4,
                label="moment-matched Gaussian")
        ax.set_title(f"pair {pair + 1}, {OBJ[j]}", fontsize=8.4, pad=5)
        ax.tick_params(labelsize=7.2)
        ax.set_yticks([])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           fontsize=8.6, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.4)
plt.savefig(os.path.join(FIG, "fig11_multimodal.png"), bbox_inches="tight")
print("wrote figs/fig11_multimodal.png")
