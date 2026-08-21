#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
Regenerates every figure in the G2M-SK manuscript in the paper's visual style.

WHY THIS SCRIPT EXISTS
----------------------
After the rerun, the accuracy / coverage / width numbers for G2M-SK changed
(see Tables 7, 8, 10, 12). Some figures are fully determined by those tables and
are reproduced here exactly (Figure 10, and the coverage panel of Figure 6).
The others (Figure 5's N-sweep, Figure 6's per-objective widths, Figure 8's
Pareto scatter, Figure 9's per-objective drift coverage) need raw per-run arrays
that live only in your experiment outputs, NOT in the manuscript. Those functions
read your saved data. Point the DATA_DIR / CSV paths below at your repo outputs
and run:

    python make_figures.py

Each figure is written to ./figures/ at the same aspect ratio used in the docx.

AUTHORITATIVE TABLE VALUES (post-rerun) are hardcoded below so the script both
(a) renders the fully-determined figures correctly out of the box, and
(b) sanity-checks your CSVs against the tables (a warning prints on mismatch).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Global style to match the manuscript figures
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",      # a clean sans; swap for the docx font if needed
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# muted seaborn-style palette used in the paper
C_F1, C_F2, C_F3, C_F4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
OBJ_COLORS = [C_F1, C_F2, C_F3, C_F4]
OBJ_LABELS = ["f1", "f2", "f3", "f4"]
# method line colors (Figure 5)
M_SK, M_NNSK, M_NN, M_G2 = "#8172B3", "#4C72B0", "#55A868", "#C44E52"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Where your rerun outputs live. Change to your paths.
DATA_DIR = os.environ.get("G2MSK_DATA", "results")

# ----------------------------------------------------------------------------
# Authoritative table values (post-rerun). DO NOT edit unless the tables change.
# ----------------------------------------------------------------------------
# Table 7: per-objective test RMSE at N=64 (mean, std). Order: f1,f2,f3,f4, normalized
TABLE7 = {
    "SK":        {"rmse": [12.1, 54.1, 60.2, 285.1], "sd": [0.7, 6.9, 9.9, 27.6], "norm": (1.510, 0.067)},
    "NN-SK":     {"rmse": [2.83, 66.7, 44.2, 107.8], "sd": [0.20, 9.2, 4.8, 16.3], "norm": (0.674, 0.028)},
    "NN":        {"rmse": [2.33, 54.0, 29.0, 76.0],  "sd": [0.25, 10.5, 2.0, 10.3], "norm": (0.494, 0.018)},
    "G2M-SK":    {"rmse": [1.47, 12.2, 28.7, 83.9],  "sd": [0.32, 1.9, 2.7, 8.8],  "norm": (0.387, 0.023)},
}
# Table 8: per-objective coverage + mean width at N=64 (the 4 methods drawn in Fig 6)
TABLE8 = {
    "G2M-SK\n(conformal)": {"cov": [0.90, 0.92, 0.92, 0.87], "mean_cov": 0.904, "mean_w": 1.42},
    "G2M-SK\n(no conf.)":  {"cov": [0.76, 0.89, 0.91, 0.94], "mean_cov": 0.876, "mean_w": 2.11},
    "NN-SK\n(Gaussian)":   {"cov": [0.40, 0.19, 0.21, 0.26], "mean_cov": 0.266, "mean_w": 0.34},
    "NN\nensemble":        {"cov": [0.79, 0.40, 0.47, 0.63], "mean_cov": 0.570, "mean_w": 0.76},
}
# Table 12: online latency (ms / 1000 evals) and storage (MB)
TABLE12 = {
    "SK":            {"online": 2212, "storage": 38.5},
    "NN\nensemble":  {"online": 4.9,  "storage": 0.43},
    "NN-SK":         {"online": 54,   "storage": 0.13},
    "G2M-SK":        {"online": 54,   "storage": 0.14},
    "G2M-SK\n(full)":{"online": 162,  "storage": 0.14},
}
# Figure 9 mean coverage anchors (from Section 6.6 text)
DRIFT_MEANS = {"In distribution": 0.918, "Severe covariate drift": 0.652, "After recalibration": 0.916}


def _load_csv(name):
    """Load a CSV from DATA_DIR if present; return None otherwise."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  [skip] {name} not found in '{DATA_DIR}/' -- provide it to render this figure.")
        return None
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] could not read {name}: {e}")
        return None


# ----------------------------------------------------------------------------
# Figure 5 -- Test RMSE vs number of training scenarios N, 4 objectives
#   NEEDS: results/rmse_by_N.csv with columns: method,N,objective,rmse,sd
#   (objective in {f1,f2,f3,f4}; method in {SK,NN-SK,NN,G2M-SK})
#   The N=64 rows are cross-checked against Table 7.
# ----------------------------------------------------------------------------
def figure5():
    print("Figure 5 (RMSE vs N)")
    df = _load_csv("rmse_by_N.csv")
    if df is None:
        return
    mcol = {"SK": M_SK, "NN-SK": M_NNSK, "NN": M_NN, "G2M-SK": M_G2}
    titles = ["f1: boarding delay (h)", "f2: unserved patients",
              "f3: overflow bed h", "f4: staff overload"]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.3))
    for j, (ax, obj, title) in enumerate(zip(axes, OBJ_LABELS, titles)):
        for m, c in mcol.items():
            sub = df[(df.method == m) & (df.objective == obj)].sort_values("N")
            if len(sub):
                ax.errorbar(sub.N, sub.rmse, yerr=sub.sd, marker="o", ms=3,
                            lw=1.2, capsize=2, color=c, label=m)
        ax.set_title(title)
        ax.set_xlabel("training scenarios N")
        if j == 0:
            ax.set_ylabel("RMSE")
        ax.set_xticks(sorted(df.N.unique()))
    axes[0].legend(loc="upper right", ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figure5_rmse_vs_N.png")
    plt.close(fig)
    # sanity check vs Table 7 at N=64
    for m in TABLE7:
        for k, obj in enumerate(OBJ_LABELS):
            row = df[(df.method == m) & (df.objective == obj) & (df.N == 64)]
            if len(row):
                got = float(row.rmse.iloc[0]); exp = TABLE7[m]["rmse"][k]
                if abs(got - exp) > 0.05 * max(exp, 1e-9):
                    print(f"  [check] {m} {obj} N=64 rmse={got} != Table 7 {exp}")
    print(f"  wrote {OUT}/figure5_rmse_vs_N.png")


# ----------------------------------------------------------------------------
# Figure 6 -- coverage (left) and normalized width (right), by method x objective
#   Coverage panel is fully determined by Table 8 (rendered here).
#   Width panel NEEDS per-objective widths: results/width_by_objective.csv
#   columns: method,objective,width  (method labels must match TABLE8 keys w/o \n)
# ----------------------------------------------------------------------------
def figure6():
    print("Figure 6 (coverage + width)")
    methods = list(TABLE8.keys())
    x = np.arange(len(methods))
    w = 0.19
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 2.4))

    # LEFT: coverage (from Table 8, exact)
    for k in range(4):
        vals = [TABLE8[m]["cov"][k] for m in methods]
        axL.bar(x + (k - 1.5) * w, vals, w, color=OBJ_COLORS[k], label=OBJ_LABELS[k])
    axL.axhline(0.90, ls="--", lw=1, color="0.4")
    axL.text(0.02, 0.905, "target 90%", transform=axL.get_yaxis_transform(),
             fontsize=6.5, color="0.35")
    axL.set_ylabel("empirical coverage")
    axL.set_ylim(0, 1.05)
    axL.set_xticks(x); axL.set_xticklabels(methods)
    axL.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))

    # RIGHT: per-objective width (needs CSV); fall back to Table 8 mean width bar
    dfw = _load_csv("width_by_objective.csv")
    if dfw is not None:
        keymap = {m.replace("\n", " "): m for m in methods}
        for k, obj in enumerate(OBJ_LABELS):
            vals = []
            for m in methods:
                sub = dfw[(dfw.method == m.replace("\n", " ")) & (dfw.objective == obj)]
                vals.append(float(sub.width.iloc[0]) if len(sub) else np.nan)
            axR.bar(x + (k - 1.5) * w, vals, w, color=OBJ_COLORS[k], label=OBJ_LABELS[k])
        axR.set_ylabel("normalized interval width")
    else:
        vals = [TABLE8[m]["mean_w"] for m in methods]
        axR.bar(x, vals, 0.55, color="0.6")
        axR.set_ylabel("mean normalized width")
        axR.set_title("(mean width shown; supply width_by_objective.csv for per-objective)",
                      fontsize=6.5, color="0.4")
    axR.set_xticks(x); axR.set_xticklabels(methods)

    fig.tight_layout()
    fig.savefig(f"{OUT}/figure6_coverage_width.png")
    plt.close(fig)
    print(f"  wrote {OUT}/figure6_coverage_width.png (coverage exact; width from CSV or mean fallback)")


# ----------------------------------------------------------------------------
# Figure 8 -- Pareto scatter for one scenario, two objective projections
#   NEEDS: results/pareto_scenario.csv with columns:
#   f1,f2,f3,f4, on_true_front(bool), g2msk_selected(bool), sk_selected(bool),
#   and optional err_lo_*/err_hi_* for calibrated bars on G2M-SK selections.
# ----------------------------------------------------------------------------
def figure8():
    print("Figure 8 (Pareto scatter)")
    df = _load_csv("pareto_scenario.csv")
    if df is None:
        return
    proj = [("f1", "f2"), ("f3", "f4")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 2.6))
    for ax, (a, b) in zip(axes, proj):
        ax.scatter(df[a], df[b], s=8, color="0.8", label="candidates", zorder=1)
        tf = df[df.on_true_front == True]
        ax.scatter(tf[a], tf[b], s=26, marker="s", facecolors="none",
                   edgecolors="k", label="true Pareto", zorder=3)
        g2 = df[df.g2msk_selected == True]
        ax.scatter(g2[a], g2[b], s=22, color=M_G2, label="G2M-SK", zorder=4)
        sk = df[df.sk_selected == True]
        ax.scatter(sk[a], sk[b], s=22, marker="^", color="#8172B3", label="SK", zorder=2)
        ax.set_xlabel(a); ax.set_ylabel(b)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figure8_pareto.png")
    plt.close(fig)
    print(f"  wrote {OUT}/figure8_pareto.png")


# ----------------------------------------------------------------------------
# Figure 9 -- coverage in distribution / under drift / after recalibration
#   NEEDS: results/drift_coverage.csv with columns: condition,objective,coverage
#   conditions: 'in_dist','drift','recal'; objective in {f1,f2,f3,f4}
#   Means are cross-checked against DRIFT_MEANS.
# ----------------------------------------------------------------------------
def figure9():
    print("Figure 9 (drift coverage)")
    df = _load_csv("drift_coverage.csv")
    if df is None:
        return
    cond_order = [("in_dist", "In distribution", "#4C72B0"),
                  ("drift", "Severe covariate drift", "#DD8452"),
                  ("recal", "After recalibration", "#55A868")]
    groups = OBJ_LABELS + ["mean"]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (ckey, clabel, c) in enumerate(cond_order):
        vals = []
        for g in OBJ_LABELS:
            sub = df[(df.condition == ckey) & (df.objective == g)]
            vals.append(float(sub.coverage.iloc[0]) if len(sub) else np.nan)
        vals.append(np.nanmean(vals))
        ax.bar(x + (i - 1) * w, vals, w, color=c, label=clabel)
        for xi, v in zip(x + (i - 1) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.axhline(0.90, ls="--", lw=1, color="0.4")
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(["f1 boarding\ndelay", "f2 unserved\npatients",
                        "f3 overflow\nbed hours", "f4 staff\noverload", "Mean"])
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(f"{OUT}/figure9_drift.png")
    plt.close(fig)
    for ckey, clabel, _ in cond_order:
        m = df[df.condition == ckey].coverage.mean()
        exp = DRIFT_MEANS[clabel]
        if abs(m - exp) > 0.01:
            print(f"  [check] {clabel} mean={m:.3f} vs text {exp}")
    print(f"  wrote {OUT}/figure9_drift.png")


# ----------------------------------------------------------------------------
# Figure 10 -- online latency (left) and storage (right), log axes
#   FULLY determined by Table 12 -> rendered exactly, no external data needed.
# ----------------------------------------------------------------------------
def figure10():
    print("Figure 10 (latency + storage)")
    methods = list(TABLE12.keys())
    x = np.arange(len(methods))
    online = [TABLE12[m]["online"] for m in methods]
    storage = [TABLE12[m]["storage"] for m in methods]
    colors = ["#8172B3", "#55A868", "#4C72B0", "#C44E52", "#C44E52"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 2.2))
    axL.bar(x, online, 0.6, color=colors)
    axL.set_yscale("log"); axL.set_ylabel("online ms / 1000 evals")
    for xi, v in zip(x, online):
        axL.text(xi, v * 1.15, f"{v:g}", ha="center", fontsize=6)
    axL.set_xticks(x); axL.set_xticklabels(methods)

    axR.bar(x, storage, 0.6, color=colors)
    axR.set_yscale("log"); axR.set_ylabel("storage (MB)")
    for xi, v in zip(x, storage):
        axR.text(xi, v * 1.15, f"{v:g}", ha="center", fontsize=6)
    axR.set_xticks(x); axR.set_xticklabels(methods)

    fig.tight_layout()
    fig.savefig(f"{OUT}/figure10_cost.png")
    plt.close(fig)
    print(f"  wrote {OUT}/figure10_cost.png  (exact, from Table 12)")


if __name__ == "__main__":
    print(f"Reading rerun data from '{DATA_DIR}/'. Set G2MSK_DATA to change.\n")
    figure5()
    figure6()
    figure8()
    figure9()
    figure10()
    print("\nDone. Fully-determined figures (10, and Figure 6 coverage) are correct as-is.")
    print("Provide the listed CSVs to render Figures 5, 8, 9 and Figure 6 widths.")
