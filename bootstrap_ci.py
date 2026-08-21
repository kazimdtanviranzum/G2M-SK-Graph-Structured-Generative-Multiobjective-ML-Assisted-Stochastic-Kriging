#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_ci.py
Recomputes the paired scenario-level bootstrap confidence intervals reported in
Section 6.2, so you can replace the stale numbers
("exceeds ... by 0.104 with a 95 percent CI of [0.074, 0.136]" and
"[0.875, 1.017] with mean 0.945") with the post-rerun values.

INPUT
-----
A CSV of per-scenario-cell normalized errors, one row per (scenario, method):
    scenario,method,norm_error
with method in {G2M-SK, NN-SK, SK, NN, GBT-CP, NN-CP, GNN-ENS-CP}.
This is exactly the per-cell quantity your evaluation already aggregates into the
"Normalized" column of Table 7. Set the path below.

The bootstrap resamples the ~120 scenario cells (with replacement), 10,000 times,
and reports the mean paired difference (baseline - G2M-SK) and its 95% CI.
A positive interval that excludes zero means G2M-SK is significantly better.
"""

import numpy as np
import pandas as pd

CSV = "results/cell_norm_errors.csv"   # <- point at your rerun output
N_BOOT = 10000
SEED = 0
BASELINES = ["NN-SK", "SK", "NN", "GBT-CP", "NN-CP", "GNN-ENS-CP"]


def paired_bootstrap(df, baseline, ref="G2M-SK", n_boot=N_BOOT, seed=SEED):
    piv = df.pivot_table(index="scenario", columns="method",
                         values="norm_error", aggfunc="mean")
    piv = piv.dropna(subset=[baseline, ref])
    diff = (piv[baseline] - piv[ref]).to_numpy()   # >0 => G2M-SK better
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return diff.mean(), lo, hi, n


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    print(f"cells: {df.scenario.nunique()} scenarios, methods: {sorted(df.method.unique())}\n")
    print(f"{'baseline':>12}  {'mean diff':>9}  {'95% CI':>18}   (positive & excludes 0 => G2M-SK better)")
    for b in BASELINES:
        if b in df.method.unique():
            m, lo, hi, n = paired_bootstrap(df, b)
            flag = "  *" if lo > 0 else ""
            print(f"{b:>12}  {m:9.3f}  [{lo:6.3f}, {hi:6.3f}]{flag}   (n={n})")
    print("\nDrop the NN-SK row into the first CI slot of Section 6.2 and the SK row "
          "into the second, replacing 0.104 [0.074, 0.136] and [0.875, 1.017] mean 0.945.")
