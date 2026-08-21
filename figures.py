"""All manuscript figures. Reads out/results.json and out/arrays.npz.

Revised layout: every figure was redrawn so that no text element overlaps
another element (larger canvases, legends outside the plotting area,
offset node labels, wider boxes, and non-crossing arrows).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300})
OBJ = ["f1: boarding delay (h)", "f2: unserved patients",
       "f3: overflow bed h", "f4: staff overload"]
OBJ_S = ["f1", "f2", "f3", "f4"]
COLS = {"G2M-SK": "#b2182b", "NN-SK": "#2166ac", "NN": "#4d9221",
        "SK": "#8073ac"}


def box(ax, x, y, w, h, text, fc="#eef3fa", ec="#2b4c7e", fs=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.25)


def arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color="#333", **kw))


def fig_framework():
    # Larger canvas, wider boxes, and arrows routed through empty corridors
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.27, 0.975, "Offline stage", fontsize=11, weight="bold",
            ha="center", color="#2b4c7e")
    ax.text(0.815, 0.975, "Online stage", fontsize=11, weight="bold",
            ha="center", color="#7a1f1f")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.535, 0.925, fc="none",
                                ec="#2b4c7e", lw=1.0, linestyle="--",
                                boxstyle="round,pad=0.005"))
    ax.add_patch(FancyBboxPatch((0.615, 0.02), 0.375, 0.925, fc="none",
                                ec="#7a1f1f", lw=1.0, linestyle="--",
                                boxstyle="round,pad=0.005"))
    # left column (data)
    box(ax, 0.03, 0.70, 0.215, 0.185, "Disruption scenario\nlibrary "
        r"$\{\theta_n\}$" "\n(hospital graphs)")
    box(ax, 0.03, 0.415, 0.215, 0.185, "Digital twin\nsimulation at design\n"
        r"pairs $(\theta_n, x_k)$")
    box(ax, 0.03, 0.13, 0.215, 0.185, "Replication outputs\n"
        r"$\{Y_m(x_k;\theta_n)\}$")
    # middle column (learned components)
    box(ax, 0.315, 0.66, 0.215, 0.235, "GNN encoder " r"$h(\theta)$" "\n"
        "+ LMC-SK heads\n" r"$\beta,\ L_q,\ \gamma_q,\ \phi_F$",
        fc="#fdeee0", ec="#a35d1f")
    box(ax, 0.315, 0.375, 0.215, 0.20, "Wasserstein\ngenerator + critic\n(anchored on SK)",
        fc="#fdeee0", ec="#a35d1f")
    box(ax, 0.315, 0.10, 0.215, 0.19, "Split-conformal\ncalibration "
        r"$\hat q_{1-\alpha}$", fc="#fdeee0", ec="#a35d1f")
    # right column (online)
    box(ax, 0.64, 0.68, 0.325, 0.185, "Observe disruption state\n"
        r"$\theta$ (graph); one GNN pass")
    box(ax, 0.64, 0.40, 0.325, 0.20, "Evaluate any policy " r"$x$:" "\n"
        "mean, predictive samples,\ncalibrated intervals")
    box(ax, 0.64, 0.11, 0.325, 0.20, "Calibrated Pareto frontier\n"
        "over candidate policies\n(interval dominance)",
        fc="#f5e8ee", ec="#7a1f1f")
    # vertical arrows in left and right columns
    arrow(ax, (0.1375, 0.70), (0.1375, 0.603))
    arrow(ax, (0.1375, 0.415), (0.1375, 0.318))
    arrow(ax, (0.8025, 0.68), (0.8025, 0.603))
    arrow(ax, (0.8025, 0.40), (0.8025, 0.313))
    # horizontal arrows left column -> middle column (straight, no crossing)
    arrow(ax, (0.245, 0.79), (0.315, 0.79))
    arrow(ax, (0.245, 0.22), (0.315, 0.20))
    arrow(ax, (0.245, 0.50), (0.315, 0.475))
    # middle column internal
    arrow(ax, (0.4225, 0.66), (0.4225, 0.578))
    arrow(ax, (0.4225, 0.375), (0.4225, 0.293))
    # middle -> right
    arrow(ax, (0.53, 0.79), (0.64, 0.775))
    arrow(ax, (0.53, 0.475), (0.64, 0.50))
    arrow(ax, (0.53, 0.19), (0.64, 0.21))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_framework.png"), bbox_inches="tight")
    plt.close()


def fig_topology():
    import twin
    G = nx.DiGraph()
    for i, nm in enumerate(twin.UNIT_NAMES):
        G.add_node(i, name=nm)
    T = twin.BASE_TRANSFER
    for u in range(19):
        for v in range(19):
            if T[u, v] > 0.01:
                G.add_edge(u, v, w=T[u, v])
    route = twin.ROUTE_AC.mean(0)
    for v in range(1, 19):
        if route[v - 1] > 0.02:
            G.add_edge(0, v, w=route[v - 1])
    # more spread-out layout so labels never collide
    pos = {0: (0.0, 0.0)}
    for i, u in enumerate((1, 2, 3)):
        pos[u] = (1.5, 1.5 - i * 1.5)          # ICUs
    for i, u in enumerate((4, 5)):
        pos[u] = (3.0, 0.8 - i * 1.6)          # SDUs
    for i, u in enumerate(range(6, 14)):        # wards, 2 x 4 grid
        pos[u] = (4.7 + (i % 2) * 1.25, 2.4 - (i // 2) * 1.6)
    pos[14] = (1.5, -3.2); pos[15] = (3.0, -3.2)   # ORs
    pos[16] = (7.4, -0.6); pos[17] = (7.4, -2.2)   # Rehab
    pos[18] = (8.7, -1.4)                          # DLounge
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    caps = twin.BASE_CAP
    colors = (["#c94f4f"] + ["#e2a13c"] * 3 + ["#e8d15a"] * 2 +
              ["#7aa9d9"] * 8 + ["#9a76b8"] * 2 + ["#77b58a"] * 2 + ["#b8b8b8"])
    nx.draw_networkx_nodes(G, pos, node_size=caps * 26, node_color=colors,
                           edgecolors="#333", linewidths=0.7, ax=ax)
    ws = np.array([G[u][v]["w"] for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, width=0.5 + 4.5 * ws, alpha=0.32,
                           arrowsize=8, ax=ax, connectionstyle="arc3,rad=0.08")
    # labels drawn above each node with a white background so they never
    # overlap the node bodies or the edges
    lbl = {i: twin.UNIT_NAMES[i].replace("Ward-", "W") for i in range(19)}
    for i, (x, y) in pos.items():
        r = 0.28 + 0.012 * caps[i]
        ax.text(x, y + r, lbl[i], ha="center", va="bottom", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.85))
    ax.set_xlim(-1.0, 9.7); ax.set_ylim(-4.1, 3.4)
    ax.axis("off"); ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_topology.png"), bbox_inches="tight")
    plt.close()


def load():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    arr = dict(np.load(os.path.join(OUT, "arrays.npz")))
    return res, arr


def fig_rmse(res):
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9))
    Ns = sorted(int(n) for n in res["rmse"]["G2M-SK"])
    for j, ax in enumerate(axes):
        for mth in ["SK", "NN-SK", "NN", "G2M-SK"]:
            mu = [np.mean([res["rmse"][mth][str(N)][s][j]
                           for s in res["rmse"][mth][str(N)]]) for N in Ns]
            sd = [np.std([res["rmse"][mth][str(N)][s][j]
                          for s in res["rmse"][mth][str(N)]]) for N in Ns]
            ax.errorbar(Ns, mu, yerr=sd, marker="o", ms=3.5, capsize=2.5,
                        lw=1.3, label=mth, color=COLS[mth])
        ax.set_title(OBJ[j], fontsize=8.6, pad=6)
        ax.set_xlabel("training scenarios N", fontsize=8)
        ax.set_xticks(Ns)
        ax.tick_params(labelsize=7.5)
        if j == 0:
            ax.set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.6)
    plt.savefig(os.path.join(FIG, "fig3_rmse.png"), bbox_inches="tight")
    plt.close()


def fig_uq(res):
    methods = ["G2M-SK (conformal)", "G2M-SK w/o conformal",
               "NN-SK (Gaussian)", "NN ensemble"]
    labels = ["G2M-SK\n(conformal)", "G2M-SK\n(no conf.)",
              "NN-SK\n(Gaussian)", "NN\nensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    xs = np.arange(4); w = 0.19
    for j in range(4):
        cov = [np.mean([res["uq"][m][s]["cov"][j] for s in res["uq"][m]])
               for m in methods]
        wid = [np.mean([res["uq"][m][s]["wid"][j] for s in res["uq"][m]])
               for m in methods]
        axes[0].bar(xs + (j - 1.5) * w, cov, w, label=OBJ_S[j])
        axes[1].bar(xs + (j - 1.5) * w, wid, w)
    axes[0].axhline(0.90, color="k", ls="--", lw=1)
    axes[0].text(-0.62, 0.925, "target 90%", fontsize=7.5, va="bottom")
    axes[0].set_ylabel("empirical coverage"); axes[0].set_ylim(0, 1.12)
    axes[1].set_ylabel("normalized interval width")
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.8)
        ax.tick_params(axis="y", labelsize=8)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(os.path.join(FIG, "fig5_uq.png"), bbox_inches="tight")
    plt.close()


def fig_density(arr):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    picks = [(0, 0), (1, 3)]
    for ax, (i, j) in zip(axes, picks):
        emp = arr[f"dens{i}_emp"][:, j]; gen = arr[f"dens{i}_gen"][:, j]
        gm, gs = arr[f"dens{i}_gm"][j], arr[f"dens{i}_gs"][j]
        lo = min(emp.min(), gen.min()); hi = max(emp.max(), gen.max())
        bins = np.linspace(lo, hi, 26)
        ax.hist(emp, bins=bins, density=True, alpha=0.45, color="#777",
                label="digital twin (200 replications)")
        ax.hist(gen, bins=bins, density=True, alpha=0.55, color="#b2182b",
                histtype="step", lw=1.8, label="G2M-SK generator")
        g = np.linspace(lo, hi, 200)
        ax.plot(g, np.exp(-(g - gm) ** 2 / (2 * gs ** 2)) /
                (gs * np.sqrt(2 * np.pi)), color="#2166ac", lw=1.6,
                label="Gaussian posterior (mean surface)")
        ax.set_xlabel(OBJ[j]); ax.set_ylabel("density")
        ax.tick_params(labelsize=8)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(os.path.join(FIG, "fig6_density.png"), bbox_inches="tight")
    plt.close()


def fig_pareto(arr):
    mu = arr["par_mu"]; tf = arr["par_true_idx"]
    sg = arr["par_sel_g2m"]; ss = arr["par_sel_sk"]
    pm = arr["par_pred_g2m"]; sig = arr["par_sig_g2m"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, (a, b) in zip(axes, [(0, 3), (2, 3)]):
        ax.scatter(mu[:, a], mu[:, b], s=8, c="#bbb",
                   label="candidates (true values)")
        ax.scatter(mu[tf, a], mu[tf, b], s=26, c="k", marker="s",
                   label="true Pareto set")
        ax.scatter(mu[sg, a], mu[sg, b], s=34, facecolors="none",
                   edgecolors="#b2182b", lw=1.6, label="G2M-SK selected")
        ax.scatter(mu[ss, a], mu[ss, b], s=52, facecolors="none",
                   edgecolors="#8073ac", marker="^", lw=1.2,
                   label="classical SK selected")
        ax.errorbar(pm[sg, a], pm[sg, b], xerr=sig[sg, a], yerr=sig[sg, b],
                    fmt="none", ecolor="#b2182b", alpha=0.35, lw=0.9)
        ax.set_xlabel(OBJ[a]); ax.set_ylabel(OBJ[b])
        ax.tick_params(labelsize=8)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(os.path.join(FIG, "fig7_pareto.png"), bbox_inches="tight")
    plt.close()


def fig_drift(res):
    d = res["drift"]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    xs = np.arange(4); w = 0.25
    ax.bar(xs - w, d["in_dist"], w, label="in-distribution", color="#4d9221")
    ax.bar(xs, d["shifted"], w, label="severe shift", color="#c94f4f")
    ax.bar(xs + w, d["recalibrated"], w,
           label="shift + recalibration (8 scenarios)", color="#2166ac")
    ax.axhline(0.90, color="k", ls="--", lw=1)
    ax.text(3.42, 0.915, "target 90%", fontsize=7.5, va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(OBJ_S)
    ax.set_ylabel("coverage of 90% interval"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.20))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(os.path.join(FIG, "fig8_drift.png"), bbox_inches="tight")
    plt.close()


def fig_cost(res):
    t = res["timing"]; s = res["storage"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    names = ["SK", "NN", "G2M-SK\n(mean only)", "G2M-SK\n(full,\n100 samples)"]
    vals = np.array([t["sk"], t["nn"], t["g2m_mean"], t["g2m_full"]]) * 1000
    bars = axes[0].bar(names, vals,
                       color=["#8073ac", "#4d9221", "#b2182b", "#b2182b"])
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v * 1.10,
                     f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("online time per 1000\npolicy evaluations (ms)")
    axes[0].set_yscale("log"); axes[0].set_ylim(1, 12000)
    n2 = ["SK", "NN ensemble", "G2M-SK"]
    v2 = np.array([s["sk"], s["nn"], s["g2m"]]) / 1e6
    bars2 = axes[1].bar(n2, v2, color=["#8073ac", "#4d9221", "#b2182b"])
    for b, v in zip(bars2, v2):
        axes[1].text(b.get_x() + b.get_width() / 2, v * 1.15,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylabel("metamodel storage (MB)")
    axes[1].set_yscale("log"); axes[1].set_ylim(0.05, 200)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.8)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(w_pad=2.0)
    plt.savefig(os.path.join(FIG, "fig9_cost.png"), bbox_inches="tight")
    plt.close()


def main():
    fig_framework(); fig_topology()
    res, arr = load()
    fig_rmse(res); fig_uq(res); fig_density(arr)
    fig_pareto(arr); fig_drift(res); fig_cost(res)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
