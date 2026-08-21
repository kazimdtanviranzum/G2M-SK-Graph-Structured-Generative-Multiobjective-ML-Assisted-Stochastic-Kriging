"""All manuscript figures. Reads out/results.json and out/arrays.npz."""
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


def box(ax, x, y, w, h, text, fc="#eef3fa", ec="#2b4c7e", fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color="#333", **kw))


def fig_framework():
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.185, 0.965, "Offline stage", fontsize=10, weight="bold",
            ha="center", color="#2b4c7e")
    ax.text(0.80, 0.965, "Online stage", fontsize=10, weight="bold",
            ha="center", color="#7a1f1f")
    ax.add_patch(FancyBboxPatch((0.01, 0.03), 0.50, 0.90, fc="none",
                                ec="#2b4c7e", lw=1.0, linestyle="--",
                                boxstyle="round,pad=0.005"))
    ax.add_patch(FancyBboxPatch((0.60, 0.03), 0.39, 0.90, fc="none",
                                ec="#7a1f1f", lw=1.0, linestyle="--",
                                boxstyle="round,pad=0.005"))
    box(ax, 0.03, 0.72, 0.20, 0.15, "Disruption scenario\nlibrary "
        r"$\{\theta_n\}$" "\n(hospital graphs)")
    box(ax, 0.03, 0.47, 0.20, 0.15, "Digital twin\nsimulation at\ndesign "
        r"pairs $(\theta_n,x_k)$")
    box(ax, 0.03, 0.22, 0.20, 0.15, "Replications\n"
        r"$\{Y_m(x_k;\theta_n)\}$")
    box(ax, 0.30, 0.66, 0.19, 0.21, "GNN encoder\n" r"$h(\theta)$" "\n+ LMC-SK heads\n"
        r"$\beta,L_q,\gamma_q,\phi_F$", fc="#fdeee0", ec="#a35d1f")
    box(ax, 0.30, 0.38, 0.19, 0.18, "Wasserstein\ngenerator + critic\n(anchored on SK)",
        fc="#fdeee0", ec="#a35d1f")
    box(ax, 0.30, 0.12, 0.19, 0.16, "Split-conformal\ncalibration\n"
        r"$\hat q_{1-\alpha}$", fc="#fdeee0", ec="#a35d1f")
    box(ax, 0.63, 0.70, 0.33, 0.16, "Observe disruption state\n"
        r"$\theta$ (graph), one GNN pass")
    box(ax, 0.63, 0.44, 0.33, 0.18, "Evaluate any policy "
        r"$x$:" "\nmean, predictive samples,\ncalibrated intervals")
    box(ax, 0.63, 0.14, 0.33, 0.20, "Calibrated Pareto frontier\nover candidate "
        "policies\n(interval dominance)", fc="#f5e8ee", ec="#7a1f1f")
    arrow(ax, (0.13, 0.72), (0.13, 0.63))
    arrow(ax, (0.13, 0.47), (0.13, 0.38))
    arrow(ax, (0.23, 0.55), (0.30, 0.74), connectionstyle="arc3,rad=0.15")
    arrow(ax, (0.23, 0.30), (0.30, 0.47), connectionstyle="arc3,rad=0.12")
    arrow(ax, (0.395, 0.66), (0.395, 0.57))
    arrow(ax, (0.395, 0.38), (0.395, 0.29))
    arrow(ax, (0.49, 0.76), (0.63, 0.78))
    arrow(ax, (0.49, 0.47), (0.63, 0.53))
    arrow(ax, (0.49, 0.20), (0.63, 0.24))
    arrow(ax, (0.795, 0.70), (0.795, 0.63))
    arrow(ax, (0.795, 0.44), (0.795, 0.35))
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
    pos = {0: (0, 0)}
    for i, u in enumerate((1, 2, 3)): pos[u] = (1.15, 1.15 - i * 1.1)
    for i, u in enumerate((4, 5)): pos[u] = (2.3, 0.6 - i * 1.2)
    for i, u in enumerate(range(6, 14)):
        pos[u] = (3.6 + (i % 2) * 0.85, 1.9 - (i // 2) * 1.15)
    pos[14] = (1.15, -2.3); pos[15] = (2.3, -2.3)
    pos[16] = (5.6, -0.4); pos[17] = (5.6, -1.5); pos[18] = (6.6, -0.95)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    caps = twin.BASE_CAP
    colors = (["#c94f4f"] + ["#e2a13c"] * 3 + ["#e8d15a"] * 2 +
              ["#7aa9d9"] * 8 + ["#9a76b8"] * 2 + ["#77b58a"] * 2 + ["#b8b8b8"])
    nx.draw_networkx_nodes(G, pos, node_size=caps * 22, node_color=colors,
                           edgecolors="#333", linewidths=0.7, ax=ax)
    ws = np.array([G[u][v]["w"] for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, width=0.6 + 5.5 * ws, alpha=0.45,
                           arrowsize=8, ax=ax, connectionstyle="arc3,rad=0.08")
    lbl = {i: twin.UNIT_NAMES[i].replace("Ward-", "W") for i in range(19)}
    nx.draw_networkx_labels(G, pos, lbl, font_size=6.8, ax=ax)
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
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5))
    Ns = sorted(int(n) for n in res["rmse"]["G2M-SK"])
    for j, ax in enumerate(axes):
        for mth in ["SK", "NN-SK", "NN", "G2M-SK"]:
            mu = [np.mean([res["rmse"][mth][str(N)][s][j]
                           for s in res["rmse"][mth][str(N)]]) for N in Ns]
            sd = [np.std([res["rmse"][mth][str(N)][s][j]
                          for s in res["rmse"][mth][str(N)]]) for N in Ns]
            ax.errorbar(Ns, mu, yerr=sd, marker="o", ms=3.5, capsize=2.5,
                        lw=1.3, label=mth, color=COLS[mth])
        ax.set_title(OBJ[j], fontsize=8.5)
        ax.set_xlabel("training scenarios N"); ax.set_xticks(Ns)
        if j == 0:
            ax.set_ylabel("RMSE"); ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_rmse.png"), bbox_inches="tight")
    plt.close()


def fig_uq(res):
    methods = ["G2M-SK (conformal)", "G2M-SK w/o conformal",
               "NN-SK (Gaussian)", "NN ensemble"]
    labels = ["G2M-SK\n(conformal)", "G2M-SK\n(no conf.)",
              "NN-SK\n(Gaussian)", "NN\nensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.7))
    xs = np.arange(4); w = 0.19
    for j in range(4):
        cov = [np.mean([res["uq"][m][s]["cov"][j] for s in res["uq"][m]])
               for m in methods]
        wid = [np.mean([res["uq"][m][s]["wid"][j] for s in res["uq"][m]])
               for m in methods]
        axes[0].bar(xs + (j - 1.5) * w, cov, w, label=OBJ_S[j])
        axes[1].bar(xs + (j - 1.5) * w, wid, w)
    axes[0].axhline(0.90, color="k", ls="--", lw=1)
    axes[0].text(3.35, 0.905, "target 90%", fontsize=7)
    axes[0].set_ylabel("empirical coverage"); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("normalized interval width")
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    axes[0].legend(fontsize=7, ncol=4, frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_uq.png"), bbox_inches="tight")
    plt.close()


def fig_density(arr):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.6))
    picks = [(0, 0), (1, 3)]
    for ax, (i, j) in zip(axes, picks):
        emp = arr[f"dens{i}_emp"][:, j]; gen = arr[f"dens{i}_gen"][:, j]
        gm, gs = arr[f"dens{i}_gm"][j], arr[f"dens{i}_gs"][j]
        lo = min(emp.min(), gen.min()); hi = max(emp.max(), gen.max())
        bins = np.linspace(lo, hi, 26)
        ax.hist(emp, bins=bins, density=True, alpha=0.45, color="#777",
                label="digital twin (200 rep.)")
        ax.hist(gen, bins=bins, density=True, alpha=0.55, color="#b2182b",
                histtype="step", lw=1.8, label="G2M-SK generator")
        g = np.linspace(lo, hi, 200)
        ax.plot(g, np.exp(-(g - gm) ** 2 / (2 * gs ** 2)) /
                (gs * np.sqrt(2 * np.pi)), color="#2166ac", lw=1.6,
                label="Gaussian posterior (mean surface)")
        ax.set_xlabel(OBJ[j]); ax.set_ylabel("density")
        ax.legend(fontsize=6.6, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig6_density.png"), bbox_inches="tight")
    plt.close()


def fig_pareto(arr):
    mu = arr["par_mu"]; tf = arr["par_true_idx"]
    sg = arr["par_sel_g2m"]; ss = arr["par_sel_sk"]
    pm = arr["par_pred_g2m"]; sig = arr["par_sig_g2m"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax, (a, b) in zip(axes, [(0, 3), (2, 3)]):
        ax.scatter(mu[:, a], mu[:, b], s=8, c="#bbb", label="candidates (true)")
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
    axes[0].legend(fontsize=6.6, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_pareto.png"), bbox_inches="tight")
    plt.close()


def fig_drift(res):
    d = res["drift"]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    xs = np.arange(4); w = 0.25
    ax.bar(xs - w, d["in_dist"], w, label="in-distribution", color="#4d9221")
    ax.bar(xs, d["shifted"], w, label="severe shift", color="#c94f4f")
    ax.bar(xs + w, d["recalibrated"], w,
           label="shift + recalibration (8 scen.)", color="#2166ac")
    ax.axhline(0.90, color="k", ls="--", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(OBJ_S)
    ax.set_ylabel("coverage of 90% interval"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower center")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig8_drift.png"), bbox_inches="tight")
    plt.close()


def fig_cost(res):
    t = res["timing"]; s = res["storage"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    names = ["SK", "NN", "G2M-SK\n(mean only)", "G2M-SK\n(full, 100 samples)"]
    vals = [t["sk"], t["nn"], t["g2m_mean"], t["g2m_full"]]
    axes[0].bar(names, np.array(vals) * 1000,
                color=["#8073ac", "#4d9221", "#b2182b", "#b2182b"])
    axes[0].set_ylabel("online time per 1000\npolicy evaluations (ms)")
    n2 = ["SK", "NN ensemble", "G2M-SK"]
    v2 = [s["sk"] / 1e6, s["nn"] / 1e6, s["g2m"] / 1e6]
    axes[1].bar(n2, v2, color=["#8073ac", "#4d9221", "#b2182b"])
    axes[1].set_ylabel("metamodel storage (MB)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    plt.tight_layout()
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
