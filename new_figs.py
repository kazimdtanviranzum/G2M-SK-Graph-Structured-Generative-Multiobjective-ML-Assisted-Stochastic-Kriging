"""Two new manuscript figures: problem setting (Sec. 3) and twin/software structure (Sec. 5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

GREY = "#f2f2f2"
DARK = "#d9d9d9"
EDGE = "#333333"


def box(ax, x, y, w, h, text, fc=GREY, fs=8.6, bold=False, ec=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, lw=1.2, color=EDGE,
                                 linestyle=ls))


# ------------------------------------------------------------------ Figure 1
fig, ax = plt.subplots(figsize=(9.4, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# panel titles
ax.text(2.45, 4.72, "Offline stage (weeks before any disruption)",
        ha="center", fontsize=10.5, fontweight="bold")
ax.text(7.55, 4.72, "Online stage (minutes after disruption onset)",
        ha="center", fontsize=10.5, fontweight="bold")
ax.plot([5.0, 5.0], [0.15, 4.55], ls="--", lw=1.0, color="#888888")

# offline column
box(ax, 0.45, 3.55, 4.0, 0.75,
    "Scenario distribution\nsample N disruption states $\\theta_1,\\ldots,\\theta_N$")
box(ax, 0.45, 2.35, 4.0, 0.75,
    "Stochastic digital twin\nsimulate K design policies $\\times$ n replications per state")
box(ax, 0.45, 1.15, 4.0, 0.75,
    "Offline library\nsample means and variances of $(f_1,\\ldots,f_4)$")
box(ax, 0.45, 0.15, 4.0, 0.62,
    "Train metamodel and calibrate uncertainty", fc=DARK, bold=True)
arrow(ax, 2.45, 3.55, 2.45, 3.12)
arrow(ax, 2.45, 2.35, 2.45, 1.92)
arrow(ax, 2.45, 1.15, 2.45, 0.79)

# online column
box(ax, 5.55, 3.55, 4.0, 0.75,
    "Observed hospital state $\\theta$\nattributed graph: 19 units, transfer corridors")
box(ax, 5.55, 2.35, 4.0, 0.75,
    "Metamodel forward pass\nscore any candidate recovery policy $x$ in $\\ll$ 1 ms",
    fc=DARK, bold=True)
box(ax, 5.55, 1.15, 4.0, 0.75,
    "Calibrated objective estimates, predictive\nsamples, and robust Pareto frontier")
box(ax, 5.55, 0.15, 4.0, 0.62,
    "Manager selects the recovery policy")
arrow(ax, 7.55, 3.55, 7.55, 3.12)
arrow(ax, 7.55, 2.35, 7.55, 1.92)
arrow(ax, 7.55, 1.15, 7.55, 0.79)

# offline -> online transfer arrow
arrow(ax, 4.45, 0.46, 5.55, 2.60)
ax.text(4.98, 1.75, "deploy\n(0.14 MB)", ha="center", fontsize=8,
        style="italic", bbox=dict(fc="white", ec="none", pad=1))

ax.text(2.45, -0.28, "hours of computation, spent in advance",
        ha="center", fontsize=8.5, style="italic")
ax.text(7.55, -0.28, "milliseconds of computation, at decision time",
        ha="center", fontsize=8.5, style="italic")
fig.savefig("figs/fig0_problem.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ------------------------------------------------------------------ Figure 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (ax1, ax2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

# ---- (a) digital twin structure
ax1.set_title("(a) Structure of the stochastic hospital digital twin",
              fontsize=10.5, fontweight="bold", pad=8)
box(ax1, 0.2, 7.6, 4.5, 2.0,
    "Scenario input $\\theta$\nseverity, unit capacity and staffing\ndegradation,"
    " arrival surge, acuity mix\nshift, corridor degradation, occupancy")
box(ax1, 5.3, 7.6, 4.5, 2.0,
    "Recovery policy $x$\n$x_1$ bed reallocation, $x_2$ staff\nredeployment,"
    " $x_3$ ED diversion,\n$x_4$ accelerated discharge")
ax1.add_patch(FancyBboxPatch((0.2, 1.9), 9.6, 4.9,
              boxstyle="round,pad=0.012", fc="white", ec=EDGE, lw=1.3))
ax1.text(5.0, 6.42, "Patient flow engine (discrete time, $\\Delta t$ = 0.25 h, 48 h horizon)",
         ha="center", fontsize=9, fontweight="bold")
box(ax1, 0.55, 4.7, 2.75, 1.3,
    "Nonstationary\narrivals\n(6 acuity classes)", fc=GREY)
box(ax1, 3.65, 4.7, 2.75, 1.3, "Emergency\ndepartment,\nboarding queue", fc=GREY)
box(ax1, 6.75, 4.7, 2.75, 1.3, "Acuity based\nrouting matrix", fc=GREY)
box(ax1, 0.55, 2.35, 2.75, 1.3, "19 care units\n(beds, staff,\nservice rates)", fc=GREY)
box(ax1, 3.65, 2.35, 2.75, 1.3, "Blocked transfer\nqueues between\nunits", fc=GREY)
box(ax1, 6.75, 2.35, 2.75, 1.3, "Discharge and\ndischarge lounge", fc=GREY)
arrow(ax1, 3.30, 5.35, 3.65, 5.35)
arrow(ax1, 6.40, 5.35, 6.75, 5.35)
arrow(ax1, 8.12, 4.70, 1.92, 3.65)
arrow(ax1, 3.30, 3.0, 3.65, 3.0)
arrow(ax1, 6.40, 3.0, 6.75, 3.0)
arrow(ax1, 2.45, 7.6, 2.45, 6.8)
arrow(ax1, 7.55, 7.6, 7.55, 6.8)
box(ax1, 1.7, 0.25, 6.6, 1.15,
    "Objectives $f_1$ boarding delay, $f_2$ diverted and unserved,\n"
    "$f_3$ overflow bed hours, $f_4$ staff overload index", fc=DARK, bold=True)
arrow(ax1, 5.0, 1.9, 5.0, 1.4)

# ---- (b) software structure
ax2.set_title("(b) Software structure of the released implementation",
              fontsize=10.5, fontweight="bold", pad=8)
box(ax2, 0.3, 8.4, 4.3, 1.25, "twin.py\ndigital twin simulator", fc=GREY)
box(ax2, 5.4, 8.4, 4.3, 1.25, "models.py\nGNN, SK layer, generator,\ncritic, baselines", fc=GREY, fs=8.2)
box(ax2, 0.3, 6.3, 4.3, 1.25, "train.py\ntraining, prediction,\nconformal calibration", fc=GREY, fs=8.2)
box(ax2, 5.4, 6.3, 4.3, 1.25, "experiments.py\nfull pipeline, 3 master\nseeds, resumable stages", fc=GREY, fs=8.2)
box(ax2, 0.3, 4.2, 4.3, 1.25, "figures.py / eqs.py\nfigure and equation\ngeneration", fc=GREY, fs=8.2)
box(ax2, 5.4, 4.2, 4.3, 1.25, "out/results.json\nout/arrays.npz\nout/numbers.json", fc="white", fs=8.2)
box(ax2, 0.3, 2.1, 4.3, 1.25, "figs/*.png\neqs/*.png", fc="white")
box(ax2, 5.4, 2.1, 4.3, 1.25, "build scripts\n(docx-js)", fc=GREY)
box(ax2, 2.6, 0.25, 4.8, 1.05, "manuscript (.docx)", fc=DARK, bold=True)
arrow(ax2, 2.45, 8.4, 2.45, 7.55)     # twin -> train
arrow(ax2, 5.4, 8.7, 4.6, 7.55)       # models -> train
arrow(ax2, 4.6, 6.92, 5.4, 6.92)      # train -> experiments
arrow(ax2, 7.55, 6.3, 7.55, 5.45)     # experiments -> out
arrow(ax2, 5.4, 4.82, 4.6, 4.82)      # out -> figures
arrow(ax2, 7.55, 4.2, 7.55, 3.35)     # out -> build scripts
arrow(ax2, 2.45, 4.2, 2.45, 3.35)     # figures -> figs
arrow(ax2, 2.45, 2.1, 4.0, 1.30)      # figs -> manuscript
arrow(ax2, 7.55, 2.1, 6.0, 1.30)      # build -> manuscript
fig.savefig("figs/fig4_twin_software.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("new figures written")
