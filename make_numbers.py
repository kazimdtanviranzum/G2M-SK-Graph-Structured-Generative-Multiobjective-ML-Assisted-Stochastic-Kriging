"""Regenerate out/numbers.json from out/results.json for the v4 manuscript.

Keeps the exact schema consumed by build_v4.js / build_v4b.js (inherited from
the v3 builders) and adds the fields for the five-seed revision: bootstrap
confidence intervals, additional conformalized baselines, the regime-switching
(multimodal) twin comparison, and the face-validity magnitudes.
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")
res = json.load(open(os.path.join(OUT, "results.json")))
old = json.load(open(os.path.join(OUT, "numbers.json"))) \
    if os.path.exists(os.path.join(OUT, "numbers.json")) else {}

SEEDS = sorted(res["nrmse"]["G2M-SK"]["64"].keys(), key=int)
NSEED = len(SEEDS)


def ms(vals, d=3):
    v = np.asarray(vals, float)
    return f"{v.mean():.{d}f} ({v.std():.{d}f})"


def ms_auto(vals):
    v = np.asarray(vals, float)
    d = 2 if v.mean() < 10 else 1
    return f"{v.mean():.{d}f} ({v.std():.{d}f})"


R = {}

# ---------- rmse64 (Table 7 rows: f1..f4 + normalized) ----------
R["rmse64"] = {}
for meth in ["SK", "NN-SK", "NN", "G2M-SK"]:
    per = np.array([res["rmse"][meth]["64"][s] for s in SEEDS])
    R["rmse64"][meth] = [ms_auto(per[:, j]) for j in range(4)] + \
        [ms([res["nrmse"][meth]["64"][s] for s in SEEDS])]

R["f2"] = {m: f"{np.mean([res['rmse'][m]['64'][s][1] for s in SEEDS]):.1f}"
           for m in ["G2M-SK", "NN-SK"]}

# ---------- nrmse by N ----------
R["nrmse_by_N"] = {m: {N: ms([res["nrmse"][m][N][s] for s in SEEDS])
                       for N in ["16", "32", "64"]}
                   for m in ["SK", "NN-SK", "NN", "G2M-SK"]}

# ---------- uq ----------
R["uq"] = {}
for name in ["G2M-SK (conformal)", "G2M-SK w/o conformal",
             "NN-SK (Gaussian)", "NN ensemble"]:
    covs = np.array([res["uq"][name][s]["cov"] for s in SEEDS])
    wids = np.array([res["uq"][name][s]["wid"] for s in SEEDS])
    R["uq"][name] = dict(cov=[f"{covs[:, j].mean():.2f}" for j in range(4)],
                         wid=[f"{wids[:, j].mean():.2f}" for j in range(4)],
                         meancov=f"{covs.mean():.3f}",
                         meanwid=f"{wids.mean():.2f}")

# ---------- new conformalized baselines ----------
R["bl"] = {}
for name, per in res.get("baselines_new", {}).items():
    ss = sorted(per.keys(), key=int)
    nr = [per[s]["nrmse"] for s in ss]
    rm = np.array([per[s]["rmse"] for s in ss])
    cov = np.array([per[s]["cov"] for s in ss])
    wid = np.array([per[s]["wid"] for s in ss])
    R["bl"][name] = dict(
        rmse=[ms_auto(rm[:, j]) for j in range(4)] + [ms(nr)],
        cov=[f"{cov[:, j].mean():.2f}" for j in range(4)],
        wid=[f"{wid[:, j].mean():.2f}" for j in range(4)],
        meancov=f"{cov.mean():.3f}", meanwid=f"{wid.mean():.2f}",
        nrm=f"{np.mean(nr):.3f}",
        t1000=f"{np.mean([per[s2]['t_pred_ms'] for s2 in ss]):.0f}")

# ---------- bootstrap ----------
R["boot"] = {k: dict(mean=f"{v['mean']:.3f}", lo=f"{v['lo']:.3f}",
                     hi=f"{v['hi']:.3f}", n=str(v["n_cells"]))
             for k, v in res.get("bootstrap", {}).items()}

# ---------- ablation (Table 11: [nrmse, cov, wid]) ----------
R["abl"] = {}
for name in ["full", "w/o GNN", "w/o generator", "w/o conformal"]:
    per = res["ablation"][name]
    ss = sorted(per.keys(), key=int)
    nr = [per[s]["nrmse"] for s in ss]
    cov = np.array([per[s]["cov"] for s in ss])
    wid = np.array([per[s]["wid"] for s in ss])
    R["abl"][name] = [ms(nr), f"{cov.mean():.3f}", f"{wid.mean():.2f}"]
full_n = float(R["abl"]["full"][0].split(" ")[0])
gnn_n = float(R["abl"]["w/o GNN"][0].split(" ")[0])
conf_w, full_w = float(R["abl"]["w/o conformal"][2]), float(R["abl"]["full"][2])

# ---------- w1 (Table 9) ----------
w1 = res["w1"]
if "mm" in w1:            # recomputed three-row form from extensions.py
    R["w1"] = dict(gen=f"{w1['gen'][0]:.2f} ({w1['gen'][1]:.2f})",
                   mm=f"{w1['mm'][0]:.2f} ({w1['mm'][1]:.2f})",
                   msf=f"{w1['msf'][0]:.2f} ({w1['msf'][1]:.2f})",
                   skew_emp=f"{w1['skew_emp']:.2f}",
                   skew_gen=f"{w1.get('skew_gen', 0):.2f}")
else:                     # base pipeline two-row form; mm added later
    R["w1"] = dict(gen=f"{w1['gen'][0]:.2f} ({w1['gen'][1]:.2f})",
                   mm=old.get("w1", {}).get("mm", ""),
                   msf=f"{w1['gauss'][0]:.2f} ({w1['gauss'][1]:.2f})",
                   skew_emp=old.get("w1", {}).get("skew_emp", ""),
                   skew_gen=old.get("w1", {}).get("skew_gen", ""))

# ---------- multimodal variant ----------
if "multimodal" in res:
    mm = res["multimodal"]
    R["mm"] = dict(gen=f"{mm['gen'][0]:.2f} ({mm['gen'][1]:.2f})",
                   mmg=f"{mm['mm_gauss'][0]:.2f} ({mm['mm_gauss'][1]:.2f})",
                   post=f"{mm['posterior'][0]:.2f} ({mm['posterior'][1]:.2f})",
                   gen_m=f"{mm['gen'][0]:.2f}",
                   mmg_m=f"{mm['mm_gauss'][0]:.2f}",
                   post_m=f"{mm['posterior'][0]:.2f}",
                   skew=f"{mm['mean_abs_skew']:.2f}",
                   kurt_emp=f"{mm.get('kurt_emp', 0):.2f}",
                   kurt_gen=f"{mm.get('kurt_gen', 0):.2f}",
                   gain=f"{(mm['mm_gauss'][0]-mm['gen'][0])/mm['mm_gauss'][0]*100:.0f}")

# ---------- pareto (Table 10: [hv, gd]) ----------
R["pareto"] = {m: [f"{res['pareto']['hv'][m][0]:.3f} ({res['pareto']['hv'][m][1]:.3f})",
                   f"{res['pareto']['gd'][m][0]:.3f} ({res['pareto']['gd'][m][1]:.3f})"]
               for m in ["SK", "NN-SK", "NN", "G2M-SK"]}

# ---------- drift ----------
dr = res["drift"]
R["drift"] = {k: [f"{v:.2f}" for v in dr[k]]
              for k in ["in_dist", "shifted", "recalibrated"]}
R["drift_mean"] = {k: f"{np.mean(dr[k]):.3f}"
                   for k in ["in_dist", "shifted", "recalibrated"]}
R["drift"]["worst_shift"] = f"{min(dr['shifted']):.2f}"

# ---------- cost (Table 12) ----------
t, s = res["timing"], res["storage"]
nnsk_store = old.get("params", {}).get("nnsk", 31782) * 4 + 24 * 4 * 4
R["cost"] = dict(
    train={"SK": f"{np.mean([res['train_time']['SK']['64'][x] for x in SEEDS]):.0f}",
           "NN-SK": f"{np.mean([res['train_time']['NN-SK']['64'][x] for x in SEEDS]):.0f}",
           "NN": f"{np.mean([res['train_time']['NN']['64'][x] for x in SEEDS]):.0f}",
           "G2M-SK": f"{np.mean([res['train_time']['G2M-SK']['64'][x] for x in SEEDS]):.0f}"},
    online={"SK": f"{t['sk']*1000:.0f}", "NN": f"{t['nn']*1000:.1f}",
            "G2M_mean": f"{t['g2m_mean']*1000:.0f}",
            "G2M_full": f"{t['g2m_full']*1000:.0f}",
            "NNSK": f"{t['g2m_mean']*1000:.0f}"},
    storage={"SK": f"{s['sk']/1e6:.1f}", "NN": f"{s['nn']/1e6:.2f}",
             "G2M": f"{s['g2m']/1e6:.2f}", "NNSK": f"{nnsk_store/1e6:.2f}"},
    speedup=f"{t['sk']/t['g2m_mean']:.0f}",
    storratio=f"{s['sk']/s['g2m']:.0f}")

# ---------- headline ----------
def mean_n(m, N="64"):
    return float(np.mean([res["nrmse"][m][N][x] for x in SEEDS]))

R["headline"] = dict(
    imp_nnsk=f"{(mean_n('NN-SK')-mean_n('G2M-SK'))/mean_n('NN-SK')*100:.0f}",
    imp_sk=f"{(mean_n('SK')-mean_n('G2M-SK'))/mean_n('SK')*100:.0f}",
    g64=f"{mean_n('G2M-SK'):.3f}",
    g16=f"{mean_n('G2M-SK','16'):.3f}",
    nnsk16=f"{mean_n('NN-SK','16'):.3f}",
    nnsk64=f"{mean_n('NN-SK'):.3f}",
    nognn_ratio=f"{gnn_n/full_n:.2f}",
    conf_wid_pct=f"{(conf_w-full_w)/full_w*100:.0f}",
    meancov=R["uq"]["G2M-SK (conformal)"]["meancov"],
    nnskcov=R["uq"]["NN-SK (Gaussian)"]["meancov"],
    hv=f"{res['pareto']['hv']['G2M-SK'][0]:.3f}",
    hv_sk=f"{res['pareto']['hv']['SK'][0]:.3f}",
    hv_nn=f"{res['pareto']['hv']['NN'][0]:.3f}",
    nseeds=str(NSEED),
    nseeds_word={3: "three", 5: "five", 7: "seven",
                 10: "ten"}.get(NSEED, str(NSEED)))

# ---------- face validity ----------
if "facevalid" in res:
    fv = res["facevalid"]
    R["fv"] = dict(bb=f"{fv['base_board'][0]:.1f}", bbs=f"{fv['base_board'][1]:.1f}",
                   bu=f"{fv['base_unserved'][0]:.0f}",
                   mb=f"{fv['mod_board'][0]:.1f}", mu=f"{fv['mod_unserved'][0]:.0f}")

# ---------- static twin/params blocks ----------
for k in ["twin", "twin_tot", "params"]:
    if k in old:
        R[k] = old[k]

json.dump(R, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print("numbers.json regenerated for", NSEED, "seeds; headline:", R["headline"])
