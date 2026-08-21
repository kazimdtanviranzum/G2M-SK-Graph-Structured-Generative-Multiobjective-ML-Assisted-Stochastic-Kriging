"""Extension experiments for the Omega revision of the G2M-SK paper.

Adds, on top of the base pipeline in experiments.py:
  baselines  -- three additional conformalized modern surrogates
                (GBT-CP, NN-CP, GNN-ENS-CP) evaluated at N = 64 for all seeds
  bootstrap  -- paired scenario-level bootstrap confidence intervals for the
                normalized-error differences between G2M-SK and every baseline
  multimodal -- regime-switching (cascade) variant of the digital twin and the
                generator-versus-Gaussian distributional comparison on it
  facevalid  -- undisrupted and moderate-severity face-validity magnitudes of
                the twin for comparison against published emergency-department
                statistics

Run after the base pipeline:  python3 extensions.py <stage>
"""
import json, os, time
import numpy as np
import torch
from scipy.stats import wasserstein_distance, skew

import twin
from twin import (sample_scenario, scenario_features, flat_covariate, simulate,
                  N_UNITS, N_STEPS, DT, BASE_CAP, BASE_STAFF, BASE_LOS,
                  BASE_TRANSFER, BASE_ARRIVAL_RATE, BASE_ACUITY_MIX,
                  AC_LOS_MULT, ROUTE_AC)
from train import (fit_g2msk, g2m_predict, conformal_calibrate,
                   nn_predict, Standardizer)
from models import GNNEncoder, mlp, M_OBJ
import torch.nn as nn

OUT = os.path.join(os.path.dirname(__file__), "out")
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.10
N_CAL, N_TEST = 24, 24
NMAX = "64"


def load_results():
    with open(os.path.join(OUT, "results.json")) as f:
        return json.load(f)


def save_results(results):
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


# ---------------------------------------------------------------- baselines
class GNNReg(nn.Module):
    """Direct graph-network regression: GNN encoder + MLP head on (h, x)."""
    def __init__(self):
        super().__init__()
        self.enc = GNNEncoder()
        self.head = mlp([32 + 4, 64, 64, M_OBJ])

    def forward(self, node, glob, A, X):
        h = self.enc(node, glob, A)                      # [B,32]
        hh = h.unsqueeze(1).expand(-1, X.shape[1], -1)   # [B,P,32]
        z = torch.cat([hh, X], dim=-1)
        return self.head(z)                              # [B,P,m]


def fit_gnn_ens(D, N, seed, n_models=5, epochs=400):
    node = torch.as_tensor(D["node_t"][:N]); glob = torch.as_tensor(D["glob_t"][:N])
    A = torch.as_tensor(D["A_t"][:N])
    Xd = torch.as_tensor(np.repeat(D["Xd"][None], N, 0), dtype=torch.float32)
    std = Standardizer().fit(D["Yb"][:N])
    Yt = torch.as_tensor(std.tf(D["Yb"][:N]), dtype=torch.float32)  # [N,K,m]
    nets = []
    for s in range(n_models):
        torch.manual_seed(seed * 300 + s)
        net = GNNReg()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        for ep in range(epochs):
            opt.zero_grad()
            loss = ((net(node, glob, A, Xd) - Yt) ** 2).mean()
            loss.backward(); opt.step()
        nets.append(net)
    return dict(nets=nets, std=std)


@torch.no_grad()
def gnn_ens_predict(fit, node, glob, A, Xq):
    node = torch.as_tensor(node); glob = torch.as_tensor(glob)
    A = torch.as_tensor(A)
    Xq = torch.as_tensor(Xq, dtype=torch.float32).unsqueeze(0)
    preds = torch.stack([net(node, glob, A, Xq)[0] for net in fit["nets"]])
    m = preds.mean(0).numpy(); s = preds.std(0).numpy() + 1e-3
    return fit["std"].inv(m), fit["std"].inv_scale(s)


def norm_conformal(pred_c, sig_c, mu_c, alpha=ALPHA):
    """Per-objective normalized split conformal quantile."""
    scores = np.abs(mu_c - pred_c) / sig_c                # [...,m]
    n = scores.reshape(-1, M_OBJ).shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    srt = np.sort(scores.reshape(-1, M_OBJ), axis=0)
    return srt[min(k, n) - 1]


def stage_baselines(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    results = load_results()
    D = torch.load(os.path.join(OUT, f"data_s{seed}.pt"), weights_only=False)
    F = torch.load(os.path.join(OUT, f"fits_s{seed}.pt"), weights_only=False)
    K = D["Xd"].shape[0]; NREP = D["Yr"].shape[2]; N = 64
    te_mu, ca_mu = D["te_mu"], D["ca_mu"]
    te_Xs, ca_Xs = D["te_X"], D["ca_X"]
    flat_t, flat_c, flat_e = D["flat_t"], D["flat_c"], D["flat_e"]
    node_c, glob_c, A_c = D["node_c"], D["glob_c"], D["A_c"]
    node_e, glob_e, A_e = D["node_e"], D["glob_e"], D["A_e"]
    truth_sd = te_mu.reshape(-1, M_OBJ).std(0)

    def summarize(name, pe, se, qhat, t_fit, t_pred):
        e = pe - te_mu
        rmse = np.sqrt((e ** 2).mean(axis=(0, 1)))
        nrm = float(np.mean(rmse / truth_sd))
        lo = pe - qhat[None, None] * se; hi = pe + qhat[None, None] * se
        cov = ((te_mu >= lo) & (te_mu <= hi)).mean(axis=(0, 1))
        wid = (hi - lo).mean(axis=(0, 1)) / truth_sd
        results.setdefault("baselines_new", {}).setdefault(name, {})[str(seed)] = dict(
            rmse=rmse.tolist(), nrmse=nrm, cov=cov.tolist(), wid=wid.tolist(),
            meancov=float(cov.mean()), meanwid=float(wid.mean()),
            t_fit=t_fit, t_pred_ms=t_pred * 1000)
        print(f"  seed{seed} {name}: nrmse={nrm:.3f} cov={cov.mean():.3f} "
              f"wid={wid.mean():.2f}", flush=True)

    # ---- GBT-CP: gradient boosted trees, per objective, split conformal ----
    covr = np.repeat(flat_t[:N, None, None], K, 1)
    covr = np.repeat(covr, NREP, 2).reshape(-1, flat_t.shape[1])
    Xr = np.tile(np.repeat(D["Xd"][None], N, 0)[:, :, None], (1, 1, NREP, 1))
    Xr = Xr.reshape(-1, 4)
    Zr = np.concatenate([covr, Xr], 1)
    Yflat = D["Yr"][:N].reshape(-1, M_OBJ)
    t0 = time.time()
    gbts = []
    for j in range(M_OBJ):
        g = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
        g.fit(Zr, Yflat[:, j]); gbts.append(g)
    t_fit = time.time() - t0

    def gbt_pred(flat_i, Xq):
        Z = np.concatenate([np.repeat(flat_i[None], Xq.shape[0], 0), Xq], 1)
        return np.stack([g.predict(Z) for g in gbts], -1)

    pc = np.stack([gbt_pred(flat_c[i], ca_Xs[i]) for i in range(N_CAL)])
    # constant-width split conformal (no model-based scale available)
    res_c = np.abs(ca_mu - pc).reshape(-1, M_OBJ)
    n = res_c.shape[0]; k = int(np.ceil((n + 1) * (1 - ALPHA)))
    q_abs = np.sort(res_c, 0)[min(k, n) - 1]
    t0 = time.time()
    pe = np.stack([gbt_pred(flat_e[i], te_Xs[i]) for i in range(N_TEST)])
    t_pred = (time.time() - t0) / (N_TEST * te_Xs.shape[1]) * 1000
    summarize("GBT-CP", pe, np.ones_like(pe), q_abs, t_fit, t_pred)

    # ---- NN-CP: conformalized deep ensemble (reuses fitted ensemble) ----
    fitn = F["fitn"]
    pc, sc_ = [], []
    for i in range(N_CAL):
        c0 = np.repeat(flat_c[i:i+1], ca_Xs.shape[1], 0)
        m, s = nn_predict(fitn, c0, ca_Xs[i]); pc.append(m); sc_.append(s)
    qn = norm_conformal(np.stack(pc), np.stack(sc_), ca_mu)
    pe, se = [], []
    t0 = time.time()
    for i in range(N_TEST):
        c0 = np.repeat(flat_e[i:i+1], te_Xs.shape[1], 0)
        m, s = nn_predict(fitn, c0, te_Xs[i]); pe.append(m); se.append(s)
    t_pred = (time.time() - t0) / (N_TEST * te_Xs.shape[1]) * 1000
    summarize("NN-CP", np.stack(pe), np.stack(se), qn, 0.0, t_pred)

    # ---- GNN-ENS-CP: conformalized graph-network regression ensemble ----
    t0 = time.time()
    fitg = fit_gnn_ens(D, 64, seed)
    t_fit = time.time() - t0
    pc, sc_ = [], []
    for i in range(N_CAL):
        m, s = gnn_ens_predict(fitg, node_c[i:i+1], glob_c[i:i+1],
                               A_c[i:i+1], ca_Xs[i])
        pc.append(m); sc_.append(s)
    qg = norm_conformal(np.stack(pc), np.stack(sc_), ca_mu)
    pe, se = [], []
    t0 = time.time()
    for i in range(N_TEST):
        m, s = gnn_ens_predict(fitg, node_e[i:i+1], glob_e[i:i+1],
                               A_e[i:i+1], te_Xs[i])
        pe.append(m); se.append(s)
    t_pred = (time.time() - t0) / (N_TEST * te_Xs.shape[1]) * 1000
    summarize("GNN-ENS-CP", np.stack(pe), np.stack(se), qg, t_fit, t_pred)
    save_results(results)


# ---------------------------------------------------------------- bootstrap
def stage_bootstrap():
    """Paired scenario-level bootstrap CIs for normalized-error differences."""
    results = load_results()
    cells = {}       # method -> list of scenario-level normalized RMSE
    for seed in SEEDS:
        D = torch.load(os.path.join(OUT, f"data_s{seed}.pt"), weights_only=False)
        F = torch.load(os.path.join(OUT, f"fits_s{seed}.pt"), weights_only=False)
        te_mu, te_Xs = D["te_mu"], D["te_X"]
        node_e, glob_e, A_e, flat_e = D["node_e"], D["glob_e"], D["A_e"], D["flat_e"]
        truth_sd = te_mu.reshape(-1, M_OBJ).std(0)
        preds = {}
        pe = []
        for i in range(N_TEST):
            m, _, _ = g2m_predict(F["fit_full"], node_e[i:i+1], glob_e[i:i+1],
                                  A_e[i:i+1], te_Xs[i])
            pe.append(m[0])
        preds["G2M-SK"] = np.stack(pe)
        pe = []
        for i in range(N_TEST):
            m, _, _ = g2m_predict(F["fitf_full"], node_e[i:i+1], glob_e[i:i+1],
                                  A_e[i:i+1], te_Xs[i])
            pe.append(m[0])
        preds["NN-SK"] = np.stack(pe)
        pe = []
        for i in range(N_TEST):
            c0 = np.repeat(flat_e[i:i+1], te_Xs.shape[1], 0)
            m, _ = nn_predict(F["fitn"], c0, te_Xs[i]); pe.append(m)
        preds["NN"] = np.stack(pe)
        muY, stdY = F["sk_scale"]
        pe = []
        for i in range(N_TEST):
            Xj = np.concatenate([np.repeat(flat_e[i:i+1], te_Xs.shape[1], 0),
                                 te_Xs[i]], 1)
            m, _ = F["sk_full"].predict(Xj)
            pe.append(m.numpy() * stdY + muY)
        preds["SK"] = np.stack(pe)
        for name, p in preds.items():
            e2 = (p - te_mu) ** 2                       # [S,Q,m]
            sc_rmse = np.sqrt(e2.mean(1))               # [S,m]
            sc_n = (sc_rmse / truth_sd[None]).mean(1)   # [S]
            cells.setdefault(name, []).append(sc_n)
        print(f"  bootstrap cells seed {seed} done", flush=True)
    for k in cells:
        cells[k] = np.concatenate(cells[k])             # [5*24]
    rng = np.random.default_rng(2026)
    n = len(cells["G2M-SK"]); B = 10000
    out = {}
    for base in ["NN-SK", "NN", "SK"]:
        diff = cells[base] - cells["G2M-SK"]            # >0: G2M-SK better
        bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[base] = dict(mean=float(diff.mean()), lo=float(lo), hi=float(hi),
                         n_cells=n)
        print(f"  G2M-SK vs {base}: mean diff {diff.mean():.3f} "
              f"[{lo:.3f}, {hi:.3f}]", flush=True)
    results["bootstrap"] = out
    save_results(results)


# ------------------------------------------------------------- multimodal
def simulate_mm(sc, X, n_rep, rng, p_regime=0.5, boost=2.4, extra_absent=0.30,
                w_lo=12.0, w_hi=28.0, w_len=8.0):
    """Regime-switching cascade variant of the twin.

    With probability p_regime a replication experiences a secondary cascade:
    an aftershock arrival wave (rate multiplied by `boost`) over a window of
    w_len hours starting at a scenario-specific time, accompanied by an
    absenteeism wave that removes an additional `extra_absent` share of staff
    for the remainder of the horizon. Mixing the two regimes produces
    bimodal replication distributions of the objective vector.
    """
    P = X.shape[0]; B = P * n_rep
    x = np.repeat(X, n_rep, axis=0)
    regime = (rng.uniform(size=B) < p_regime)
    t_on = rng.uniform(w_lo, w_hi)          # scenario-specific cascade onset

    cap = BASE_CAP * sc["cap_factor"]; cap = np.tile(cap, (B, 1))
    ward_pool = 0.20 * cap[:, 6:14].sum(1) * x[:, 0]
    cap[:, 6:14] -= (ward_pool / 8.0)[:, None]
    crit_short = np.maximum(0.0, BASE_CAP[1:6] - BASE_CAP[1:6] * sc["cap_factor"][1:6])
    w_crit = crit_short + 1e-6; w_crit = w_crit / w_crit.sum()
    cap[:, 1:6] += ward_pool[:, None] * w_crit[None, :]
    staff = BASE_STAFF * sc["staff_factor"]; staff = np.tile(staff, (B, 1))
    mu = 1.0 / BASE_LOS
    ac_mult = float(AC_LOS_MULT @ sc["acuity_mix"])
    lam0 = BASE_ARRIVAL_RATE * sc["surge"]
    n = np.floor(cap * sc["init_occ"][None, :]).astype(float)
    q_ed = np.full(B, 4.0 * sc["severity"] * 10)
    q_unit = np.zeros((B, N_UNITS))
    T = BASE_TRANSFER * sc["edge_factor"]
    p_dis = np.clip(1.0 - T.sum(1), 0.02, 1.0)
    wait_ph = np.zeros(B); admitted = np.zeros(B); diverted = np.zeros(B)
    overflow_h = np.zeros(B); overload_h = np.zeros(B); early_pen = np.zeros(B)
    route18 = (ROUTE_AC.T @ sc["acuity_mix"]); route18 = route18 / route18.sum()
    route = np.concatenate([[0.0], route18])

    for t in range(N_STEPS):
        tt = t * DT
        lam = lam0 * (1.0 + 0.35 * np.sin(2 * np.pi * tt / 24.0))
        lam_b = np.full(B, lam)
        in_win = (tt >= t_on) & (tt < t_on + w_len)
        if in_win:
            lam_b = np.where(regime, lam * boost, lam_b)
        stf = staff.copy()
        if tt >= t_on:
            stf[regime] *= (1.0 - extra_absent)
        arr = rng.poisson(lam_b * DT).astype(float)
        ed_load = (n[:, 0] + q_ed) / (cap[:, 0] + 1e-9)
        div_frac = x[:, 2] * 0.8 * np.clip(ed_load - 0.75, 0, None)
        d_now = np.minimum(arr, rng.binomial(arr.astype(int),
                                             np.clip(div_frac, 0, 0.85)))
        diverted += d_now; q_ed += arr - d_now
        util = (n + q_unit) / (cap + 1e-9)
        w = np.clip(util - 1.0, 0, None) + 1e-6
        w = w / w.sum(1, keepdims=True)
        staff_eff = stf + x[:, 1:2] * 0.20 * stf.sum(1, keepdims=True) * w \
            - x[:, 1:2] * 0.20 * stf * (stf / stf.sum(1, keepdims=True))
        staff_eff = np.clip(staff_eff, 0.3 * BASE_STAFF[None, :], None)
        sf = np.clip(staff_eff / BASE_STAFF[None, :], 0.35, 1.5) ** 0.6
        mu_eff = mu[None, :] * sf / ac_mult
        mu_eff[:, 6:14] *= (1.0 + 0.30 * x[:, 3:4])
        p_srv = 1.0 - np.exp(-mu_eff * DT)
        done = rng.binomial(n.astype(int), np.clip(p_srv, 0, 1)).astype(float)
        n -= done
        early_pen += (x[:, 3] ** 1.5) * done[:, 6:14].sum(1) * 0.60
        ed_done = done[:, 0]
        for v in range(1, N_UNITS):
            q_unit[:, v] += ed_done * route[v]
        for u in range(1, N_UNITS):
            du = done[:, u]
            if du.max() == 0:
                continue
            trans = du * (1.0 - p_dis[u])
            row = T[u] / max(T[u].sum(), 1e-9)
            for v in np.nonzero(T[u])[0]:
                q_unit[:, v] += trans * row[v]
        free = np.clip(cap - n, 0, None)
        adm = np.minimum(q_unit, free)
        q_unit -= adm; n += adm
        free_ed = np.clip(cap[:, 0] - n[:, 0], 0, None)
        a_ed = np.minimum(q_ed, free_ed)
        q_ed -= a_ed; n[:, 0] += a_ed; admitted += a_ed
        wait_ph += (q_ed + q_unit.sum(1)) * DT
        overflow_h += np.clip((n + q_unit) / (cap + 1e-9) - 0.95, 0, None).sum(1) * DT
        workload = (n * (1.0 / sf)) / (staff_eff + 1e-9)
        overload_h += np.clip(workload - 2.2, 0, None).sum(1) * DT

    f1 = wait_ph / np.maximum(admitted, 1.0)
    f2 = diverted + q_ed + q_unit.sum(1)
    f3 = overflow_h
    f4 = overload_h + early_pen
    return np.stack([f1, f2, f3, f4], axis=1).reshape(P, n_rep, 4)


def stage_multimodal(seed=0):
    from scipy.stats import qmc
    results = load_results()
    arrays = dict(np.load(os.path.join(OUT, "arrays.npz")))
    rng = np.random.default_rng(7000 + seed)
    K, NREP, N = 24, 6, 64
    sam = qmc.LatinHypercube(d=4, seed=seed)
    Xd = sam.random(K).astype(np.float32)
    tr = [sample_scenario(rng) for _ in range(N)]
    te = [sample_scenario(rng) for _ in range(6)]
    print("  simulating regime-switching library ...", flush=True)
    Yb, Vb, Yr = [], [], []
    for s in tr:
        Y = simulate_mm(s, Xd, NREP, rng)
        Yr.append(Y); Yb.append(Y.mean(1)); Vb.append(Y.var(1, ddof=1) / NREP)
    Yb, Vb, Yr = (np.stack(Yb).astype(np.float32),
                  np.stack(Vb).astype(np.float32),
                  np.stack(Yr).astype(np.float32))

    def pack(scens):
        node = np.stack([scenario_features(s)[0] for s in scens])
        glob = np.stack([scenario_features(s)[1] for s in scens])
        A = np.stack([scenario_features(s)[2] for s in scens])
        return node, glob, A
    node_t, glob_t, A_t = pack(tr)
    node_e, glob_e, A_e = pack(te)
    gs = np.array([3.0, 1.0, 1.0], np.float32)
    glob_t, glob_e = glob_t / gs, glob_e / gs
    d = dict(node=node_t, glob=glob_t, A=A_t, Ybar=Yb, Vbar=Vb, Yrep=Yr, Xd=Xd)
    print("  fitting G2M-SK on regime-switching twin ...", flush=True)
    # Longer adversarial training: the cascade twin has strongly bimodal
    # replication noise, which needs more critic/generator iterations to fit.
    fit = fit_g2msk(d, encoder="gnn", use_gen=True, seed=seed, gan_iters=2000)
    fit_ng = fit_g2msk(d, encoder="gnn", use_gen=False, seed=seed)

    n_x, n_emp = 8, 200
    w1_gen, w1_mm, w1_gau = [], [], []
    skews, dips = [], []
    kurts_e, kurts_g = [], []
    dens = []
    for i, s in enumerate(te):
        Xq = rng.uniform(0, 1, (n_x, 4)).astype(np.float32)
        Yemp = simulate_mm(s, Xq, n_emp, rng)
        m, sg, S = g2m_predict(fit, node_e[i:i+1], glob_e[i:i+1],
                               A_e[i:i+1], Xq, n_z=200)
        m2, s2, _ = g2m_predict(fit_ng, node_e[i:i+1], glob_e[i:i+1],
                                A_e[i:i+1], Xq)
        for k in range(n_x):
            for j in range(M_OBJ):
                emp = Yemp[k, :, j]; sd = emp.std() + 1e-8
                skews.append(float(skew(emp)))
                gen = S[0, k, :, j]
                from scipy.stats import kurtosis as _k
                kurts_e.append(float(_k(emp))); kurts_g.append(float(_k(gen)))
                w1_gen.append(wasserstein_distance(gen, emp) / sd)
                mm = rng.normal(gen.mean(), gen.std() + 1e-9, 200)
                w1_mm.append(wasserstein_distance(mm, emp) / sd)
                gau = rng.normal(m2[0, k, j], s2[0, k, j], 200)
                w1_gau.append(wasserstein_distance(gau, emp) / sd)
            if i < 2 and k == 0:
                dens.append(dict(emp=Yemp[k], gen=S[0, k]))
    results["multimodal"] = dict(
        gen=[float(np.mean(w1_gen)), float(np.std(w1_gen))],
        mm_gauss=[float(np.mean(w1_mm)), float(np.std(w1_mm))],
        posterior=[float(np.mean(w1_gau)), float(np.std(w1_gau))],
        mean_abs_skew=float(np.mean(np.abs(skews))),
        kurt_emp=float(np.mean(kurts_e)), kurt_gen=float(np.mean(kurts_g)))
    for i, dsx in enumerate(dens):
        arrays[f"mmdens{i}_emp"] = dsx["emp"]; arrays[f"mmdens{i}_gen"] = dsx["gen"]
    print(f"  W1 gen={np.mean(w1_gen):.3f} moment-matched={np.mean(w1_mm):.3f} "
          f"posterior={np.mean(w1_gau):.3f} |skew|={np.mean(np.abs(skews)):.2f}",
          flush=True)
    save_results(results)
    np.savez_compressed(os.path.join(OUT, "arrays.npz"),
                        **{k: np.asarray(v) for k, v in arrays.items()})


# ------------------------------------------------------------------- w1
def stage_w1(seed=0):
    """Recompute Table 9 with three rows: generator samples, Gaussian moment
    matched to the generator, and the Gaussian posterior of the mean surface,
    plus the skewness diagnostics."""
    results = load_results()
    arrays = dict(np.load(os.path.join(OUT, "arrays.npz")))
    D = torch.load(os.path.join(OUT, f"data_s{seed}.pt"), weights_only=False)
    F = torch.load(os.path.join(OUT, f"fits_s{seed}.pt"), weights_only=False)
    AB = torch.load(os.path.join(OUT, "abl_s0.pt"), weights_only=False)
    rng = np.random.default_rng(4242)
    te_scens, te_Xs = D["te_scens"], D["te_X"]
    node_e, glob_e, A_e = D["node_e"], D["glob_e"], D["A_e"]
    n_sc, n_x, n_emp = 6, 8, 200
    w1_gen, w1_mm, w1_msf = [], [], []
    sk_emp, sk_gen = [], []
    for i in range(n_sc):
        Xq = te_Xs[i][:n_x]
        Yemp = simulate(te_scens[i], Xq, n_emp, rng)
        _, _, S = g2m_predict(F["fit_full"], node_e[i:i+1], glob_e[i:i+1],
                              A_e[i:i+1], Xq, n_z=200)
        m2, s2, _ = g2m_predict(AB["fitA2"], node_e[i:i+1], glob_e[i:i+1],
                                A_e[i:i+1], Xq)
        for k in range(n_x):
            for j in range(M_OBJ):
                emp = Yemp[k, :, j]; sd = emp.std() + 1e-8
                gen = S[0, k, :, j]
                sk_emp.append(skew(emp)); sk_gen.append(skew(gen))
                w1_gen.append(wasserstein_distance(gen, emp) / sd)
                mmx = rng.normal(gen.mean(), gen.std() + 1e-9, 200)
                w1_mm.append(wasserstein_distance(mmx, emp) / sd)
                gau = rng.normal(m2[0, k, j], s2[0, k, j], 200)
                w1_msf.append(wasserstein_distance(gau, emp) / sd)
    results["w1"] = dict(
        gen=[float(np.mean(w1_gen)), float(np.std(w1_gen))],
        mm=[float(np.mean(w1_mm)), float(np.std(w1_mm))],
        msf=[float(np.mean(w1_msf)), float(np.std(w1_msf))],
        skew_emp=float(np.mean(np.abs(sk_emp))),
        skew_gen=float(np.mean(np.abs(sk_gen))))
    print(f"  w1 gen={np.mean(w1_gen):.2f} mm={np.mean(w1_mm):.2f} "
          f"msf={np.mean(w1_msf):.2f} |skew|={np.mean(np.abs(sk_emp)):.2f}",
          flush=True)
    save_results(results)


# -------------------------------------------------------------- facevalid
def stage_facevalid():
    results = load_results()
    rng = np.random.default_rng(99)
    base = dict(severity=0.0, cap_factor=np.ones(N_UNITS),
                staff_factor=np.ones(N_UNITS), surge=1.0,
                acuity_mix=BASE_ACUITY_MIX.copy(),
                init_occ=np.full(N_UNITS, 0.75),
                edge_factor=np.ones((N_UNITS, N_UNITS)))
    X0 = np.zeros((1, 4), np.float32)
    Y0 = simulate(base, X0, 300, rng)[0]
    mod = dict(base); mod = {k: (v.copy() if hasattr(v, "copy") else v)
                             for k, v in base.items()}
    mod["severity"] = 0.40; mod["surge"] = 1.55
    mod["staff_factor"] = np.ones(N_UNITS) * 0.85
    mod["init_occ"] = np.full(N_UNITS, 0.90)
    Ym = simulate(mod, X0, 300, rng)[0]
    results["facevalid"] = dict(
        base_board=[float(Y0[:, 0].mean()), float(Y0[:, 0].std())],
        base_unserved=[float(Y0[:, 1].mean()), float(Y0[:, 1].std())],
        mod_board=[float(Ym[:, 0].mean()), float(Ym[:, 0].std())],
        mod_unserved=[float(Ym[:, 1].mean()), float(Ym[:, 1].std())])
    print("  facevalid:", results["facevalid"], flush=True)
    save_results(results)


if __name__ == "__main__":
    import sys
    st = sys.argv[1]
    if st.startswith("baselines"):
        stage_baselines(int(st[9:]))
    elif st == "bootstrap":
        stage_bootstrap()
    elif st == "multimodal":
        stage_multimodal()
    elif st == "w1":
        stage_w1()
    elif st == "facevalid":
        stage_facevalid()
    print(f"EXT STAGE {st} DONE", flush=True)
