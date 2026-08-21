"""End-to-end experiments for the G2M-SK paper. Writes out/results.json and
out/arrays.npz consumed by figures.py and the manuscript builder."""
import json, os, time
import numpy as np
import torch
from scipy.stats import qmc, wasserstein_distance
import twin
from twin import sample_scenario, scenario_features, flat_covariate, simulate
from train import (fit_g2msk, g2m_predict, conformal_calibrate,
                   fit_nn_ensemble, nn_predict)
from models import ClassicalSK, count_params, M_OBJ

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

K = 24; NREP = 6
N_LEVELS = [16, 32, 64]
SEEDS = [0, 1, 2]
N_CAL, N_TEST = 24, 24
Q_CAL, R_CAL = 30, 30
Q_TEST, R_TEST = 60, 50
ALPHA = 0.10


def pack_scen(scens):
    node = np.stack([scenario_features(s)[0] for s in scens])
    glob = np.stack([scenario_features(s)[1] for s in scens])
    A = np.stack([scenario_features(s)[2] for s in scens])
    flat = np.stack([flat_covariate(s) for s in scens]).astype(np.float32)
    return node, glob, A, flat


def sim_at_design(scens, Xd, nrep, rng):
    Yb, Vb, Yr = [], [], []
    for s in scens:
        Y = simulate(s, Xd, nrep, rng)              # [K,nrep,m]
        Yr.append(Y)
        Yb.append(Y.mean(1))
        Vb.append(Y.var(1, ddof=1) / nrep)
    return (np.stack(Yb).astype(np.float32),
            np.stack(Vb).astype(np.float32),
            np.stack(Yr).astype(np.float32))


def truth_at(scens, Xlist, reps, rng, return_reps=False):
    mus, rep_out = [], []
    for s, Xq in zip(scens, Xlist):
        Y = simulate(s, Xq, reps, rng)
        mus.append(Y.mean(1))
        if return_reps:
            rep_out.append(Y)
    mus = np.stack(mus).astype(np.float32)
    return (mus, np.stack(rep_out)) if return_reps else mus


def nondominated(F):
    n = F.shape[0]
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            keep[i] = False
    return np.where(keep)[0]


def hypervolume_mc(front, ref, lo, n_mc=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    pts = rng.uniform(lo, ref, size=(n_mc, M_OBJ))
    dom = np.zeros(n_mc, bool)
    for f in front:
        dom |= np.all(pts >= f, axis=1)
    return dom.mean() * np.prod(ref - lo)


def stage_data(seed):
    print(f"===== SEED {seed} data =====", flush=True)
    import torch as _t
    rng = np.random.default_rng(1000 + seed)
    sam = qmc.LatinHypercube(d=4, seed=seed)
    Xd = sam.random(K).astype(np.float32)
    tr_scens = [sample_scenario(rng) for _ in range(max(N_LEVELS))]
    ca_scens = [sample_scenario(rng) for _ in range(N_CAL)]
    te_scens = [sample_scenario(rng) for _ in range(N_TEST)]
    t0 = time.time()
    Yb, Vb, Yr = sim_at_design(tr_scens, Xd, NREP, rng)
    ca_X = [rng.uniform(0, 1, (Q_CAL, 4)) for _ in range(N_CAL)]
    ca_mu = truth_at(ca_scens, ca_X, R_CAL, rng)
    te_X = [rng.uniform(0, 1, (Q_TEST, 4)) for _ in range(N_TEST)]
    te_mu = truth_at(te_scens, te_X, R_TEST, rng)
    print(f"  sims done in {time.time()-t0:.0f}s", flush=True)
    node_t, glob_t, A_t, flat_t = pack_scen(tr_scens)
    node_c, glob_c, A_c, flat_c = pack_scen(ca_scens)
    node_e, glob_e, A_e, flat_e = pack_scen(te_scens)
    glob_scale = np.array([3.0, 1.0, 1.0], np.float32)
    glob_t, glob_c, glob_e = glob_t/glob_scale, glob_c/glob_scale, glob_e/glob_scale
    fmu, fsd = flat_t.mean(0), np.maximum(flat_t.std(0), 0.05)
    flat_t, flat_c, flat_e = [(f - fmu)/fsd for f in (flat_t, flat_c, flat_e)]
    D = dict(rng=rng, Xd=Xd, tr_scens=tr_scens, ca_scens=ca_scens,
             te_scens=te_scens, Yb=Yb, Vb=Vb, Yr=Yr,
             ca_X=np.stack(ca_X), ca_mu=ca_mu, te_X=np.stack(te_X),
             te_mu=te_mu, node_t=node_t, glob_t=glob_t, A_t=A_t,
             flat_t=flat_t, node_c=node_c, glob_c=glob_c, A_c=A_c,
             flat_c=flat_c, node_e=node_e, glob_e=glob_e, A_e=A_e,
             flat_e=flat_e, glob_scale=glob_scale)
    _t.save(D, os.path.join(OUT, f"data_s{seed}.pt"))


def stage_fit(seed, N, results, arrays):
    import torch as _t
    D = _t.load(os.path.join(OUT, f"data_s{seed}.pt"), weights_only=False)
    (Xd, Yb, Vb, Yr) = (D["Xd"], D["Yb"], D["Vb"], D["Yr"])
    node_t, glob_t, A_t, flat_t = D["node_t"], D["glob_t"], D["A_t"], D["flat_t"]
    node_c, glob_c, A_c = D["node_c"], D["glob_c"], D["A_c"]
    node_e, glob_e, A_e, flat_e = D["node_e"], D["glob_e"], D["A_e"], D["flat_e"]
    ca_Xs, ca_mu, te_Xs, te_mu = D["ca_X"], D["ca_mu"], D["te_X"], D["te_mu"]
    truth_sd = te_mu.reshape(-1, M_OBJ).std(0)

    def eval_rmse(pred):
        e = pred - te_mu
        rmse = np.sqrt((e ** 2).mean(axis=(0, 1)))
        return rmse, float(np.mean(rmse / truth_sd))

    def cov_width(pred, sig, q):
        lo = pred - q[None, None] * sig; hi = pred + q[None, None] * sig
        cov = ((te_mu >= lo) & (te_mu <= hi)).mean(axis=(0, 1))
        wid = (hi - lo).mean(axis=(0, 1)) / truth_sd
        return cov, wid

    d = dict(node=node_t[:N], glob=glob_t[:N], A=A_t[:N],
             Ybar=Yb[:N], Vbar=Vb[:N], Yrep=Yr[:N], Xd=Xd)
    # ---- G2M-SK full ----
    t0 = time.time()
    fit = fit_g2msk(d, encoder="gnn", use_gen=True, seed=seed)
    t_fit = time.time() - t0
    pc, sc_ = [], []
    for i in range(N_CAL):
        m, s, _ = g2m_predict(fit, node_c[i:i+1], glob_c[i:i+1],
                              A_c[i:i+1], ca_Xs[i])
        pc.append(m[0]); sc_.append(s[0])
    qhat = conformal_calibrate(np.stack(pc), np.stack(sc_), ca_mu, ALPHA)
    pe, se, Samp_e = [], [], []
    for i in range(N_TEST):
        m, s, S = g2m_predict(fit, node_e[i:i+1], glob_e[i:i+1],
                              A_e[i:i+1], te_Xs[i])
        pe.append(m[0]); se.append(s[0]); Samp_e.append(S[0])
    pe, se = np.stack(pe), np.stack(se)
    rmse, nrm = eval_rmse(pe)
    results["rmse"].setdefault("G2M-SK", {}).setdefault(str(N), {})[str(seed)] = rmse.tolist()
    results["nrmse"].setdefault("G2M-SK", {}).setdefault(str(N), {})[str(seed)] = nrm
    results["train_time"].setdefault("G2M-SK", {}).setdefault(str(N), {})[str(seed)] = t_fit
    if N == max(N_LEVELS):
        cov, wid = cov_width(pe, se, qhat)
        results["uq"].setdefault("G2M-SK (conformal)", {})[str(seed)] = \
            dict(cov=cov.tolist(), wid=wid.tolist())
        cov3, wid3 = cov_width(pe, se, np.full(M_OBJ, 1.645))
        results["uq"].setdefault("G2M-SK w/o conformal", {})[str(seed)] = \
            dict(cov=cov3.tolist(), wid=wid3.tolist())
        arrays[f"qhat_s{seed}"] = qhat

    print('    g2m done', flush=True)
    # ---- NN-SK ----
    t0 = time.time()
    fitf = fit_g2msk(d, encoder="flat", use_gen=False, seed=seed)
    t_f = time.time() - t0
    pef, sef = [], []
    for i in range(N_TEST):
        m, s, _ = g2m_predict(fitf, node_e[i:i+1], glob_e[i:i+1],
                              A_e[i:i+1], te_Xs[i])
        pef.append(m[0]); sef.append(s[0])
    pef, sef = np.stack(pef), np.stack(sef)
    rmse, nrm = eval_rmse(pef)
    results["rmse"].setdefault("NN-SK", {}).setdefault(str(N), {})[str(seed)] = rmse.tolist()
    results["nrmse"].setdefault("NN-SK", {}).setdefault(str(N), {})[str(seed)] = nrm
    results["train_time"].setdefault("NN-SK", {}).setdefault(str(N), {})[str(seed)] = t_f
    if N == max(N_LEVELS):
        cov, wid = cov_width(pef, sef, np.full(M_OBJ, 1.645))
        results["uq"].setdefault("NN-SK (Gaussian)", {})[str(seed)] = \
            dict(cov=cov.tolist(), wid=wid.tolist())

    print('    nnsk done', flush=True)
    # ---- NN ensemble ----
    covr = np.repeat(flat_t[:N, None, None], K, 1)
    covr = np.repeat(covr, NREP, 2).reshape(-1, flat_t.shape[1])
    Xr = np.tile(np.repeat(Xd[None], N, 0)[:, :, None], (1, 1, NREP, 1))
    Xr = Xr.reshape(-1, 4)
    Yflat = Yr[:N].reshape(-1, M_OBJ)
    t0 = time.time()
    fitn = fit_nn_ensemble(covr, Xr, Yflat, seed=seed)
    t_n = time.time() - t0
    pen, sen = [], []
    for i in range(N_TEST):
        c0 = np.repeat(flat_e[i:i+1], te_Xs.shape[1], 0)
        m, s = nn_predict(fitn, c0, te_Xs[i])
        pen.append(m); sen.append(s)
    pen, sen = np.stack(pen), np.stack(sen)
    rmse, nrm = eval_rmse(pen)
    results["rmse"].setdefault("NN", {}).setdefault(str(N), {})[str(seed)] = rmse.tolist()
    results["nrmse"].setdefault("NN", {}).setdefault(str(N), {})[str(seed)] = nrm
    results["train_time"].setdefault("NN", {}).setdefault(str(N), {})[str(seed)] = t_n
    if N == max(N_LEVELS):
        cov, wid = cov_width(pen, sen, np.full(M_OBJ, 1.645))
        results["uq"].setdefault("NN ensemble", {})[str(seed)] = \
            dict(cov=cov.tolist(), wid=wid.tolist())

    print('    nn done', flush=True)
    # ---- classical SK ----
    Xj = np.concatenate([np.repeat(flat_t[:N], K, 0), np.tile(Xd, (N, 1))], 1)
    Yj = Yb[:N].reshape(-1, M_OBJ)
    stdY = Yj.std(0) + 1e-8; muY = Yj.mean(0)
    Vj = Vb[:N].reshape(-1, M_OBJ) / stdY ** 2
    t0 = time.time()
    sk = ClassicalSK().fit(Xj, (Yj - muY) / stdY, Vj)
    t_s = time.time() - t0
    pes = []
    for i in range(N_TEST):
        Xq = np.concatenate([np.repeat(flat_e[i:i+1], te_Xs.shape[1], 0),
                             te_Xs[i]], 1)
        m, s = sk.predict(Xq)
        pes.append(m.numpy() * stdY + muY)
    pes = np.stack(pes)
    rmse, nrm = eval_rmse(pes)
    results["rmse"].setdefault("SK", {}).setdefault(str(N), {})[str(seed)] = rmse.tolist()
    results["nrmse"].setdefault("SK", {}).setdefault(str(N), {})[str(seed)] = nrm
    results["train_time"].setdefault("SK", {}).setdefault(str(N), {})[str(seed)] = t_s
    print(f"  seed{seed} N={N}  nrmse: G2M={results['nrmse']['G2M-SK'][str(N)][str(seed)]:.3f} "
          f"NN-SK={results['nrmse']['NN-SK'][str(N)][str(seed)]:.3f} "
          f"NN={results['nrmse']['NN'][str(N)][str(seed)]:.3f} "
          f"SK={results['nrmse']['SK'][str(N)][str(seed)]:.3f}", flush=True)
    if N == max(N_LEVELS):
        st = dict(fit_full=fit, fitf_full=fitf, fitn=fitn, sk_full=sk,
                  sk_scale=(muY, stdY), pe_full=pe, se_full=se,
                  Samp_full=np.stack(Samp_e), qhat=qhat)
        _t.save(st, os.path.join(OUT, f"fits_s{seed}.pt"))


def stage_abl(seed, results, arrays):
    import torch as _t
    D = _t.load(os.path.join(OUT, f"data_s{seed}.pt"), weights_only=False)
    F = _t.load(os.path.join(OUT, f"fits_s{seed}.pt"), weights_only=False)
    Xd, Yb, Vb, Yr = D["Xd"], D["Yb"], D["Vb"], D["Yr"]
    node_t, glob_t, A_t = D["node_t"], D["glob_t"], D["A_t"]
    node_c, glob_c, A_c = D["node_c"], D["glob_c"], D["A_c"]
    node_e, glob_e, A_e = D["node_e"], D["glob_e"], D["A_e"]
    ca_Xs, ca_mu, te_Xs, te_mu = D["ca_X"], D["ca_mu"], D["te_X"], D["te_mu"]
    truth_sd = te_mu.reshape(-1, M_OBJ).std(0)
    d = dict(node=node_t, glob=glob_t, A=A_t, Ybar=Yb, Vbar=Vb, Yrep=Yr, Xd=Xd)

    def cov_width(pred, sig, q):
        lo = pred - q[None, None] * sig; hi = pred + q[None, None] * sig
        cov = ((te_mu >= lo) & (te_mu <= hi)).mean(axis=(0, 1))
        wid = (hi - lo).mean(axis=(0, 1)) / truth_sd
        return cov, wid

    def eval_fit(fitX):
        pc, sc_ = [], []
        for i in range(N_CAL):
            m, s, _ = g2m_predict(fitX, node_c[i:i+1], glob_c[i:i+1],
                                  A_c[i:i+1], ca_Xs[i])
            pc.append(m[0]); sc_.append(s[0])
        q = conformal_calibrate(np.stack(pc), np.stack(sc_), ca_mu, ALPHA)
        pe, se = [], []
        for i in range(N_TEST):
            m, s, _ = g2m_predict(fitX, node_e[i:i+1], glob_e[i:i+1],
                                  A_e[i:i+1], te_Xs[i])
            pe.append(m[0]); se.append(s[0])
        pe, se = np.stack(pe), np.stack(se)
        e = pe - te_mu
        rmse = np.sqrt((e ** 2).mean(axis=(0, 1)))
        nrm = float(np.mean(rmse / truth_sd))
        cov, wid = cov_width(pe, se, q)
        return rmse, nrm, cov, wid

    fitA1 = fit_g2msk(d, encoder="flat", use_gen=True, seed=seed)
    rmse, nrm, cov, wid = eval_fit(fitA1)
    results["ablation"].setdefault("w/o GNN", {})[str(seed)] = dict(
        rmse=rmse.tolist(), nrmse=nrm, cov=cov.tolist(), wid=wid.tolist())
    fitA2 = fit_g2msk(d, encoder="gnn", use_gen=False, seed=seed)
    rmse, nrm, cov, wid = eval_fit(fitA2)
    results["ablation"].setdefault("w/o generator", {})[str(seed)] = dict(
        rmse=rmse.tolist(), nrmse=nrm, cov=cov.tolist(), wid=wid.tolist())
    covF, widF = cov_width(F["pe_full"], F["se_full"], F["qhat"])
    Nmax = str(max(N_LEVELS))
    results["ablation"].setdefault("full", {})[str(seed)] = dict(
        rmse=results["rmse"]["G2M-SK"][Nmax][str(seed)],
        nrmse=results["nrmse"]["G2M-SK"][Nmax][str(seed)],
        cov=covF.tolist(), wid=widF.tolist())
    a3 = results["uq"]["G2M-SK w/o conformal"][str(seed)]
    results["ablation"].setdefault("w/o conformal", {})[str(seed)] = dict(
        rmse=results["rmse"]["G2M-SK"][Nmax][str(seed)],
        nrmse=results["nrmse"]["G2M-SK"][Nmax][str(seed)],
        cov=a3["cov"], wid=a3["wid"])
    if seed == 0:
        _t.save(dict(fitA2=fitA2), os.path.join(OUT, "abl_s0.pt"))
    print(f"  seed{seed} ablations done", flush=True)


def extras_a(results, arrays, rng, fit_full, fitA2, fitf_full,
             sk_full, sk_scale, fitn, node_e, glob_e, A_e, flat_e,
             te_scens, te_Xs, te_mu, pe_full, se_full, Samp_full,
             Xd, glob_scale, seed):
    # ---------- distributional fit (Wasserstein-1) ----------
    print("  W1 distributional evaluation ...", flush=True)
    n_sc, n_x, n_emp = min(6, len(te_scens)), 8, 200
    w1_gen, w1_gau = [], []
    dens_store = []
    for i in range(n_sc):
        Xq = te_Xs[i][:n_x]
        Yemp = simulate(te_scens[i], Xq, n_emp, rng)      # [n_x,n_emp,m]
        m, s, S = g2m_predict(fit_full, node_e[i:i+1], glob_e[i:i+1],
                              A_e[i:i+1], Xq, n_z=200)
        m2, s2, _ = g2m_predict(fitA2, node_e[i:i+1], glob_e[i:i+1],
                                A_e[i:i+1], Xq)
        for k in range(n_x):
            for j in range(M_OBJ):
                emp = Yemp[k, :, j]
                sd = emp.std() + 1e-8
                w1_gen.append(wasserstein_distance(S[0, k, :, j], emp) / sd)
                gau = rng.normal(m2[0, k, j], s2[0, k, j], 200)
                w1_gau.append(wasserstein_distance(gau, emp) / sd)
            if i < 2 and k == 0:
                dens_store.append(dict(emp=Yemp[k], gen=S[0, k],
                                       gau_m=m2[0, k], gau_s=s2[0, k]))
    results["w1"] = dict(gen=[float(np.mean(w1_gen)), float(np.std(w1_gen))],
                         gauss=[float(np.mean(w1_gau)), float(np.std(w1_gau))])
    for i, dsx in enumerate(dens_store):
        arrays[f"dens{i}_emp"] = dsx["emp"]; arrays[f"dens{i}_gen"] = dsx["gen"]
        arrays[f"dens{i}_gm"] = dsx["gau_m"]; arrays[f"dens{i}_gs"] = dsx["gau_s"]

    # ---------- Pareto experiment ----------
    print("  Pareto evaluation ...", flush=True)
    n_ps, n_cand = min(8, len(te_scens)), 150
    hv, gd = {mth: [] for mth in ["G2M-SK", "NN-SK", "NN", "SK"]}, \
             {mth: [] for mth in ["G2M-SK", "NN-SK", "NN", "SK"]}
    frontier_sizes = []
    for i in range(n_ps):
        Xc = rng.uniform(0, 1, (n_cand, 4)).astype(np.float32)
        mu_true = simulate(te_scens[i], Xc, 50, rng).mean(1)
        lo = mu_true.min(0); ref = mu_true.max(0) * 1.05 + 1e-6
        tf_idx = nondominated(mu_true)
        hv_true = hypervolume_mc(mu_true[tf_idx], ref, lo, rng=rng)
        preds = {}
        m, s, _ = g2m_predict(fit_full, node_e[i:i+1], glob_e[i:i+1],
                              A_e[i:i+1], Xc)
        preds["G2M-SK"] = (m[0], s[0])
        mf, sf, _ = g2m_predict(fitf_full, node_e[i:i+1], glob_e[i:i+1],
                                A_e[i:i+1], Xc)
        preds["NN-SK"] = (mf[0], sf[0])
        covr = np.repeat(flat_e[i:i+1], n_cand, 0)
        mn, sn = nn_predict(fitn, covr, Xc)
        preds["NN"] = (mn, sn)
        Xj = np.concatenate([covr, Xc], 1)
        ms, ss = sk_full.predict(Xj)
        muY, stdY = sk_scale
        preds["SK"] = (ms.numpy() * stdY + muY, ss.numpy() * stdY)
        for mth, (pm, psig) in preds.items():
            sel = nondominated(pm)
            hv_sel = hypervolume_mc(mu_true[sel], ref, lo, rng=rng)
            hv[mth].append(float(hv_sel / hv_true))
            tv = mu_true[sel][:, None] - mu_true[tf_idx][None]
            gdist = np.sqrt((tv ** 2).sum(-1)).min(1).mean()
            gd[mth].append(float(gdist / np.linalg.norm(ref - lo)))
        if i == 0:
            arrays["par_mu"] = mu_true; arrays["par_true_idx"] = tf_idx
            arrays["par_sel_g2m"] = nondominated(preds["G2M-SK"][0])
            arrays["par_sel_sk"] = nondominated(preds["SK"][0])
            arrays["par_pred_g2m"] = preds["G2M-SK"][0]
            arrays["par_sig_g2m"] = preds["G2M-SK"][1] * arrays[f"qhat_s{seed}"]
    results["pareto"] = dict(
        hv={k: [float(np.mean(v)), float(np.std(v))] for k, v in hv.items()},
        gd={k: [float(np.mean(v)), float(np.std(v))] for k, v in gd.items()})

    return


def extras_b(results, arrays, rng, fit_full, fitA2, fitf_full,
             sk_full, sk_scale, fitn, node_e, glob_e, A_e, flat_e,
             te_scens, te_Xs, te_mu, pe_full, se_full, Samp_full,
             Xd, glob_scale, seed):
    # ---------- covariate drift ----------
    print("  drift stress test ...", flush=True)
    sev_ca = [sample_scenario(rng, (0.60, 0.90)) for _ in range(8)]
    sev_te = [sample_scenario(rng, (0.60, 0.90)) for _ in range(16)]
    node_sc, glob_sc, A_sc, _ = pack_scen(sev_ca)
    node_se, glob_se, A_se, _ = pack_scen(sev_te)
    glob_sc, glob_se = glob_sc / glob_scale, glob_se / glob_scale
    Xsc = [rng.uniform(0, 1, (20, 4)) for _ in range(8)]
    mu_sc = truth_at(sev_ca, Xsc, 30, rng)
    Xse = [rng.uniform(0, 1, (40, 4)) for _ in range(16)]
    mu_se = truth_at(sev_te, Xse, 40, rng)
    Xsc_s, Xse_s = np.stack(Xsc), np.stack(Xse)
    pS, sS = [], []
    for i in range(16):
        m, s, _ = g2m_predict(fit_full, node_se[i:i+1], glob_se[i:i+1],
                              A_se[i:i+1], Xse_s[i])
        pS.append(m[0]); sS.append(s[0])
    pS, sS = np.stack(pS), np.stack(sS)
    q0 = arrays[f"qhat_s{seed}"]
    lo, hi = pS - q0 * sS, pS + q0 * sS
    cov_sh = ((mu_se >= lo) & (mu_se <= hi)).mean(axis=(0, 1))
    pC, sC = [], []
    for i in range(8):
        m, s, _ = g2m_predict(fit_full, node_sc[i:i+1], glob_sc[i:i+1],
                              A_sc[i:i+1], Xsc_s[i])
        pC.append(m[0]); sC.append(s[0])
    q_re = conformal_calibrate(np.stack(pC), np.stack(sC), mu_sc, ALPHA)
    lo, hi = pS - q_re * sS, pS + q_re * sS
    cov_re = ((mu_se >= lo) & (mu_se <= hi)).mean(axis=(0, 1))
    wid_sh = (2 * q0 * sS).mean(axis=(0, 1))
    wid_re = (2 * q_re * sS).mean(axis=(0, 1))
    in_cov = results["uq"]["G2M-SK (conformal)"][str(seed)]["cov"]
    results["drift"] = dict(in_dist=list(map(float, in_cov)),
                            shifted=cov_sh.tolist(),
                            recalibrated=cov_re.tolist(),
                            wid_shift=wid_sh.tolist(),
                            wid_recal=wid_re.tolist())

    # ---------- online time and storage ----------
    print("  timing/storage ...", flush=True)
    Xbig = rng.uniform(0, 1, (1000, 4)).astype(np.float32)
    t0 = time.time()
    for _ in range(3):
        g2m_predict(fit_full, node_e[:1], glob_e[:1], A_e[:1], Xbig, n_z=100)
    t_g2m = (time.time() - t0) / 3
    t0 = time.time()
    for _ in range(3):
        m, s, _ = fit_full["model"].predict(
            torch.as_tensor(node_e[:1]), torch.as_tensor(glob_e[:1]),
            torch.as_tensor(A_e[:1]), Xbig)
    t_g2m_mean = (time.time() - t0) / 3
    Xj = np.concatenate([np.repeat(flat_e[:1], 1000, 0), Xbig], 1)
    t0 = time.time()
    for _ in range(3):
        sk_full.predict(Xj)
    t_sk = (time.time() - t0) / 3
    covr = np.repeat(flat_e[:1], 1000, 0)
    t0 = time.time()
    for _ in range(3):
        nn_predict(fitn, covr, Xbig)
    t_nn = (time.time() - t0) / 3
    mods = [fit_full["model"], fit_full["gen"]]
    st_g2m = count_params(*mods) * 4 + Xd.size * 4
    st_sk = sk_full.storage_bytes() + 64 * K * NREP * M_OBJ * 4 \
        + 64 * K * (98 + 4) * 4
    st_nn = count_params(*fitn["nets"]) * 4
    results["timing"] = dict(g2m_full=t_g2m, g2m_mean=t_g2m_mean,
                             sk=t_sk, nn=t_nn)
    results["storage"] = dict(g2m=int(st_g2m), sk=int(st_sk), nn=int(st_nn))
    arrays["te_mu0"] = te_mu; arrays["pe_full0"] = pe_full
    arrays["se_full0"] = se_full


def load_results():
    rp = os.path.join(OUT, "results.json")
    if os.path.exists(rp):
        with open(rp) as f:
            return json.load(f)
    return dict(rmse={}, nrmse={}, uq={}, ablation={}, train_time={})


def save_results(results):
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


def main(stage):
    results = load_results()
    arrays = {}
    ap = os.path.join(OUT, "arrays.npz")
    if os.path.exists(ap):
        arrays.update(dict(np.load(ap)))
    if stage.startswith("data"):
        stage_data(int(stage[4:]))
    elif stage.startswith("fit"):
        seed, N = stage[3:].split("_")
        stage_fit(int(seed), int(N), results, arrays)
    elif stage.startswith("abl"):
        stage_abl(int(stage[3:]), results, arrays)
    elif stage in ("extras_a", "extras_b"):
        import torch as _t
        D = _t.load(os.path.join(OUT, "data_s0.pt"), weights_only=False)
        F = _t.load(os.path.join(OUT, "fits_s0.pt"), weights_only=False)
        AB = _t.load(os.path.join(OUT, "abl_s0.pt"), weights_only=False)
        arrays["qhat_s0"] = F["qhat"]
        args = (results, arrays, D["rng"], F["fit_full"], AB["fitA2"],
                F["fitf_full"], F["sk_full"], F["sk_scale"], F["fitn"],
                D["node_e"], D["glob_e"], D["A_e"], D["flat_e"],
                D["te_scens"], D["te_X"], D["te_mu"], F["pe_full"],
                F["se_full"], F["Samp_full"], D["Xd"], D["glob_scale"], 0)
        if stage == "extras_a":
            extras_a(*args)
        else:
            extras_b(*args)
    else:
        run_seed(int(stage), results, arrays)
    save_results(results)
    np.savez_compressed(ap, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"STAGE {stage} DONE", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
