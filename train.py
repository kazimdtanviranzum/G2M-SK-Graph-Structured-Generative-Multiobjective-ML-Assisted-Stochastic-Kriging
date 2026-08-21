"""Offline learning, conformal calibration, and prediction wrappers."""
import numpy as np
import torch
import torch.nn as nn
from models import (G2MSK, Generator, Critic, NNOnly, M_OBJ, Z_DIM,
                    count_params)


class Standardizer:
    def fit(self, Y):                       # Y [..., m]
        f = Y.reshape(-1, M_OBJ)
        self.mu = f.mean(0)
        self.sd = f.std(0) + 1e-8
        return self

    def tf(self, Y):
        return (Y - self.mu) / self.sd

    def inv(self, Y):
        return Y * self.sd + self.mu

    def inv_scale(self, S):
        return S * self.sd


def fit_g2msk(data, encoder="gnn", use_gen=True, epochs_sk=260,
              epochs_F=250, gan_iters=550, seed=0, verbose=False):
    """data: dict with node,glob,A [N,...], Ybar [N,K,m], Vbar [N,K,m]
    (variance of the mean), Yrep [N,K,n,m], Xd [K,4]."""
    torch.manual_seed(seed); np.random.seed(seed)
    K = data["Xd"].shape[0]
    std = Standardizer().fit(data["Yrep"])
    node = torch.as_tensor(data["node"]); glob = torch.as_tensor(data["glob"])
    A = torch.as_tensor(data["A"])
    Yb = torch.as_tensor(std.tf(data["Ybar"]), dtype=torch.float32)
    # noise on standardized scale: var/sd^2, arranged objective-major [N,mK]
    Vs = torch.as_tensor(data["Vbar"] / (std.sd ** 2)[None, None, :],
                         dtype=torch.float32)
    Ybar_v = Yb.permute(0, 2, 1).reshape(Yb.shape[0], -1)          # [N,mK]
    Sige = Vs.permute(0, 2, 1).reshape(Vs.shape[0], -1).clamp(min=1e-5)

    model = G2MSK(K, encoder=encoder)
    model.set_design(data["Xd"])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs_sk):
        opt.zero_grad()
        model.refresh_design()
        loss = model.nll(node, glob, A, Ybar_v, Sige)
        loss.backward(); opt.step()
        if verbose and ep % 50 == 0:
            print(f"  [SK] ep{ep} nll={loss.item():.3f}")

    # F-head distillation on kriging posterior means (base-paper style)
    Ftar = model.posterior_F(node, glob, A, Ybar_v, Sige).detach()
    optF = torch.optim.Adam(model.head_F.parameters(), lr=2e-3)
    for ep in range(epochs_F):
        optF.zero_grad()
        h = model.enc(node, glob, A).detach()
        lossF = ((model.head_F(h) - Ftar) ** 2).mean()
        lossF.backward(); optF.step()

    G = Ck = None
    if use_gen:
        G, Ck = Generator(), Critic()
        og = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
        oc = torch.optim.Adam(Ck.parameters(), lr=1e-3, betas=(0.5, 0.9))
        with torch.no_grad():
            mean_d, std_d, hvec = model.predict(node, glob, A, data["Xd"])
        N, K = mean_d.shape[0], mean_d.shape[1]
        nrep = data["Yrep"].shape[2]
        Yr = torch.as_tensor(std.tf(data["Yrep"]),
                             dtype=torch.float32).reshape(N, K, nrep, M_OBJ)
        idxN = np.arange(N)
        for it in range(gan_iters):
            for _ in range(3):                                    # critic steps
                bi = np.random.choice(idxN, 256); bk = np.random.randint(0, K, 256)
                br = np.random.randint(0, nrep, 256)
                y = Yr[bi, bk, br]
                hb = hvec[bi]; xb = model.Xd[bk]
                an = mean_d[bi, bk]; sc = std_d[bi, bk]
                z = torch.randn(256, Z_DIM)
                yf = G(z, xb, hb, an, sc).detach()
                eps = torch.rand(256, 1)
                ymix = (eps * y + (1 - eps) * yf).requires_grad_(True)
                d_mix = Ck(ymix, xb, hb)
                gp = torch.autograd.grad(d_mix.sum(), ymix,
                                         create_graph=True)[0]
                gp = ((gp.norm(2, dim=1) - 1) ** 2).mean()
                lossC = Ck(yf, xb, hb).mean() - Ck(y, xb, hb).mean() + 10 * gp
                oc.zero_grad(); lossC.backward(); oc.step()
            bi = np.random.choice(idxN, 256); bk = np.random.randint(0, K, 256)
            hb = hvec[bi]; xb = model.Xd[bk]
            an = mean_d[bi, bk]; sc = std_d[bi, bk]
            z = torch.randn(256, Z_DIM)
            yf = G(z, xb, hb, an, sc)
            anchor_pen = ((yf.mean(0) - an.mean(0)) ** 2).mean()
            lossG = -Ck(yf, xb, hb).mean() + 2.0 * anchor_pen
            og.zero_grad(); lossG.backward(); og.step()
    return dict(model=model, gen=G, critic=Ck, std=std)


@torch.no_grad()
def g2m_predict(fit, node, glob, A, Xq, n_z=200):
    """Returns mean [B,Q,m], sigma_hat [B,Q,m] (natural scale) and, if the
    generator is present, predictive samples [B,Q,n_z,m]."""
    model, std = fit["model"], fit["std"]
    node = torch.as_tensor(node); glob = torch.as_tensor(glob)
    A = torch.as_tensor(A)
    mean, sk_std, h = model.predict(node, glob, A, Xq)
    Bn, Q = mean.shape[0], mean.shape[1]
    if fit["gen"] is None:
        return (std.inv(mean.numpy()), std.inv_scale(sk_std.numpy()), None)
    G = fit["gen"]
    Xq_t = torch.as_tensor(Xq, dtype=torch.float32)
    xb = Xq_t.unsqueeze(0).expand(Bn, -1, -1).reshape(Bn * Q, 4)
    hb = h.unsqueeze(1).expand(-1, Q, -1).reshape(Bn * Q, -1)
    an = mean.reshape(Bn * Q, M_OBJ); sc = sk_std.reshape(Bn * Q, M_OBJ)
    samp = []
    for _ in range(n_z // 50):
        z = torch.randn(Bn * Q, 50, Z_DIM)
        yb = G(z.reshape(-1, Z_DIM),
               xb.repeat_interleave(50, 0), hb.repeat_interleave(50, 0),
               an.repeat_interleave(50, 0), sc.repeat_interleave(50, 0))
        samp.append(yb.reshape(Bn * Q, 50, M_OBJ))
    S = torch.cat(samp, 1)                                        # [BQ,nz,m]
    g_mean = S.mean(1); g_std = S.std(1)
    sig = torch.sqrt(g_std ** 2 + 0.25 * sk_std.reshape(Bn * Q, M_OBJ) ** 2)
    return (std.inv(g_mean.reshape(Bn, Q, M_OBJ).numpy()),
            std.inv_scale(sig.reshape(Bn, Q, M_OBJ).numpy()),
            std.inv(S.reshape(Bn, Q, -1, M_OBJ).numpy()))


def conformal_calibrate(pred_mean, sigma, truth, alpha=0.10):
    """Normalized split-conformal quantile per objective.
    pred_mean, sigma, truth: [Ncal, Q, m] arrays."""
    s = np.abs(truth - pred_mean) / (sigma + 1e-8)
    q = np.zeros(M_OBJ)
    for j in range(M_OBJ):
        sj = np.sort(s[..., j].ravel())
        n = len(sj)
        k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
        q[j] = sj[k]
    return q


def fit_nn_ensemble(cov, X, Y, n_models=5, epochs=350, seed=0):
    """NN-only baseline: deep ensemble on joint input, MSE on replications."""
    torch.manual_seed(seed)
    cov = torch.as_tensor(cov, dtype=torch.float32)
    X = torch.as_tensor(X, dtype=torch.float32)
    std = Standardizer().fit(Y)
    Yt = torch.as_tensor(std.tf(Y), dtype=torch.float32)
    nets = []
    n = cov.shape[0]
    for s in range(n_models):
        torch.manual_seed(seed * 100 + s)
        net = NNOnly(cov.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        for ep in range(epochs):
            bi = torch.randint(0, n, (512,))
            opt.zero_grad()
            loss = ((net(cov[bi], X[bi]) - Yt[bi]) ** 2).mean()
            loss.backward(); opt.step()
        nets.append(net)
    return dict(nets=nets, std=std)


@torch.no_grad()
def nn_predict(fit, cov, X):
    cov = torch.as_tensor(cov, dtype=torch.float32)
    X = torch.as_tensor(X, dtype=torch.float32)
    preds = torch.stack([net(cov, X) for net in fit["nets"]])
    m = preds.mean(0).numpy(); s = preds.std(0).numpy() + 1e-3
    return fit["std"].inv(m), fit["std"].inv_scale(s)
