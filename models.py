"""Model components for G2M-SK and all baselines (PyTorch, CPU)."""
import math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
DEV = "cpu"
M_OBJ = 4
R_LAT = 2          # latent LMC kernels
H_DIM = 32         # graph embedding size
PSI_DIM = 8        # deep-kernel feature size
Z_DIM = 8
JIT = 1e-4


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class GNNEncoder(nn.Module):
    """Two-round message passing + attention pooling on the hospital graph."""
    def __init__(self, node_dim=5, glob_dim=3, h=H_DIM):
        super().__init__()
        d_in = node_dim + glob_dim
        self.w1n, self.w1s = nn.Linear(d_in, 48), nn.Linear(d_in, 48)
        self.w2n, self.w2s = nn.Linear(48, h), nn.Linear(48, h)
        self.att = nn.Linear(h, 1)

    def forward(self, node, glob, A):
        # node [B,19,5]  glob [B,3]  A [B,19,19]
        g = glob.unsqueeze(1).expand(-1, node.shape[1], -1)
        X = torch.cat([node, g], dim=-1)
        H1 = torch.relu(torch.bmm(A, self.w1n(X)) + self.w1s(X))
        H2 = torch.relu(torch.bmm(A, self.w2n(H1)) + self.w2s(H1))
        a = torch.softmax(self.att(H2), dim=1)                    # [B,19,1]
        return (a * H2).sum(1)                                    # [B,h]


class FlatEncoder(nn.Module):
    """Vector-covariate encoder used by NN-SK and the no-GNN ablation."""
    def __init__(self, d_in=98, h=H_DIM):
        super().__init__()
        self.net = mlp([d_in, 96, 64, h])

    def forward(self, node, glob, A):
        B = node.shape[0]
        flat = torch.cat([node.reshape(B, -1), glob], dim=-1)
        return torch.relu(self.net(flat))


class G2MSK(nn.Module):
    """Graph-structured generative multiobjective ML-SK core.

    Heads map the covariate embedding h(theta) to LMC-SK elements:
      beta(theta) in R^m, L_q(theta) in R^m (q=1..r), gamma_q(theta)>0,
    plus the fast performance-vector head phi_F(theta) in R^{m x K}."""
    def __init__(self, K, encoder="gnn"):
        super().__init__()
        self.K = K
        self.enc = GNNEncoder() if encoder == "gnn" else FlatEncoder()
        self.psi = mlp([4, 32, PSI_DIM])                          # deep kernel
        self.head_beta = mlp([H_DIM, 32, M_OBJ])
        self.head_L = mlp([H_DIM, 32, R_LAT * M_OBJ])
        self.head_gam = mlp([H_DIM, 32, R_LAT])
        self.head_F = mlp([H_DIM, 64, 48, M_OBJ * K])

    def elements(self, node, glob, A):
        h = self.enc(node, glob, A)                               # [B,H]
        beta = self.head_beta(h)                                  # [B,m]
        L = self.head_L(h).view(-1, R_LAT, M_OBJ)                 # [B,r,m]
        gam = torch.nn.functional.softplus(self.head_gam(h)) + 0.05
        return h, beta, L, gam

    def kernel_mats(self, gam, PsiX):
        # PsiX [K,psi]; gam [B,r] -> R [B,r,K,K]
        d2 = torch.cdist(PsiX, PsiX) ** 2                         # [K,K]
        return torch.exp(-gam[:, :, None, None] * d2[None, None])

    def joint_cov(self, L, gam, PsiX):
        R = self.kernel_mats(gam, PsiX)                           # [B,r,K,K]
        Bq = torch.einsum("brm,brn->brmn", L, L)                  # [B,r,m,m]
        # kron(B_q, R_q): Sigma[(j,k),(j',k')] = sum_q Bq[j,j'] Rq[k,k']
        Sig = torch.einsum("brmn,brkl->bmknl", Bq, R)
        B = Sig.shape[0]
        mk = M_OBJ * self.K
        return Sig.reshape(B, mk, mk)

    def nll(self, node, glob, A, Ybar, Sige_diag, PsiX=None):
        """Ybar [B, m*K] standardized sample means; Sige_diag [B, m*K]."""
        if PsiX is None:
            PsiX = self.psi_design
        h, beta, L, gam = self.elements(node, glob, A)
        Sig = self.joint_cov(L, gam, PsiX)
        mk = Sig.shape[1]
        Sig = Sig + torch.diag_embed(Sige_diag) + JIT * torch.eye(mk)
        mu = beta.repeat_interleave(self.K, dim=1)                # [B,mK]
        r = (Ybar - mu).unsqueeze(-1)
        Lc = torch.linalg.cholesky(Sig)
        alpha = torch.cholesky_solve(r, Lc)
        quad = (r * alpha).sum(dim=(1, 2))
        logdet = 2 * torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum(1)
        return 0.5 * (quad + logdet).mean()

    def set_design(self, Xd):
        self.Xd = torch.as_tensor(Xd, dtype=torch.float32)
        with torch.no_grad():
            self.psi_design = self.psi(self.Xd)

    def refresh_design(self):
        self.psi_design = self.psi(self.Xd)

    @torch.no_grad()
    def posterior_F(self, node, glob, A, Ybar, Sige_diag):
        """Kriging posterior mean of F at the design points (training targets
        for phi_F), following the base ML-SK construction."""
        self.refresh_design()
        h, beta, L, gam = self.elements(node, glob, A)
        Sig = self.joint_cov(L, gam, self.psi_design)
        mk = Sig.shape[1]
        S = Sig + torch.diag_embed(Sige_diag) + JIT * torch.eye(mk)
        mu = beta.repeat_interleave(self.K, dim=1)
        r = (Ybar - mu).unsqueeze(-1)
        Lc = torch.linalg.cholesky(S)
        alpha = torch.cholesky_solve(r, Lc)
        return (mu.unsqueeze(-1) + Sig @ alpha).squeeze(-1)       # [B,mK]

    @torch.no_grad()
    def predict(self, node, glob, A, Xq):
        """Online predictor: mean f_hat [B,Q,m] and predictive std [B,Q,m]
        for query decisions Xq [Q,4], using the learned F-head (OSOA mode:
        no simulated samples needed)."""
        self.refresh_design()
        h, beta, L, gam = self.elements(node, glob, A)
        Fh = self.head_F(h)                                       # [B,mK]
        PsiQ = self.psi(torch.as_tensor(Xq, dtype=torch.float32))
        d2 = torch.cdist(PsiQ, self.psi_design) ** 2              # [Q,K]
        kq = torch.exp(-gam[:, :, None, None] * d2[None, None])   # [B,r,Q,K]
        Bq = torch.einsum("brm,brn->brmn", L, L)
        # cross covariance C [B, Q, m, m*K]
        C = torch.einsum("brmn,brqk->bqmnk", Bq, kq)
        Bn, Q = C.shape[0], C.shape[1]
        C = C.reshape(Bn, Q, M_OBJ, M_OBJ * self.K)
        Sig = self.joint_cov(L, gam, self.psi_design)
        mk = Sig.shape[1]
        S = Sig + JIT * torch.eye(mk)
        Lc = torch.linalg.cholesky(S)
        mu = beta.repeat_interleave(self.K, dim=1)
        r = (Fh - mu).unsqueeze(-1)
        alpha = torch.cholesky_solve(r, Lc).squeeze(-1)           # [B,mK]
        mean = beta.unsqueeze(1) + torch.einsum("bqmk,bk->bqm", C, alpha)
        # predictive variance per objective
        Bsum = Bq.sum(1)                                          # [B,m,m]
        prior_var = torch.diagonal(Bsum, dim1=1, dim2=2)          # [B,m]
        Ct = C.reshape(Bn * Q, M_OBJ, mk)
        sol = torch.cholesky_solve(Ct.transpose(1, 2),
                                   Lc.repeat_interleave(Q, 0))    # [BQ,mk,m]
        red = torch.einsum("nmk,nkm->nm", Ct, sol).reshape(Bn, Q, M_OBJ)
        var = torch.clamp(prior_var.unsqueeze(1) - red, min=1e-5)
        return mean, torch.sqrt(var), h


class Generator(nn.Module):
    """Conditional generator producing objective-vector samples around the
    SK anchor prediction (residual parameterization)."""
    def __init__(self):
        super().__init__()
        self.embx = mlp([4, 16, 16])
        self.net = mlp([Z_DIM + 16 + H_DIM + 2 * M_OBJ, 96, 96, M_OBJ])

    def forward(self, z, x, h, anchor, scale):
        e = torch.relu(self.embx(x))
        inp = torch.cat([z, e, h, anchor, scale], dim=-1)
        return anchor + self.net(inp) * (scale + 0.05)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.embx = mlp([4, 16, 16])
        self.net = mlp([M_OBJ + 16 + H_DIM, 96, 96, 1])

    def forward(self, y, x, h):
        e = torch.relu(self.embx(x))
        return self.net(torch.cat([y, e, h], dim=-1)).squeeze(-1)


class NNOnly(nn.Module):
    """Pure neural metamodel baseline on the joint (theta, x) input."""
    def __init__(self, d_cov=98):
        super().__init__()
        self.net = mlp([d_cov + 4, 128, 64, M_OBJ])

    def forward(self, cov, x):
        return self.net(torch.cat([cov, x], dim=-1))


class ClassicalSK:
    """Per-objective stochastic kriging on the joint 102-dim input
    (base-paper baseline). Constant trend, Gaussian kernel, MLE by Adam."""
    def __init__(self):
        self.params = None

    def fit(self, Xjoint, Ybar, Vnoise, iters=None, lr=None):
        X = torch.as_tensor(Xjoint, dtype=torch.float32)
        Y = torch.as_tensor(Ybar, dtype=torch.float32)             # [P,m]
        V = torch.as_tensor(Vnoise, dtype=torch.float32)
        n = X.shape[0]
        if iters is None:
            iters = 60 if n <= 800 else 30
        if lr is None:
            lr = 0.08 if n <= 800 else 0.13
        d2 = torch.cdist(X, X) ** 2
        self.models = []
        for j in range(M_OBJ):
            b = torch.tensor(0.0, requires_grad=True)
            lt = torch.tensor(0.0, requires_grad=True)             # log tau2
            lg = torch.tensor(math.log(0.05), requires_grad=True)  # log gamma
            opt = torch.optim.Adam([b, lt, lg], lr=lr)
            for _ in range(iters):
                opt.zero_grad()
                S = torch.exp(lt) * torch.exp(-torch.exp(lg) * d2) \
                    + torch.diag(V[:, j]) + 1e-4 * torch.eye(n)
                Lc = torch.linalg.cholesky(S)
                r = (Y[:, j] - b).unsqueeze(-1)
                a = torch.cholesky_solve(r, Lc)
                nll = 0.5 * (r * a).sum() + torch.log(torch.diagonal(Lc)).sum()
                nll.backward(); opt.step()
            with torch.no_grad():
                S = torch.exp(lt) * torch.exp(-torch.exp(lg) * d2) \
                    + torch.diag(V[:, j]) + 1e-4 * torch.eye(n)
                Sinv = torch.linalg.inv(S)
                a = Sinv @ (Y[:, j] - b)
            self.models.append((b.detach(), lt.detach(), lg.detach(),
                                Sinv, a, X))
        return self

    @torch.no_grad()
    def predict(self, Xq):
        Xq = torch.as_tensor(Xq, dtype=torch.float32)
        out, std = [], []
        for (b, lt, lg, Sinv, a, X) in self.models:
            k = torch.exp(lt) * torch.exp(-torch.exp(lg)
                                          * torch.cdist(Xq, X) ** 2)
            out.append(b + k @ a)
            v = torch.clamp(torch.exp(lt) - (k * (k @ Sinv)).sum(1), min=1e-6)
            std.append(torch.sqrt(v))
        return torch.stack(out, 1), torch.stack(std, 1)

    def storage_bytes(self):
        n = self.models[0][3].shape[0]
        return M_OBJ * (n * n + n) * 4 + 3 * M_OBJ * 4


def count_params(*mods):
    return sum(p.numel() for m in mods for p in m.parameters())
