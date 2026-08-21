"""
Stochastic hospital patient-flow digital twin.

19 units, 6 acuity classes, discrete-time stochastic network simulation
(dt = 0.25 h, horizon 48 h). Vectorized over a batch of runs so that
thousands of replications execute in seconds.

Units (index): 0 ED | 1-3 ICU-A/B/C | 4-5 StepDown | 6-13 Ward-1..8
               14-15 OR/PACU | 16-17 Rehab | 18 DischargeLounge
"""
import numpy as np

N_UNITS = 19
N_ACUITY = 6
DT = 0.25          # hours
HORIZON = 48.0     # hours
N_STEPS = int(HORIZON / DT)

UNIT_NAMES = (["ED"] + [f"ICU-{c}" for c in "ABC"] + ["SDU-1", "SDU-2"] +
              [f"Ward-{i}" for i in range(1, 9)] + ["OR-1", "OR-2",
              "Rehab-1", "Rehab-2", "DLounge"])

# base capacities (beds) and base staffing (nurse FTE per shift)
BASE_CAP = np.array([30, 10, 10, 8, 12, 12, 24, 24, 24, 24, 20, 20, 20, 20,
                     6, 6, 15, 15, 12], dtype=float)
BASE_STAFF = np.array([18, 10, 10, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
                       6, 6, 5, 5, 3], dtype=float)
# base mean length of stay in each unit (hours) for an average-acuity patient
BASE_LOS = np.array([3.0, 46, 46, 44, 26, 26, 30, 30, 30, 30, 28, 28, 28, 28,
                     5, 5, 60, 60, 4], dtype=float)

# acuity -> initial placement probability after ED treatment (rows sum to 1)
# columns map to units 1..18: ICU-A,B,C | SDU1,2 | Ward1..8 | OR1,2 | Rehab1,2 | DLounge
ROUTE_AC = np.array([
    [0.28, 0.28, 0.24, 0.06, 0.06, *([0.005] * 8), 0.02, 0.02, 0.00, 0.00, 0.00],
    [0.16, 0.16, 0.14, 0.14, 0.14, *([0.020] * 8), 0.05, 0.05, 0.00, 0.00, 0.00],
    [0.05, 0.05, 0.04, 0.13, 0.13, *([0.060] * 8), 0.05, 0.05, 0.005, 0.005, 0.01],
    [0.01, 0.01, 0.01, 0.07, 0.07, *([0.093] * 8), 0.03, 0.03, 0.008, 0.008, 0.015],
    [0.00, 0.00, 0.00, 0.03, 0.03, *([0.100] * 8), 0.01, 0.01, 0.030, 0.030, 0.06],
    [0.00, 0.00, 0.00, 0.01, 0.01, *([0.065] * 8), 0.00, 0.00, 0.090, 0.090, 0.28],
])
ROUTE_AC = ROUTE_AC / ROUTE_AC.sum(axis=1, keepdims=True)

# internal transfer topology: on unit departure, prob of transfer to target
# unit (remaining mass = discharge home). Sparse, clinically ordered flows.
BASE_TRANSFER = np.zeros((N_UNITS, N_UNITS))
for icu in (1, 2, 3):
    BASE_TRANSFER[icu, 4] = 0.35; BASE_TRANSFER[icu, 5] = 0.35
    for w in range(6, 14):
        BASE_TRANSFER[icu, w] = 0.02
for sdu in (4, 5):
    for w in range(6, 14):
        BASE_TRANSFER[sdu, w] = 0.09
    BASE_TRANSFER[sdu, 16] = 0.04; BASE_TRANSFER[sdu, 17] = 0.04
for w in range(6, 14):
    BASE_TRANSFER[w, 16] = 0.06; BASE_TRANSFER[w, 17] = 0.06
    BASE_TRANSFER[w, 18] = 0.20
    BASE_TRANSFER[w, 4] = 0.015; BASE_TRANSFER[w, 5] = 0.015  # deterioration
for orx in (14, 15):
    BASE_TRANSFER[orx, 1] = 0.15; BASE_TRANSFER[orx, 2] = 0.15
    BASE_TRANSFER[orx, 4] = 0.15; BASE_TRANSFER[orx, 5] = 0.15
    for w in range(6, 14):
        BASE_TRANSFER[orx, w] = 0.05
for rb in (16, 17):
    BASE_TRANSFER[rb, 18] = 0.30
# ED routes handled separately via ROUTE_AC

BASE_ARRIVAL_RATE = 9.0  # patients per hour into ED (baseline)
BASE_ACUITY_MIX = np.array([0.05, 0.09, 0.16, 0.25, 0.25, 0.20])
# acuity LOS multiplier (higher acuity -> longer stay in downstream units)
AC_LOS_MULT = np.array([1.9, 1.6, 1.3, 1.0, 0.85, 0.7])


def sample_scenario(rng, severity_range=(0.15, 0.60)):
    """Draw one disruption scenario (the covariate theta).

    Returns dict with node features, edge factors, globals, severity."""
    sev = rng.uniform(*severity_range)
    n_hit = rng.integers(2, 6)                      # units disrupted
    hit = rng.choice(np.arange(1, N_UNITS), size=n_hit, replace=False)
    cap_factor = np.ones(N_UNITS)
    staff_factor = np.ones(N_UNITS)
    cap_factor[hit] *= (1.0 - sev * rng.uniform(0.4, 1.0, size=n_hit))
    staff_factor[hit] *= (1.0 - sev * rng.uniform(0.3, 0.9, size=n_hit))
    staff_factor *= (1.0 - 0.25 * sev * rng.uniform(0.3, 1.0))  # system-wide strain
    surge = 1.0 + sev * rng.uniform(0.8, 2.2)       # arrival surge multiplier
    ac_shift = sev * rng.uniform(0.2, 1.0)          # shift toward high acuity
    mix = BASE_ACUITY_MIX.copy()
    mix[:3] += ac_shift * np.array([0.10, 0.08, 0.05])
    mix = mix / mix.sum()
    init_occ = np.clip(rng.uniform(0.55, 0.85) + sev * rng.uniform(0.0, 0.25)
                       + rng.normal(0, 0.05, N_UNITS), 0.3, 1.0)
    # disrupted transfer corridors (e.g., closed elevators / contaminated route)
    edge_factor = np.ones((N_UNITS, N_UNITS))
    n_edges_hit = rng.integers(2, 8)
    src = rng.integers(0, N_UNITS, n_edges_hit)
    dst = rng.integers(0, N_UNITS, n_edges_hit)
    edge_factor[src, dst] *= rng.uniform(0.2, 0.7, n_edges_hit)
    return dict(severity=sev, cap_factor=cap_factor, staff_factor=staff_factor,
                surge=surge, acuity_mix=mix, init_occ=init_occ,
                edge_factor=edge_factor)


def scenario_features(sc):
    """Node-feature matrix [19 x 5] + global vector [3] + adjacency [19 x 19]."""
    inflow_share = np.concatenate([[1.0], ROUTE_AC.T @ sc["acuity_mix"]])  # 19
    sevidx = (sc["acuity_mix"][:3].sum()) * np.ones(N_UNITS)
    node = np.stack([sc["init_occ"], sc["staff_factor"], sc["cap_factor"],
                     inflow_share, sevidx], axis=1)              # [19,5]
    glob = np.array([sc["surge"], sc["severity"],
                     sc["acuity_mix"][:2].sum()])                # [3]
    T = BASE_TRANSFER * sc["edge_factor"]
    A = T + T.T
    A = A + np.eye(N_UNITS)
    d = A.sum(1)
    A_hat = A / np.sqrt(np.outer(d, d))                          # sym-normalized
    return node.astype(np.float32), glob.astype(np.float32), A_hat.astype(np.float32)


def flat_covariate(sc):
    node, glob, _ = scenario_features(sc)
    return np.concatenate([node.reshape(-1), glob])              # 98-dim


def simulate(sc, X, n_rep, rng):
    """Simulate the twin under scenario sc for decision matrix X [P,4],
    n_rep replications each. Returns objectives Y [P, n_rep, 4] (all costs).

    Decision x = (bed reallocation, staff redeployment,
                  diversion aggressiveness, early-discharge intensity), in [0,1]^4.
    """
    P = X.shape[0]
    B = P * n_rep
    x = np.repeat(X, n_rep, axis=0)                              # [B,4]

    cap = BASE_CAP * sc["cap_factor"]                            # [19]
    cap = np.tile(cap, (B, 1))
    # x1: shift up to 20% of ward beds to ICU/SDU (proportional to shortfall)
    ward_pool = 0.20 * cap[:, 6:14].sum(1) * x[:, 0]             # [B]
    cap[:, 6:14] -= (ward_pool / 8.0)[:, None]
    crit_short = np.maximum(0.0, BASE_CAP[1:6] - BASE_CAP[1:6] * sc["cap_factor"][1:6])
    w_crit = crit_short + 1e-6
    w_crit = w_crit / w_crit.sum()
    cap[:, 1:6] += ward_pool[:, None] * w_crit[None, :]

    staff = BASE_STAFF * sc["staff_factor"]
    staff = np.tile(staff, (B, 1))

    mu = 1.0 / BASE_LOS                                          # base rates
    ac_mult = float(AC_LOS_MULT @ sc["acuity_mix"])              # scenario acuity load
    lam0 = BASE_ARRIVAL_RATE * sc["surge"]

    n = np.floor(cap * sc["init_occ"][None, :]).astype(float)    # occupancy [B,19]
    q_ed = np.full(B, 4.0 * sc["severity"] * 10)                 # ED boarding queue
    q_unit = np.zeros((B, N_UNITS))                              # bed-wait queues

    T = BASE_TRANSFER * sc["edge_factor"]                        # [19,19]
    p_dis = np.clip(1.0 - T.sum(1), 0.02, 1.0)                   # discharge prob

    wait_ph = np.zeros(B)      # boarding patient-hours
    admitted = np.zeros(B)
    diverted = np.zeros(B)
    overflow_h = np.zeros(B)
    overload_h = np.zeros(B)
    early_pen = np.zeros(B)

    route18 = (ROUTE_AC.T @ sc["acuity_mix"])                    # [18] placement probs
    route18 = route18 / route18.sum()
    route = np.concatenate([[0.0], route18])                     # index by unit id

    for t in range(N_STEPS):
        tt = t * DT
        lam = lam0 * (1.0 + 0.35 * np.sin(2 * np.pi * tt / 24.0))
        arr = rng.poisson(lam * DT, size=B).astype(float)
        # diversion: if ED load high, divert low-acuity share scaled by x3
        ed_load = (n[:, 0] + q_ed) / (cap[:, 0] + 1e-9)
        div_frac = x[:, 2] * 0.8 * np.clip(ed_load - 0.75, 0, None)
        d_now = np.minimum(arr, rng.binomial(arr.astype(int), np.clip(div_frac, 0, 0.85)))
        diverted += d_now
        q_ed += arr - d_now

        # staff redeployment: shift staff toward most loaded units
        util = (n + q_unit) / (cap + 1e-9)
        w = np.clip(util - 1.0, 0, None) + 1e-6
        w = w / w.sum(1, keepdims=True)
        staff_eff = staff * (1 - 0.25 * x[:, 0:1] * 0) + \
            x[:, 1:2] * 0.20 * staff.sum(1, keepdims=True) * w \
            - x[:, 1:2] * 0.20 * staff * (staff / staff.sum(1, keepdims=True))
        staff_eff = np.clip(staff_eff, 0.3 * BASE_STAFF[None, :], None)
        sf = np.clip(staff_eff / BASE_STAFF[None, :], 0.35, 1.5) ** 0.6

        # service completions (thinning of exponential service)
        mu_eff = mu[None, :] * sf / ac_mult
        mu_eff[:, 6:14] *= (1.0 + 0.30 * x[:, 3:4])              # early discharge
        p_srv = 1.0 - np.exp(-mu_eff * DT)
        done = rng.binomial(n.astype(int), np.clip(p_srv, 0, 1)).astype(float)
        n -= done
        early_pen += (x[:, 3] ** 1.5) * done[:, 6:14].sum(1) * 0.60

        # ED completions route to downstream bed queues
        ed_done = done[:, 0]
        for v in range(1, N_UNITS):
            q_unit[:, v] += ed_done * route[v]
        # internal transfers
        for u in range(1, N_UNITS):
            du = done[:, u]
            if du.max() == 0:
                continue
            trans = du * (1.0 - p_dis[u])
            row = T[u] / max(T[u].sum(), 1e-9)
            for v in np.nonzero(T[u])[0]:
                q_unit[:, v] += trans * row[v]
        # admissions from queues into free beds
        free = np.clip(cap - n, 0, None)
        adm = np.minimum(q_unit, free)
        q_unit -= adm
        n += adm
        # ED admits from its own queue
        free_ed = np.clip(cap[:, 0] - n[:, 0], 0, None)
        a_ed = np.minimum(q_ed, free_ed)
        q_ed -= a_ed
        n[:, 0] += a_ed
        admitted += a_ed

        wait_ph += (q_ed + q_unit.sum(1)) * DT
        overflow_h += np.clip((n + q_unit) / (cap + 1e-9) - 0.95, 0, None).sum(1) * DT
        workload = (n * (1.0 / sf)) / (staff_eff + 1e-9)
        overload_h += np.clip(workload - 2.2, 0, None).sum(1) * DT

    f1 = wait_ph / np.maximum(admitted, 1.0)                     # mean boarding delay (h)
    f2 = diverted + q_ed + q_unit.sum(1)                         # unserved count
    f3 = overflow_h                                              # overflow bed-hours
    f4 = overload_h + early_pen                                  # staff overload index
    Y = np.stack([f1, f2, f3, f4], axis=1).reshape(P, n_rep, 4)
    return Y
