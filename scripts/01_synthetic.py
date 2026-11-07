"""Synthetic graph model with confounding built in by construction.

Nodes: n = 3000. Conditioning attribute g: uniform over 4 blocks. Tested attribute x (4 values):
with probability eta, x = g; otherwise uniform. Arcs (i, j), i != j, are drawn as independent
Bernoulli variables with
    P_ij = min(1, c * s_i^out * s_j^in * (9 if g_i = g_j else 1) * (1 + lambda if x_i = x_j else 1)),
where c is chosen so that sum_ij c*(...) = n * 10 (mean out-degree 10 before capping).
Homogeneous degrees: s = 1. Heterogeneous degrees: s^out, s^in i.i.d. lognormal(0, 1).
With lambda = 0, x affects tie formation only through g.
This script draws ONE graph per (degrees, eta, lambda) cell and compares the permutation mean
with the closed forms on that graph; 01b_synthetic_replications.py repeats this over many graphs."""
import numpy as np, pandas as pd
from common import RES
from ca_core import mixing, newman, perm_within, closed_form_r

N, KG, KX, DEG = 3000, 4, 4, 10
CELLS = [(h, eta, lam) for h in (False, True) for eta in (0.3, 0.6, 0.9) for lam in (0, 1, 3)]

def generate(hetero, eta, lam, rng, n=N):
    g = rng.integers(KG, size=n)
    x = np.where(rng.random(n) < eta, g % KX, rng.integers(KX, size=n))
    s_out = rng.lognormal(0, 1.0, n) if hetero else np.ones(n)
    s_in = rng.lognormal(0, 1.0, n) if hetero else np.ones(n)
    W = np.outer(s_out, s_in) * np.where(g[:, None] == g[None], 9.0, 1.0) * np.where(x[:, None] == x[None], 1 + lam, 1.0)
    np.fill_diagonal(W, 0); P = W * (n * DEG / W.sum())
    capped = float((P > 1).mean()); P = np.minimum(P, 1)
    u, v = np.nonzero(rng.random((n, n)) < P)
    return g, x, u, v, np.ones(len(u)), capped

def cramer_v2(x, g):
    t = np.zeros((g.max() + 1, x.max() + 1)); np.add.at(t, (g, x), 1)
    E = t.sum(1)[:, None] * t.sum(0)[None] / t.sum()
    return ((t - E) ** 2 / E).sum() / t.sum() / (min(t.shape) - 1)

if __name__ == "__main__":
    rng = np.random.default_rng(11); rows = []
    for hetero, eta, lam in CELLS:
        g, x, u, v, w, capped = generate(hetero, eta, lam, rng)
        rX = newman(mixing(x, u, v, w, KX)); rG = newman(mixing(g, u, v, w, KG))
        nul = np.array([newman(mixing(perm_within(x, g, rng), u, v, w, KX)) for _ in range(300)])
        rows.append(dict(degrees="heterogeneous" if hetero else "homogeneous", eta=eta, lam=lam, arcs=len(u),
                         capped_share=capped, r_G=rG, V2=cramer_v2(x, g), r_obs=rX, r_perm=nul.mean(), sd_null=nul.std(),
                         r_cf=closed_form_r(x, g, u, v, w, True), r_cf_leading=closed_form_r(x, g, u, v, w, False),
                         rG_V2=rG * cramer_v2(x, g)))
        rows[-1]["borrowed"] = rows[-1]["r_cf"] / rX
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/synthetic.csv", index=False)
    print(df.round(4).to_string(index=False))
