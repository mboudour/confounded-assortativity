"""Planted model: X affects tie formation only through G (lambda=0) or also directly (lambda>0)."""
import numpy as np, pandas as pd
from common import RES
from ca_core import mixing, newman, perm_within, null_r

rng = np.random.default_rng(11)
n, KG, KX, deg = 3000, 4, 4, 10

def cramer_v2(x, g):
    t = np.zeros((g.max() + 1, x.max() + 1)); np.add.at(t, (g, x), 1)
    E = t.sum(1)[:, None] * t.sum(0)[None] / t.sum()
    return ((t - E) ** 2 / E).sum() / t.sum() / (min(t.shape) - 1)

rows = []
for hetero in (False, True):
    for eta in (0.3, 0.6, 0.9):
        for lam in (0, 1, 3):
            g = rng.integers(KG, size=n)
            x = np.where(rng.random(n) < eta, g % KX, rng.integers(KX, size=n))
            s_out = rng.lognormal(0, 1.0, n) if hetero else np.ones(n)
            s_in = rng.lognormal(0, 1.0, n) if hetero else np.ones(n)
            W = np.outer(s_out, s_in) * np.where(g[:, None] == g[None], 9.0, 1.0) * np.where(x[:, None] == x[None], 1 + lam, 1.0)
            np.fill_diagonal(W, 0); P = np.minimum(W * (n * deg / W.sum()), 1)
            u, v = np.nonzero(rng.random((n, n)) < P); w = np.ones(len(u))
            rX = newman(mixing(x, u, v, w, KX)); rG = newman(mixing(g, u, v, w, KG))
            nul = np.array([newman(mixing(perm_within(x, g, rng), u, v, w, KX)) for _ in range(300)])
            rows.append(dict(degrees="heterogeneous" if hetero else "homogeneous", eta=eta, lam=lam, arcs=len(u),
                             r_G=rG, V2=cramer_v2(x, g), r_obs=rX, r_null_perm=nul.mean(), sd_null=nul.std(),
                             r_null_exact=null_r(x, g, u, v, w, True), r_null_leading=null_r(x, g, u, v, w, False),
                             rG_V2=rG * cramer_v2(x, g)))
            rows[-1]["borrowed"] = rows[-1]["r_null_exact"] / rX
df = pd.DataFrame(rows); df.to_csv(f"{RES}/planted.csv", index=False)
print(df.round(3).to_string(index=False))
