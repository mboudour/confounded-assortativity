"""Lazega: sensitivity of age/tenure borrowing from status to discretisation.
Categorical codings: tertiles (main text), quartiles, fixed-width bands; plus the continuous
attribute with Pearson assortativity, where the borrowed value is the permutation mean."""
import numpy as np, pandas as pd
from common import DATA, RES
from ca_core import mixing, newman, perm_within, closed_form_r

rng = np.random.default_rng(71)
at = np.loadtxt(f"{DATA}/lazega/ELattr.dat", dtype=int)
status = at[:, 1] - 1; years, age = at[:, 4].astype(float), at[:, 5].astype(float)
def q(z, k): return np.digitize(z, np.quantile(z, np.arange(1, k) / k), right=True)
def bands(z, edges): return np.digitize(z, edges)
codings = {
    "age": {"tertiles": q(age, 3), "quartiles": q(age, 4), "10-year bands (<35,35-44,45-54,55+)": bands(age, [35, 45, 55])},
    "years in firm": {"tertiles": q(years, 3), "quartiles": q(years, 4), "5-year bands (<5,5-9,10-19,20+)": bands(years, [5, 10, 20])},
}
def pearson(z, u, v):
    return np.corrcoef(z[u], z[v])[0, 1]
rows = []
for net, f in [("friendship", "ELfriend"), ("advice", "ELadv")]:
    M = np.loadtxt(f"{DATA}/lazega/{f}.dat"); u, v = np.nonzero(M); w = np.ones(len(u))
    for var, cods in codings.items():
        for name, x in cods.items():
            x = pd.factorize(pd.Series(x))[0]; K = x.max() + 1
            r = newman(mixing(x, u, v, w, K))
            nul = np.array([newman(mixing(perm_within(x, status, rng), u, v, w, K)) for _ in range(2000)])
            rc = closed_form_r(x, status, u, v, w)
            rows.append(dict(network=net, X=var, coding=name, K=K, r_obs=r, r_cf=rc, r_perm=nul.mean(),
                             borrowed_cf=rc / r, p_upper=(1 + np.sum(nul >= r)) / 2001))
        z = age if var == "age" else years
        r = pearson(z, u, v)
        nul = np.array([pearson(perm_within(z, status, rng), u, v) for _ in range(2000)])
        rows.append(dict(network=net, X=var, coding="continuous (Pearson)", K=np.nan, r_obs=r, r_cf=np.nan,
                         r_perm=nul.mean(), borrowed_cf=nul.mean() / r, p_upper=(1 + np.sum(nul >= r)) / 2001))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/lazega_sensitivity.csv", index=False)
print(df.round(3).to_string(index=False))
