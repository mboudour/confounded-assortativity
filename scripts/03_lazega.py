"""Lazega law firm: age/seniority homophily conditioned on status; office as negative control."""
import numpy as np, pandas as pd, scipy.sparse as sp
from common import DATA, RES
from ca_core import operator_arcsets, borrowed_table, newman, mixing

rng = np.random.default_rng(1988)
at = np.loadtxt(f"{DATA}/lazega/ELattr.dat", dtype=int)   # id status gender office years age practice school
n = len(at)
tert = lambda z: np.digitize(z, np.quantile(z, [1/3, 2/3]), right=True)
att = {"status": at[:, 1] - 1, "gender": at[:, 2] - 1, "office": at[:, 3] - 1,
       "years3": tert(at[:, 4]), "age3": tert(at[:, 5]), "practice": at[:, 6] - 1, "school": at[:, 7] - 1}
cases = [("age3", "status"), ("years3", "status"), ("status", "age3"),
         ("status", "office"), ("practice", "office"), ("gender", "office"), ("age3", "office")]
rows = []
for net, f in [("friendship", "ELfriend"), ("advice", "ELadv")]:
    A = sp.csr_matrix(np.loadtxt(f"{DATA}/lazega/{f}.dat", dtype=float))
    for X, G in cases:
        x, g = att[X], att[G]
        arcs, codes, bd = operator_arcsets(A, g, n)
        u, v, w = arcs["A"]; rG = newman(mixing(g, u, v, w, g.max() + 1))
        for r in borrowed_table(x, g, arcs, codes, rng, 2000):
            rows.append(dict(network=net, X=X, G=G, r_G=rG, n_interior=int((~bd).sum()), n=n, **r))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/lazega.csv", index=False)
print(df.query("stratum=='all'")[["network","X","G","r_G","operator","n_arcs","r_obs","r_null_perm","r_null_exact","p_upper","p_lower","excess","borrowed"]].round(3).to_string(index=False))
