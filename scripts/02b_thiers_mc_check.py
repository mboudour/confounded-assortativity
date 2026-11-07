"""Lycee Thiers friendship, gender | class, adjacency operator: long Monte Carlo check of the
permutation means of Q, D and Q/D against the closed forms (300,000 within-block permutations)."""
import numpy as np, pandas as pd
from common import DATA, RES
from ca_core import mixing, perm_within, null_expectation

md = pd.read_csv(f"{DATA}/thiers/metadata_2013.txt", sep="\t", header=None, names=["id", "cls", "sex"])
md = md[md.sex != "Unknown"].reset_index(drop=True)
idx = {a: i for i, a in enumerate(md.id)}; code = lambda s: pd.factorize(s)[0]; x, g = code(md.sex), code(md.cls)
E = pd.read_csv(f"{DATA}/thiers/Friendship-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v"]).drop_duplicates()
E = E[E.u.isin(idx) & E.v.isin(idx) & (E.u != E.v)]
u, v = E.u.map(idx).values, E.v.map(idx).values; w = np.ones(len(u))
rng = np.random.default_rng(987654); B = 300000; Q = np.empty(B); D = np.empty(B)
for b in range(B):
    e = mixing(perm_within(x, g, rng), u, v, w, 2); e /= e.sum(); a, bb = e.sum(1), e.sum(0)
    Q[b], D[b] = np.trace(e) - a @ bb, 1 - a @ bb
mQ, mD = null_expectation(x, g, u, v, w); R = Q / D
out = pd.DataFrame([dict(B=B, EQ_perm=Q.mean(), EQ_exact=mQ, ED_perm=D.mean(), ED_exact=mD,
                         r_perm=R.mean(), r_perm_mc_se=R.std() / np.sqrt(B), r_cf=mQ / mD)])
out.to_csv(f"{RES}/thiers_mc_check.csv", index=False); print(out.round(5).to_string(index=False))
