"""Lycee Thiers friendship, gender | class: r_cf = E[Q]/E[D] versus r_perm = E[Q/D] and the
second-order expansion  E[Q/D] ~ mu_Q/mu_D - Cov(Q,D)/mu_D^2 + mu_Q Var(D)/mu_D^3  (20,000 permutations)."""
import numpy as np, pandas as pd, scipy.sparse as sp
from common import DATA, RES
from ca_core import operator_arcsets, mixing, perm_within, null_expectation, STRATA

rng = np.random.default_rng(20000)
md = pd.read_csv(f"{DATA}/thiers/metadata_2013.txt", sep="\t", header=None, names=["id", "cls", "sex"])
md = md[md.sex != "Unknown"].reset_index(drop=True)
idx = {a: i for i, a in enumerate(md.id)}; n = len(md); code = lambda s: pd.factorize(s)[0]
x, g = code(md.sex), code(md.cls)
E = pd.read_csv(f"{DATA}/thiers/Friendship-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v"]).drop_duplicates()
E = E[E.u.isin(idx) & E.v.isin(idx) & (E.u != E.v)]
A = sp.csr_matrix((np.ones(len(E)), (E.u.map(idx), E.v.map(idx))), shape=(n, n))
arcs, codes, _ = operator_arcsets(A, g, n)
perms = [perm_within(x, g, rng) for _ in range(20000)]
rows = []
for op in "ATO":
    u0, v0, w0 = arcs[op]
    for st, m in [("all", np.ones(len(u0), bool))] + [(STRATA[t], codes[op] == t) for t in range(4)]:
        u, v, w = u0[m], v0[m], w0[m]
        QD = []
        for y in perms:
            e = mixing(y, u, v, w, 2); e /= e.sum(); a, b = e.sum(1), e.sum(0)
            QD.append((np.trace(e) - a @ b, 1 - a @ b))
        Q, D = np.array(QD).T; ok = D > 1e-12
        mQ, mD = null_expectation(x, g, u, v, w)
        rows.append(dict(operator=op, stratum=st, n_arcs=int(m.sum()), r_cf=mQ / mD,
                         second_order=mQ / mD - np.cov(Q, D)[0, 1] / mD ** 2 + mQ * D.var() / mD ** 3,
                         r_perm=np.mean(Q[ok] / D[ok]), mc_se=np.std(Q[ok] / D[ok]) / np.sqrt(ok.sum())))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/thiers_ratio_check.csv", index=False); print(df.round(4).to_string(index=False))
