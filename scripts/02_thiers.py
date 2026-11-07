"""Lycee Thiers 2013 (SocioPatterns): gender homophily conditioned on class / track."""
import numpy as np, pandas as pd, scipy.sparse as sp
from common import DATA, RES
from ca_core import operator_arcsets, borrowed_table, newman, mixing

rng = np.random.default_rng(2013)
md = pd.read_csv(f"{DATA}/thiers/metadata_2013.txt", sep="\t", header=None, names=["id", "cls", "sex"])
md = md[md.sex != "Unknown"].reset_index(drop=True)
md["track"] = md.cls.str.replace(r"\d|\*", "", regex=True)
idx = {a: i for i, a in enumerate(md.id)}; n = len(md)
code = lambda s: pd.factorize(s)[0]
att = {"sex": code(md.sex), "class": code(md.cls), "track": code(md.track)}

nets = {
 "friendship": pd.read_csv(f"{DATA}/thiers/Friendship-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v"]).drop_duplicates().assign(w=1.0),
 "diaries": pd.read_csv(f"{DATA}/thiers/Contact-diaries-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v", "w"]),
}
rows = []
for net, E in nets.items():
    E = E[E.u.isin(idx) & E.v.isin(idx) & (E.u != E.v)]
    A = sp.csr_matrix((E.w.astype(float), (E.u.map(idx), E.v.map(idx))), shape=(n, n))
    for X, G in [("sex", "class"), ("sex", "track"), ("class", "sex")]:
        x, g = att[X], att[G]
        arcs, codes, bd = operator_arcsets(A, g, n)
        u, v, w = arcs["A"]; rG = newman(mixing(g, u, v, w, g.max() + 1))
        for r in borrowed_table(x, g, arcs, codes, rng, 2000):
            rows.append(dict(network=net, X=X, G=G, r_G=rG, n_interior=int((~bd).sum()), n=n, **r))
pd.DataFrame(rows).to_csv(f"{RES}/thiers.csv", index=False)
print(pd.DataFrame(rows).query("stratum=='all'").round(3).to_string(index=False))
