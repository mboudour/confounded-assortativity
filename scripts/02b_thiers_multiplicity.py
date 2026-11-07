"""Lycee Thiers: multiplicity adjustment for the excess-homophily family of each analysis
(3 operators x {all, II, IB, BI, BB}). Max-T (Westfall-Young) and Benjamini-Hochberg."""
import numpy as np, pandas as pd, scipy.sparse as sp
from common import DATA, RES
from ca_core import operator_arcsets, permutation_family, maxT_adjust

rng = np.random.default_rng(4242)
md = pd.read_csv(f"{DATA}/thiers/metadata_2013.txt", sep="\t", header=None, names=["id", "cls", "sex"])
md = md[md.sex != "Unknown"].reset_index(drop=True)
md["track"] = md.cls.str.replace(r"\d|\*", "", regex=True)
idx = {a: i for i, a in enumerate(md.id)}; n = len(md); code = lambda s: pd.factorize(s)[0]
att = {"sex": code(md.sex), "class": code(md.cls), "track": code(md.track)}
nets = {"friendship": pd.read_csv(f"{DATA}/thiers/Friendship-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v"]).drop_duplicates().assign(w=1.0),
        "diaries": pd.read_csv(f"{DATA}/thiers/Contact-diaries-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v", "w"])}
rows = []
for net, E in nets.items():
    E = E[E.u.isin(idx) & E.v.isin(idx) & (E.u != E.v)]
    A = sp.csr_matrix((E.w.astype(float), (E.u.map(idx), E.v.map(idx))), shape=(n, n))
    for X, G in [("sex", "class"), ("sex", "track")]:
        arcs, codes, _ = operator_arcsets(A, att[G], n)
        labels, obs, draws = permutation_family(att[X], att[G], arcs, codes, rng, 4000)
        z, p_raw, p_maxT, bh = maxT_adjust(obs, draws)
        for (op, st), o, zz, a, b, c in zip(labels, obs, z, p_raw, p_maxT, bh):
            rows.append(dict(network=net, X=X, G=G, operator=op, stratum=st, family_size=len(labels),
                             r_obs=o, z=zz, p_raw=a, p_maxT=b, p_BH=c))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/thiers_multiplicity.csv", index=False)
print(df[df.p_raw < 0.05].round(4).to_string(index=False))
