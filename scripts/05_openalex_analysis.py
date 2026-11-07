"""OpenAlex author-to-author citation network (network-science topics, 2015-2024):
topic homophily conditioned on country and vice versa (negative control at scale)."""
import gzip, json, collections, numpy as np, pandas as pd
from common import DATA, RES
from ca_core import mixing, newman, perm_within, null_r, ib_strata, STRATA

works, wa = {}, {}
atop, acty = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for line in gzip.open(f"{DATA}/openalex/oa_works.jsonl.gz", "rt"):
    w = json.loads(line); wid = w["id"].split("/")[-1]; top = w["primary_topic"]["id"].split("/")[-1]
    works[wid] = [r.split("/")[-1] for r in w["referenced_works"]]; au = []
    for a in w["authorships"]:
        aid = (a.get("author") or {}).get("id")
        if not aid: continue
        aid = aid.split("/")[-1]; au.append(aid); atop[aid][top] += 1
        for c in set(a.get("countries") or []): acty[aid][c] += 1
    wa[wid] = list(dict.fromkeys(au))
pairs = collections.Counter()
for a, refs in works.items():
    for b in refs:
        if b in works:
            for s in wa[a]:
                for t in wa[b]:
                    if s != t: pairs[(s, t)] += 1
nodes = [a for a in atop if acty[a]]
idx = {a: i for i, a in enumerate(nodes)}; n = len(nodes)
topic = pd.factorize(pd.Series([atop[a].most_common(1)[0][0] for a in nodes]))[0]
country = pd.factorize(pd.Series([acty[a].most_common(1)[0][0] for a in nodes]))[0]
E = np.array([(idx[s], idx[t], c) for (s, t), c in pairs.items() if s in idx and t in idx], dtype=float)
u, v, w = E[:, 0].astype(int), E[:, 1].astype(int), E[:, 2]
print("works", len(works), "authors", n, "arcs", len(u))
rng = np.random.default_rng(2024); rows = []
for X, G, x, g in [("topic", "country", topic, country), ("country", "topic", country, topic)]:
    K = x.max() + 1; bd, s = ib_strata(g, u, v, n); perms = [perm_within(x, g, rng) for _ in range(200)]
    rG = newman(mixing(g, u, v, w, g.max() + 1))
    for st, m in [("all", np.ones(len(u), bool))] + [(STRATA[t], s == t) for t in range(4)]:
        if m.sum() == 0: continue
        r = newman(mixing(x, u[m], v[m], w[m], K))
        nul = np.array([newman(mixing(y, u[m], v[m], w[m], K)) for y in perms]); nul = nul[~np.isnan(nul)]
        ex = null_r(x, g, u[m], v[m], w[m], True)
        rows.append(dict(X=X, G=G, r_G=rG, n=n, n_interior=int((~bd).sum()), operator="A", stratum=st,
                         n_arcs=int(m.sum()), weight=w[m].sum(), r_obs=r, r_null_perm=nul.mean(), sd_null=nul.std(),
                         p_upper=(1 + np.sum(nul >= r)) / (1 + len(nul)), r_null_exact=ex, excess=r - ex, borrowed=ex / r))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/openalex.csv", index=False); print(df.round(3).to_string(index=False))
