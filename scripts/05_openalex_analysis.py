"""OpenAlex author-to-author citation network (network-science topics, 2015-2024):
topic homophily conditioned on country and vice versa (negative control at scale).

Construction. Every citation from paper P (authors a_1..a_A with OpenAlex author ids) to paper P'
(authors b_1..b_B) inside the corpus is expanded to the A*B author pairs (a_i, b_j). Pairs with
a_i = b_j (author self-citation) are dropped, so the graph has no loops. Repeated pairs are
aggregated. Two weightings: 'full' (each pair adds 1) and 'fractional' (each pair adds 1/(A*B)).
Author topic = most frequent primary topic of the author's papers; author country = most frequent
country over the author's authorships (each authorship counts a country once even with several
affiliations there; authors with several countries in one authorship contribute to each).
Ties are broken alphabetically, so the assignment is deterministic. Authors without any recorded country are dropped."""
import gzip, json, collections, numpy as np, pandas as pd
from common import DATA, RES
from ca_core import mixing, newman, perm_within, closed_form_r, ib_strata, STRATA

works, wa = {}, {}
atop, acty = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for line in gzip.open(f"{DATA}/openalex/oa_works.jsonl.gz", "rt"):
    w = json.loads(line); wid = w["id"].split("/")[-1]; top = w["primary_topic"]["id"].split("/")[-1]
    works[wid] = [r.split("/")[-1] for r in w["referenced_works"]]; au = []
    for a in w["authorships"]:
        aid = (a.get("author") or {}).get("id")
        if not aid: continue
        aid = aid.split("/")[-1]; au.append(aid); atop[aid][top] += 1
        for c in sorted(set(a.get("countries") or [])): acty[aid][c] += 1
    wa[wid] = list(dict.fromkeys(au))
pairs, fpairs = collections.Counter(), collections.Counter()
for a, refs in works.items():
    for b in refs:
        if b in works and wa[a] and wa[b]:
            frac = 1.0 / (len(wa[a]) * len(wa[b]))
            for s in wa[a]:
                for t in wa[b]:
                    if s != t: pairs[(s, t)] += 1; fpairs[(s, t)] += frac
nodes = [a for a in atop if acty[a]]
idx = {a: i for i, a in enumerate(nodes)}; n = len(nodes)
def mode(cnt):  # most frequent value; ties broken alphabetically (deterministic across runs)
    return min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
topic = pd.factorize(pd.Series([mode(atop[a]) for a in nodes]))[0]
country = pd.factorize(pd.Series([mode(acty[a]) for a in nodes]))[0]
keys = [(s, t) for (s, t) in pairs if s in idx and t in idx]
u = np.array([idx[s] for s, _ in keys]); v = np.array([idx[t] for _, t in keys])
W = {"full": np.array([pairs[k] for k in keys], float), "fractional": np.array([fpairs[k] for k in keys], float)}
print("works", len(works), "authors", n, "arcs", len(u))
rng = np.random.default_rng(2024); rows = []
for (X, G, x, g), wname in [(c, wn) for c in [("topic", "country", topic, country), ("country", "topic", country, topic)] for wn in W]:
    w = W[wname]
    K = x.max() + 1; bd, s = ib_strata(g, u, v, n); perms = [perm_within(x, g, rng) for _ in range(200)]
    rG = newman(mixing(g, u, v, w, g.max() + 1))
    for st, m in [("all", np.ones(len(u), bool))] + [(STRATA[t], s == t) for t in range(4)]:
        if m.sum() == 0: continue
        r = newman(mixing(x, u[m], v[m], w[m], K))
        nul = np.array([newman(mixing(y, u[m], v[m], w[m], K)) for y in perms]); nul = nul[~np.isnan(nul)]
        ex = closed_form_r(x, g, u[m], v[m], w[m], True)
        rows.append(dict(X=X, G=G, weighting=wname, r_G=rG, n=n, n_interior=int((~bd).sum()), operator="A", stratum=st,
                         n_arcs=int(m.sum()), weight=w[m].sum(), r_obs=r, r_perm=nul.mean(), sd_null=nul.std(),
                         p_upper=(1 + np.sum(nul >= r)) / (1 + len(nul)), r_cf=ex, excess=r - ex, borrowed=ex / r))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/openalex.csv", index=False); print(df.round(3).to_string(index=False))
