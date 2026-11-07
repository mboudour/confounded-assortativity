"""Write the LaTeX tables of the paper from results/*.csv."""
import pandas as pd
from common import RES
import os
OUT = os.path.join(os.path.dirname(RES), "paper", "tables"); os.makedirs(OUT, exist_ok=True)
f3 = lambda z: "--" if pd.isna(z) else f"{z:.3f}".replace("-", "$-$")
def fp(p):
    return "$\\le 0.001$" if p <= 0.0011 else f"{p:.3f}"
def pct(z): return "--" if pd.isna(z) else ("0\\%" if z <= 0 else f"{100*z:.0f}\\%")

# Table: planted
d = pd.read_csv(f"{RES}/planted.csv")
L = []
for _, r in d.iterrows():
    L.append(f"{r.degrees[:3]}. & {r.eta:.1f} & {int(r.lam)} & {f3(r.r_G)} & {f3(r.r_obs)} & {f3(r.r_null_perm)} ({r.sd_null:.3f}) & {f3(r.r_null_exact)} & {f3(r.r_null_leading)} & {pct(r.borrowed)} \\\\")
open(f"{OUT}/planted.tex", "w").write("\n".join(L) + "\n\\bottomrule")

# Table: Thiers overall
d = pd.read_csv(f"{RES}/thiers.csv"); a = d[d.stratum == "all"]
L = []
lab = {("sex", "class"): "gender $\\mid$ class", ("sex", "track"): "gender $\\mid$ track", ("class", "sex"): "class $\\mid$ gender"}
for net in ["friendship", "diaries"]:
    L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{'Friendship survey' if net=='friendship' else 'Contact diaries'}}}}} \\\\")
    for (X, G), name in lab.items():
        for op in "ATO":
            r = a[(a.network == net) & (a.X == X) & (a.G == G) & (a.operator == op)].iloc[0]
            L.append(f"{name if op=='A' else ''} & ${op}$ & {f3(r.r_obs)} & {f3(r.r_null_exact)} & {r.sd_null:.3f} & {f3(r.excess)} & {fp(min(r.p_upper, r.p_lower)*2 if min(r.p_upper,r.p_lower)*2<1 else 1)} & {(('$>$100\\%' if r.borrowed > 1.1 else pct(r.borrowed)) if r.r_obs>0.05 else '--')} \\\\")
open(f"{OUT}/thiers_all.tex", "w").write("\n".join(L) + "\n\\bottomrule")

# Table: Thiers IB profile gender|class
s = d[(d.X == "sex") & (d.G == "class") & (d.stratum != "all")]
L = []
for net in ["friendship", "diaries"]:
    for op in "ATO":
        cells = []
        for st in ["II", "IB", "BI", "BB"]:
            r = s[(s.network == net) & (s.operator == op) & (s.stratum == st)].iloc[0]
            p = min(1, 2 * min(r.p_upper, r.p_lower))
            star = "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")
            cells.append(f"{f3(r.r_obs)}{star} / {f3(r.r_null_exact)} [{int(r.n_arcs)}]")
        L.append(f"{('Friendship' if net=='friendship' else 'Diaries') if op=='A' else ''} & ${op}$ & " + " & ".join(cells) + " \\\\")
open(f"{OUT}/thiers_ib.tex", "w").write("\n".join(L) + "\n\\bottomrule")

# Table: Lazega
d = pd.read_csv(f"{RES}/lazega.csv"); a = d[d.stratum == "all"]
name = {"age3": "age", "years3": "years in firm", "status": "status", "practice": "practice", "gender": "gender", "office": "office"}
L = []
for X, G in [("age3", "status"), ("years3", "status"), ("status", "age3"), ("status", "office"), ("practice", "office"), ("gender", "office"), ("age3", "office")]:
    if G == "office" and X == "status": L.append("\\midrule")
    cells = []
    for net in ["friendship", "advice"]:
        r = a[(a.network == net) & (a.X == X) & (a.G == G) & (a.operator == "A")].iloc[0]
        cells.append(f"{f3(r.r_obs)} & {f3(r.r_null_exact)} & {fp(r.p_upper)} & {pct(r.borrowed) if r.r_null_exact>0 else '0\\%'}")
    L.append(f"{name[X]} $\\mid$ {name[G]} & " + " & ".join(cells) + " \\\\")
open(f"{OUT}/lazega.tex", "w").write("\n".join(L) + "\n\\bottomrule")

# Table: OpenAlex
d = pd.read_csv(f"{RES}/openalex.csv")
L = []
for (X, G), g in d.groupby(["X", "G"], sort=False):
    for _, r in g.iterrows():
        L.append(f"{(X + ' $\\mid$ ' + G) if r.stratum=='all' else ''} & {r.stratum} & {int(r.n_arcs):,} & {f3(r.r_obs)} & {f3(r.r_null_perm)} & {f3(r.r_null_exact)} & {pct(r.borrowed)} \\\\")
open(f"{OUT}/openalex.tex", "w").write("\n".join(L) + "\n\\bottomrule")
print("tables written")
