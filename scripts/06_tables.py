"""Write the LaTeX table bodies of the paper (paper/tables/*.tex) from results/*.csv.
Each file holds the rows and the closing \\bottomrule; the tabular head stays in main.tex."""
import os
import pandas as pd
from common import RES

OUT = os.path.join(os.path.dirname(RES), "paper", "tables"); os.makedirs(OUT, exist_ok=True)
f3 = lambda z: "--" if pd.isna(z) else f"{z:.3f}".replace("-", "$-$")
f4 = lambda z: "--" if pd.isna(z) else f"{z:.4f}".replace("-", "$-$")
def fp(p): return "--" if pd.isna(p) else ("$\\le 0.001$" if p <= 0.0011 else f"{p:.3f}")
def pct(z): return "--" if pd.isna(z) else ("0\\%" if z <= 0 else ("$>$100\\%" if z > 1.1 else f"{100*z:.0f}\\%"))
def write(name, lines): open(f"{OUT}/{name}.tex", "w").write("\n".join(lines) + "\n\\bottomrule")

# Synthetic graph model, one graph per cell
d = pd.read_csv(f"{RES}/synthetic.csv")
write("synthetic", [f"{r.degrees[:3]}. & {r.eta:.1f} & {int(r.lam)} & {f3(r.r_G)} & {f3(r.r_obs)} & {f3(r.r_perm)} ({r.sd_null:.3f}) & "
                    f"{f3(r.r_cf)} & {f3(r.r_cf_leading)} & {pct(r.borrowed)} \\\\" for _, r in d.iterrows()])

# Synthetic replications
d = pd.read_csv(f"{RES}/synthetic_replications.csv")
write("synthetic_reps", [f"{r.degrees[:3]}. & {r.eta:.1f} & {int(r.lam)} & {f3(r.mean_r_obs)} & {f3(r.mean_r_cf)} ({r.sd_r_cf:.3f}) & "
                         f"{f4(r.bias)} & {f4(r.rmse)} & {f4(r.max_abs)} & {f4(r.mean_mc_se)} \\\\" for _, r in d.iterrows()])

# Thiers, whole arc sets
d = pd.read_csv(f"{RES}/thiers.csv"); a = d[d.stratum == "all"]
lab = {("sex", "class"): "gender $\\mid$ class", ("sex", "track"): "gender $\\mid$ track", ("class", "sex"): "class $\\mid$ gender"}
L = []
for net in ["friendship", "diaries"]:
    L.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{'Friendship survey' if net == 'friendship' else 'Contact diaries'}}}}} \\\\")
    for (X, G), name in lab.items():
        for op in "ATO":
            r = a[(a.network == net) & (a.X == X) & (a.G == G) & (a.operator == op)].iloc[0]
            L.append(f"{name if op == 'A' else ''} & ${op}$ & {f3(r.r_obs)} & {f3(r.r_cf)} & {f3(r.r_perm)} & {r.sd_null:.3f} & "
                     f"{f3(r.excess)} & {fp(r.p_two)} & {pct(r.borrowed) if r.r_obs > 0.05 else '--'} \\\\")
write("thiers_all", L)

# Thiers IB profile, gender | class, with max-T adjusted p
m = pd.read_csv(f"{RES}/thiers_multiplicity.csv")
s = d[(d.X == "sex") & (d.G == "class") & (d.stratum != "all")]
L = []
for net in ["friendship", "diaries"]:
    for op in "ATO":
        cells = []
        for st in ["II", "IB", "BI", "BB"]:
            r = s[(s.network == net) & (s.operator == op) & (s.stratum == st)].iloc[0]
            mm = m[(m.network == net) & (m.G == "class") & (m.operator == op) & (m.stratum == st)]
            pa = mm.p_maxT.iloc[0] if len(mm) else float("nan")
            star = "$^{\\dagger}$" if pa < 0.05 else ("$^{*}$" if r.p_two < 0.05 else "")
            cells.append(f"{f3(r.r_obs)}{star} / {f3(r.r_cf)} [{int(r.n_arcs)}]")
        L.append(f"{('Friendship' if net == 'friendship' else 'Diaries') if op == 'A' else ''} & ${op}$ & " + " & ".join(cells) + " \\\\")
write("thiers_ib", L)

# Multiplicity: all family members with raw p < 0.05
mm = m[m.p_raw < 0.05].copy()
write("thiers_mult", [f"{'friendship' if r.network == 'friendship' else 'diaries'} & gender $\\mid$ {r.G} & ${r.operator}$ & {r.stratum} & "
                      f"{f3(r.r_obs)} & {f"{r.z:+.2f}".replace("-", "$-$")} & {fp(r.p_raw)} & {fp(r.p_maxT)} & {fp(r.p_BH)} \\\\" for _, r in mm.iterrows()])

# Lazega main table
d = pd.read_csv(f"{RES}/lazega.csv"); a = d[(d.stratum == "all") & (d.operator == "A")]
name = {"age3": "age", "years3": "years in firm", "status": "status", "practice": "practice", "gender": "gender", "office": "office"}
L = []
for X, G in [("age3", "status"), ("years3", "status"), ("status", "age3"), ("status", "office"), ("practice", "office"), ("gender", "office"), ("age3", "office")]:
    if G == "office" and X == "status": L.append("\\midrule")
    cells = []
    for net in ["friendship", "advice"]:
        r = a[(a.network == net) & (a.X == X) & (a.G == G)].iloc[0]
        cells.append(f"{f3(r.r_obs)} & {f3(r.r_cf)} & {fp(r.p_upper)} & {pct(r.borrowed) if r.r_cf > 0 else '0\\%'}")
    L.append(f"{name[X]} $\\mid$ {name[G]} & " + " & ".join(cells) + " \\\\")
write("lazega", L)

# Lazega sensitivity
d = pd.read_csv(f"{RES}/lazega_sensitivity.csv")
L = []
for (X, coding), g in d.groupby(["X", "coding"], sort=False):
    cells = []
    for net in ["friendship", "advice"]:
        r = g[g.network == net].iloc[0]
        bor = r.r_cf if r.r_cf == r.r_cf else r.r_perm
        cells.append(f"{f3(r.r_obs)} & {f3(bor)} & {pct(r.borrowed_cf)}")
    L.append(f"{X} & {coding} & " + " & ".join(cells) + " \\\\")
write("lazega_sens", L)

# OpenAlex: full vs fractional counting
d = pd.read_csv(f"{RES}/openalex.csv")
L = []
for (X, G), g in d.groupby(["X", "G"], sort=False):
    for st in ["all", "II", "IB", "BI", "BB"]:
        rf = g[(g.weighting == "full") & (g.stratum == st)].iloc[0]; rr = g[(g.weighting == "fractional") & (g.stratum == st)].iloc[0]
        L.append(f"{(X + ' $\\mid$ ' + G) if st == 'all' else ''} & {st} & {int(rf.n_arcs):,} & {f3(rf.r_obs)} & {f3(rf.r_perm)} & {f3(rf.r_cf)} & {pct(rf.borrowed)} & "
                 f"{f3(rr.r_obs)} & {f3(rr.r_cf)} & {pct(rr.borrowed)} \\\\")
write("openalex", L)
print("tables written to", OUT)
