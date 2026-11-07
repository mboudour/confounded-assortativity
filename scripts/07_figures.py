"""Figures of the paper (written to paper/figures/).

Fig. 1  Quotient graph of the Lycee Thiers friendship network by class (networkx + pygraphviz).
Fig. 2  Observed vs borrowed assortativity across all cases (from results/*.csv).
Fig. 3  Excess homophily of gender given class by operator x interior-boundary stratum.
Fig. 4  Planted model: borrowed share vs direct homophily.
Fig. 1 needs data/thiers; Figs. 2-4 need only results/.
"""
import os
import numpy as np, pandas as pd, networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from common import DATA, RES
from ca_core import mixing, newman

FIG = os.path.join(os.path.dirname(RES), "paper", "figures"); os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#ffffff"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated)
BLUE, RED, MID = "#1c5cab", "#d03b3b", "#f0efec"  # diverging poles + neutral midpoint
DIV = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

# ---------------------------------------------------------------- Fig. 1
def fig_quotient():
    md = pd.read_csv(f"{DATA}/thiers/metadata_2013.txt", sep="\t", header=None, names=["id", "cls", "sex"])
    md = md[md.sex != "Unknown"]
    E = pd.read_csv(f"{DATA}/thiers/Friendship-network_data_2013.csv.gz", sep=" ", header=None, names=["u", "v"]).drop_duplicates()
    cls = dict(zip(md.id, md.cls)); E = E[E.u.isin(cls) & E.v.isin(cls) & (E.u != E.v)]
    E = E.assign(cu=E.u.map(cls), cv=E.v.map(cls))
    W = E.groupby(["cu", "cv"]).size()
    size = md.cls.value_counts(); fem = md.groupby("cls").sex.apply(lambda s: (s == "F").mean())
    # weighted attribute assortativity of the quotient graph (self-loops = within-class arcs)
    lab = {c: i for i, c in enumerate(sorted(size.index))}
    e = np.zeros((len(lab), len(lab)))
    for (a, b), w in W.items(): e[lab[a], lab[b]] += w
    r_q = newman(e)
    Q = nx.DiGraph()
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    for c in size.index:
        within = int(W.get((c, c), 0))
        Q.add_node(c, label=f"{c}\n{size[c]}", width=f"{0.16 * np.sqrt(size[c]):.2f}", shape="circle", style="filled",
                   fillcolor=to_hex(DIV(norm(fem[c]))), color=INK2, penwidth="0.8", fixedsize="true")
    for (a, b), w in W.items():
        if a != b:
            Q.add_edge(a, b, penwidth=f"{0.4 + 0.35 * w:.2f}", color="#8a8984", arrowsize="0.6", weight=str(w))
    A = nx.nx_agraph.to_agraph(Q)
    A.graph_attr.update(overlap="false", splines="true", fontname="Times-Roman",
                        outputorder="edgesfirst", pad="0.3", sep="+20", bgcolor="white")
    A.node_attr.update(fontname="Times-Roman", fontsize="10", fontcolor=INK)
    A.layout(prog="neato", args="-Gstart=7")
    A.draw(os.path.join(FIG, "fig1_quotient_thiers.pdf"))
    # colour bar for the share of girls, drawn separately and placed under the graph in LaTeX
    fig, ax = plt.subplots(figsize=(3.2, 0.42))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=ax, orientation="horizontal")
    cb.set_label("share of girls in class", color=INK); cb.outline.set_edgecolor(INK2)
    fig.savefig(os.path.join(FIG, "fig1_colorbar.pdf"), bbox_inches="tight"); plt.close(fig)
    return r_q

# ---------------------------------------------------------------- Fig. 2
def fig_dotplot():
    t = pd.read_csv(f"{RES}/thiers.csv"); l = pd.read_csv(f"{RES}/lazega.csv"); o = pd.read_csv(f"{RES}/openalex.csv")
    def pick(d, **k):
        q = d[(d.stratum == "all") & (d.operator == "A")]
        for a, b in k.items(): q = q[q[a] == b]
        return q.iloc[0]
    rows = [
        ("Thiers friendship: gender | class", pick(t, network="friendship", X="sex", G="class"), "school"),
        ("Thiers friendship: gender | track", pick(t, network="friendship", X="sex", G="track"), "school"),
        ("Thiers diaries: gender | class", pick(t, network="diaries", X="sex", G="class"), "school"),
        ("Thiers friendship: class | gender", pick(t, network="friendship", X="class", G="sex"), "school"),
        ("Lazega friendship: age | status", pick(l, network="friendship", X="age3", G="status"), "firm"),
        ("Lazega friendship: years | status", pick(l, network="friendship", X="years3", G="status"), "firm"),
        ("Lazega advice: age | status", pick(l, network="advice", X="age3", G="status"), "firm"),
        ("Lazega friendship: status | office", pick(l, network="friendship", X="status", G="office"), "firm"),
        ("Lazega advice: practice | office", pick(l, network="advice", X="practice", G="office"), "firm"),
        ("OpenAlex citation: topic | country", pick(o, X="topic", G="country", weighting="full"), "cite"),
        ("OpenAlex citation: country | topic", pick(o, X="country", G="topic", weighting="full"), "cite"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    y = np.arange(len(rows))[::-1]
    for yi, (name, r, _) in zip(y, rows):
        ax.plot([r.r_cf, r.r_obs], [yi, yi], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.errorbar(r.r_cf, yi, xerr=1.96 * r.sd_null, fmt="none", ecolor=CAT[1], elinewidth=1, capsize=2, zorder=2)
        ax.scatter(r.r_cf, yi, s=36, facecolor=SURF, edgecolor=CAT[1], linewidth=1.6, zorder=3, marker="D")
        ax.scatter(r.r_obs, yi, s=40, color=CAT[0], edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows], color=INK)
    for yy in (y[3] - 0.5, y[8] - 0.5): ax.axhline(yy, color=GRID, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Newman assortativity  $r$"); ax.set_xlim(-0.1, 1.0)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.scatter([], [], s=40, color=CAT[0], label="observed $r(X)$")
    ax.scatter([], [], s=36, facecolor=SURF, edgecolor=CAT[1], linewidth=1.6, marker="D",
               label="borrowed $r_{\\mathrm{cf}}(X\\mid G)$ ($\\pm$1.96 perm. SD)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, labelcolor=INK)
    fig.savefig(os.path.join(FIG, "fig2_observed_vs_borrowed.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3
def fig_heatmap():
    t = pd.read_csv(f"{RES}/thiers.csv"); s = t[(t.X == "sex") & (t.G == "class") & (t.stratum != "all")]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.35)
    for ax, net, title in zip(axes, ["friendship", "diaries"], ["Friendship survey", "Contact diaries"]):
        M = np.full((3, 4), np.nan); P = np.ones((3, 4))
        for i, op in enumerate("ATO"):
            for j, st in enumerate(["II", "IB", "BI", "BB"]):
                r = s[(s.network == net) & (s.operator == op) & (s.stratum == st)].iloc[0]
                M[i, j] = r.excess; P[i, j] = r.p_two
        ax.imshow(M, cmap=DIV, norm=norm, aspect="auto")
        for i in range(3):
            for j in range(4):
                star = "**" if P[i, j] < 0.01 else ("*" if P[i, j] < 0.05 else "")
                ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(["II", "IB", "BI", "BB"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["$A$", "$T$", "$O$"])
        ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
        ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, fontsize=9, color=INK)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("excess  $r - r_0$", color=INK); cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, "fig3_excess_profile.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4
def fig_synthetic():
    d = pd.read_csv(f"{RES}/synthetic.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), sharey=True)
    marks = ["o", "s", "^"]
    for ax, deg, title in zip(axes, ["homogeneous", "heterogeneous"], ["Homogeneous degrees", "Heterogeneous degrees"]):
        q = d[d.degrees == deg]
        for k, (eta, g) in enumerate(q.groupby("eta")):
            ax.plot(g.lam, g.borrowed, color=CAT[k], lw=2, marker=marks[k], ms=6, mec=SURF, mew=1.2)
            ax.annotate(f"$\\eta={eta:.1f}$", (g.lam.iloc[-1], g.borrowed.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1, 3]); ax.set_xlabel("direct homophily  $\\lambda$"); ax.set_xlim(-0.2, 3.9)
        ax.set_ylim(0, 1.12); ax.axhline(1, color=INK2, lw=0.6, ls=":")
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("borrowed share  $r_0/r$")
    fig.savefig(os.path.join(FIG, "fig4_synthetic.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. S1
def fig_replications():
    d = pd.read_csv(f"{RES}/synthetic_replications_all.csv"); d["diff"] = d.r_cf - d.r_perm
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for ax, deg, title in zip(axes, ["homogeneous", "heterogeneous"], ["Homogeneous degrees", "Heterogeneous degrees"]):
        q = d[d.degrees == deg]; cells = list(q.groupby(["eta", "lam"]))
        data = [g["diff"].values * 1e3 for _, g in cells]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=2.5, mfc=INK2, mec="none"),
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        for k, b in enumerate(bp["boxes"]):
            b.set_facecolor(CAT[k // 3]); b.set_alpha(0.55); b.set_edgecolor(INK2)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(range(1, len(cells) + 1)); ax.set_xticklabels([f"{int(l)}" for (e, l), _ in cells], fontsize=7)
        for k, eta in enumerate(sorted(q.eta.unique())):
            ax.text(2 + 3 * k, 1.0, f"$\\eta={eta:.1f}$", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xlabel("direct homophily $\\lambda$ (grouped by $\\eta$)"); ax.set_title(title, fontsize=9, color=INK, pad=14)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("$r_{\\mathrm{cf}} - \\bar r_{\\mathrm{perm}}$  ($\\times 10^{-3}$)")
    fig.savefig(os.path.join(FIG, "figS1_replications.pdf"), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_dotplot(); fig_heatmap(); fig_synthetic()
    if os.path.exists(f"{RES}/synthetic_replications_all.csv"): fig_replications()
    if os.path.exists(f"{DATA}/thiers/metadata_2013.txt"):
        print(f"quotient-graph class assortativity r = {fig_quotient():.3f}")
    print("figures written to", FIG)
