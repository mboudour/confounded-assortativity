# Confounded Assortativity

Code for *Confounded Assortativity in Online Attention Networks* (M. Boudourides, manuscript, 2026).

**Question.** Is the observed homophily of a node attribute X genuine, or is it borrowed
from another attribute G that both structures the network and is associated with X?
(The motivating case: partisan homophily in online attention networks that may be
produced by geography, Quintana-Mathé, Guo, Grinberg & Lazer, 2026.)

**Method.**
1. Newman's categorical assortativity r(X) on a directed (weighted) graph.
2. Null model: permute X uniformly *within* the blocks of the partition induced by G.
3. Closed-form expectation of the null (no simulation needed), `ca_core.null_expectation`:
   leading order E[Q] ≈ ⟨B^G, ΠΠᵀ⟩ (B^G = modularity matrix of G's mixing,
   Π = within-block distribution of X), plus exact finite-sample terms.
4. Interior–boundary (IB) stratification of arcs by G (Boudourides, *DCDS-B*, 2026) and the
   triadic / open-wedge decomposition W = T + O of two-paths (Boudourides, *J. Complex Netw.*, 2026);
   the null is computed per operator and per stratum.

## Layout
```
src/ca_core.py              core routines (mixing, IB strata, T/O operators, nulls)
scripts/01_planted.py       planted-model validation
scripts/02_thiers.py        Lycée Thiers 2013: gender | class, gender | track
scripts/03_lazega.py        Lazega law firm: age | status; office as negative control
scripts/04_openalex_download.py   OpenAlex download (API key needed)
scripts/05_openalex_analysis.py   OpenAlex citation network: topic | country (negative control)
scripts/06_tables.py        LaTeX tables of the paper from results/
results/                    CSV outputs used in the paper
paper/                      LaTeX source
data/README.md              where and how to download the (public) data
```

## Reproduce
```bash
pip install -r requirements.txt
# put the data in data/ as described in data/README.md
cd scripts
python 01_planted.py
python 02_thiers.py
python 03_lazega.py
python 04_openalex_download.py   # optional, ~10 min
python 05_openalex_analysis.py
python 06_tables.py              # then: cd ../paper && latexmk -pdf main.tex
```
Random seeds are fixed; permutation results agree with the closed-form nulls to Monte Carlo error.

## Output columns (results/*.csv)
`r_obs` observed assortativity; `r_null_perm` mean over within-G permutations (`sd_null` its SD);
`r_null_exact` closed-form null; `excess = r_obs − r_null_exact`; `borrowed = r_null_exact / r_obs`;
`p_upper`, `p_lower` one-sided permutation p-values; `operator` ∈ {A, T, O};
`stratum` ∈ {all, II, IB, BI, BB}.
