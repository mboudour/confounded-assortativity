# Confounded Assortativity

Code for *Confounded Assortativity in Online Attention Networks* (M. Boudourides, manuscript, 2026).

**Question.** Is the observed homophily of a node attribute X genuine, or is it borrowed
from another attribute G that both structures the network and is associated with X?
(The motivating case: partisan homophily in online attention networks that may be
produced by geography, Quintana-Mathé, Guo, Grinberg & Lazer, 2026.)

**Method.**
1. Newman's categorical assortativity r(X) on a directed (weighted) graph.
2. Null model: permute X uniformly *within* the blocks of the partition induced by G.
3. Closed-form borrowed assortativity r_cf = E[Q]/E[D] (exact ratio of expectations; not E[Q/D], which is
   estimated by permutation and reported as r_perm), `ca_core.null_expectation` / `closed_form_r`:
   leading order E[Q] ≈ ⟨B^G, ΠΠᵀ⟩ (B^G = modularity matrix of G's mixing,
   Π = within-block distribution of X), plus exact finite-sample terms.
4. Interior–boundary (IB) stratification of arcs by G (Boudourides, *DCDS-B*, 2026) and the
   triadic / open-wedge decomposition W = T + O of two-paths (Boudourides, *J. Complex Netw.*, 2026);
   the null is computed per operator and per stratum.

## Layout
```
src/ca_core.py              core routines (mixing, IB strata, T/O operators, nulls)
scripts/01_synthetic.py     synthetic-graph validation (confounding built in by construction), one graph per cell
scripts/01b_synthetic_replications.py   100 graphs per cell: bias/RMSE of r_cf vs mean permutation r (~25 min)
scripts/02_thiers.py        Lycée Thiers 2013: gender | class, gender | track; one 20,000-permutation set per analysis
                            shared by all operators/strata (r_perm, SDs, raw / max-T / BH p-values, 2nd-order check)
scripts/02b_thiers_mc_check.py       300,000-permutation check of E[Q], E[D], E[Q/D] against the closed forms
scripts/03_lazega.py        Lazega law firm: age | status; office as negative control
scripts/03b_lazega_sensitivity.py    quartiles, fixed-width bands, continuous (Pearson) codings
scripts/04_openalex_download.py   OpenAlex download (API key needed)
scripts/05_openalex_analysis.py   OpenAlex citation network: topic | country (negative control), full and fractional counting
scripts/06_tables.py        LaTeX table bodies (written to paper/tables/, not tracked)
scripts/07_figures.py       figures (written to paper/figures/, not tracked)
tests/test_ca_core.py       unit tests (brute-force enumeration, edge cases, operators, p-values)
results/                    CSV outputs used in the paper
data/README.md              where and how to download the (public) data
```

## Reproduce
```bash
pip install -r requirements.txt pytest
# put the data in data/ as described in data/README.md
cd scripts
python -m pytest -q ../tests
python 01_synthetic.py
python 01b_synthetic_replications.py
python 02_thiers.py
python 02b_thiers_mc_check.py
python 03_lazega.py
python 03b_lazega_sensitivity.py
python 04_openalex_download.py   # optional, ~10 min
python 05_openalex_analysis.py
python 06_tables.py
python 07_figures.py
```
Random seeds are fixed. On whole networks r_cf and r_perm agree to Monte Carlo error; on very small strata they can differ (column `second_order` in results/thiers.csv).

## Output columns (results/*.csv)
`r_obs` observed assortativity; `r_perm` mean permutation assortativity E[Q/D] over within-G permutations (`sd_null` its SD);
`r_cf` closed-form borrowed assortativity E[Q]/E[D] (Theorem in the paper); `excess = r_obs − r_cf`; `borrowed = r_cf / r_obs`;
`p_upper`, `p_lower` one-sided and `p_two` two-sided (standardized) plus-one permutation p-values; `operator` ∈ {A, T, O};
`stratum` ∈ {all, II, IB, BI, BB}.
