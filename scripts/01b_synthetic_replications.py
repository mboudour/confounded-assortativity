"""Replication study for the synthetic graph model (see 01_synthetic.py for the generator).
For each cell, R independent graphs are drawn. On each graph b we compute the closed-form
borrowed assortativity r_cf,b = E[Q]/E[D] and the mean permutation assortativity rbar_perm,b
over B within-block permutations, and report bias, RMSE and max |r_cf,b - rbar_perm,b| across
graphs, together with the Monte Carlo SE of rbar_perm and the spread of r_cf across graphs."""
import importlib.util, os, numpy as np, pandas as pd
from common import RES
from ca_core import mixing, newman, perm_within, closed_form_r
spec = importlib.util.spec_from_file_location("syn", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_synthetic.py"))
syn = importlib.util.module_from_spec(spec); spec.loader.exec_module(syn)

R, B = 100, 200
rng = np.random.default_rng(20261107)
rows, reps = [], []
for hetero, eta, lam in syn.CELLS:
    d = []
    for b in range(R):
        g, x, u, v, w, capped = syn.generate(hetero, eta, lam, rng)
        r_obs = newman(mixing(x, u, v, w, syn.KX))
        nul = np.array([newman(mixing(perm_within(x, g, rng), u, v, w, syn.KX)) for _ in range(B)])
        rc = closed_form_r(x, g, u, v, w, True)
        d.append(dict(rep=b, r_obs=r_obs, r_cf=rc, r_perm=nul.mean(), mc_se=nul.std() / np.sqrt(B), capped=capped))
    d = pd.DataFrame(d); diff = d.r_cf - d.r_perm
    lab = dict(degrees="heterogeneous" if hetero else "homogeneous", eta=eta, lam=lam)
    reps.append(d.assign(**lab))
    rows.append(dict(**lab, R=R, B=B, mean_r_obs=d.r_obs.mean(), mean_r_cf=d.r_cf.mean(), sd_r_cf=d.r_cf.std(),
                     bias=diff.mean(), rmse=np.sqrt((diff ** 2).mean()), max_abs=diff.abs().max(),
                     mean_mc_se=d.mc_se.mean(), mean_borrowed=(d.r_cf / d.r_obs).mean(), max_capped=d.capped.max()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/synthetic_replications.csv", index=False)
pd.concat(reps).to_csv(f"{RES}/synthetic_replications_all.csv", index=False)
