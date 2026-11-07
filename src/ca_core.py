"""Core routines for confounded (borrowed) assortativity on directed graphs."""
import numpy as np

STRATA = ("II", "IB", "BI", "BB")

def mixing(x, u, v, w, K):
    e = np.zeros((K, K)); np.add.at(e, (x[u], x[v]), w); return e

def newman(e):
    """Categorical Newman coefficient from an (unnormalised) mixing matrix."""
    m = e.sum()
    if m <= 0: return np.nan
    e = e / m; a, b = e.sum(1), e.sum(0); d = 1 - a @ b
    return (np.trace(e) - a @ b) / d if d > 1e-12 else np.nan

def ib_strata(g, u, v, n):
    """Interior/boundary status of nodes w.r.t. the partition g (any incident arc
    crossing blocks makes a node boundary) and stratum code 0..3 = II,IB,BI,BB of arcs."""
    cross = g[u] != g[v]
    bd = np.zeros(n, bool); bd[u[cross]] = True; bd[v[cross]] = True
    return bd, 2 * bd[u].astype(int) + bd[v].astype(int)

def perm_within(x, g, rng):
    y = x.copy()
    for c in np.unique(g):
        i = np.flatnonzero(g == c); y[i] = x[rng.permutation(i)]
    return y

def null_expectation(x, g, u, v, w, finite=True):
    """Expected numerator Q and denominator D of Newman's r for x under uniform
    permutation of x within the blocks of g.  finite=False gives the leading-order
    formula <B^G, Pi Pi^T>; finite=True adds the exact without-replacement terms."""
    n = len(x); KG, KX = g.max() + 1, x.max() + 1
    m = w.sum()
    f = np.zeros((KG, KG)); np.add.at(f, (g[u], g[v]), w); f /= m
    nc = np.bincount(g, minlength=KG).astype(float)
    Pi = np.zeros((KG, KX)); np.add.at(Pi, (g, x), 1.0); Pi /= nc[:, None]
    kout = np.bincount(u, weights=w, minlength=n) / m
    kin = np.bincount(v, weights=w, minlength=n) / m
    aG = np.bincount(g, weights=kout, minlength=KG); bG = np.bincount(g, weights=kin, minlength=KG)
    Ea, Eb = aG @ Pi, bG @ Pi
    K = Pi @ Pi.T
    if not finite:
        Q = np.sum((f - np.outer(aG, bG)) * K)
        return Q, 1 - Ea @ Eb
    # E[trace e]: arcs between distinct nodes of the same block get the
    # without-replacement probability; self-loops excluded (none in our data).
    s2 = (Pi ** 2).sum(1)
    Kd = K.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        Kd[np.diag_indices(KG)] = np.where(nc > 1, (nc * s2 - 1) / (nc - 1), 1.0)
    Etr = np.sum(f * Kd)
    # E[sum_p a_p b_p] = sum_p Ea_p Eb_p + sum_p Cov(a_p, b_p)
    cov = 0.0
    for c in range(KG):
        i = np.flatnonzero(g == c)
        if len(i) < 2: continue
        ko, ki = kout[i], kin[i]
        same = ko @ ki; other = ko.sum() * ki.sum() - same
        var = Pi[c] * (1 - Pi[c])
        cov += var.sum() * (same - other / (len(i) - 1))
    Eab = Ea @ Eb + cov
    return Etr - Eab, 1 - Eab

def null_r(x, g, u, v, w, finite=True):
    Q, D = null_expectation(x, g, u, v, w, finite); return Q / D

def two_walk_operators(A):
    """Two-walk decomposition of the *binary* structure of A:
    W = B^2 with zero diagonal (B = 1{A>0}); T = B o W counts directed two-paths i->k->j
    closing on an existing arc i->j; O = W - T counts open two-paths between non-adjacent i, j."""
    B = (A > 0).astype(float).tocsr(); B.setdiag(0); B.eliminate_zeros()
    W = (B @ B).tolil(); W.setdiag(0); W = W.tocsr(); W.eliminate_zeros()
    T = B.multiply(W).tocsr(); T.eliminate_zeros(); O = (W - T).tocsr(); O.eliminate_zeros()
    return T, O


def borrowed_table(x, g, arcsets, ib_codes, rng, n_perm=2000):
    """arcsets: dict name -> (u, v, w); ib_codes: dict name -> stratum code per arc (or None).
    Returns list of dict rows: operator, stratum, n_arcs, r_obs, r_null_perm, sd, p, r_null_exact, borrowed."""
    K = x.max() + 1
    perms = [perm_within(x, g, rng) for _ in range(n_perm)]
    rows = []
    for op, (u, v, w) in arcsets.items():
        codes = ib_codes.get(op)
        subsets = [("all", np.ones(len(u), bool))]
        if codes is not None:
            subsets += [(STRATA[t], codes == t) for t in range(4)]
        for st, msk in subsets:
            if msk.sum() == 0:
                continue
            uu, vv, ww = u[msk], v[msk], w[msk]
            r = newman(mixing(x, uu, vv, ww, K))
            nul = np.array([newman(mixing(y, uu, vv, ww, K)) for y in perms])
            nul = nul[~np.isnan(nul)]
            ex = null_r(x, g, uu, vv, ww, True)
            rows.append(dict(operator=op, stratum=st, n_arcs=int(msk.sum()), weight=float(ww.sum()),
                             r_obs=r, r_null_perm=nul.mean() if len(nul) else np.nan,
                             sd_null=nul.std() if len(nul) else np.nan,
                             p_upper=(1 + np.sum(nul >= r - 1e-12)) / (1 + len(nul)) if len(nul) and r == r else np.nan,
                             p_lower=(1 + np.sum(nul <= r + 1e-12)) / (1 + len(nul)) if len(nul) and r == r else np.nan,
                             r_null_exact=ex, excess=r - ex,
                             borrowed=(ex / r) if (r == r and abs(r) > 1e-9) else np.nan))
    return rows


def operator_arcsets(A, g, n):
    """A: scipy.sparse CSR adjacency (weighted, zero diagonal). Returns arcsets for A, T, O and
    their IB stratum codes w.r.t. g (node status always computed from A)."""
    T, O = two_walk_operators(A)
    Au, Av = A.nonzero(); bd, _ = ib_strata(g, Au, Av, n)
    out, codes = {}, {}
    for name, M in (("A", A), ("T", T), ("O", O)):
        M = M.tocoo(); u, v, w = M.row, M.col, M.data.astype(float)
        out[name] = (u, v, w); codes[name] = 2 * bd[u].astype(int) + bd[v].astype(int)
    return out, codes, bd
