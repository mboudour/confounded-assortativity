"""Unit tests for ca_core. Run with:  python -m pytest -q tests"""
import itertools, os, sys
import numpy as np
import pytest
import scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ca_core import (mixing, newman, null_expectation, closed_form_r, perm_within,
                     two_walk_operators, ib_strata, operator_arcsets, borrowed_table)


def QD(x, u, v, w, K):
    e = mixing(x, u, v, w, K); e = e / e.sum(); a, b = e.sum(1), e.sum(0)
    return np.trace(e) - a @ b, 1 - a @ b


def enumerate_null(x, g, u, v, w):
    """Exact averages of Q, D and Q/D over ALL within-block permutations (brute force)."""
    K = x.max() + 1
    blocks = [np.flatnonzero(g == c) for c in np.unique(g)]
    per_block = [list(itertools.permutations(b)) for b in blocks]
    Qs, Ds, Rs = [], [], []
    for combo in itertools.product(*per_block):
        y = x.copy()
        for b, p in zip(blocks, combo):
            y[b] = x[list(p)]
        q, d = QD(y, u, v, w, K); Qs.append(q); Ds.append(d); Rs.append(q / d if d > 1e-12 else np.nan)
    return np.mean(Qs), np.mean(Ds), np.nanmean(Rs)


def random_graph(n, p, rng, weighted=False, loops=False):
    M = (rng.random((n, n)) < p).astype(float)
    if not loops: np.fill_diagonal(M, 0)
    if weighted: M *= rng.gamma(0.5, 2.0, (n, n))      # non-integer, concentrated weights
    u, v = np.nonzero(M); return u, v, M[u, v]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("weighted", [False, True])
def test_theorem_matches_bruteforce(seed, weighted):
    rng = np.random.default_rng(seed)
    n = 9
    g = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])[rng.permutation(n)] if seed % 2 else np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    x = rng.integers(0, 3, n)
    u, v, w = random_graph(n, 0.35, rng, weighted)
    if len(u) < 3: pytest.skip("too few arcs")
    EQ, ED, ER = enumerate_null(x, g, u, v, w)
    Q, D = null_expectation(x, g, u, v, w, finite=True)
    assert Q == pytest.approx(EQ, abs=1e-12)
    assert D == pytest.approx(ED, abs=1e-12)
    # r_cf = E[Q]/E[D] is exact for that ratio, but is NOT E[Q/D] in general
    assert closed_form_r(x, g, u, v, w) == pytest.approx(EQ / ED, abs=1e-12)


def test_ratio_of_expectations_differs_from_expected_ratio():
    rng = np.random.default_rng(3); diffs = []
    for s in range(10):
        n = 8; g = np.array([0, 0, 0, 0, 1, 1, 1, 1]); x = rng.integers(0, 2, n)
        u, v, w = random_graph(n, 0.4, rng, True)
        if len(u) < 3: continue
        EQ, ED, ER = enumerate_null(x, g, u, v, w)
        if np.isfinite(ER): diffs.append(abs(EQ / ED - ER))
    assert max(diffs) > 1e-6        # the two estimands are genuinely different


def test_singleton_blocks():
    rng = np.random.default_rng(1)
    g = np.array([0, 1, 1, 1, 2, 2, 3]); x = np.array([0, 1, 0, 1, 1, 0, 1])   # blocks 0 and 3 are singletons
    u, v, w = random_graph(7, 0.5, rng)
    y = perm_within(x, g, rng); assert y[0] == x[0] and y[6] == x[6]
    with np.errstate(all="raise"):
        Q, D = null_expectation(x, g, u, v, w)
    EQ, ED, _ = enumerate_null(x, g, u, v, w)
    assert Q == pytest.approx(EQ, abs=1e-12) and D == pytest.approx(ED, abs=1e-12)


def test_undefined_cases_return_nan():
    u = np.array([0, 1]); v = np.array([1, 0]); w = np.ones(2)
    assert np.isnan(newman(mixing(np.array([0, 0]), u, v, w, 1)))     # one category: D = 0
    assert np.isnan(newman(np.zeros((2, 2))))                            # empty arc set
    assert np.isnan(closed_form_r(np.array([0, 0]), np.array([0, 1]), u, v, w))   # E[D] = 0


def test_directed_marginals_not_symmetrised():
    # a directed star: all arcs leave node 0; a != b must be respected
    u = np.array([0, 0, 0]); v = np.array([1, 2, 3]); w = np.ones(3); x = np.array([0, 1, 1, 0])
    e = mixing(x, u, v, w, 2) / 3
    assert np.allclose(e.sum(1), [1, 0]) and np.allclose(e.sum(0), [1 / 3, 2 / 3])
    g = np.array([0, 0, 1, 1])
    EQ, ED, _ = enumerate_null(x, g, u, v, w)
    Q, D = null_expectation(x, g, u, v, w)
    assert Q == pytest.approx(EQ) and D == pytest.approx(ED)


def test_two_walk_decomposition_and_status():
    rng = np.random.default_rng(5)
    M = (rng.random((30, 30)) < 0.15).astype(float); np.fill_diagonal(M, 0)
    M *= rng.uniform(0.5, 3, M.shape)                         # weights must not leak into T, O
    A = sp.csr_matrix(M); B = (M > 0).astype(float)
    W = B @ B; np.fill_diagonal(W, 0)
    T, O = two_walk_operators(A)
    assert np.allclose(T.toarray() + O.toarray(), W)
    assert np.all(T.diagonal() == 0) and np.all(O.diagonal() == 0)
    assert np.all(T.toarray()[B == 0] == 0) and np.all(O.toarray()[B == 1] == 0)
    g = rng.integers(0, 3, 30)
    arcs, codes, bd = operator_arcsets(A, g, 30)
    bdA, _ = ib_strata(g, *np.nonzero(M), 30)
    assert np.array_equal(bd, bdA)                            # status always from the original graph
    for op in "TO":
        u, v, _ = arcs[op]
        assert np.array_equal(codes[op], 2 * bdA[u].astype(int) + bdA[v].astype(int))


def test_empty_strata_and_operators():
    # two disjoint directed 2-cycles in different blocks: every node interior, so IB, BI, BB are empty;
    # no two-paths close on an arc (T is empty) and O has only i->j->i loops removed
    M = np.zeros((4, 4)); M[0, 1] = M[1, 0] = M[2, 3] = M[3, 2] = 1
    A = sp.csr_matrix(M); g = np.array([0, 0, 1, 1]); x = np.array([0, 1, 0, 1])
    arcs, codes, bd = operator_arcsets(A, g, 4)
    assert not bd.any() and len(arcs["T"][0]) == 0
    rows = borrowed_table(x, g, arcs, codes, np.random.default_rng(0), n_perm=50)
    strata = {(r["operator"], r["stratum"]) for r in rows}
    assert ("A", "II") in strata and ("A", "BB") not in strata and not any(o == "T" for o, _ in strata)


def test_permutation_pvalues_never_zero():
    rng = np.random.default_rng(2)
    g = np.repeat([0, 1], 10); x = g.copy()                   # observed is extreme
    u, v, w = random_graph(20, 0.3, rng)
    A = sp.csr_matrix((w, (u, v)), shape=(20, 20))
    arcs, codes, _ = operator_arcsets(A, rng.integers(0, 2, 20), 20)
    for r in borrowed_table(x, rng.integers(0, 2, 20), {"A": arcs["A"]}, {"A": None}, rng, n_perm=99):
        assert r["p_upper"] >= 1 / 100 and r["p_lower"] >= 1 / 100


def test_maxT_is_conservative():
    from ca_core import maxT_adjust
    rng = np.random.default_rng(9); draws = rng.normal(size=(999, 6)); obs = np.array([3.5, 0.1, -2.2, 0.0, 1.0, -0.4])
    z, p_raw, p_maxT, bh = maxT_adjust(obs, draws)
    assert np.all(p_maxT >= p_raw - 1e-12) and np.all(bh >= p_raw - 1e-12) and np.all(p_raw >= 1 / 1000)
