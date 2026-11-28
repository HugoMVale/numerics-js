# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy", "numpy"]
# ///
"""Generate SciPy/LAPACK reference values for BandedMatrix, tridiagonalLU and solveTridiagonal.

Run with: uv run benchmarks/linalg/generate_banded_fixtures.py

Each case here must have a matching entry in benchmarks/linalg/BandedMatrix.scipy.test.ts
(same id). The matrices themselves are stored in the fixture (generated with a seeded NumPy
generator), so the test does not need to reproduce the random numbers.

References: `scipy.linalg.solve_banded` for the solution, `scipy.linalg.lapack.dgbtrf` for the
LU factors and row interchanges of a general band matrix, `dgttrf` for the tridiagonal ones.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.linalg.lapack import dgbtrf, dgttrf

# id, n, kl, ku, diagonal scale (small = the diagonal is not dominant, so rows are interchanged)
CASES = [
    ("n1", 1, 0, 0, 1.0),
    ("diagonal", 6, 0, 0, 1.0),
    ("lower_bidiagonal", 7, 1, 0, 0.1),
    ("upper_bidiagonal", 7, 0, 1, 0.1),
    ("tridiagonal_dominant", 12, 1, 1, 4.0),
    ("tridiagonal_pivoting", 15, 1, 1, 0.02),
    ("pentadiagonal_pivoting", 20, 2, 2, 0.05),
    ("kl3_ku1_pivoting", 25, 3, 1, 0.05),
    ("kl1_ku4_pivoting", 25, 1, 4, 0.05),
    ("wide_band_pivoting", 30, 6, 5, 0.05),
    ("full_band", 8, 7, 7, 0.1),
    ("large_pivoting", 200, 4, 3, 0.05),
    ("large_dominant", 300, 2, 2, 5.0),
]


def make_case(case_id, n, kl, ku, diag_scale, seed):
    rng = np.random.default_rng(seed)
    ab = np.zeros((kl + ku + 1, n))
    dense = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            v = rng.uniform(-1.0, 1.0)
            if i == j:
                v *= diag_scale
            dense[i, j] = v
            ab[ku + i - j, j] = v
    b = rng.uniform(-1.0, 1.0, size=n)
    return ab, dense, b


def normalize_ipiv(ipiv, n):
    """LAPACK pivots are 1-based; some SciPy wrappers return them 0-based. Return 0-based."""
    ipiv = np.asarray(ipiv, dtype=int)
    if ipiv[-1] == n:  # 1-based: the last pivot is the last row
        ipiv = ipiv - 1
    assert ipiv[-1] == n - 1
    assert all(ipiv[j] >= j for j in range(n))
    return ipiv.tolist()


def main() -> None:
    results = []
    for index, (case_id, n, kl, ku, diag_scale) in enumerate(CASES):
        ab, dense, b = make_case(case_id, n, kl, ku, diag_scale, seed=1000 + index)
        x = solve_banded((kl, ku), ab, b)

        ab_ext = np.zeros((2 * kl + ku + 1, n))
        ab_ext[kl:, :] = ab
        lu, ipiv, info = dgbtrf(ab_ext, kl, ku)
        assert info == 0, f"{case_id}: dgbtrf info = {info}"

        cond1 = float(np.linalg.cond(dense, 1))
        residual = float(np.linalg.norm(dense @ x - b, np.inf) / np.linalg.norm(b, np.inf))
        item = {
            "id": case_id,
            "n": n,
            "kl": kl,
            "ku": ku,
            "ab": ab.tolist(),
            "b": b.tolist(),
            "x": x.tolist(),
            "cond1": cond1,
            "scipyResidual": residual,
            "lu": lu.tolist(),
            "ipiv": normalize_ipiv(ipiv, n),
        }
        if kl == 1 and ku == 1:
            dl = np.diag(dense, -1).copy()
            d = np.diag(dense).copy()
            du = np.diag(dense, 1).copy()
            gdl, gd, gdu, gdu2, gipiv, ginfo = dgttrf(dl, d, du)
            assert ginfo == 0
            item["tridiagonal"] = {
                "dl": dl.tolist(),
                "d": d.tolist(),
                "du": du.tolist(),
                "luDl": gdl.tolist(),
                "luD": gd.tolist(),
                "luDu": gdu.tolist(),
                "luDu2": gdu2.tolist(),
                "ipiv": normalize_ipiv(gipiv, n),
            }
        results.append(item)

    out_path = Path(__file__).resolve().with_name("BandedMatrix.scipy.json")
    out_path.write_text(json.dumps(results, indent=1) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
