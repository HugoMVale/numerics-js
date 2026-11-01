# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the Matrix SVD family test fixture.

Run with: uv run benchmarks/linalg/generate_matrix_svd_fixtures.py

References: `scipy.linalg.svd` (LAPACK gesdd) for the singular values,
`scipy.linalg.lstsq(lapack_driver="gelsd")` for the minimum-norm least-squares
solution and the effective rank, `scipy.linalg.pinv` for the pseudoinverse,
and `numpy.linalg.matrix_rank` / `numpy.linalg.cond` for the rank and the
2-norm condition number.

The matrices are stored in the generated JSON fixture, which is read by
benchmarks/linalg/Matrix.svd.scipy.test.ts; no case definitions are
duplicated in the TypeScript test. Rerun this script after changing CASES.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq, pinv, svd


def kahan_like(n: int) -> list[list[float]]:
    """Unit upper triangular matrix with -1 above the diagonal (cond_2 ~ 2^n)."""
    return [[1.0 if i == j else (-1.0 if j > i else 0.0) for j in range(n)] for i in range(n)]


def lauchli(d: float) -> list[list[float]]:
    return [[1.0, 1.0, 1.0], [d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, d]]


CASES = [
    {
        "id": "well_conditioned",
        "description": "well-conditioned 3x3 matrix",
        "matrix": [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]],
        "rhs": [1.0, -2.0, 3.0],
    },
    {
        "id": "tall",
        "description": "3x2 tall rectangular matrix",
        "matrix": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "rhs": [1.0, 0.0, -1.0],
    },
    {
        "id": "wide",
        "description": "2x3 wide rectangular matrix",
        "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "rhs": [1.0, 2.0],
    },
    {
        "id": "tall_unequal_columns",
        "description": "8x4 tall matrix whose column norms differ by orders of magnitude",
        "matrix": [
            [1.0 * (i + 1) ** 0.5, 1e3 * np.sin(i + 1.0), 1e-2 * (i - 3.5) ** 2, np.cos(2.0 * i) + 2.0]
            for i in range(8)
        ],
        "rhs": [float(np.sin(0.7 * i)) for i in range(8)],
    },
    {
        "id": "hilbert_8",
        "description": "8x8 Hilbert matrix (cond_2 = 1.5e10)",
        "matrix": [[1.0 / (i + j + 1) for j in range(8)] for i in range(8)],
        "rhs": [1.0] * 8,
    },
    {
        "id": "hilbert_10x6",
        "description": "10x6 Hilbert-type matrix (ill-conditioned, overdetermined)",
        "matrix": [[1.0 / (i + j + 1) for j in range(6)] for i in range(10)],
        "rhs": [float(np.cos(i)) for i in range(10)],
    },
    {
        "id": "rank_deficient_tall",
        "description": "6x4 matrix of rank 2 (dependent columns)",
        "matrix": [[float(a), float(b), float(a + b), float(a - 2 * b)] for a, b in [(1, 2), (2, 1), (3, 5), (0, 1), (4, 4), (1, 0)]],
        "rhs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    {
        "id": "rank_deficient_wide",
        "description": "3x6 matrix of rank 2 (dependent rows)",
        "matrix": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 3.0, 3.0, 5.0, 5.0, 7.0]],
        "rhs": [1.0, 1.0, 2.0],
    },
    {
        "id": "wide_dynamic_range",
        "description": "4x4 matrix with entries spanning twelve orders of magnitude",
        "matrix": [[1e-12, 2.0, -1.0, 0.5], [3e8, -1e-4, 2.0, 1.0], [1e-3, 4e4, 1.0, -2e2], [2e6, 0.5, -3e-6, 1e2]],
        "rhs": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "id": "lauchli",
        "description": "Lauchli 4x3 matrix with d = 1e-8 (squaring A loses the small singular values)",
        "matrix": lauchli(1e-8),
        "rhs": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "id": "kahan_like_50",
        "description": "50x50 unit upper triangular matrix with -1 above the diagonal (numerical rank 49)",
        "matrix": kahan_like(50),
        "rhs": [1.0] * 50,
    },
]


def main() -> None:
    results = []
    for case in CASES:
        a = np.array(case["matrix"], dtype=float)
        b = np.array(case["rhs"], dtype=float)
        s = svd(a, compute_uv=False)
        # scipy's lstsq defaults to cond = eps, whereas numpy.linalg.lstsq, scipy.linalg.pinv and
        # numpy.linalg.matrix_rank (and Matrix.lstsq) use max(rows, cols) * eps: pass it explicitly.
        cond = max(a.shape) * np.finfo(float).eps
        x, _, rank_gelsd, _ = lstsq(a, b, cond=cond, lapack_driver="gelsd")
        # The pseudoinverse of the large case is not stored, to keep the fixture small.
        p = pinv(a) if max(a.shape) <= 10 else None
        # Reconstruction scale of the reference itself, to size the tolerance of the test.
        u, sv, vt = svd(a, full_matrices=False)
        scipy_residual = float(np.linalg.norm(u @ np.diag(sv) @ vt - a) / np.linalg.norm(a))
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "matrix": case["matrix"],
                "rhs": case["rhs"],
                "scipyS": s.tolist(),
                "scipyReconstructionResidual": scipy_residual,
                "scipyRank": int(np.linalg.matrix_rank(a)),
                "scipyCond2": float(np.linalg.cond(a)) if np.linalg.matrix_rank(a) == min(a.shape) else None,
                "scipyLstsqX": x.tolist(),
                "scipyLstsqRank": int(rank_gelsd),
                "scipyPinv": p.tolist() if p is not None else None,
            }
        )
    out = Path(__file__).with_name("Matrix.svd.scipy.json")
    out.write_text(json.dumps(results, indent=4) + "\n")
    print(f"wrote {len(results)} cases to {out}")


if __name__ == "__main__":
    main()
