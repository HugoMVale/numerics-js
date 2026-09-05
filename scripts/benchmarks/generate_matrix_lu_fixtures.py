# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the Matrix.lu() test fixture.

Run with: uv run scripts/benchmarks/generate_matrix_lu_fixtures.py

Each case here must have a matching matrix in
tests/linalg/Matrix.lu.scipy.test.ts (same id and entries).
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu


CASES = [
    {
        "id": "well_conditioned",
        "description": "well-conditioned 3x3 matrix",
        "matrix": [
            [4.0, 3.0, 0.0],
            [3.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ],
    },
    {
        "id": "row_pivoting",
        "description": "3x3 matrix requiring row pivoting",
        "matrix": [
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
    },
    {
        "id": "scaled_pivots",
        "description": "4x4 matrix with mixed pivot scales",
        "matrix": [
            [1e-4, 2.0, -1.0, 0.5],
            [3.0, -1.0, 2.0, 1.0],
            [1.0, 4.0, 1.0, -2.0],
            [2.0, 0.5, -3.0, 1.0],
        ],
    },
    {
        "id": "hilbert_6x6",
        "description": "6x6 Hilbert matrix",
        "matrix": [[1.0 / (i + j + 1) for j in range(6)] for i in range(6)],
    },
    {
        "id": "wide_dynamic_range",
        "description": "4x4 matrix with entries spanning twelve orders of magnitude",
        "matrix": [
            [1e-12, 2.0, -1.0, 0.5],
            [3e8, -1e-4, 2.0, 1.0],
            [1e-3, 4e4, 1.0, -2e2],
            [2e6, 0.5, -3e-6, 1e2],
        ],
    },
]


def main() -> None:
    results = []
    for case in CASES:
        matrix = np.array(case["matrix"], dtype=float)
        permutation, lower, upper = lu(matrix)

        # SciPy returns A = P * L * U. The library returns P_local * A = L * U,
        # so each local permutation row selects the source row in P.T.
        local_permutation = np.argmax(permutation.T, axis=1).tolist()
        local_permuted = matrix[local_permutation, :]
        scipy_residual = np.linalg.norm(local_permuted - lower @ upper) / np.linalg.norm(matrix)
        inversions = sum(
            local_permutation[i] > local_permutation[j]
            for i in range(len(local_permutation))
            for j in range(i + 1, len(local_permutation))
        )

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "matrix": case["matrix"],
                "scipyL": lower.tolist(),
                "scipyU": upper.tolist(),
                "scipyPerm": local_permutation,
                "scipySign": 1 if inversions % 2 == 0 else -1,
                "scipyResidual": float(scipy_residual),
            }
        )

    out_path = Path(__file__).resolve().parents[2] / "tests" / "array" / "fixtures" / "Matrix.lu.scipy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()