# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the Matrix.qrPivot() test fixture.

Run with: uv run benchmarks/linalg/generate_matrix_qr_pivot_fixtures.py

The matrices are stored in the generated JSON fixture, which is read by
benchmarks/linalg/Matrix.qrPivot.scipy.test.ts; no case definitions are
duplicated in the TypeScript test. Rerun this script after changing CASES.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr


CASES = [
    {
        "id": "well_conditioned",
        "description": "well-conditioned 3x3 matrix",
        "matrix": [
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ],
    },
    {
        "id": "tall",
        "description": "3x2 tall rectangular matrix",
        "matrix": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    },
    {
        "id": "tall_unequal_columns",
        "description": "8x4 tall matrix whose column norms differ by orders of magnitude",
        "matrix": [
            [1.0 * (i + 1) ** 0.5, 1e3 * np.sin(i + 1.0), 1e-2 * (i - 3.5) ** 2, np.cos(2.0 * i) + 2.0]
            for i in range(8)
        ],
    },
    {
        "id": "wide",
        "description": "2x3 wide rectangular matrix",
        "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    },
    {
        "id": "hilbert_6x4",
        "description": "6x4 Hilbert matrix (ill-conditioned)",
        "matrix": [[1.0 / (i + j + 1) for j in range(4)] for i in range(6)],
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
    {
        "id": "rank_deficient",
        "description": "4x3 matrix with dependent columns (rank 2)",
        "matrix": [[1.0, 2.0, 0.0], [2.0, 4.0, 1.0], [3.0, 6.0, 0.0], [4.0, 8.0, 1.0]],
    },
]


def main() -> None:
    results = []
    for case in CASES:
        matrix = np.array(case["matrix"], dtype=float)
        (compact, tau), upper, perm = qr(matrix, mode="raw", pivoting=True)
        q, r, _ = qr(matrix, mode="economic", pivoting=True)
        reconstruction_residual = np.linalg.norm(matrix[:, perm] - q @ r) / np.linalg.norm(matrix)

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "matrix": matrix.tolist(),
                "scipyQR": compact.tolist(),
                "scipyTau": tau.tolist(),
                "scipyPerm": [int(p) for p in perm],
                "scipyR": r.tolist(),
                "scipyResidual": float(reconstruction_residual),
            }
        )

    out_path = Path(__file__).resolve().with_name("Matrix.qrPivot.scipy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
