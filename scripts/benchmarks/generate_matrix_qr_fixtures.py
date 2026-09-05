# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the Matrix.qr() test fixture.

Run with: uv run scripts/benchmarks/generate_matrix_qr_fixtures.py

Each case here must have a matching matrix in
tests/linalg/Matrix.qr.scipy.test.ts (same id and entries).
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
        "id": "wide",
        "description": "2x3 wide rectangular matrix",
        "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
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
    {
        "id": "rank_deficient",
        "description": "3x2 matrix with dependent columns",
        "matrix": [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
    },
]


def main() -> None:
    results = []
    for case in CASES:
        matrix = np.array(case["matrix"], dtype=float)
        orthogonal, upper = qr(matrix, mode="full")
        reconstruction_residual = np.linalg.norm(matrix - orthogonal @ upper) / np.linalg.norm(matrix)
        orthogonality_residual = np.linalg.norm(orthogonal.T @ orthogonal - np.eye(matrix.shape[0]))

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "matrix": case["matrix"],
                "scipyQ": orthogonal.tolist(),
                "scipyR": upper.tolist(),
                "scipyResidual": float(reconstruction_residual),
                "scipyOrthogonalityResidual": float(orthogonality_residual),
            }
        )

    out_path = Path(__file__).resolve().parents[2] / "tests" / "linalg" / "fixtures" / "Matrix.qr.scipy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()