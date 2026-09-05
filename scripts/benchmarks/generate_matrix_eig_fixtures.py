# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the Matrix.eig() test fixture.

Run with: uv run scripts/benchmarks/generate_matrix_eig_fixtures.py

Each case here must have a matching matrix in
tests/linalg/Matrix.eig.scipy.test.ts (same id and entries).
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig


CASES = [
    {
        "id": "well_conditioned",
        "description": "well-conditioned symmetric 3x3 matrix",
        "matrix": [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]],
    },
    {
        "id": "complex_pair",
        "description": "4x4 matrix with two distinct complex-conjugate pairs",
        "matrix": [
            [0.0, -2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0, 0.0],
        ],
    },
    {
        "id": "nonsymmetric",
        "description": "nonsymmetric 4x4 matrix with distinct real eigenvalues",
        "matrix": [
            [4.0, 1.0, 2.0, 0.0],
            [1.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, 5.0, 2.0],
            [0.0, 1.0, 2.0, 6.0],
        ],
    },
    {
        "id": "hilbert_5x5",
        "description": "5x5 Hilbert matrix",
        "matrix": [[1.0 / (i + j + 1) for j in range(5)] for i in range(5)],
    },
]


def main() -> None:
    results = []
    for case in CASES:
        matrix = np.array(case["matrix"], dtype=float)
        values, vectors = eig(matrix)
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "matrix": case["matrix"],
                "scipyValues": [{"re": float(value.real), "im": float(value.imag)} for value in values],
                "scipyVectorsRe": vectors.real.T.tolist(),
                "scipyVectorsIm": vectors.imag.T.tolist(),
            }
        )

    out_path = Path(__file__).resolve().parents[2] / "tests" / "linalg" / "fixtures" / "Matrix.eig.scipy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()