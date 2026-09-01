# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy", "numpy"]
# ///
"""Generate SciPy Nelder-Mead reference values for the optimize.nelderMead test fixture.

Run with: uv run scripts/benchmarks/generate_nelder_mead_fixtures.py

Each case here must have a matching hand-written objective and initial point
in tests/optimize/nelderMead.scipy.test.ts (same id, math, and starting
point), mirroring TEST_FUNCTIONS_MULTIVAR in tests/optimize/testFunctions.ts.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Shared with tests/optimize/nelderMead.scipy.test.ts so both solvers use the
# same requested tolerances and their evaluation counts are comparable.
# fatol is set far below xatol so both solvers terminate on the x-tolerance
# criterion, avoiding any need to reconcile xatol with nelderMead's tolF.
TOL_X = 1e-8
TOL_F = 1e-14

N = 2  # matches the dimensionality used in tests/optimize/nelderMead.test.ts


def ellipsoid_coeffs(n: int) -> np.ndarray:
    denom = n - 1 if n > 1 else 1
    return np.array([1e6 ** (i / denom) for i in range(n)])


def zakharov_u(x: np.ndarray) -> float:
    return 0.5 * sum((i + 1) * x[i] for i in range(len(x)))


CASES = [
    {
        "id": "sphere",
        "description": "sum(x_i^2)",
        "f": lambda x: float(np.sum(x**2)),
        "x0": np.full(N, 5.0),
    },
    {
        "id": "ellipsoid",
        "description": "sum(c_i * x_i^2) with c_i = 1e6^(i/(n-1))",
        "f": lambda x: float(np.sum(ellipsoid_coeffs(len(x)) * x**2)),
        "x0": np.linspace(1.0, 2.0, N),
    },
    {
        "id": "rosenbrock",
        "description": "sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)",
        "f": lambda x: float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
        ),
        "x0": np.full(N, -1.2),
    },
    {
        "id": "zakharov",
        "description": "sum(x_i^2) + u^2 + u^4, u = 0.5*sum((i+1)*x_i)",
        "f": lambda x: float(np.sum(x**2) + zakharov_u(x) ** 2 + zakharov_u(x) ** 4),
        "x0": np.full(N, 1.5),
    },
]


def main() -> None:
    results = []
    for case in CASES:
        result = minimize(
            case["f"],
            case["x0"],
            method="Nelder-Mead",
            options={"xatol": TOL_X, "fatol": TOL_F, "adaptive": True},
        )
        if not result.success:
            raise RuntimeError(f"SciPy Nelder-Mead failed for {case['id']}: {result.message}")
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "x0": case["x0"].tolist(),
                "tolX": TOL_X,
                "tolF": TOL_F,
                "scipyX": result.x.tolist(),
                "scipyFx": result.fun,
                "scipyEvaluations": result.nfev,
            }
        )

    out_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "optimize"
        / "fixtures"
        / "nelderMead.scipy.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
