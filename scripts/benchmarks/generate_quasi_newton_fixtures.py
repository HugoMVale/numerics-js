# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate SciPy hybr reference values for the roots.quasiNewton test fixture.

Run with: uv run scripts/benchmarks/generate_quasi_newton_fixtures.py

Each case here must mirror the function and starting point in
tests/roots/quasiNewton.scipy.test.ts, which in turn uses
tests/roots/testFunctions.ts.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import root

TOL_X = 1e-10
TOL_F = 0.0
MAX_FEVAL = 1000
RESIDUAL_TOLERANCE = 1e-7


def rosenbrock(x: np.ndarray) -> np.ndarray:
    values = np.zeros_like(x)
    for i in range(0, len(x), 2):
        values[i] = 1.0 - x[i]
        values[i + 1] = 10.0 * (x[i + 1] - x[i] ** 2)
    return values


def powell_singular(x: np.ndarray) -> np.ndarray:
    values = np.zeros_like(x)
    for i in range(0, len(x), 4):
        values[i] = x[i] - 10.0 * x[i + 1]
        values[i + 1] = np.sqrt(5.0) * (x[i + 2] - x[i + 3])
        values[i + 2] = (x[i + 1] - 2.0 * x[i + 2]) ** 2
        values[i + 3] = np.sqrt(10.0) * (x[i] - x[i + 3]) ** 2
    return values


def trigonometric(x: np.ndarray) -> np.ndarray:
    """Trigonometric function (Moré, Garbow & Hillstrom 1981, problem 26).

    f_i(x) = n - sum_j cos(x_j) + i * (1 - cos(x_i)) - sin(x_i), i = 1, ..., n.
    """
    n = len(x)
    sum_cos = np.cos(x).sum()
    return np.array([n - sum_cos + (i + 1) * (1.0 - np.cos(x[i])) - np.sin(x[i]) for i in range(n)])


def example65(x: np.ndarray) -> np.ndarray:
    x1, x2 = x
    return np.array([x1**2 + x2**2 - 2.0, np.exp(x1 - 1.0) + x2**3 - 2.0])


CASES = [
    {
        "id": "rosenbrock",
        "description": "Extended Rosenbrock function (4 variables)",
        "f": rosenbrock,
        "x0": np.array([-1.2, 1.0, -1.2, 1.0]),
    },
    {
        "id": "powell_singular",
        "description": "Extended Powell singular function (4 variables)",
        "f": powell_singular,
        "x0": np.array([3.0, -1.0, 0.0, 1.0]),
    },
    {
        "id": "trigonometric_5",
        "description": "Trigonometric function (5 variables)",
        "f": trigonometric,
        "x0": np.full(5, 0.01),
    },
    {
        "id": "trigonometric",
        "description": "Trigonometric function (10 variables)",
        "f": trigonometric,
        "x0": np.full(10, 0.01),
    },
    {
        "id": "example65",
        "description": "Example 6.5 of Dennis & Schnabel (1996), x0 = [1.5, 1]",
        "f": example65,
        "x0": np.array([1.5, 1.0]),
    },
]


def main() -> None:
    results = []
    for case in CASES:
        result = root(
            case["f"],
            case["x0"],
            method="hybr",
            options={"xtol": TOL_X, "maxfev": MAX_FEVAL},
        )
        residual_norm = float(np.linalg.norm(result.fun, ord=np.inf))
        if residual_norm > RESIDUAL_TOLERANCE:
            if not result.success:
                raise RuntimeError(f"SciPy hybr failed for {case['id']}: {result.message}")
            raise RuntimeError(
                f"SciPy hybr residual for {case['id']} was {residual_norm}, "
                f"above {RESIDUAL_TOLERANCE}"
            )
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "x0": case["x0"].tolist(),
                # f(x0) lets the TypeScript test verify that its test function mirrors this one.
                "fx0": case["f"](case["x0"]).tolist(),
                # `tolx` and `tolf` are settings for the numerics-js solver (SciPy's hybr has no
                # `tolf`); `tolx` is also the `xtol` passed to SciPy.
                "tolx": TOL_X,
                "tolf": TOL_F,
                "scipyX": result.x.tolist(),
                "scipyFx": result.fun.tolist(),
                "scipyEvaluations": result.nfev,
                "scipySuccess": bool(result.success),
                "scipyMessage": str(result.message),
            }
        )

    out_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "roots"
        / "fixtures"
        / "quasiNewton.scipy.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()