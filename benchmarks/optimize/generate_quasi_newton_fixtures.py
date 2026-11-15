# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy", "numpy"]
# ///
"""Generate reference values for the optimize.quasiNewton benchmark fixture.

Run with: uv run benchmarks/optimize/generate_quasi_newton_fixtures.py

The reference is scipy.optimize.minimize, using the closest SciPy algorithm for each
solver mode:

- BFGS with the analytic gradient: line-search quasi-Newton method with BFGS updates of
  the inverse Hessian (cf. globalMethod "line-search", bfgsMethod "inverse").
- BFGS with finite-difference gradient (SciPy's default when `jac` is not given): cf. the
  default usage of the solver, with only the function available.
- dogleg with analytic gradient and Hessian: dogleg trust-region method with exact
  Hessian (cf. globalMethod "dogleg" with `hess`). Only for problems with a Hessian.

Each problem here must have a matching objective, gradient, Hessian and starting
point in tests/optimize/gradientTestFunctions.ts (same id and math).
"""

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    return np.array(
        [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    )


def rosenbrock_hess(x):
    return np.array(
        [[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]], [-400 * x[0], 200.0]]
    )


def wood(x):
    return (
        100 * (x[0] ** 2 - x[1]) ** 2
        + (x[0] - 1) ** 2
        + (x[2] - 1) ** 2
        + 90 * (x[2] ** 2 - x[3]) ** 2
        + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
        + 19.8 * (x[1] - 1) * (x[3] - 1)
    )


def wood_grad(x):
    return np.array(
        [
            400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1),
            -200 * (x[0] ** 2 - x[1]) + 20.2 * (x[1] - 1) + 19.8 * (x[3] - 1),
            2 * (x[2] - 1) + 360 * x[2] * (x[2] ** 2 - x[3]),
            -180 * (x[2] ** 2 - x[3]) + 20.2 * (x[3] - 1) + 19.8 * (x[1] - 1),
        ]
    )


def beale_terms(x):
    return (
        1.5 - x[0] + x[0] * x[1],
        2.25 - x[0] + x[0] * x[1] ** 2,
        2.625 - x[0] + x[0] * x[1] ** 3,
    )


def beale(x):
    return sum(t * t for t in beale_terms(x))


def beale_grad(x):
    a, b, c = beale_terms(x)
    return np.array(
        [
            2 * a * (x[1] - 1) + 2 * b * (x[1] ** 2 - 1) + 2 * c * (x[1] ** 3 - 1),
            2 * a * x[0] + 4 * b * x[0] * x[1] + 6 * c * x[0] * x[1] ** 2,
        ]
    )


A = np.array([[10.0, 2, 0], [2, 5, 1], [0, 1, 1]])
B = np.array([1.0, 2, 3])


def quadratic(x):
    return 0.5 * x @ A @ x - B @ x


def quadratic_grad(x):
    return A @ x - B


def quadratic_hess(x):
    return A.copy()


CASES = [
    {
        "id": "rosenbrock",
        "f": rosenbrock,
        "grad": rosenbrock_grad,
        "hess": rosenbrock_hess,
        "x0": [-1.2, 1.0],
    },
    {"id": "wood", "f": wood, "grad": wood_grad, "hess": None, "x0": [-3.0, -1.0, -3.0, -1.0]},
    {"id": "beale", "f": beale, "grad": beale_grad, "hess": None, "x0": [1.0, 1.0]},
    {
        "id": "quadratic",
        "f": quadratic,
        "grad": quadratic_grad,
        "hess": quadratic_hess,
        "x0": [1.0, 1.0, 1.0],
    },
]

# Gradient tolerance (`gtol`, unscaled infinity norm). Finite-difference gradients are only
# accurate to about 1e-8, so the tolerance must be looser.
GTOL = {"analytic": 1e-10, "finite-difference": 1e-6}


def reference(case: dict, method: str, derivatives: str) -> dict:
    options = {"gtol": GTOL[derivatives], "maxiter": 1000}
    kwargs: dict = {"method": method, "options": options}
    if derivatives == "analytic":
        kwargs["jac"] = case["grad"]
    if method == "dogleg":
        kwargs["hess"] = case["hess"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(case["f"], np.array(case["x0"]), **kwargs)
    if not r.success:
        raise RuntimeError(f"SciPy {method} failed for {case['id']}: {r.message}")
    return {
        "id": case["id"],
        "method": method,
        "derivatives": derivatives,
        "x0": case["x0"],
        "gtol": GTOL[derivatives],
        "x": r.x.tolist(),
        "fx": float(r.fun),
        # for finite differences, `nfev` includes the evaluations used by the gradient
        "evaluations": int(r.nfev),
        "evaluationsGradient": int(r.njev),
        "evaluationsHessian": int(getattr(r, "nhev", 0)),
        "iterations": int(r.nit),
    }


def main() -> None:
    results = []
    for case in CASES:
        results.append(reference(case, "BFGS", "analytic"))
        results.append(reference(case, "BFGS", "finite-difference"))
        if case["hess"] is not None:
            results.append(reference(case, "dogleg", "analytic"))

    path = Path(__file__).resolve().parent / "quasiNewton.scipy.json"
    path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {path}")


if __name__ == "__main__":
    main()
