# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference fixtures for the adaptive Runge-Kutta solvers.

Run with: uv run scripts/benchmarks/generate_runge_kutta_adaptive_fixtures.py

Each case must have a matching derivative function in
tests/ode/rungeKuttaAdaptive.scipy.test.ts (same id, math, span, and state).
"""

import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp

ATOL = 1e-8
RTOL = 1e-8

CASES = [
    {
        "id": "exponential_decay",
        "description": "y' = -y on [0, 5]",
        "f": lambda _t, y: [-y[0]],
        "t0": 0.0,
        "tEnd": 5.0,
        "y0": [1.0],
    },
    {
        "id": "logistic_growth",
        "description": "y' = 10 y (1 - y) on [0, 1]",
        "f": lambda _t, y: [10.0 * y[0] * (1.0 - y[0])],
        "t0": 0.0,
        "tEnd": 1.0,
        "y0": [0.01],
    },
    {
        "id": "harmonic_oscillator",
        "description": "x' = v, v' = -x on [0, 2 pi]",
        "f": lambda _t, y: [y[1], -y[0]],
        "t0": 0.0,
        "tEnd": 2.0 * math.pi,
        "y0": [1.0, 0.0],
    },
    {
        "id": "backward_exponential",
        "description": "y' = -2 y from t = 2 to t = -1",
        "f": lambda _t, y: [-2.0 * y[0]],
        "t0": 2.0,
        "tEnd": -1.0,
        "y0": [math.exp(-4.0)],
    },
]

METHODS = {"rk23": "RK23", "rk45": "RK45"}


def main() -> None:
    results = []
    for case in CASES:
        scipy = {}
        for method_id, scipy_method in METHODS.items():
            result = solve_ivp(
                case["f"],
                (case["t0"], case["tEnd"]),
                case["y0"],
                method=scipy_method,
                atol=ATOL,
                rtol=RTOL,
            )
            if not result.success:
                raise RuntimeError(f"SciPy {scipy_method} failed for {case['id']}: {result.message}")
            scipy[method_id] = {
                "finalY": result.y[:, -1].tolist(),
                "evaluations": result.nfev,
            }

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "t0": case["t0"],
                "tEnd": case["tEnd"],
                "y0": case["y0"],
                "atol": ATOL,
                "rtol": RTOL,
                "scipy": scipy,
            }
        )

    out_path = Path(__file__).resolve().parents[2] / "tests" / "ode" / "fixtures" / "rungeKuttaAdaptive.scipy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()