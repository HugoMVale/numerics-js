# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference fixtures for the adaptive TR-BDF2 solver.

Run with: uv run benchmarks/ode/generate_tr_bdf2_adaptive_fixtures.py

SciPy has no TR-BDF2 solver, so the reference solution of each case is a very tight
`Radau` (order 5) integration, and the final state of `trBdf2Adaptive` is compared
with it. SciPy's `Radau`, `BDF` and `LSODA` solvers are also run at the benchmark
tolerances; their cost counters are recorded for context only.

Each case must have a matching derivative function in
benchmarks/ode/trBdf2Adaptive.scipy.test.ts (same id, math, span, and state).
"""

import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp

REFERENCE_ATOL = 1e-12
REFERENCE_RTOL = 1e-11

# Tolerance settings used for every case: the library defaults, which are also the defaults of
# SciPy, MATLAB and Julia, and a tighter setting.
SETTINGS = [
    {"name": "default", "atol": 1e-6, "rtol": 1e-3},
    {"name": "tight", "atol": 1e-8, "rtol": 1e-5},
]

LAMBDA = -1e5


def prothero_robinson(t, y):
    return [LAMBDA * (y[0] - math.sin(t)) + math.cos(t)]


def robertson(_t, y):
    a, b, c = y
    return [
        -0.04 * a + 1e4 * b * c,
        0.04 * a - 1e4 * b * c - 3e7 * b * b,
        3e7 * b * b,
    ]


def van_der_pol(_t, y):
    mu = 1000.0
    return [y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]


def stiff_linear(_t, y):
    # y1' = -1000 y1 + y2,  y2' = -y2 + y3,  y3' = -0.1 y3
    return [-1000.0 * y[0] + y[1], -y[1] + y[2], -0.1 * y[2]]


CASES = [
    {
        "id": "prothero_robinson",
        "description": "y' = -1e5 (y - sin t) + cos t, y(0) = 1, on [0, 2]",
        "f": prothero_robinson,
        "t0": 0.0,
        "tEnd": 2.0,
        "y0": [1.0],
    },
    {
        "id": "robertson",
        "description": "Robertson chemical kinetics on [0, 1e4]",
        "f": robertson,
        "t0": 0.0,
        "tEnd": 1e4,
        "y0": [1.0, 0.0, 0.0],
    },
    {
        "id": "van_der_pol_mu1000",
        "description": "Van der Pol oscillator, mu = 1000, on [0, 1500]",
        "f": van_der_pol,
        "t0": 0.0,
        "tEnd": 1500.0,
        "y0": [2.0, 0.0],
    },
    {
        "id": "stiff_linear_chain",
        "description": "Linear chain with rates -1000, -1, -0.1 on [0, 20]",
        "f": stiff_linear,
        "t0": 0.0,
        "tEnd": 20.0,
        "y0": [1.0, 1.0, 1.0],
    },
]

METHODS = {"radau": "Radau", "bdf": "BDF", "lsoda": "LSODA"}


def main() -> None:
    results = []
    for case in CASES:
        span = (case["t0"], case["tEnd"])

        reference = solve_ivp(
            case["f"], span, case["y0"], method="Radau", atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL
        )
        if not reference.success:
            raise RuntimeError(f"Reference Radau failed for {case['id']}: {reference.message}")

        settings = []
        for setting in SETTINGS:
            scipy = {}
            for method_id, scipy_method in METHODS.items():
                result = solve_ivp(
                    case["f"], span, case["y0"], method=scipy_method, atol=setting["atol"], rtol=setting["rtol"]
                )
                if not result.success:
                    raise RuntimeError(f"SciPy {scipy_method} failed for {case['id']}: {result.message}")
                scipy[method_id] = {
                    "finalY": result.y[:, -1].tolist(),
                    "steps": int(len(result.t) - 1),
                    "evaluations": int(result.nfev),
                    "jacobians": int(result.njev),
                    "decompositions": int(result.nlu),
                }
            settings.append({**setting, "scipy": scipy})

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "t0": case["t0"],
                "tEnd": case["tEnd"],
                "y0": case["y0"],
                "reference": {
                    "method": "Radau",
                    "atol": REFERENCE_ATOL,
                    "rtol": REFERENCE_RTOL,
                    "finalY": reference.y[:, -1].tolist(),
                },
                "settings": settings,
            }
        )

    out_path = Path(__file__).resolve().with_name("trBdf2Adaptive.scipy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
