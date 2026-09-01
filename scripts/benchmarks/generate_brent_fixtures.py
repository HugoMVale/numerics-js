# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy Brent reference values for the optimize.brent test fixture.

Run with: uv run scripts/benchmarks/generate_brent_fixtures.py

Each case here must have a matching hand-written objective in
tests/optimize/brent.scipy.test.ts (same id, math, and bounds).
"""

import json
import math
from pathlib import Path

from scipy.optimize import minimize_scalar

# Shared with tests/optimize/brent.scipy.test.ts so both solvers use the same
# requested x tolerance and their evaluation counts are comparable.
TOL_X = 1e-8

CASES = [
    {
        "id": "quadratic",
        "description": "(x - 2)^2 on [-5, 5]",
        "f": lambda x: (x - 2.0) ** 2,
        "xa": -5.0,
        "xb": 5.0,
    },
    {
        "id": "quartic",
        "description": "x^4 - x + 1 on [-3, 3]",
        "f": lambda x: x**4 - x + 1.0,
        "xa": -3.0,
        "xb": 3.0,
    },
    {
        "id": "absolute_value",
        "description": "abs(x - 0.5) on [-1, 2]",
        "f": lambda x: abs(x - 0.5),
        "xa": -1.0,
        "xb": 2.0,
    },
    {
        "id": "oscillatory_local_minimum",
        "description": "sin(5 x) + (x - 1)^2 on [0, 2]",
        "f": lambda x: math.sin(5.0 * x) + (x - 1.0) ** 2,
        "xa": 0.0,
        "xb": 2.0,
    },
    {
        "id": "sharp_minimum",
        "description": "exp(50 (x - 0.3)^2) on [0, 1]",
        "f": lambda x: math.exp(50.0 * (x - 0.3) ** 2),
        "xa": 0.0,
        "xb": 1.0,
    },
]


def main() -> None:
    results = []
    for case in CASES:
        result = minimize_scalar(
            case["f"],
            bracket=(case["xa"], (case["xa"] + case["xb"]) / 2, case["xb"]),
            method="brent",
            options={"xtol": TOL_X},
        )
        if not result.success:
            raise RuntimeError(f"SciPy Brent failed for {case['id']}: {result.message}")
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "xa": case["xa"],
                "xb": case["xb"],
                "tolX": TOL_X,
                "scipyX": result.x,
                "scipyFx": result.fun,
                "scipyEvaluations": result.nfev,
            }
        )

    out_path = Path(__file__).resolve().parents[2] / "tests" / "optimize" / "fixtures" / "brent.scipy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()