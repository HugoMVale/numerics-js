# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "scipy"]
# ///
"""Generate the mpmath reference fixture for the special.besselJ benchmark.

Run with: uv run benchmarks/special/generate_besselJ_mpmath_fixtures.py

The reference is mpmath.besselj at 60 digits, stored as a double-double pair
(hi, lo) as in generate_erf_fixtures.py. SciPy (AMOS) values are stored
alongside for the benchmark report. Each case id must match
benchmarks/special/besselJ.mpmath.test.ts.
"""

import json
from pathlib import Path

import mpmath
from scipy.special import jv

mpmath.mp.dps = 60


def grid(orders, args, keep):
    return [(n, x) for n in orders for x in args if keep(n, x)]


CASES = [
    {
        "id": "small_x",
        "description": "|x| < 1, power-series region, tiny and moderate arguments",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50],
            [0.0, 1e-300, 1e-100, 1e-10, 1e-3, 0.1, 0.5, 0.99],
            lambda n, x: True,
        ),
    },
    {
        "id": "decaying_n_ge_x",
        "description": "n >= x >= 1, where J_n decays monotonically (Miller recurrence)",
        "points": grid(
            [1, 2, 3, 5, 10, 20, 50, 100, 200, 500],
            [1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
            lambda n, x: n >= x,
        ),
    },
    {
        "id": "oscillatory_n_lt_x",
        "description": "1 <= n < x <= 5000, oscillatory region (trapezoid + forward recurrence)",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500],
            [1.5, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0],
            lambda n, x: n < x,
        ),
    },
    {
        "id": "turning_point",
        "description": "x within a few n^(1/3) of the transition point x = n",
        "points": [
            (n, round(n + d * n ** (1 / 3), 6))
            for n in [10, 50, 100, 300, 1000]
            for d in [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0]
        ],
    },
    {
        "id": "large_x",
        "description": "1e4 <= x <= 1e5 (beyond the old fixed sampling cap)",
        "points": grid([0, 1, 2, 10, 100], [1e4, 2e4, 5e4, 1e5], lambda n, x: True),
    },
    {
        "id": "negative_x",
        "description": "negative arguments, J_n(-x) = (-1)^n J_n(x)",
        "points": grid([0, 1, 2, 3, 10], [-0.5, -2.5, -10.0, -100.0], lambda n, x: True),
    },
]


def mpmath_reference(n: int, x: float) -> dict[str, float]:
    value = mpmath.besselj(n, mpmath.mpf(x))
    hi = float(value)
    return {"hi": hi, "lo": float(value - mpmath.mpf(hi))}


def main() -> None:
    results = []
    for case in CASES:
        points = [
            {
                "n": n,
                "x": x,
                "mpmath": mpmath_reference(n, x),
                "scipy": float(jv(float(n), x)),
            }
            for n, x in case["points"]
        ]
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "points": points,
            }
        )

    out_path = Path(__file__).resolve().with_name("besselJ.mpmath.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
