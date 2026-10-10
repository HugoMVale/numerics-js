# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the special.besselJ test fixture.

Run with: uv run benchmarks/special/generate_besselJ_fixtures.py

The reference is scipy.special.jv (AMOS) evaluated at integer orders. Each
group of (n, x) points here must have a matching tolerance entry in
benchmarks/special/besselJ.scipy.test.ts (same id).
"""

import json
from pathlib import Path

from scipy.special import jv

# Results below this magnitude are skipped: they are denormal or nearly so, where
# neither implementation keeps full relative accuracy. SciPy also flushes some
# values far above the underflow threshold to exactly 0 (e.g. J_200(5) ~ 4.8e-296),
# so a zero reference is only kept where J_n(x) is exactly zero (n > 0, x = 0).
TINY = 1e-290


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


def main() -> None:
    results = []
    for case in CASES:
        points = []
        for n, x in case["points"]:
            ref = float(jv(float(n), x))
            if abs(ref) < TINY and not (ref == 0.0 and x == 0.0):
                continue
            points.append({"n": n, "x": x, "scipy": ref})
        results.append(
            {"id": case["id"], "description": case["description"], "points": points}
        )

    out_path = Path(__file__).resolve().with_name("besselJ.scipy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
