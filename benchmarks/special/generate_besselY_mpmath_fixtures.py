# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "scipy"]
# ///
"""Generate the mpmath reference fixture for the special.besselY benchmark.

Run with: uv run benchmarks/special/generate_besselY_mpmath_fixtures.py

The reference is mpmath.bessely at 60 digits, stored as a double-double pair
(hi, lo) as in generate_besselJ_mpmath_fixtures.py. SciPy (Cephes `y0`/`y1`
plus forward recurrence) values are stored alongside for the benchmark report.
Points whose true value overflows a double (|Y_n(x)| > 1e300) are dropped, since
no finite reference exists there. Each case id must match
benchmarks/special/besselY.mpmath.test.ts.
"""

import json
from pathlib import Path

import mpmath
from scipy.special import yn

mpmath.mp.dps = 60

OVERFLOW = 1e300


def grid(orders, args, keep):
    return [(n, x) for n in orders for x in args if keep(n, x)]


CASES = [
    {
        "id": "small_x",
        "description": "0 < x < 3, ascending-series region, tiny to moderate arguments",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50],
            [1e-300, 1e-100, 1e-10, 1e-3, 0.1, 0.5, 0.8936, 1.0, 1.5, 2.0, 2.5, 2.99],
            lambda n, x: True,
        ),
    },
    {
        "id": "mid_x",
        "description": "3 <= x < 20, Neumann-series region (includes zeros of Y_0 and Y_1)",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50],
            [3.0, 3.5, 3.9577, 4.0, 5.0, 5.4297, 7.5, 10.0, 12.5, 15.0, 17.5, 19.99],
            lambda n, x: True,
        ),
    },
    {
        "id": "large_x",
        "description": "x >= 20, Hankel-expansion region, up to 1e15 (O(1) cost)",
        "points": grid(
            [0, 1, 2, 5, 10, 100],
            [20.0, 25.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1e3, 1e4, 1e5, 1e6, 1e8, 1e12, 1e15],
            lambda n, x: True,
        ),
    },
    {
        "id": "growing_n_ge_x",
        "description": "n >= x >= 1, where |Y_n| grows with n (forward recurrence is stable)",
        "points": grid(
            [1, 2, 3, 5, 10, 20, 50, 100, 200, 500],
            [1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
            lambda n, x: n >= x,
        ),
    },
    {
        "id": "oscillatory_n_lt_x",
        "description": "1 <= n < x <= 5000, oscillatory region (recurrence through the oscillation)",
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
        "id": "dense_n01",
        "description": "n = 0, 1 on a dense grid: linear 0.075 <= x <= 29.925 and log-spaced 1e-3 <= x <= 1e5",
        "points": [
            (n, x)
            for n in (0, 1)
            for x in [0.075 + 0.15 * i for i in range(200)]
            + [10 ** (-3 + 8 * i / 59) for i in range(60)]
        ],
    },
]


def mpmath_reference(n: int, x: float) -> dict[str, float]:
    value = mpmath.bessely(n, mpmath.mpf(x))
    hi = float(value)
    return {"hi": hi, "lo": float(value - mpmath.mpf(hi))}


def main() -> None:
    results = []
    for case in CASES:
        points = []
        for n, x in case["points"]:
            reference = mpmath_reference(n, x)
            if abs(reference["hi"]) > OVERFLOW:
                continue
            points.append(
                {
                    "n": n,
                    "x": x,
                    "mpmath": reference,
                    "scipy": float(yn(n, x)),
                }
            )
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "points": points,
            }
        )

    out_path = Path(__file__).resolve().with_name("besselY.mpmath.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
