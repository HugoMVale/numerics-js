# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "scipy"]
# ///
"""Generate the mpmath reference fixture for the special.besselI benchmark.

Run with: uv run benchmarks/special/generate_besselI_mpmath_fixtures.py

The reference is mpmath.besseli at 60 digits, stored as a double-double pair
(hi, lo) as in generate_besselJ_mpmath_fixtures.py. Both the plain value
I_n(x) ("mpmath") and the exponentially scaled value exp(-|x|) I_n(x)
("mpmathScaled") are stored; points where I_n(x) overflows a double have
"mpmath" set to null. SciPy (AMOS, iv/ive) values are stored alongside for the
benchmark report. Each case id must match benchmarks/special/besselI.mpmath.test.ts.
"""

import json
import math
from pathlib import Path

import mpmath
from scipy.special import iv, ive

mpmath.mp.dps = 60

DBL_MAX = mpmath.mpf(1.7976931348623157e308)


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
        "id": "moderate_x",
        "description": "1 <= x <= 100, all orders from 0 up to beyond x",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50, 100, 200],
            [1.0, 1.5, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0],
            lambda n, x: True,
        ),
    },
    {
        "id": "large_x",
        "description": "100 < x <= 700, last decade before I_0 overflows",
        "points": grid(
            [0, 1, 2, 10, 50, 100, 300],
            [150.0, 300.0, 500.0, 700.0],
            lambda n, x: True,
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
        "id": "high_order",
        "description": "large n relative to x, deep in the decaying (underflow) tail",
        "points": grid(
            [500, 1000, 5000],
            [1.0, 10.0, 100.0, 500.0, 1000.0],
            lambda n, x: True,
        ),
    },
    {
        "id": "huge_x_scaled",
        "description": "1e3 <= x <= 1e6, only meaningful through the scaled function",
        "points": grid(
            [0, 1, 2, 10, 100, 1000],
            [1e3, 1e4, 1e5, 1e6],
            lambda n, x: True,
        ),
    },
    {
        "id": "negative_x",
        "description": "negative arguments, I_n(-x) = (-1)^n I_n(x)",
        "points": grid([0, 1, 2, 3, 10], [-0.5, -2.5, -10.0, -100.0], lambda n, x: True),
    },
]


def split(value):
    hi = float(value)
    return {"hi": hi, "lo": float(value - mpmath.mpf(hi))}


def reference(n: int, x: float):
    xm = mpmath.mpf(x)
    value = mpmath.besseli(n, xm)
    scaled = value * mpmath.exp(-abs(xm))
    plain = split(value) if abs(value) <= DBL_MAX else None
    return plain, split(scaled)


def finite_or_none(v: float):
    return v if math.isfinite(v) else None


def main() -> None:
    results = []
    for case in CASES:
        points = []
        for n, x in case["points"]:
            plain, scaled = reference(n, x)
            points.append(
                {
                    "n": n,
                    "x": x,
                    "mpmath": plain,
                    "mpmathScaled": scaled,
                    "scipy": finite_or_none(float(iv(float(n), x))),
                    "scipyScaled": float(ive(float(n), x)),
                }
            )
        results.append(
            {"id": case["id"], "description": case["description"], "points": points}
        )

    out_path = Path(__file__).resolve().with_name("besselI.mpmath.json")
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
