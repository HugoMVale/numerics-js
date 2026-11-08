# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "scipy"]
# ///
"""Generate the mpmath reference fixture for the special.besselK benchmark.

Run with: uv run benchmarks/special/generate_besselK_mpmath_fixtures.py

The reference is K_n(x) at 60 digits, stored as a double-double pair (hi, lo) as
in generate_besselJ_mpmath_fixtures.py. Both the plain value K_n(x) ("mpmath")
and the exponentially scaled value exp(x) K_n(x) ("mpmathScaled") are stored;
points where K_n(x) underflows to zero or overflows a double have "mpmath" set
to null (the sign of "log10K", log10 K_n(x), tells the two apart), and points
where exp(x) K_n(x) overflows have "mpmathScaled" set to null. For n >= 2 the 60-digit reference comes from the forward recurrence
started from mpmath.besselk(0, x) and mpmath.besselk(1, x) at 110 digits (exact
to well beyond double precision, since K_n is the dominant solution), because
mpmath.besselk itself fails for some large (n, x). SciPy (AMOS, kv/kve) values
are stored alongside for the benchmark report. Each case id must match
benchmarks/special/besselK.mpmath.test.ts.
"""

import json
import math
from pathlib import Path

import mpmath
from scipy.special import kv, kve

mpmath.mp.dps = 110

DBL_MAX = mpmath.mpf(1.7976931348623157e308)
DBL_MIN = mpmath.mpf(2.2250738585072014e-308)


def grid(orders, args):
    return [(n, x) for n in orders for x in args]


CASES = [
    {
        "id": "small_x",
        "description": "0 < x < 1, ascending series region, tiny and moderate arguments",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50],
            [5e-324, 1e-300, 1e-100, 1e-10, 1e-3, 0.1, 0.5, 0.9, 0.99, 0.999999],
        ),
    },
    {
        "id": "moderate_x",
        "description": "1 <= x <= 100, trapezoidal-rule region, orders from 0 up to beyond x",
        "points": grid(
            [0, 1, 2, 3, 5, 10, 20, 50, 100, 200],
            [1.0, 1.000001, 1.5, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0],
        ),
    },
    {
        "id": "large_x",
        "description": "100 < x <= 745, last decade before K_0 underflows",
        "points": grid(
            [0, 1, 2, 10, 50, 100, 300, 1000],
            [150.0, 300.0, 500.0, 700.0, 708.0, 745.0],
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
        "description": "large n relative to x, up to the overflow of K_n(x)",
        "points": grid(
            [150, 151, 152, 500, 1000, 5000],
            [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0],
        ),
    },
    {
        "id": "huge_x_scaled",
        "description": "1e3 <= x <= 1e300, only meaningful through the scaled function",
        "points": grid(
            [0, 1, 2, 10, 100, 1000],
            [1e3, 1e4, 1e5, 1e6, 1e8, 1e12, 1e100, 1e300],
        ),
    },
]


def split(value):
    hi = float(value)
    return {"hi": hi, "lo": float(value - mpmath.mpf(hi))}


def k_values(n: int, x: float):
    """K_n(x) at 110 digits via the (stable) forward recurrence."""
    xm = mpmath.mpf(x)
    prev, cur = mpmath.besselk(0, xm), mpmath.besselk(1, xm)
    if n == 0:
        return prev
    for k in range(1, n):
        prev, cur = cur, prev + (2 * k / xm) * cur
    return cur


def reference(n: int, x: float):
    value = k_values(n, x)
    scaled = value * mpmath.exp(mpmath.mpf(x))
    plain = split(value) if DBL_MIN <= value <= DBL_MAX else None
    return plain, split(scaled) if scaled <= DBL_MAX else None, float(mpmath.log10(value))


def finite_or_none(v: float):
    return v if math.isfinite(v) else None


def main() -> None:
    results = []
    for case in CASES:
        points = []
        for n, x in case["points"]:
            plain, scaled, log10_k = reference(n, x)
            points.append(
                {
                    "n": n,
                    "x": x,
                    "mpmath": plain,
                    "mpmathScaled": scaled,
                    "log10K": log10_k,
                    "scipy": finite_or_none(float(kv(float(n), x))),
                    "scipyScaled": finite_or_none(float(kve(float(n), x))),
                }
            )
        results.append(
            {"id": case["id"], "description": case["description"], "points": points}
        )

    out_path = Path(__file__).resolve().with_name("besselK.mpmath.json")
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
