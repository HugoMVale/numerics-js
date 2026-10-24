# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "scipy"]
# ///
"""Generate the mpmath reference fixture for the special.erf / special.erfc benchmark.

Run with: uv run benchmarks/special/generate_erf_fixtures.py

The reference is mpmath at 60 digits, stored as a double-double pair (hi, lo):
hi is the nearest double and lo the nearest double to the remainder, so the
test measures numerics-js error without the rounding of a single-double
reference. SciPy (Cephes) values are stored alongside for the benchmark report.
Each case id and function must match benchmarks/special/erf.mpmath.test.ts.
"""

import json
from pathlib import Path

import mpmath
from scipy.special import erf as scipy_erf, erfc as scipy_erfc

mpmath.mp.dps = 60

MPMATH_FUNCTIONS = {"erf": mpmath.erf, "erfc": mpmath.erfc}
SCIPY_FUNCTIONS = {"erf": scipy_erf, "erfc": scipy_erfc}


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def both_signs(xs):
    return [-x for x in reversed(xs)] + xs


CASES = [
    {
        "id": "erf_small",
        "description": "|x| < 0.84375 (rational approximation), including tiny arguments",
        "fn": "erf",
        "points": both_signs([1e-300, 1e-100, 1e-10, 2.0**-28, 1e-5] + linspace(0.001, 0.84, 60)),
    },
    {
        "id": "erf_mid",
        "description": "0.84375 <= |x| < 1.25 (expansion around 1)",
        "fn": "erf",
        "points": both_signs(linspace(0.85, 1.24, 40)),
    },
    {
        "id": "erf_tail",
        "description": "1.25 <= |x| < 6 (asymptotic form, 1 - erfc)",
        "fn": "erf",
        "points": both_signs(linspace(1.25, 5.9, 80)),
    },
    {
        "id": "erf_saturated",
        "description": "|x| >= 6, where erf rounds to exactly +-1",
        "fn": "erf",
        "points": both_signs([6.0, 6.5, 8.0, 10.0, 30.0, 100.0]),
    },
    {
        "id": "erfc_small",
        "description": "|x| < 0.84375, including the x = 0.25 sub-branch",
        "fn": "erfc",
        "points": both_signs([1e-300, 1e-100, 1e-10, 1e-5] + linspace(0.001, 0.84, 60)),
    },
    {
        "id": "erfc_mid",
        "description": "0.84375 <= |x| < 1.25 (expansion around 1)",
        "fn": "erfc",
        "points": both_signs(linspace(0.85, 1.24, 40)),
    },
    {
        "id": "erfc_tail",
        "description": "1.25 <= x < 6 (asymptotic form)",
        "fn": "erfc",
        "points": linspace(1.25, 5.99, 80),
    },
    {
        "id": "erfc_far_tail",
        "description": "6 <= x <= 26, deep tail down to 1e-296 (full relative accuracy required)",
        "fn": "erfc",
        "points": linspace(6.0, 26.0, 60),
    },
    {
        "id": "erfc_negative",
        "description": "x <= -1.25, where erfc(x) = 2 - erfc(|x|) approaches 2",
        "fn": "erfc",
        "points": linspace(-8.0, -1.25, 50),
    },
    {
        "id": "erfc_subnormal",
        "description": "erfc(x) subnormal, 26.55 <= x <= 27.4 (SciPy flushes most of these to 0)",
        "fn": "erfc",
        "points": [round(26.55 + 0.01 * i, 2) for i in range(86)],
    },
]


def mpmath_reference(fn_name: str, x: float) -> dict[str, float]:
    value = MPMATH_FUNCTIONS[fn_name](mpmath.mpf(x))
    hi = float(value)
    return {"hi": hi, "lo": float(value - mpmath.mpf(hi))}


def main() -> None:
    results = []
    for case in CASES:
        fn_name = case["fn"]
        points = [
            {
                "x": x,
                "mpmath": mpmath_reference(fn_name, x),
                "scipy": float(SCIPY_FUNCTIONS[fn_name](x)),
            }
            for x in case["points"]
        ]
        results.append(
            {
                "id": case["id"],
                "function": fn_name,
                "description": case["description"],
                "points": points,
            }
        )

    out_path = Path(__file__).resolve().with_name("erf.mpmath.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
