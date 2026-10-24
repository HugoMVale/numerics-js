# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "scipy"]
# ///
"""Generate the mpmath reference fixture for the special.gamma / special.lgamma benchmark.

Run with: uv run benchmarks/special/generate_gamma_fixtures.py

The reference is mpmath at 60 digits (gamma, and re(loggamma) for lgamma), stored
as a double-double pair (hi, lo) as in generate_erf_fixtures.py. Conversion to
double is correctly rounded, including the subnormal range. SciPy (Cephes) values
are stored alongside for the benchmark report (null where SciPy returns a non-finite value). Each case id and function must
match benchmarks/special/gamma.mpmath.test.ts.

For lgamma on the negative axis the result is a difference of two terms,
ln(pi / |x sin(pi x)|) - lgamma(-x), so its error is only meaningful relative to
the larger term. Those points carry a `scale` field with that magnitude, and the
test measures the error in ulps of `scale` instead of ulps of the result.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import mpmath
from scipy.special import gamma as scipy_gamma, gammaln as scipy_gammaln

mpmath.mp.dps = 60

# Largest x for which gamma(x) is finite in double precision.
GAMMA_OVERFLOW = 171.6243769563027
# Largest x for which lgamma(x) is finite in double precision.
LGAMMA_OVERFLOW = 2.5599833278516383e305
# Boundaries of the sub-intervals of (0, 2) used by the fdlibm kernel of lgamma
# (high words 0x3FCDA661, 0x3FE76944, 0x3FECCCCD, 0x3FF3B4C4, 0x3FFBB4C3).
LGAMMA_KERNEL_BOUNDS = [
    0.23163998126983643,
    0.7315998077392578,
    0.9000000953674316,
    1.2316322326660156,
    1.7316312789916992,
]


def to_double(value: mpmath.mpf) -> float:
    """Nearest double to an mpf, correctly rounded also for subnormal results."""
    if value == 0:
        return 0.0
    sign, man, exp, _ = value._mpf_
    fraction = Fraction(int(man)) * Fraction(2) ** int(exp)
    return float(-fraction if sign else fraction)


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def geomspace(a, b, n):
    return [math.exp(math.log(a) + (math.log(b) - math.log(a)) * i / (n - 1)) for i in range(n)]


def away_from_integers(xs, tol=1e-3):
    return [x for x in xs if abs(x - round(x)) > tol]


def around_poles(n_max, offsets=(1e-8, 1e-6, 1e-3, 1e-1)):
    pts = []
    for n in range(1, n_max + 1):
        for d in offsets:
            pts += [-n + d, -n - d]
    return pts


def around(center, offsets):
    return [center + s * d for d in offsets for s in (-1, 1)]


CASES = [
    {
        "id": "gamma_tiny",
        "description": "0 < x < 1e-3, where gamma(x) ~ 1/x (down to x = 1e-300)",
        "fn": "gamma",
        "points": [1e-300, 1e-100, 1e-20, 1e-10] + geomspace(1e-8, 1e-3, 40),
    },
    {
        "id": "gamma_small",
        "description": "0.001 <= x < 3, polynomial core for 1/gamma(1 + t)",
        "fn": "gamma",
        "points": linspace(0.001, 3.0, 100) + around(1.0, [1e-9, 1e-4]) + around(2.0, [1e-9, 1e-4]),
    },
    {
        "id": "gamma_mid",
        "description": "3 <= x < 10, recurrence down to [1, 2)",
        "fn": "gamma",
        "points": linspace(3.0, 10.0, 60),
    },
    {
        "id": "gamma_large",
        "description": "10 <= x < 143, Stirling series",
        "fn": "gamma",
        "points": linspace(10.0, 143.0, 100),
    },
    {
        "id": "gamma_large_split",
        "description": "143 <= x <= 171.6243769563027 (overflow edge), split power in the Stirling form",
        "fn": "gamma",
        "points": linspace(143.0, GAMMA_OVERFLOW, 60) + [GAMMA_OVERFLOW],
    },
    {
        "id": "gamma_integers",
        "description": "x = 1 .. 171, factorials from the exact table",
        "fn": "gamma",
        "points": [float(n) for n in range(1, 172)],
    },
    {
        "id": "gamma_negative",
        "description": "-10 < x < 0, away from the poles",
        "fn": "gamma",
        "points": away_from_integers(linspace(-10.0, -0.001, 150)) + [-1e-300, -1e-10],
    },
    {
        "id": "gamma_negative_large",
        "description": "-170.5 <= x <= -10, reflection formula",
        "fn": "gamma",
        "points": away_from_integers(linspace(-170.5, -10.0, 150)),
    },
    {
        "id": "gamma_poles",
        "description": "1e-8 .. 1e-1 from the poles at -1 .. -30",
        "fn": "gamma",
        "points": around_poles(30),
    },
    {
        "id": "gamma_subnormal",
        "description": "-184 < x < -170.5675, results subnormal (SciPy flushes all of these to 0)",
        "fn": "gamma",
        "points": away_from_integers(linspace(-183.99, -170.57, 120), tol=1e-6),
    },
    {
        "id": "lgamma_tiny",
        "description": "|x| < 2^-56 gives -ln|x|; also subnormal arguments",
        "fn": "lgamma",
        "points": [5e-324, 1e-310, 1e-300, 1e-100, 1e-20, 2.0**-57, 2.0**-56] + geomspace(1e-16, 1e-8, 20),
    },
    {
        "id": "lgamma_small",
        "description": "0 < x < 2, all three kernels of the fdlibm (0, 2) region and their boundaries",
        "fn": "lgamma",
        "points": linspace(1e-6, 2.0, 120)
        + [b for bound in LGAMMA_KERNEL_BOUNDS for b in around(bound, [0, 1e-12, 1e-6])],
    },
    {
        "id": "lgamma_zeros",
        "description": "1e-15 .. 1e-1 around the zeros x = 1 and x = 2 (relative accuracy required)",
        "fn": "lgamma",
        "points": [c + s * d for c in (1.0, 2.0) for d in geomspace(1e-15, 1e-1, 20) for s in (-1, 1)],
    },
    {
        "id": "lgamma_mid",
        "description": "2 <= x < 8, rational kernel plus recurrence",
        "fn": "lgamma",
        "points": linspace(2.0, 8.0, 80),
    },
    {
        "id": "lgamma_large",
        "description": "8 <= x < 2^56, asymptotic form",
        "fn": "lgamma",
        "points": linspace(8.0, 100.0, 60) + geomspace(100.0, 2.0**56, 60),
    },
    {
        "id": "lgamma_huge",
        "description": "2^56 <= x <= 2.5599833278516383e305 (overflow edge), x (ln x - 1)",
        "fn": "lgamma",
        "points": geomspace(2.0**56, 1e300, 30) + [LGAMMA_OVERFLOW],
    },
    {
        "id": "lgamma_negative",
        "description": "-100 < x < 0, away from the poles (error in ulps of the larger reflection term)",
        "fn": "lgamma",
        "points": away_from_integers(linspace(-100.0, -0.001, 250)),
    },
    {
        "id": "lgamma_negative_poles",
        "description": "1e-8 .. 1e-1 from the poles at -1 .. -40 (error in ulps of the larger reflection term)",
        "fn": "lgamma",
        "points": around_poles(40),
    },
    {
        "id": "lgamma_negative_large",
        "description": "-1e15 < x <= -100 (error in ulps of the larger reflection term)",
        "fn": "lgamma",
        "points": away_from_integers(linspace(-1000.0, -100.0, 60))
        + [-v - 0.5 for v in geomspace(1e3, 1e15, 30)],
    },
]


def mpmath_value(fn_name: str, x: float) -> mpmath.mpf:
    xm = mpmath.mpf(x)
    if fn_name == "gamma":
        return mpmath.gamma(xm)
    return mpmath.re(mpmath.loggamma(xm))


def scipy_value(fn_name: str, x: float) -> float | None:
    """SciPy result, or None where it is not finite (JSON has no Infinity / NaN)."""
    value = float((scipy_gamma if fn_name == "gamma" else scipy_gammaln)(x))
    return value if math.isfinite(value) else None


def reflection_scale(x: float) -> float:
    """Magnitude of the larger term in ln|Gamma(x)| = ln(pi / |x sin(pi x)|) - lgamma(-x), x < 0."""
    xm = mpmath.mpf(x)
    adjustment = mpmath.log(mpmath.pi / abs(xm * mpmath.sinpi(xm)))
    return to_double(max(abs(adjustment), abs(mpmath.re(mpmath.loggamma(-xm)))))


def main() -> None:
    results = []
    for case in CASES:
        fn_name = case["fn"]
        points = []
        for x in case["points"]:
            value = mpmath_value(fn_name, x)
            hi = to_double(value)
            if not math.isfinite(hi):
                continue  # e.g. gamma of a subnormal x overflows: no finite reference
            point = {
                "x": x,
                "mpmath": {"hi": hi, "lo": to_double(value - mpmath.mpf(hi))},
                "scipy": scipy_value(fn_name, x),
            }
            if fn_name == "lgamma" and x < 0:
                point["scale"] = reflection_scale(x)
            points.append(point)
        results.append(
            {
                "id": case["id"],
                "function": fn_name,
                "description": case["description"],
                "points": points,
            }
        )

    out_path = Path(__file__).resolve().with_name("gamma.mpmath.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["points"]) for r in results)
    print(f"Wrote {len(results)} groups ({total} points) to {out_path}")


if __name__ == "__main__":
    main()
