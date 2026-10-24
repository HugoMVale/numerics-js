# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Derive the constants used by src/special/gamma.ts.

Run with: uv run scripts/derive_gamma_coefficients.py

Two tables are produced, both correctly rounded to double precision from
60-digit arithmetic, so no third-party coefficient table is involved:

- RECIPROCAL_GAMMA_P: ascending coefficients of P in 1/Gamma(1 + t) = 1 + t P(t)
  on [0, 1] (degree-14 Chebyshev fit, mpmath.chebyfit).
- STIRLING: sqrt(2 pi) * a_n, n = 0..13, where a_n are the exact rational
  coefficients of the series Gamma(x) = sqrt(2 pi) x^(x-1/2) e^-x sum_n a_n x^-n
  (a_0 = 1, a_1 = 1/12, a_2 = 1/288, a_3 = -139/51840, ...), obtained by
  exponentiating the Bernoulli series of ln Gamma with Fraction arithmetic.

The fit and truncation errors are printed so the choice of degree / number of
terms can be audited. The output is meant to be pasted into gamma.ts.
"""

from fractions import Fraction
from math import comb

import mpmath as mp

mp.mp.dps = 60

DEGREE = 14
TERMS = 14


def bernoulli(n: int) -> list[Fraction]:
    b = [Fraction(0)] * (n + 1)
    b[0] = Fraction(1)
    for m in range(1, n + 1):
        b[m] = -sum(Fraction(comb(m + 1, k)) * b[k] for k in range(m)) / (m + 1)
    return b


def stirling_coefficients(n_terms: int) -> list[Fraction]:
    """a_n with sum a_n z^n = exp(sum_k B_2k / (2k (2k-1)) z^(2k-1))."""
    b = bernoulli(2 * n_terms + 2)
    log_series = [Fraction(0)] * n_terms
    for k in range(1, n_terms // 2 + 2):
        n = 2 * k - 1
        if n < n_terms:
            log_series[n] = b[2 * k] / (2 * k * (2 * k - 1))
    a = [Fraction(0)] * n_terms
    a[0] = Fraction(1)
    for n in range(1, n_terms):  # n a_n = sum_{k=1..n} k L_k a_(n-k)
        a[n] = sum(k * log_series[k] * a[n - k] for k in range(1, n + 1)) / n
    return a


def to_mpf(q: Fraction) -> mp.mpf:
    return mp.mpf(q.numerator) / q.denominator


def js(v: float) -> str:
    return repr(float(v))


def main() -> None:
    # --- 1/Gamma(1 + t) = 1 + t P(t)
    def f(t):
        return (1 / mp.gamma(1 + t) - 1) / t if t != 0 else mp.euler

    coeffs_desc, max_err = mp.chebyfit(f, [0, 1], DEGREE + 1, error=True)
    p = [float(c) for c in reversed(coeffs_desc)]
    assert abs(p[0] - float(mp.euler)) < 1e-15, "P(0) must equal the Euler-Mascheroni constant"
    print(f"// RECIPROCAL_GAMMA_P (degree {DEGREE}); max abs fit error on [0, 1]: {mp.nstr(max_err, 3)}")
    print("[" + ", ".join(js(c) for c in p) + "]\n")

    # --- Stirling
    a = stirling_coefficients(TERMS)
    assert a[1] == Fraction(1, 12) and a[2] == Fraction(1, 288) and a[3] == Fraction(-139, 51840)
    s2pi = mp.sqrt(2 * mp.pi)
    stirling = [float(s2pi * to_mpf(q)) for q in a]

    def truncation_error(x: float) -> mp.mpf:
        x = mp.mpf(x)
        series = sum(to_mpf(q) / x**n for n, q in enumerate(a))
        approx = s2pi * x ** (x - mp.mpf(1) / 2) * mp.exp(-x) * series
        return abs(approx / mp.gamma(x) - 1)

    print(f"// STIRLING ({TERMS} terms); relative truncation error at x = 10: {mp.nstr(truncation_error(10), 3)}")
    print("[" + ", ".join(js(c) for c in stirling) + "]")


if __name__ == "__main__":
    main()
