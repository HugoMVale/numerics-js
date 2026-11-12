# besselK vs mpmath Benchmark Results

Reference: `K_n(x)` at 60 decimal digits, stored as a double-double pair `hi + lo`,
where `hi` is the nearest double and `lo` is the double nearest to the remainder.
`mpmath.besselk` is used for `n <= 1`; for `n >= 2` the reference is the forward
recurrence `K_{k+1} = K_{k-1} + (2k/x) K_k` carried out at 110 digits from the
mpmath values of `K_0` and `K_1`, because `mpmath.besselk` itself fails for some
large `(n, x)`. The recurrence is exact to far beyond double precision since
`K_n` is its dominant solution. The scaled reference `exp(x) K_n(x)` is computed
from the same value. SciPy (`scipy.special.kve`, SciPy 1.18.1) is measured against
the same scaled references, so the columns are comparable.

Points where `K_n(x)` overflows a double have no plain reference (the test instead
requires `besselK` to return `Infinity`); likewise when `K_n(x)` is below the
smallest normal double (the test then only requires a result below `2.3e-308`).
Points where `exp(x) K_n(x)` itself overflows have no scaled reference (the test
requires `besselKScaled` to return `Infinity`).

Error is the relative error `|actual - reference| / |reference|`. `K_n(x)` has no
zeros for `x > 0`, so no envelope term is needed.

The test (`besselK.mpmath.test.ts`) checks only numerics-js against the mpmath
references, with `rtol = 1e-14` for both functions in every group. It does not read
the SciPy values.

| Group           | Region                                | Points | `besselK` max rel. error | `besselKScaled` max rel. error | SciPy `kve` max rel. error |
| --------------- | ------------------------------------- | -----: | -----------------------: | -----------------------------: | -------------------------: |
| `small_x`       | ascending series                      |     80 |                  2.8e-15 |                        2.9e-15 |                    1.7e-14 |
| `moderate_x`    | trapezoidal rule                      |     90 |                  1.1e-15 |                        1.0e-15 |                    4.5e-14 |
| `large_x`       | trapezoidal rule                      |     48 |                  2.5e-15 |                        2.6e-15 |                    3.3e-13 |
| `turning_point` | trapezoidal rule, `x ~ n`             |     35 |                  2.9e-15 |                        3.0e-15 |                    1.8e-13 |
| `high_order`    | recurrence up to the overflow of `K_n` |     36 |                  1.5e-15 |                        1.6e-15 |                    9.6e-14 |
| `huge_x_scaled` | trapezoidal rule, scaled only         |     48 |                  1.5e-15 |                        1.9e-15 |                    1.5e-13 |

`besselK` columns exclude points whose plain value overflows or underflows below
the normal range (`small_x`: 62 compared, `moderate_x`: 86, `large_x`: 35,
`high_order`: 15, `huge_x_scaled`: 1). Everything is at the round-off level.

Algorithm: `K_0` and `K_1` come from the ascending series for `x < 1` (all terms
positive, no cancellation) and from the trapezoidal rule on
`e^x K_nu(x) = integral_0^inf exp(-2 x sinh^2(t/2)) cosh(nu t) dt` for `x >= 1`
(positive integrand, geometric convergence, at most 21 terms for any `x`).
Higher orders use the stable forward recurrence. The factor `e^-x` of the plain
function is removed by an exact reduction `x = m ln 2 + r`, so no accuracy is lost
even when `e^x K_n(x)` overflows while `K_n(x)` is still representable.

Cost: measured on one machine (Node 22), `besselK(0, x)` takes about 0.1 us for
`x < 1` and 0.4 to 0.6 us for `x >= 1`, independent of `x` (`besselIScaled(0, 1e6)`,
for comparison, takes about 80 us); the recurrence adds about 3 us per 1000 orders.
