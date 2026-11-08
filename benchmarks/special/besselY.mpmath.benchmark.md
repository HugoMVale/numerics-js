# besselY vs mpmath Benchmark Results

Reference: `mpmath.bessely(n, x)` evaluated at 60 decimal digits and stored as a
double-double pair `hi + lo`, where `hi` is the nearest double and `lo` is the
double nearest to the remainder. SciPy (`scipy.special.yn`, Cephes `y0`/`y1` plus
forward recurrence) is measured against the same references, so both columns are
comparable. Points whose true value exceeds `1e300` in magnitude are not in the
fixture.

Error is reported in envelope units, `|actual - hi - lo| / (2^-52 S)`, where
`S = |Y_n(x)|` for growing points (`n >= x`) and
`S = max(|Y_n(x)|, min(1, sqrt(2 / (pi x))))` for oscillatory points (`n < x`).
The envelope term keeps the unit meaningful near zeros of `Y_n`. Points with
`|Y_n(x)| <= 1e-290` are excluded from the unit columns. Absolute error is
reported alongside, restricted to points with `|Y_n| <= 1`, because the unit
shrinks at large `x`, which inflates the unit values there.

The test (`besselY.mpmath.test.ts`) checks only numerics-js against the mpmath
references: per group, `|actual - mpmath| <= atol + rtol * |mpmath|` with
`rtol = 2e-14`, plus one test that every point meets the accuracy documented in the
`besselY` docstring. It does not read the SciPy values.

| Group                | Region                                       | Points | numerics-js max error (units) | SciPy max error (units) | numerics-js max abs error (`\|Y\| <= 1`) | SciPy max abs error (`\|Y\| <= 1`) |
| :------------------- | :------------------------------------------- | -----: | ----------------------------: | ----------------------: | ---------------------------------------: | ---------------------------------: |
| `small_x`            | `0 < x < 3`, ascending series                |     84 |                          13.9 |                    3.97 |                                 1.11e-16 |                           4.44e-16 |
| `mid_x`              | `3 <= x < 20`, Neumann series                |     96 |                          10.8 |                    9.55 |                                 8.88e-16 |                           7.22e-16 |
| `large_x`            | `20 <= x <= 1e15`, Hankel expansion          |     84 |                          8.45 |                 8.19e13 |                                 4.16e-17 |                           4.59e-10 |
| `growing_n_ge_x`     | `n >= x >= 1`, growing                       |     46 |                          19.8 |                    31.1 |                                 4.30e-16 |                           9.16e-16 |
| `oscillatory_n_lt_x` | `1 <= n < x <= 5000`, oscillatory            |     87 |                           3.5 |                     753 |                                 1.39e-16 |                           1.89e-15 |
| `turning_point`      | `x` within a few `n^(1/3)` of `x = n`        |     35 |                          31.6 |                     170 |                                 1.28e-15 |                           1.75e-15 |
| `dense_n01`          | `n = 0, 1`, 200 linear + 60 log-spaced `x`   |    520 |                          5.52 |                 1.70e4 |                                 2.52e-16 |                           1.30e-14 |

Points violating the documented accuracy claim (`src/special/bessel.ts`) for
numerics-js: none.

## Notes

- Large arguments are where the two differ most. numerics-js evaluates
  `sin(x)` and `cos(x)` of the exact input and never forms the phase
  `x - (nu/2 + 1/4) pi`, so its absolute error does not grow with `x`. SciPy's
  absolute error reaches `4.6e-10` at `x = 1e15` (the `large_x` unit column is huge
  only because the envelope shrinks like `x^(-1/2)`).
- Relative error for `|Y_n| > 1` is at most `3.3e-16` for `n <= 1` (`dense_n01`) and
  `7.0e-15` at the turning points of `n` up to 1000, where the recurrence round-off
  is largest.
- Timing (one run, Node, microseconds per call): `Y_0` 0.09 (`x = 0.7`, series),
  0.49 to 0.57 (`x = 10` to `19.9`, Neumann series), 0.1 to 0.3 (`x >= 25`,
  Hankel, independent of `x` up to `1e15`); `Y_10(50)` 0.18; `Y_1000(1000)` 6.6
  (cost is linear in `n`). The Neumann region is the most expensive part because
  Miller's recurrence starts at roughly `x + 12 x^(1/3) + 60` terms.
- Regenerate the fixture with
  `uv run benchmarks/special/generate_besselY_mpmath_fixtures.py`.
