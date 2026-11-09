# CubicSplineInterpolator1D/ND vs SciPy Benchmark Results

Reference: `scipy.interpolate.CubicSpline` with `bc_type` `'not-a-knot'`,
`'natural'` and `'clamped'` (first derivatives prescribed at both ends; random
values in the fixture). Every dataset is run with all three boundary types
(24 cases). For each case the test compares the spline value and its first
derivative at the knots, the interval midpoints and random points, plus three
definite integrals (the whole range, a sub-range, and a short interval). The
vector-valued dataset exercises `CubicSplineInterpolatorND`, including
per-component end derivatives. Queries stay inside the data range, because
SciPy extrapolates the end polynomials where numerics-js clamps to constants.
The test enforces `|actual - reference| <= 1e-11 + 1e-11 |reference|`.

| Dataset              | Knots | Description                                          | Max abs. error |
| -------------------- | ----: | ---------------------------------------------------- | -------------: |
| `uniform_sine_12`    |    12 | 12 uniform knots, y = sin(x) + 0.1 x                 |        1.0e-15 |
| `graded_25`          |    25 | 25 knots, spacing growing by 17 % per step           |        3.1e-15 |
| `random_noisy`       |    35 | random non-uniform knots, y ~ N(0, 1)                |        9.8e-15 |
| `tiny_2`             |     2 | smallest system (straight line or end-slope Hermite) |        2.2e-15 |
| `tiny_3`             |     3 | closed-form parabola for not-a-knot                  |        3.3e-15 |
| `tiny_4`             |     4 | smallest general tridiagonal system                  |        1.3e-15 |
| `tiny_5`             |     5 | small general system                                 |        6.7e-15 |
| `vector_valued_16x4` |    16 | 16 knots shared by 4 output components               |        5.7e-14 |

The largest observed error is `5.7e-14` (`vector_valued_16x4`, whose values
reach about 200), i.e. the splines agree with SciPy to round-off for values,
first derivatives and integrals. Each case's maximum is taken over the three
boundary types and the three compared quantities.

Regenerate the fixture with
`uv run benchmarks/interpolate/generate_cubic_spline_fixtures.py`.
