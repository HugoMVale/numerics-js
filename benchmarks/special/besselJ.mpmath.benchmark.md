# besselJ vs mpmath Benchmark Results

Reference: `mpmath.besselj(n, x)` evaluated at 60 decimal digits and stored as a
double-double pair `hi + lo`, where `hi` is the nearest double and `lo` is the
double nearest to the remainder. SciPy (`scipy.special.jv`, SciPy 1.18.1) is
measured against the same references, so both columns are comparable. The
mpmath values agree with a 100-digit evaluation to within `2.7e-63` in a
26-point probe, so the 60-digit reference is not the limiting factor.

Error is reported in envelope units, `|actual - hi - lo| / (2^-52 S)`, where
`S = |J_n(x)|` for decaying points (`n >= |x|`) and
`S = max(|J_n(x)|, min(1, sqrt(2 / (pi |x|))))` for oscillatory points
(`n < |x|`). The envelope term keeps the unit meaningful near zeros of `J_n`.
Points with `|J_n(x)| <= 1e-290` are excluded from the unit columns, since the
unit is undefined there; their absolute error is still included in the absolute
columns. Absolute error is reported alongside because the envelope unit shrinks
at large `|x|`, which inflates the unit values in that region.

The test (`besselJ.mpmath.test.ts`) checks only numerics-js against the mpmath
references, with a per-group tolerance `|actual - mpmath| <= atol + rtol * |mpmath|`.
`rtol` is `2e-14` for all groups. `atol` is `0` for `small_x`, `decaying_n_ge_x` and
`negative_x`, `1e-14` for `oscillatory_n_lt_x` and `turning_point`, and `1e-13` for
`large_x`. The test does not read the SciPy values.

| Group                | Region                                       | Points | numerics-js max error (units) | SciPy max error (units) | numerics-js max abs error | SciPy max abs error |
| :------------------- | :------------------------------------------- | -----: | ----------------------------: | ----------------------: | ------------------------: | ------------------: |
| `small_x`            | `\|x\| < 1`, power series                    |     64 |                          3.71 |                   177.0 |                  1.11e-16 |            1.11e-16 |
| `decaying_n_ge_x`    | `n >= x >= 1`, decaying                      |     56 |                          10.6 |                   508.0 |                  1.80e-16 |            5.00e-16 |
| `oscillatory_n_lt_x` | `1 <= n < x <= 5000`, oscillatory            |     87 |                         1880 |                    9150 |                  7.46e-15 |            2.29e-14 |
| `turning_point`      | `x` within a few `n^(1/3)` of `x = n`        |     35 |                           961 |                     128 |                  8.45e-15 |            9.71e-16 |
| `large_x`            | `1e4 <= x <= 1e5`                            |     20 |                       1.19e5 |                   0.692 |                  6.66e-14 |            8.67e-19 |
| `negative_x`         | negative `x`                                 |     20 |                          12.7 |                    5.14 |                  2.26e-16 |            1.60e-16 |

Points violating the documented accuracy claim (`src/special/bessel.ts`) for
numerics-js: `decaying_n_ge_x` 5, `oscillatory_n_lt_x` 32, `turning_point` 16,
`large_x` 1. The other groups have none. The test uses the tolerances above, not
the claim, so these points pass the test.

With `rtol = 1e-14`, `negative_x` failed at `J_0(-100)` (relative error about
`1.1e-14`), so the shared `rtol` was set to `2e-14`.

## Notes

- The documented accuracy claim is violated in four groups: `decaying_n_ge_x`,
  `oscillatory_n_lt_x`, `turning_point` and `large_x`. The test checks the measured
  tolerances instead, and all six groups pass.
- Oscillatory region: `J_0(200)`, `J_0(1000)` and `J_1(20000)` show errors of
  about `2e-15`, `1.9e-15` and `1.5e-14`. The claim is "a few `1e-16`" up to
  `|x| = 1e3`, so these exceed it. At `|x| = 2000`, numerics-js errors reach
  `6.6e-15`, about three times the `2e-15` quoted at `|x| = 5e3`. The claim does
  not match the measured error in that range.
- Turning-point region: decaying points just above the transition show relative
  error well above round-off. For example, `J_300(293.31)` has absolute error
  `1.18e-16` on a value of about `1.8e-2`, which is about `29` eps relative. The
  claim says round-off level when `n > |x|`.
- The claim gives anchor values only (`1e3`, `5e3`, `1e5`). Between the anchors it
  defines no bound, so the violations at `|x| = 2000` and `|x| = 2e4` depend on how
  the anchors are interpolated.
- Large `x`: SciPy is far more accurate than numerics-js here, with absolute error
  `8.67e-19` against `6.66e-14` at `1e5`. This is the main region where SciPy
  leads in absolute terms.
- Oscillatory region: in absolute terms numerics-js is more accurate than SciPy
  (`7.46e-15` against `2.29e-14`), but the unit column ranks SciPy as worse
  because the envelope unit is small at large `|x|`. Both columns should be read
  together.
- Small `x`: both implementations agree at round-off level in absolute terms
  (`1.11e-16`). Numerics-js is `3.71` units against SciPy at `177` units.
- Negative `x`: both are at round-off level in absolute terms (`2.26e-16` for
  numerics-js, `1.60e-16` for SciPy). In units, numerics-js is `12.7` against
  SciPy at `5.14`, so SciPy is ahead on this group by the unit measure.
