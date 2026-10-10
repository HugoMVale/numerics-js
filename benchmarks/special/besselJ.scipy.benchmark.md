# besselJ vs SciPy Benchmark Results

Reference: `scipy.special.jv` (AMOS) evaluated at integer orders. The test
accepts a point when `|actual - scipy| <= atol + rtol * |scipy|`.

| Group                | Region                                         | Points | Max abs error | Max rel error | `atol` | `rtol` |
| :------------------- | :--------------------------------------------- | -----: | ------------: | ------------: | -----: | -----: |
| `small_x`            | `\|x\| < 1` (power series)                     |     51 |       6.9e-18 |       3.9e-14 |      0 |  1e-12 |
| `decaying_n_ge_x`    | `n >= x >= 1` (Miller recurrence)              |     45 |       5.1e-16 |       1.1e-13 |      0 |  1e-12 |
| `oscillatory_n_lt_x` | `1 <= n < x <= 5000` (trapezoid + forward)     |     87 |       2.2e-14 |       8.7e-12 | 1e-13  |  1e-12 |
| `turning_point`      | `x = n + d n^(1/3)`, `d` in `[-3, 3]`          |     35 |       8.3e-15 |       1.9e-13 | 5e-14  |  1e-12 |
| `large_x`            | `1e4 <= x <= 1e5`                              |     20 |       6.7e-14 |       3.6e-11 | 3e-13  |  1e-12 |
| `negative_x`         | `x < 0` (parity identity)                      |     20 |       2.3e-16 |       1.1e-14 | 5e-14  |  1e-12 |

The maximum relative errors in the oscillatory groups occur next to zero
crossings of `J_n`, where only the absolute error is meaningful.

## Notes on the reference

- Where `J_n` decays (`n > x`), SciPy's own relative error is up to about
  `3e-13` (checked against 40-digit mpmath values); numerics-js is within
  about `3e-15` of the mpmath values at the same points, so the differences
  reported for `decaying_n_ge_x` are mostly SciPy's.
- SciPy flushes some very small values to exactly `0` well above the
  floating-point underflow threshold (for example `J_200(5)`, about `4.8e-296`,
  where numerics-js returns the correct value). Such points are excluded from
  the fixture; a zero reference is kept only for `x = 0`.
- Absolute error in the oscillatory region grows slowly with `x` because the
  phase `x * sin(t)` is rounded; the cost is proportional to `max(n, |x|)`.
