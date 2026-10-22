# erf / erfc vs SciPy Benchmark Results

Reference: `scipy.special.erf` and `scipy.special.erfc` (Cephes, SciPy 1.18.1).
The test accepts a point when `|actual - scipy| <= atol + rtol * |scipy|`. The
reference algorithm is independent of the FreeBSD/fdlibm one used by
numerics-js.

| Group           | Function | Region                                     | Points | Max abs error | Max rel error | `atol` |   `rtol` |
| :-------------- | :------- | :----------------------------------------- | -----: | ------------: | ------------: | -----: | -------: |
| `erf_small`     | `erf`    | `\|x\| < 0.84375`                          |    130 |       1.1e-16 |       2.2e-16 |      0 |    1e-15 |
| `erf_mid`       | `erf`    | `0.84375 <= \|x\| < 1.25`                  |     80 |       2.2e-16 |       2.7e-16 |      0 |    1e-15 |
| `erf_tail`      | `erf`    | `1.25 <= \|x\| < 6`                        |    160 |             0 |             0 |      0 |    1e-15 |
| `erf_saturated` | `erf`    | `\|x\| >= 6` (exactly `±1`)                |     12 |             0 |             0 |      0 |        0 |
| `erfc_small`    | `erfc`   | `\|x\| < 0.84375`                          |    128 |       2.2e-16 |       4.7e-16 |      0 |    2e-15 |
| `erfc_mid`      | `erfc`   | `0.84375 <= \|x\| < 1.25`                  |     80 |       2.2e-16 |       1.1e-15 |      0 |    4e-15 |
| `erfc_tail`     | `erfc`   | `1.25 <= x < 6`                            |     80 |       1.4e-17 |       3.4e-15 |      0 |    1e-14 |
| `erfc_far_tail` | `erfc`   | `6 <= x <= 26` (down to `1e-296`)          |     60 |       3.1e-33 |       5.5e-14 |      0 |    2e-13 |
| `erfc_negative` | `erfc`   | `-8 <= x <= -1.25`                         |     50 |             0 |             0 |      0 |    1e-15 |

## Notes on the reference

- In the `erfc` tail, the differences are SciPy's. Checked against 60-digit
  mpmath values at the same points, SciPy's maximum relative error is `3.6e-15`
  for `erfc_tail` and `5.5e-14` for `erfc_far_tail`, whereas numerics-js is
  within `2.7e-16` and `3.6e-16` respectively (about 1 to 3 ulp). The `rtol`
  values for those groups are therefore set by SciPy's error.
- The remaining groups agree with SciPy to within a few ulp, and bit for bit
  at many points (e.g. all of `erf_tail`, `erf_saturated` and `erfc_negative`).
- Independently of SciPy, numerics-js was also checked against mpmath at 45,000
  points per function across every branch and branch boundary, plus a fine scan
  of the subnormal range of `erfc` (`26 <= x <= 27.4`): maximum error 1 ulp for
  `erf` and 3 ulp for `erfc`.
