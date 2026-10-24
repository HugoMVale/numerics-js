# erf / erfc vs mpmath Benchmark Results

Reference: `mpmath.erf` and `mpmath.erfc` evaluated at 60 decimal digits and
stored as a double-double pair `hi + lo`, where `hi` is the nearest double and
`lo` is the double nearest to the remainder. SciPy (`scipy.special.erf` and
`scipy.special.erfc`, Cephes, SciPy 1.18.1) is measured against the same
references, so both columns are comparable.

Error is measured in units in the last place (ulp) of the reference:
`|actual - hi - lo| / ulp(hi)`, with `ulp(v) = max(2^(floor(log2|v|) - 52), 2^-1074)`.
For the subnormal group the unit is `2^-1074`.

The test (`erf.mpmath.test.ts`) checks only numerics-js against the mpmath
references, with the bounds documented in `src/special/erf.ts`: `erf` within
1 ulp and `erfc` within 3 ulp. It does not read the SciPy values.

| Group            | Function | Region                                            | Points | numerics-js max error (ulp) | SciPy max error (ulp) |
| :--------------- | :------- | :------------------------------------------------ | -----: | --------------------------: | --------------------: |
| `erf_small`      | `erf`    | `\|x\| < 0.84375`                                 |    130 |                        0.56 |                  1.50 |
| `erf_mid`        | `erf`    | `0.84375 <= \|x\| < 1.25`                         |     80 |                        0.49 |                  1.62 |
| `erf_tail`       | `erf`    | `1.25 <= \|x\| < 6`                               |    160 |                        0.50 |                  0.50 |
| `erf_saturated`  | `erf`    | `\|x\| >= 6` (exactly `±1`)                       |     12 |                        0.10 |                  0.10 |
| `erfc_small`     | `erfc`   | `\|x\| < 0.84375`                                 |    128 |                        2.00 |                  2.56 |
| `erfc_mid`       | `erfc`   | `0.84375 <= \|x\| < 1.25`                         |     80 |                        1.07 |                  6.49 |
| `erfc_tail`      | `erfc`   | `1.25 <= x < 6`                                   |     80 |                        2.04 |                  28.1 |
| `erfc_far_tail`  | `erfc`   | `6 <= x <= 26` (down to `1e-296`)                 |     60 |                        1.78 |                   335 |
| `erfc_negative`  | `erfc`   | `x <= -1.25`                                      |     50 |                        0.50 |                  0.50 |
| `erfc_subnormal` | `erfc`   | `26.55 <= x <= 27.4` (subnormal, `2^-1074` units) |     86 |                        0.00 |               1.53e13 |

## Notes

- Errors of about 0.5 ulp are the floor of this measure. The reference is stored
  as `hi + lo`, so a correctly rounded result already shows up at up to 0.5 ulp.
  This is why `erf_tail`, `erf_saturated` and `erfc_negative` read 0.5 for both
  implementations.
- In the `erfc` tail, SciPy's error grows to 28 ulp on `erfc_tail` and 335 ulp on
  `erfc_far_tail`. numerics-js stays within 2.04 ulp and 1.78 ulp respectively.
- In the subnormal range, SciPy flushes 76 of 86 results to `0`, which is off by
  up to `1.53e13` units of `2^-1074`. numerics-js is exact at every point.
- Claimed bounds: `erf` within 1 ulp (observed maximum 0.56) and `erfc` within
  3 ulp (observed maximum 2.04). The subnormal group is measured in `2^-1074`
  units and is not covered by the ulp bound at its scale.
