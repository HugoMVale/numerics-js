# gamma / lgamma vs mpmath Benchmark Results

Reference: `mpmath.gamma` and `mpmath.re(mpmath.loggamma)` evaluated at 60 decimal
digits and stored as a double-double pair `hi + lo`, where `hi` is the nearest
double and `lo` is the double nearest to the remainder (correctly rounded, also in
the subnormal range). SciPy (`scipy.special.gamma` and `scipy.special.gammaln`,
Cephes, SciPy 1.18.1) is measured against the same references, so both columns
are comparable.

Error is measured in units in the last place (ulp) of the reference:
`|actual - hi - lo| / ulp(hi)`, with `ulp(v) = max(2^(floor(log2|v|) - 52), 2^-1074)`.
For the subnormal group the unit is `2^-1074`.

For `lgamma` on the negative axis the result is a difference of two terms,
`ln(pi / |x sin(pi x)|) - lgamma(-x)`, and can be much smaller than either (near the
zeros of `ln|Gamma|`, at about `-2.4570`, `-2.7477`, ...). Those points (the three
`lgamma_negative*` groups) carry a `scale` field with the magnitude of the larger
term, and the error is measured in ulps of `scale` instead of ulps of `hi`.

The test (`gamma.mpmath.test.ts`) checks only numerics-js against the mpmath
references, with the bounds documented in `src/special/gamma.ts` and
`src/special/lgamma.ts`: `gamma` within 6 ulp; `lgamma` within 2 ulp for `x > 0` and
within 5 ulp of the larger reflection term for `x < 0`. It does not read the SciPy
values.

| Group                   | Function | Region                                                    | Points | numerics-js max error (ulp) | SciPy max error (ulp) | SciPy non-finite |
| :---------------------- | :------- | :-------------------------------------------------------- | -----: | --------------------------: | --------------------: | ---------------: |
| `gamma_tiny`            | `gamma`  | `1e-300 <= x < 1e-3`                                      |     44 |                        1.59 |                  1.98 |                0 |
| `gamma_small`           | `gamma`  | `0.001 <= x < 3`                                          |    108 |                        1.43 |                  3.06 |                0 |
| `gamma_mid`             | `gamma`  | `3 <= x < 10`                                             |     60 |                        2.23 |                  2.91 |                0 |
| `gamma_large`           | `gamma`  | `10 <= x < 143`                                           |    100 |                        3.02 |                  2.87 |                0 |
| `gamma_large_split`     | `gamma`  | `143 <= x <= 171.6243769563027`                           |     61 |                        3.08 |                  3.14 |                2 |
| `gamma_integers`        | `gamma`  | `x = 1 .. 171` (factorials)                               |    171 |                        0.50 |                  2.67 |                0 |
| `gamma_negative`        | `gamma`  | `-10 < x < 0`, away from poles                            |    150 |                        3.14 |                  3.19 |                0 |
| `gamma_negative_large`  | `gamma`  | `-170.5 <= x <= -10`                                      |    149 |                        4.08 |                  6.34 |                0 |
| `gamma_poles`           | `gamma`  | `1e-8 .. 1e-1` from the poles at `-1 .. -30`              |    240 |                        3.33 |                  4.98 |                0 |
| `gamma_subnormal`       | `gamma`  | `-184 < x < -170.5675` (subnormal, `2^-1074` units)       |    120 |                        1.00 |              3.12e+15 |                0 |
| `lgamma_tiny`           | `lgamma` | subnormal `x` up to `1e-8`, and `2^-57`, `2^-56`          |     27 |                        0.48 |                  0.61 |                2 |
| `lgamma_small`          | `lgamma` | `0 < x < 2`, incl. fdlibm sub-interval boundaries         |    150 |                        1.45 |                 191.4 |                0 |
| `lgamma_zeros`          | `lgamma` | `1e-15 .. 1e-1` around `x = 1` and `x = 2`                |     80 |                        0.67 |              1.38e+15 |                0 |
| `lgamma_mid`            | `lgamma` | `2 <= x < 8`                                              |     80 |                        1.05 |                  1.60 |                0 |
| `lgamma_large`          | `lgamma` | `8 <= x < 2^56`                                           |    120 |                        1.66 |                  2.16 |                0 |
| `lgamma_huge`           | `lgamma` | `2^56 <= x <= 2.5599833278516383e305`                     |     31 |                        0.49 |                  0.75 |                1 |
| `lgamma_negative`       | `lgamma` | `-100 < x < 0`, away from poles (ulp of `scale`)          |    249 |                        2.22 |                  2.48 |                0 |
| `lgamma_negative_poles` | `lgamma` | `1e-8 .. 1e-1` from the poles at `-1 .. -40` (ulp of `scale`) |  320 |                        1.51 |                  1.99 |                0 |
| `lgamma_negative_large` | `lgamma` | `-1e15 < x <= -100` (ulp of `scale`)                      |     88 |                        1.67 |                  2.55 |                0 |

## Notes

- Errors of about 0.5 ulp are the floor of this measure. The reference is stored as
  `hi + lo`, so a correctly rounded result already shows up at up to 0.5 ulp, as in
  `gamma_integers` (exact factorial table).
- "SciPy non-finite" counts the points where SciPy returns `Infinity` instead of a
  finite value; those points are left out of the SciPy error column. `gamma` returns
  `Infinity` for `x > 171.6144...` (Cephes threshold), so it loses the last two points of
  `gamma_large_split` although `gamma(171.6243769563027)` is finite. `gammaln` fails
  for the subnormal arguments `5e-324` and `1e-310` and at the overflow edge of `lgamma`.
- In the subnormal range of `gamma`, SciPy flushes 118 of 120 results to `0`, although
  66 of the true values are non-zero (down to the last subnormal at `x = -184`); the
  error is up to `3.1e15` units of `2^-1074`. numerics-js is within 1 unit at every point.
- Near its zeros at `x = 1` and `x = 2`, `gammaln` has an absolute error of about `1e-16`.
  That is a relative error of 4% at a distance of `1e-15` from the zero, `4e-3` at
  `1e-12`, and `1e-6` at `1e-9`, and it is up to 191 ulp over `lgamma_small`.
  numerics-js keeps the relative accuracy (within 0.67 ulp in `lgamma_zeros`).
- Elsewhere both implementations are within a few ulp. SciPy is slightly less accurate
  on the negative axis (`gamma_negative_large`: 6.34 against 4.08 ulp).
- Claimed bounds: `gamma` within 6 ulp (observed maximum 4.08 here, and 5.1 over about 45,000
  points in a wider offline scan), `lgamma` within 2 ulp for `x > 0` (observed 1.66) and
  within 5 ulp of the larger reflection term for `x < 0` (observed 2.22 here, 4.8 offline).
