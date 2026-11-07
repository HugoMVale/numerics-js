# besselI vs mpmath Benchmark Results

Reference: `mpmath.besseli(n, x)` evaluated at 60 decimal digits and stored as a
double-double pair `hi + lo`, where `hi` is the nearest double and `lo` is the
double nearest to the remainder. The scaled reference `exp(-|x|) I_n(x)` is
computed from the same 60-digit value. SciPy (`scipy.special.ive`, SciPy 1.18.1)
is measured against the same scaled references, so the columns are comparable.
Points where `I_n(x)` exceeds the double range have no plain reference (the test
instead requires `besselI` to return `Infinity`).

Error is the relative error `|actual - reference| / |reference|`. `I_n(x)` has no
zeros for `x > 0`, so no envelope term is needed and a pure relative tolerance is
meaningful everywhere, including deep in the underflow tail.

The test (`besselI.mpmath.test.ts`) checks only numerics-js against the mpmath
references. `rtol` is `1e-14` for the scaled function in every group. For the plain
function it is `1e-14` for `small_x`, `large_x`, `turning_point`, `huge_x_scaled` and
`negative_x`, `2e-14` for `moderate_x`, and `5e-13` for `high_order`, where the
value is recovered through the log domain because `exp(-x) I_n(x)` underflows. The
test does not read the SciPy values.

| Group           | Region                            | Points | `besselI` max rel. error | `besselIScaled` max rel. error | SciPy `ive` max rel. error |
| --------------- | --------------------------------- | -----: | -----------------------: | -----------------------------: | -------------------------: |
| `small_x`       | power series                      |     64 |                  1.1e-15 |                        1.2e-15 |                    4.7e-14 |
| `moderate_x`    | Miller recurrence                 |     80 |                  6.4e-15 |                        1.3e-15 |                    7.7e-14 |
| `large_x`       | Miller recurrence                 |     28 |                  1.4e-15 |                        1.4e-15 |                    6.0e-14 |
| `turning_point` | Miller recurrence                 |     35 |                  4.2e-15 |                        4.2e-15 |                    1.8e-13 |
| `high_order`    | Miller recurrence, underflow tail |     15 |                  7.0e-14 |                        2.1e-15 |                    9.7e-14 |
| `huge_x_scaled` | Miller recurrence, scaled only    |     24 |                  2.4e-15 |                        2.7e-15 |                    9.7e-14 |
| `negative_x`    | parity rule                       |     20 |                  4.3e-16 |                        4.7e-16 |                    1.1e-15 |

`besselI` columns exclude points whose plain value overflows. The only group
above the round-off level is `high_order` for `besselI` (`I_500(100) ~ 3.6e-283`,
error `7.0e-14`): there `exp(-100) I_500(100)` is below the smallest double, so the
value is evaluated as `exp(x + ln u_n - ln sum)` and loses about `|exponent| * eps`.
`besselIScaled` is at round-off level in the same group.

Cost: the Miller start order exceeds `n` by only `sqrt(100 max(n, x))`, so the
work is `O(n + sqrt(x))`. Measured on one machine (Node 22), `besselIScaled(0, 1e6)`
takes about 95 us, whereas `besselJ(0, 1e5)` takes about 5.6 ms.
