# quad vs SciPy Benchmark Results

Reference: `scipy.integrate.quad` (QUADPACK QAGS for finite intervals and QAGI
for infinite intervals). Both solvers use an absolute and a relative tolerance
of `1e-8` for every case (`epsabs = epsrel = 1e-8` in SciPy; the default
`atol = rtol = 1e-8` in numerics-js).

| Test problem                        | SciPy evaluations | numerics-js evaluations | Ratio |
| :---------------------------------- | ----------------: | ----------------------: | ----: |
| Sine                                |                21 |                      15 | 0.714 |
| Rational pi                         |                21 |                      15 | 0.714 |
| Oscillatory sine                    |                21 |                      15 | 0.714 |
| Gaussian                            |                63 |                      45 | 0.714 |
| Runge                               |               147 |                     165 | 1.122 |
| High-frequency cosine               |               147 |                     105 | 0.714 |
| Lorentzian peak                     |               399 |                     405 | 1.015 |
| Exponential growth                  |                21 |                      15 | 0.714 |
| Piecewise constant, breakpoint at 1 |                42 |                      30 | 0.714 |

The test enforces `EVALUATION_FACTOR = 1.2` for non-singular integrands; the
highest observed ratio is `1.122` for the Runge case.

## Infinite Domains

The benchmark also covers `quad`'s variable transformations for positive
semi-infinite, negative semi-infinite, and fully infinite intervals.

| Test problem                         | SciPy evaluations | numerics-js evaluations | Ratio |
| :----------------------------------- | ----------------: | ----------------------: | ----: |
| `exp(-x)` on [0, infinity)           |               135 |                     105 | 0.778 |
| `exp(x)` on (-infinity, 0]           |               135 |                     105 | 0.778 |
| `exp(-x^2)` on (-infinity, infinity) |               270 |                     225 | 0.833 |

The breakpoint case passes the same split point to SciPy's `points` argument
and `quad`'s `breakpoints` option. Infinite bounds are encoded as strings in
the JSON fixture because JSON has no native representation for Infinity.

## Singular Cases

Endpoint-singular integrands are checked for value accuracy but excluded from
the evaluation-count comparison: this implementation does not provide QAGS's
singularity-aware extrapolation.

| Test problem            | SciPy evaluations | numerics-js evaluations | Ratio |
| :---------------------- | ----------------: | ----------------------: | ----: |
| Inverse square root     |               231 |                   1,305 | 5.649 |
| Negative logarithm      |               231 |                     585 | 2.532 |
