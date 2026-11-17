# quad vs SciPy Benchmark Results

Reference: `scipy.integrate.quad` (QUADPACK QAGS for finite intervals, QAGP with
breakpoints and QAGI for infinite intervals; 21-point Gauss-Kronrod rule, whereas
`quad` uses the 15-point rule). Both solvers use an absolute and a relative tolerance
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

The test enforces `EVALUATION_FACTOR = 1.2` for every case, singular ones
included; the highest observed ratio is `1.122` for the Runge case.

## Infinite Domains

The benchmark also covers `quad`'s variable transformations for positive
semi-infinite, negative semi-infinite, and fully infinite intervals.

| Test problem                         | SciPy evaluations | numerics-js evaluations | Ratio |
| :----------------------------------- | ----------------: | ----------------------: | ----: |
| `exp(-x)` on [0, infinity)           |               135 |                     135 | 1.000 |
| `exp(x)` on (-infinity, 0]           |               135 |                     135 | 1.000 |
| `exp(-x^2)` on (-infinity, infinity) |               270 |                     285 | 1.056 |

The breakpoint case passes the same split point to SciPy's `points` argument
and `quad`'s `breakpoints` option. Infinite bounds are encoded as strings in
the JSON fixture because JSON has no native representation for Infinity.

## Singular Cases

Integrable endpoint singularities are handled by the extrapolation (Wynn's epsilon
algorithm), as in QAGS, and are checked for both value accuracy and evaluation counts.

| Test problem            | SciPy evaluations | numerics-js evaluations | Ratio |
| :---------------------- | ----------------: | ----------------------: | ----: |
| Inverse square root     |               231 |                     165 | 0.714 |
| Negative logarithm      |               231 |                     135 | 0.584 |
