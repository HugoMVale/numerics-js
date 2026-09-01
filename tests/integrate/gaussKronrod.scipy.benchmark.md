# Gauss-Kronrod SciPy Benchmark Results

Reference: `scipy.integrate.quad` (QUADPACK QAGS). Both solvers use the
default absolute tolerance `1e-8` for every case.

| Test problem            | SciPy evaluations | numerics-js evaluations | Ratio |
| :---------------------- | ----------------: | ----------------------: | ----: |
| Sine                    |                21 |                      15 | 0.714 |
| Rational pi             |                21 |                      15 | 0.714 |
| Oscillatory sine        |                21 |                      15 | 0.714 |
| Gaussian                |                63 |                      75 | 1.190 |
| Runge                   |               147 |                     165 | 1.122 |
| High-frequency cosine   |               147 |                     105 | 0.714 |
| Lorentzian peak         |               399 |                     465 | 1.165 |
| Exponential growth      |                21 |                      15 | 0.714 |

The test enforces `EVALUATION_FACTOR = 1.2` for non-singular integrands; the
highest observed ratio is `1.190` for the Gaussian case.

## Singular Cases

Endpoint-singular integrands are checked for value accuracy but excluded from
the evaluation-count comparison: this implementation does not provide QAGS's
singularity-aware extrapolation.

| Test problem            | SciPy evaluations | numerics-js evaluations | Ratio |
| :---------------------- | ----------------: | ----------------------: | ----: |
| Inverse square root     |               231 |                   1,395 | 6.039 |
| Negative logarithm      |               231 |                     615 | 2.662 |