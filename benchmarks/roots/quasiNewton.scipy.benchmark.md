# Quasi-Newton vs SciPy HYBRD Benchmark Results

Reference: `scipy.optimize.root(method="hybr")` with `xtol = 1e-10`.
The numerics-js solver uses `globalMethod = "dogleg"`, `tolx = 1e-10`, and
`tolf = 0`, with Broyden updates enabled. SciPy HYBRD uses a scaled relative
iterate-change test; quasiNewton uses `||Δx/max(x, 1/sclx)||∞`, so the step-size
tests are not mathematically identical. An exactly zero residual can satisfy
quasiNewton's `tolf = 0` check before its `tolx` check. Meeting `tolx` reports
success, though the residual is also checked independently.

The Python generator and TypeScript tests use the same eight cases and initial
points. The test verifies that the TypeScript function values at each initial
point, SciPy solution, and generic mirror points match the Python references
within `MIRROR_TOLERANCE = 1e-12`. Residuals below are unscaled infinity norms
`||f(x)||∞` at each method's returned solution. SciPy `nfev` includes calls
used for finite-difference Jacobian estimates; numerics-js function evaluations
also include finite-difference Jacobian calls.

SciPy reports no progress for Powell singular, but returns a residual of
`1.607e-35`, below the benchmark acceptance threshold of `1e-7`. Its status is
reported as unsuccessful rather than treated as a convergence success.

The test enforces `EVALUATION_FACTOR = 1.5`; the highest observed evaluation
ratio is `1.083` for the helical valley problem.

| Test problem                                           | SciPy evaluations | SciPy residual | numerics-js evaluations | numerics-js residual | Ratio | numerics-js stop            |
| :----------------------------------------------------- | -----------------:| --------------:| -----------------------:| --------------------:| -----:|:--------------------------  |
| Extended Rosenbrock (4 variables)                      |                32 |              0 |                      12 |                    0 | 0.375 | Exact residual (`tolf = 0`) |
| Extended Powell singular (4 variables)                 |               112 |      1.607e-35 |                     104 |            2.317e-13 | 0.929 | `tolx`                      |
| Trigonometric (10 variables)                           |               125 |              0 |                      21 |            1.213e-14 | 0.168 | `tolx`                      |
| Trigonometric (5 variables)                            |                83 |              0 |                      15 |            1.865e-16 | 0.181 | `tolx`                      |
| Extended Rosenbrock (100 variables)                    |               224 |              0 |                     108 |                    0 | 0.482 | Exact residual (`tolf = 0`) |
| Powell badly scaled (2 variables)                      |               181 |              0 |                     102 |            2.442e-15 | 0.564 | `tolx`                      |
| Helical valley (3 variables)                           |                24 |              0 |                      26 |            4.041e-15 | 1.083 | `tolx`                      |
| Example 6.5 of Dennis & Schnabel (1996), x0 = [1.5, 1] |                12 |      3.175e-14 |                      11 |            2.709e-14 | 0.917 | `tolx`                      |

All numerics-js results report success; both solvers' returned residuals are
checked against `1e-7`. SciPy residuals shown as `0` are exactly zero in the
stored reference fixtures.
