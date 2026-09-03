# Brent vs SciPy Benchmark Results

Reference: `scipy.optimize.minimize_scalar(method="brent")` with `xtol = 1e-8`.
The numerics-js solver uses `tolX = 1e-8` for every case.

| Test problem                       | SciPy evaluations | numerics-js evaluations | Ratio |
| :--------------------------------- | :---------------: | :---------------------: | ----: |
| `(x - 2)^2` on `[-5, 5]`           |         8         |            6            | 0.750 |
| `x^4 - x + 1` on `[-3, 3]`         |        18         |           19            | 1.056 |
| `abs(x - 0.5)` on `[-1, 2]`        |        21         |           19            | 0.905 |
| `sin(5 x) + (x - 1)^2` on `[0, 2]` |        15         |           11            | 0.733 |
| `exp(50 (x - 0.3)^2)` on `[0, 1]`  |        14         |           10            | 0.714 |

The test enforces `EVALUATION_FACTOR = 1.1`; the highest observed ratio is
`1.056` for `x^4 - x + 1`.
