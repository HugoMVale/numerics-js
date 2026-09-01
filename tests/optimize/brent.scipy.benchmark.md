# Brent SciPy Benchmark Results

Reference: `scipy.optimize.minimize_scalar(method="brent")` with `xtol = 1e-8`.
The numerics-js solver uses `tolX = 1e-8` for every case.

| Test problem                       | SciPy evaluations | numerics-js evaluations | Ratio |
| :--------------------------------- | :---------------: | :---------------------: | ----: |
| `(x - 2)^2` on `[-5, 5]`           |         8         |            6            | 0.750 |
| `x^4 - x + 1` on `[-3, 3]`         |        16         |           19            | 1.188 |
| `abs(x - 0.5)` on `[-1, 2]`        |        20         |           19            | 0.950 |
| `sin(5 x) + (x - 1)^2` on `[0, 2]` |        12         |           11            | 0.917 |
| `exp(50 (x - 0.3)^2)` on `[0, 1]`  |        12         |           10            | 0.833 |

The test enforces `EVALUATION_FACTOR = 1.2`; the highest observed ratio is
`1.188` for `x^4 - x + 1`.