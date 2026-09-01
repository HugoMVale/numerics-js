# Adaptive Runge-Kutta SciPy Benchmark Results

Reference: `scipy.integrate.solve_ivp` with the matching `RK23` or `RK45`
method. Both solvers use `atol = 1e-8` and `rtol = 1e-8` for every case.

| Test problem         | Solver | SciPy evaluations | numerics-js evaluations | Ratio |
| :------------------- | :----: | ----------------: | ----------------------: | ----: |
| Exponential decay    |  RK23  |               965 |                     962 | 0.997 |
| Exponential decay    |  RK45  |               212 |                     212 | 1.000 |
| Logistic growth      |  RK23  |             1,184 |                   1,187 | 1.003 |
| Logistic growth      |  RK45  |               296 |                     290 | 0.980 |
| Harmonic oscillator  |  RK23  |             2,144 |                   2,141 | 0.999 |
| Harmonic oscillator  |  RK45  |               356 |                     356 | 1.000 |
| Backward exponential |  RK23  |             1,625 |                   1,625 | 1.000 |
| Backward exponential |  RK45  |               284 |                     284 | 1.000 |

The test enforces `EVALUATION_FACTOR = 1.01`; the highest observed ratio is
`1.003` for RK23 logistic growth.
