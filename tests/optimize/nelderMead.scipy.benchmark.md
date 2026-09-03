# Nelder-Mead vs SciPy Benchmark Results

Reference: `scipy.optimize.minimize(method="Nelder-Mead", options={"adaptive": True})`
with `xatol = 1e-8` and `fatol = 1e-14`. The numerics-js solver uses the
matching `tolX = 1e-8`, `tolF = 1e-14` (both adaptive) for every case.
`fatol`/`tolF` is set far below `xatol`/`tolX` so both solvers terminate on
the x-tolerance criterion; nelderMead's default (auto-inferred) `scale` is
used, unmodified, as it would be by a typical caller. Test problems and
starting points mirror `TEST_FUNCTIONS_MULTIVAR` in
[testFunctions.ts](../../tests/optimize/testFunctions.ts), evaluated at `N = 2`.

| Test problem  | SciPy evaluations  | numerics-js evaluations  | Ratio |
| :------------ | :----------------: | :----------------------: | ----: |
| `sphere`      |        137         |           119            | 0.869 |
| `ellipsoid`   |        240         |           221            | 0.921 |
| `rosenbrock`  |        202         |           173            | 0.856 |
| `zakharov`    |        149         |           132            | 0.886 |

The test enforces `EVALUATION_FACTOR = 1.1`; the highest observed ratio is
`0.921` for `ellipsoid`.

Despite biasing `fatol`/`tolF` toward the x-tolerance criterion, both solvers
still report `tolF`-driven termination messages: near a flat minimum
(`f ~ ||x||^2`), function values already sit at floating-point noise (`~1e-15`)
once `x` differences reach `~1e-8`, so `fatol`/`tolF` still ends up binding
first in practice. The test tolerances (`X_TOLERANCE = 2e-7`,
`F_TOLERANCE_FACTOR = 10`, i.e. `1e-13`) are calibrated to the resulting
observed differences rather than to a specific termination criterion.
