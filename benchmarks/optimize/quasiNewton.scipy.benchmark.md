# Quasi-Newton minimization vs SciPy Benchmark Results

Reference: `scipy.optimize.minimize`, using for each solver mode the closest SciPy algorithm. The test
problems and starting points are defined in
[gradientTestFunctions.ts](../../tests/optimize/gradientTestFunctions.ts). The tables list the number
of function (`nfev`), gradient (`njev`) and Hessian (`nhev`) evaluations, as named by SciPy; for
numerics-js they are the `evaluations`, `evaluationsGradient` and `evaluationsHessian` of the result.
The numerics-js solver is called with `gradCheck: false`.

## BFGS with analytic gradient

Reference: `method="BFGS"` with `jac` and `gtol=1e-10`. The solver is run with `tolg = 1e-8`, the
strictest tolerance at which both strategies below terminate successfully on every problem (the line
search cannot always resolve a decrease of `f` at `1e-10`).

Evaluations (`nfev` / `njev`):

| Test problem | SciPy BFGS | numerics-js, dogleg + factored BFGS (default) | numerics-js, line search + inverse BFGS |
| :----------- | :--------: | :-------------------------------------------: | :-------------------------------------: |
| `rosenbrock` |  41 / 41   |                    44 / 34                    |                 34 / 22                 |
| `wood`       |  108 / 108 |                    84 / 58                    |                 54 / 32                 |
| `beale`      |  19 / 19   |                    19 / 17                    |                 24 / 21                 |
| `quadratic`  |   9 / 9    |                     9 / 8                     |                  9 / 7                  |

## BFGS with finite-difference gradient

Reference: `method="BFGS"` without `jac` (SciPy approximates the gradient by forward differences),
with `gtol=1e-6`, as the finite-difference gradient is only accurate to about `1e-8`. The solver is
run without `grad` and with its default `tolg = 1e-5`.

Function evaluations (`nfev`), including those used by the finite-difference gradient:

| Test problem | SciPy BFGS | numerics-js, dogleg + factored BFGS (default) | numerics-js, line search + inverse BFGS |
| :----------- | :--------: | :-------------------------------------------: | :-------------------------------------: |
| `rosenbrock` |    120     |                      114                      |                   75                    |
| `wood`       |    520     |                      302                      |                   172                   |
| `beale`      |     51     |                      47                       |                   63                    |
| `quadratic`  |     32     |                      29                       |                   26                    |

## Dogleg with exact Hessian

Reference: `method="dogleg"` with `jac`, `hess` and `gtol=1e-10`, for the problems with an analytic
Hessian. The solver is run with `hess`, the dogleg strategy and `tolg = 1e-8`.

Evaluations (`nfev` / `njev` / `nhev`):

| Test problem | SciPy dogleg | numerics-js, dogleg + exact Hessian |
| :----------- | :----------: | :---------------------------------: |
| `rosenbrock` | 26 / 23 / 22 |            25 / 22 / 21             |
| `quadratic`  |  4 / 4 / 3   |              4 / 2 / 1              |

## Tolerances

The test requires agreement of the solution with SciPy within `1e-7` for analytic derivatives
(largest observed difference: `2.3e-9`) and `1e-4` for finite differences (largest observed
difference: `1.8e-6`, limited by the accuracy of the finite-difference gradients of both solvers).
The function value must agree within `1e-12` and `1e-8`, respectively. In all cases, the number of
function, gradient and Hessian evaluations must not exceed `1.5` times SciPy's (highest observed
ratio: `1.26`, for `beale` with the line search).
