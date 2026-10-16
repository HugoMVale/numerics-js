# Levenberg-Marquardt vs SciPy `leastsq` Benchmark Results

Reference: `scipy.optimize.leastsq` (MINPACK `lmdif` with a finite-difference
Jacobian, and `lmder` with an analytic one), with `ftol = xtol = 1.49012e-8`,
`gtol = 0`, `factor = 100`, and SciPy's default evaluation budget
(`100*(n+1)` for `lmdif`, `200*(n+1)` for `lmder`). numerics-js runs with the
same tolerances and budget and with `jacCheck: false`. Both solvers use
MINPACK's adaptive scaling, so the trust-region logic is the same; the only
algorithmic difference is the finite-difference step (SciPy: `sqrt(eps)*|x|`;
numerics-js: `sqrt(eps)*max(|x|, 1/sclx)`), which matters only near `x = 0`.

The test problems are Moré-Garbow-Hillstrom problems and a rank-deficient
linear problem. The Python generator and the TypeScript test use the same
problems (functions and Jacobians), and the test verifies that the TypeScript
functions agree with the Python ones at the initial point and at generic points
within `MIRROR_TOLERANCE = 1e-12`.

The test checks that numerics-js reports success, reaches a sum of squares no
larger than SciPy's (within a relative `1e-6`) as well as the known minimum,
finds the same minimizer when it is unique, and needs no more than
`EVALUATION_FACTOR = 1.1` times SciPy's evaluations of `f` (and of the
Jacobian). The observed ratio is at most `1.0` on all cases.

Notes:

- **Freudenstein and Roth:** from the standard starting point, both solvers
  converge to the local minimum `||f||² = 48.98425` at `(11.412, -0.8968)`, not
  to the zero at `(5, 4)`.
- **Watson:** SciPy stops with `ier = 1` at `||f||² = 2.606e-3`, whereas
  numerics-js reaches the literature minimum `2.288e-3` in the same number of
  evaluations. The paths differ because of the finite-difference step at the
  starting point `x0 = 0`. Only the sum of squares is compared on this problem.
- **Powell singular:** the Jacobian is singular at the solution, so the minimizer is only determined to about `1e-8` (it is compared with an absolute tolerance of `1e-5`) and convergence
  is linear. With finite differences, SciPy exhausts its budget (`ier = 5`)
  after 503 evaluations; numerics-js converges after 278. With the analytic
  Jacobian, both stop at machine precision (`ier = 8`, `||f||² = 3.7e-67`).
  numerics-js reports this as success, because the residual is negligible
  compared to the initial one; SciPy returns `ier = 8`, which is not in its
  list of success codes.
- **Rank-deficient problem:** the minimizer is not unique, so only the minimum
  is compared.
