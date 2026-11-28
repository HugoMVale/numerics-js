# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, minor releases may contain breaking changes.

## [Unreleased]

### Added

- `linalg`: `BandedMatrix`, a square matrix with `kl` subdiagonals and `ku` superdiagonals in LAPACK band storage (`fromMatrix`, `fromBands`, `get`, `set`, `mulVec`, `toMatrix`). `lu()` is a partial-pivoted LU (`dgbtrf` semantics, `O(n * kl * (kl + ku))`) returning a `BandedLUDecomposition` that `BandedMatrix.solveLU()` reuses for any number of right-hand sides (`O(n * (kl + ku))`, allocation-free with `out`); `solve()` does both. A zero pivot throws `Error`, a `NaN` entry gives an all-`NaN` result, an infinite entry throws `RangeError`, and entries near the limits of the double range are scaled by a power of two, as in the dense `Matrix.lu()`.
- `linalg`: `tridiagonalLU`, `solveTridiagonalLU` and `solveTridiagonal`, the pivoted tridiagonal LU (`dgttrf`) that the cubic splines use, now public with input validation and the same non-finite and scaling policy.
- Benchmarks: SciPy/LAPACK comparison of the banded and tridiagonal LU, and `benchmarks/linalg/banded.perf.mjs`, a timing check of the predicted scaling.
- `ode`: `trBdf2Adaptive` takes the options `kl` and `ku` (numbers of sub- and superdiagonals of the Jacobian) and then works in band storage, for method-of-lines systems on 1D and 2D grids: the Jacobian, the iteration matrix `I - h D J` and its LU factors take `O(n (kl + ku))` memory instead of `O(n^2)`, and each Newton iteration `O(n (kl + ku))` time instead of `O(n^2)`; no `n x n` array is allocated. A finite-difference Jacobian costs `min(kl + ku + 1, n) + 1` evaluations of `f` per refresh instead of `n + 1` (all counted in `evaluations`), and an analytic one is a new `BandedJacobianFunction` that fills a `BandedMatrix` (`BandedMatrix.set` throws a `RangeError` for any entry outside the band, even `0`). A band with `kl + ku >= n - 1` runs banded on the full band. The steps are those of the dense mode up to rounding. The band is not verified, and periodic boundary conditions add corner entries to the Jacobian and are not banded. Fisher-KPP, rtol 1e-4, analytic Jacobian: 0.75 s to 0.044 s (1D, n = 800), 3.1 s to 0.054 s (1D, n = 1600) and 17.7 s to 0.62 s (2D, 50 x 50). Without `kl` and `ku` the results are bit-identical to before. `benchmarks/ode/trBdf2Adaptive.banded.perf.mjs` is the hand-run timing script.
- `numdiff`: `jacobianForwardBanded`, the forward-difference Jacobian of a function with a banded Jacobian, as a `BandedMatrix`. Columns `kl + ku + 1` apart are perturbed together (Curtis-Powell-Reid), so it needs `min(kl + ku + 1, n)` evaluations of `f` instead of `n`, with the step sizes of `jacobianForward`. The band is not verified.

### Changed

- `ode`: the `jac` option of `trBdf2Adaptive` is typed `JacobianFunction | BandedJacobianFunction`, so that it can serve both layouts. A TypeScript caller that passes an inline callback with untyped parameters (`jac: (t, y, J) => ...`) must now annotate it (`jac: ((t, y, J) => ...) satisfies JacobianFunction`) or declare it as a `JacobianFunction`; callbacks declared with their type, and JavaScript callers, are unaffected, and the dense behavior is unchanged.
- `interpolate`: the tridiagonal solver moved from `src/interpolate/tridiagonal.ts` to `src/linalg/tridiagonal.ts`; the cubic splines are unchanged and use its allocation-free internal kernels.

## [0.1.0] - 2026-11-27

First public release.

### Added

- `linalg`: `Vector`, `Matrix` and `Vec3`, with solvers, factorizations (LU, QR, Cholesky, SVD), least squares, pseudoinverse and eigenvalue methods.
- `integrate`: sampled-data rules and adaptive quadrature, including infinite intervals.
- `interpolate`: linear, PCHIP and cubic spline interpolation for scalar and vector-valued data.
- `numdiff`: finite-difference derivatives, gradients, Jacobians and Hessians.
- `ode`: fixed-step and adaptive Runge-Kutta solvers, an adaptive implicit solver for stiff problems, event detection and velocity Verlet integration.
- `optimize`: derivative-free and quasi-Newton minimization, Levenberg-Marquardt nonlinear least squares and curve fitting with parameter uncertainties.
- `roots`: bisection, Brent and secant methods, plus quasi-Newton system solving.
- `special`: Bessel functions of integer order, the error function and its complement, and the gamma function and its logarithm.
- `math`: clipping, signed magnitude, approximate equality and seeded random numbers.
- Native ESM package with a package-root entry point and focused subpath exports, bundled type declarations and no runtime dependencies.

[Unreleased]: https://github.com/HugoMVale/numerics-js/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HugoMVale/numerics-js/releases/tag/v0.1.0
