# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, minor releases may contain breaking changes.

## [Unreleased]

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
