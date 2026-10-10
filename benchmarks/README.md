# Benchmarks

These suites compare `numerics-js` algorithms with independent SciPy reference
implementations. Each module directory contains the TypeScript comparison
tests, the generated JSON reference data, the standalone Python generator, and
the available Markdown results.

The comparison tests run as part of the normal test suite and can be run alone
with:

```bash
npm run test:benchmarks
```

SciPy is only required when regenerating reference data. Run an individual
generator from the repository root with `uv`:

```bash
uv run benchmarks/integrate/generate_quad_fixtures.py
```

The Python generator and its TypeScript test intentionally duplicate the case
definitions. Keep their case ids, functions, bounds, and solver settings in
sync when changing a benchmark.

## Suites

| Module                       | Algorithm                    | Reference                                                                                     |
| ---------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------- |
| [`integrate/`](./integrate/) | Adaptive quadrature          | [`quad.scipy.benchmark.md`](./integrate/quad.scipy.benchmark.md)                              |
| [`linalg/`](./linalg/)       | Eigenvalues and eigenvectors | [`Matrix.eig.scipy.benchmark.md`](./linalg/Matrix.eig.scipy.benchmark.md)                     |
| [`linalg/`](./linalg/)       | LU factorization             | [`Matrix.lu.scipy.benchmark.md`](./linalg/Matrix.lu.scipy.benchmark.md)                       |
| [`linalg/`](./linalg/)       | QR factorization             | [`Matrix.qr.scipy.benchmark.md`](./linalg/Matrix.qr.scipy.benchmark.md)                       |
| [`ode/`](./ode/)             | Adaptive Runge-Kutta         | [`rungeKuttaAdaptive.scipy.benchmark.md`](./ode/rungeKuttaAdaptive.scipy.benchmark.md)        |
| [`optimize/`](./optimize/)   | Brent minimization           | [`brent.scipy.benchmark.md`](./optimize/brent.scipy.benchmark.md)                             |
| [`optimize/`](./optimize/)   | Nelder-Mead minimization     | [`nelderMead.scipy.benchmark.md`](./optimize/nelderMead.scipy.benchmark.md)                   |
| [`roots/`](./roots/)         | Quasi-Newton systems         | [`quasiNewton.scipy.benchmark.md`](./roots/quasiNewton.scipy.benchmark.md)                    |
| [`special/`](./special/)     | Bessel function `J_n(x)`     | [`besselJ.scipy.benchmark.md`](./special/besselJ.scipy.benchmark.md)                          |
| [`special/`](./special/)     | Bessel zeros `j_{n,m}`       | [`besselJZero.scipy.benchmark.md`](./special/besselJZero.scipy.benchmark.md)                  |
