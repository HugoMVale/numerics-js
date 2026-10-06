# numerics-js

[![CI](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml/badge.svg)](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/HugoMVale/numerics-js/graph/badge.svg?token=WrJkecJXR0)](https://codecov.io/gh/HugoMVale/numerics-js)

A small, dependency-free numerical methods library for JavaScript and
TypeScript.

## Why?

Scientific code often needs a handful of reliable building blocks without the
weight of a full application framework. numerics-js provides focused numerical
algorithms, typed data structures, and native ESM modules that work in Node.js
and the browser.

The library favors explicit APIs, predictable results, and composable methods:
use a solver with `Vector` and `Matrix`, pass its output to an interpolator or
quadrature routine, and keep the surrounding code ordinary TypeScript.

## Install

```bash
npm install numerics-js
```

## Example

This example models cooling, samples the solution, interpolates between steps,
and computes the accumulated temperature above ambient.

```ts
import { linalg, integrate, interpolate, ode } from 'numerics-js';

const solution = ode.rungeKuttaAdaptive(
    'rk45',
    (_t, y, dydt) => dydt.set([-(y.data[0] - 20) * 0.4]),
    0,
    10,
    new linalg.Vector([100]),
);

const times = solution.t;
const temperatures = solution.y.col(0);
const temperature = new interpolate.PchipInterpolator1D(times, temperatures);
const accumulatedHeat = integrate.quad((time) => temperature.eval(time) - 20, 0, 10);

console.log(temperature.eval(2.25));
console.log(accumulatedHeat.value);
```

## What’s included

- `linalg`: `Vector`, `Matrix`, and `Vec3`, with solvers, factorizations, and eigenvalue methods.
- `integrate`: sampled-data rules and adaptive quadrature, including infinite intervals.
- `interpolate`: linear and shape-preserving interpolation for scalar and vector-valued data.
- `numdiff`: finite-difference derivatives and Jacobians.
- `ode`: fixed-step and adaptive Runge-Kutta solvers, plus velocity Verlet integration.
- `optimize`: scalar Brent minimization and multidimensional Nelder-Mead optimization.
- `roots`: scalar bisection, Brent, and secant methods, plus quasi-Newton system solving.
- `special`: Bessel functions of the first kind and their positive zeros.
- `math`: clipping, signed magnitude, approximate equality, and seeded random numbers.

The package is native ESM and exposes both the package root and focused module
subpaths such as `numerics-js/linalg`, `numerics-js/ode`, and
`numerics-js/special/bessel`.

## Documentation

Read the [API reference](https://hugomvale.github.io/numerics-js/) for complete
signatures, options, guarantees, and examples.

## Development

```bash
npm install
npm test
npm run typecheck
npm run lint
npm run build
npm run docs
```

The generated API site is written to `docs/`.
