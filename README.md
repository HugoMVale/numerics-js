# numerics-js

[![CI](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml/badge.svg)](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml)

Small, dependency-free numerical building blocks for modern JavaScript.

## Install

```bash
npm install numerics-js
```

For local development:

```bash
npm install
npm test
```

## Import

The package publishes native ESM with export conditions for browser bundlers and Node ESM. The package root mirrors the source-module layout, and focused subpaths are available for each module.

```ts
import { array, ode, roots, special } from 'numerics-js';

const vector = new array.Array1D([1, 2, 3]);
const root = roots.bisection((x) => x * x - 2, 1, 2);
const solution = ode.rk4Integrate(/* ... */);
const value = special.bessel.J(0, 1.5);
```

The same modules can be imported through `numerics-js/array`,
`numerics-js/roots`, `numerics-js/ode`, `numerics-js/ode/rk4`,
`numerics-js/ode/verlet`, `numerics-js/special`, and
`numerics-js/special/bessel`.

## Package layout

- ESM only: the package is published as native ES modules.
- Export conditions: browser and Node consumers resolve the compiled `dist` modules through the package export map.

## API

### Vectors and matrices

- `Array1D` is an N-component vector backed by `Float64Array`. It provides addition, subtraction, scaling, norms, dot products, distances, min/max/sum, closeness checks, conversion, iteration, and in-place operations.
- `Array2D` is a row-major matrix backed by `Float64Array`. It provides element/row/column access, arithmetic, matrix and vector multiplication, transpose, trace, rank, determinant, inverse, linear-system solving, reductions, conversion, iteration, and in-place row operations.
- `Vec3` is a three-component vector for positions, directions, and velocities. It provides the same core vector operations as `Array1D`, plus tuple conversion and static constructors.

`Array1D` component indices and `Array2D` row and column indices are 1-based
through their `get` and `set` accessors. The public `data` properties expose
the underlying zero-based `Float64Array` storage. `Array1D` and `Vec3`
operations that do not end in `Self` return new values; methods ending in
`Self` mutate the instance.

```ts
const a = new Array1D([1, 2, 3]);
const b = new Array1D([4, 5, 6]);
const sum = a.add(b);

const matrix = new Array2D(2, 2, [1, 2, 3, 4]);
const product = matrix.mulVec(new Array1D([10, 20]));
const direction = new Vec3(3, 4, 0).normalize();
const solved = matrix.solve(new Array1D([5, 11]));
```

`Array1D` and `Array2D` also provide `zero` and `from` constructors. `Vec3`
components are accessed through its `x`, `y`, and `z` properties.

### Root finding

`bisection` requires a bracketing interval whose endpoint values have opposite signs. `secant` starts from two guesses and does not require a bracket; it can fail to converge for some functions.

```ts
const root = bisection((x) => x * x - 2, 1, 2);
const otherRoot = secant((x) => Math.cos(x) - x, 0, 1);
```

Both functions accept optional `tolerance` and `maxIter` arguments. If
the iteration limit is reached, they warn and return the latest approximation.
`bisection` throws when the interval does not bracket a root; `secant` throws
when its guesses are equal or its denominator becomes zero.

### Bessel functions

`bessel.J(n, x)` evaluates the Bessel function of the first kind for a non-negative integer order. `bessel.getZero(n, m)` returns the m-th positive zero, with `m` counted from 1.

```ts
const value = bessel.J(0, 1.5);
const firstZero = bessel.getZero(0, 1);
```

### Ordinary differential equations

`rk4Step` advances one state by the classic fourth-order Runge-Kutta method.
`rk4Integrate` uses a constant step size, records the initial state and every
step through `tEnd`, and uses a final partial step when needed. Both accept a
derivative callback with the signature `(t, y, dydt) => dydt`; the callback
must write into `dydt`, return it, and must not mutate `y`.

```ts
const initial = new Array1D([1]);
const solution = rk4Integrate(
    (t, y, dydt) => dydt.set([-y.data[0]]),
    0,
    1,
    initial,
    0.1,
);

console.log(solution.t.data[solution.t.size - 1]); // 1
console.log(solution.y.row(solution.y.rows));    // state at t = 1
```

`dp54Step` and `dp54Integrate` implement adaptive Dormand-Prince 5(4)
(RK45). `dp54Step` computes one proposed fifth-order step. `dp54Integrate`
records the initial state and every accepted adaptive step, supports backward
integration, and accepts `{ atol, rtol, h0, hMax, hMin, maxSteps, safety,
minScale, maxScale }` options. `atol` defaults to `1e-6`, `rtol` to `1e-3`,
and `h0` is estimated when omitted.

```ts
const adaptive = dp54Integrate(
    (t, y, dydt) => dydt.set([-y.data[0]]),
    0,
    1,
    new Array1D([1]),
    { atol: 1e-10, rtol: 1e-10 },
);
```

Both integration functions return `{ t: Array1D, y: Array2D }`, where `t`
contains recorded times and row `i` of `y` contains the corresponding state.
For allocating derivative functions with the simpler `(t, y) => Array1D`
signature, use `wrapAllocatingDerivative`; the direct callback form avoids
per-stage allocations in solver loops.

`createVelocityVerlet` creates a mutable `step(state, dt)` function for position and velocity updates. A state contains `u`, `v`, `a`, and a reusable `aNext` array.

```ts
const step = createVelocityVerlet((u, v, aOut) => {
    for (let i = 0; i < aOut.length; i++) aOut[i] = -9.81;
});

const state = { u: [0], v: [0], a: [-9.81], aNext: [0] };
step(state, 0.016);
```

The callback receives positions and the current velocity and writes the new
acceleration into `aOut`. The step mutates `u`, `v`, and `a`; `aNext` must be a
reusable array of the same length. The implementation evaluates acceleration
with the old velocity, which is exact for velocity-independent forces but is
only first-order for velocity-dependent forces such as drag.

## Documentation

The API reference is available at [hugomvale.github.io/numerics-js](https://hugomvale.github.io/numerics-js/).
Generate the documentation locally with:

```bash
npm run docs
```

The generated site is written to `docs/`.

## Development

```bash
npm test          # run the test suite once
npm run test:watch
```

The source is written in TypeScript. Run `npm run build` to emit the package files into `dist/`.
