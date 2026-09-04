# numerics-js

[![CI](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml/badge.svg)](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml)

A small, high-quality, mathematically literate numerical toolbox for modern JavaScript/TypeScript.

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
import { array, integrate, interpolate, math, ode, optimize, roots, special } from 'numerics-js';

const vector = new array.Array1D([1, 2, 3]);
const root = roots.bisection((x) => x * x - 2, 1, 2);
const solution = ode.rungeKuttaFixed('rk4', /* ... */);
console.log(root.x);
const value = special.bessel.J(0, 1.5);
```

The same modules can be imported through `numerics-js/array`,
`numerics-js/integrate`, `numerics-js/interpolate`, `numerics-js/math`,
`numerics-js/roots`, `numerics-js/ode`, `numerics-js/optimize`, and
`numerics-js/special`.

## Package layout

- ESM only: the package is published as native ES modules.
- Export conditions: browser and Node consumers resolve the compiled `dist` modules through the package export map.

## API

### Vectors and matrices

- `Array1D` is an N-component vector backed by `Float64Array`. It provides addition, subtraction, scaling, norms, dot products, distances, min/max/sum, closeness checks, conversion, iteration, and in-place operations.
- `Matrix` is a row-major matrix backed by `Float64Array`. It provides element/row/column access, arithmetic, matrix and vector multiplication, transpose, trace, rank, determinant, inverse, linear-system solving, reductions, conversion, iteration, and in-place row operations.
- `Vec3` is a three-component vector for positions, directions, and velocities. It provides the same core vector operations as `Array1D`, plus tuple conversion and static constructors.

`Array1D` component indices and `Matrix` row and column indices are zero-based,
including through their `get`, `set`, `row`, and `col` accessors. Their public
`data` properties expose the underlying zero-based `Float64Array` storage. `Array1D` and `Vec3`
operations that do not end in `Self` return new values; methods ending in
`Self` mutate the instance.

```ts
const a = new Array1D([1, 2, 3]);
const b = new Array1D([4, 5, 6]);
const sum = a.add(b);

const matrix = new Matrix(2, 2, [1, 2, 3, 4]);
const product = matrix.mulVec(new Array1D([10, 20]));
const direction = new Vec3(3, 4, 0).normalize();
const solved = matrix.solve(new Array1D([5, 11]));
```

`Array1D` and `Matrix` also provide `zero` and `from` constructors. `Vec3`
components are accessed through its `x`, `y`, and `z` properties.

### Integration

`trapezoid` and `simpson` integrate sampled `Array1D` values. They accept optional `Array1D` sample locations for non-uniform spacing, or a uniform `dx`
spacing that defaults to `1`.

`gaussKronrod` integrates a scalar function over an interval with adaptive 7-15 Gauss-Kronrod quadrature and returns
`{ value, error, evaluations, converged, subintervals }`.

```ts
const x = new array.Array1D([0, 1, 3]);
const y = new array.Array1D([0, 1, 9]);
const area = integrate.simpson(y, x);
const adaptiveArea = integrate.gaussKronrod((x) => Math.exp(-x * x), 0, 1);
```

### Interpolation

`interp` performs one-dimensional linear interpolation of a scalar or vector of
query points. `LinearInterpolator` and `PchipInterpolator` prevalidate and
reuse data for repeated evaluation; both expose `eval` and `derivative`.
`PchipInterpolator` is shape-preserving and avoids overshoots between monotonic
data points. Values outside the input range are clamped by default or can use
caller-supplied left and right values.

```ts
const linear = new interpolate.LinearInterpolator([0, 1, 2], [0, 1, 4]);
const smooth = new interpolate.PchipInterpolator([0, 1, 2], [0, 1, 4]);
const value = smooth.eval(1.5);
```

### Root finding

`bisection` and root-finding `brent` require a bracketing interval whose
endpoint values have opposite signs. Brent's method combines bisection,
secant, and inverse quadratic interpolation. `secant` starts from two guesses
and does not require a bracket; it can fail to converge for some functions.

All three return a `RootResult`: `{ method, evaluations, x, fx }`, giving the
approximate root, the function value there, the method name, and the number of
function evaluations used.

```ts
const root = bisection((x) => x * x - 2, 1, 2);
const robustRoot = brent((x) => Math.cos(x) - x, 0, 1);
const otherRoot = secant((x) => Math.cos(x) - x, 0, 1);
console.log(root.x, root.fx, root.evaluations);
```

All root finders accept optional tolerances and an iteration limit. `bisection`
and `secant` warn and return their latest approximation when the iteration limit
is reached; `brent` returns its latest approximation. The bracketing methods
throw for an invalid interval, while `secant` throws when its guesses are equal
or its denominator becomes zero.

### Optimization

`optimize.brent` finds a local minimum of a scalar function within an interval,
returning its location, value, evaluation count, iteration count, and convergence
status. `optimize.nelderMead` minimizes an unconstrained multivariate function
from an `Array1D` or array initial guess and returns the same convergence data.

```ts
const scalarMinimum = optimize.brent((x) => (x - 2) ** 2, -1, 5);
const vectorMinimum = optimize.nelderMead(
    (x) => (x.get(0) - 1) ** 2 + (x.get(1) + 1) ** 2,
    [0, 0],
);
```

### Bessel functions

`bessel.J(n, x)` evaluates the Bessel function of the first kind for a non-negative integer order. `bessel.getZero(n, m)` returns the m-th positive zero, with `m` counted from 1.

```ts
const value = bessel.J(0, 1.5);
const firstZero = bessel.getZero(0, 1);
```

### Ordinary differential equations

`rungeKuttaFixed` integrates using one of `'euler'`, `'midpoint'`,
`'trapezoid'`, or classic fourth-order `'rk4'` with a constant step size. It
records the initial state and every step through `tEnd`, using a final partial
step when needed.

`rungeKuttaAdaptive` integrates with the adaptive
Bogacki-Shampine 3(2) (`'rk23'`) or Dormand-Prince 5(4) (`'rk45'`) method. It
supports forward and backward integration and accepts positional controls for
absolute tolerance, relative tolerance, initial and maximum step size, minimum
step size, iteration limit, and controller scaling.

Both solvers accept a derivative callback with the signature
`(t, y, dydt) => dydt`; it must write into `dydt`, return it, and must not
mutate `y`.

```ts
const initial = new Array1D([1]);
const solution = rungeKuttaFixed(
    'rk4',
    (t, y, dydt) => dydt.set([-y.data[0]]),
    0,
    1,
    initial,
    0.1,
);

console.log(solution.t.data[solution.t.size - 1]); // 1
console.log(solution.y.row(solution.y.rows - 1)); // state at t = 1
```

`rungeKuttaAdaptive` records the initial state and every accepted adaptive
step. Its `atol` and `rtol` defaults are `1e-6` and `1e-3`, and it estimates
`h0` when it is omitted.

```ts
const adaptive = rungeKuttaAdaptive(
    'rk45',
    (t, y, dydt) => dydt.set([-y.data[0]]),
    0,
    1,
    new Array1D([1]),
    1e-10,
    1e-10,
);
```

Both integration functions return `{ t: Array1D, y: Matrix, method }`, where `t`
contains recorded times, row `i` of `y` contains the corresponding state, and
`method` identifies the solver that produced the result.
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
