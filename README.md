# numerics-js

[![CI](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml/badge.svg)](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml)

Small, dependency-free numerical building blocks for modern JavaScript.

The library uses ES modules and stores numeric array data in `Float64Array` where appropriate. The public API is exported from the package root.

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

```ts
import {
    Array1D,
    Array2D,
    Vec3,
    bessel,
    bisection,
    secant,
    rk4Integrate,
    createVelocityVerlet,
} from 'numerics-js';
```

## API

### Vectors and matrices

- `Array1D` is an N-component vector backed by `Float64Array`. It provides arithmetic, norms, dot products, comparisons, and in-place operations.
- `Array2D` is a row-major matrix with arithmetic, matrix multiplication, vector multiplication, transpose, elimination, solving, and related linear-algebra operations.
- `Vec3` is a small three-component vector for positions, directions, and velocities.

`Array2D` row and column indices are 1-based. `Array1D` and `Vec3` operations that do not end in `Self` return new values; methods ending in `Self` mutate the instance.

```ts
const a = new Array1D([1, 2, 3]);
const b = new Array1D([4, 5, 6]);
const sum = a.add(b);

const matrix = new Array2D(2, 2, [1, 2, 3, 4]);
const product = matrix.mulVec(new Array1D([10, 20]));
const direction = new Vec3(3, 4, 0).normalize();
```

### Root finding

`bisection` requires a bracketing interval whose endpoint values have opposite signs. `secant` starts from two guesses and does not require a bracket; it can fail to converge for some functions.

```ts
const root = bisection((x) => x * x - 2, 1, 2);
const otherRoot = secant((x) => Math.cos(x) - x, 0, 1);
```

Both functions accept optional `tolerance` and `maxIterations` arguments.

### Bessel functions

`bessel.J(n, x)` evaluates the Bessel function of the first kind for a non-negative integer order. `bessel.getZero(n, m)` returns the m-th positive zero, with `m` counted from 1.

```ts
const value = bessel.J(0, 1.5);
const firstZero = bessel.getZero(0, 1);
```

### Ordinary differential equations

`rk4Step` advances one state, while `rk4Integrate` records every state from `t0` through `tEnd`. Derivative functions write into the supplied output vector and return it.

```ts
const initial = new Array1D([1]);
const solution = rk4Integrate(
    (t, y, dydt) => dydt.set([-y.data[0]]),
    0,
    1,
    initial,
    0.1,
);

console.log(solution.t.data[solution.t.dim - 1]); // 1
console.log(solution.y.row(solution.y.rows));    // state at t = 1
```

`createVelocityVerlet` creates a mutable `step(state, dt)` function for position and velocity updates. A state contains `u`, `v`, `a`, and a reusable `aNext` array.

```ts
const step = createVelocityVerlet((u, v, aNext) => {
    for (let i = 0; i < aNext.length; i++) aNext[i] = -9.81;
});

const state = { u: [0], v: [0], a: [-9.81], aNext: [0] };
step(state, 0.016);
```

## Development

```bash
npm test          # run the test suite once
npm run test:watch
```

The source is written in TypeScript. Run `npm run build` to emit the package files into `dist/`.
