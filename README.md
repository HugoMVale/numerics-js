# numerics-js

[![CI](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml/badge.svg)](https://github.com/HugoMVale/numerics-js/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/HugoMVale/numerics-js/graph/badge.svg?token=WrJkecJXR0)](https://codecov.io/gh/HugoMVale/numerics-js)

A small, dependency-free numerical methods library for JavaScript and
TypeScript.

## Why?

numerics-js started with a science blog post. Live, interactive animations of a
physical system need real numerics running in the reader's browser: integrating
differential equations, finding roots, fitting and interpolating data. In Python,
NumPy and SciPy cover this ground. In JavaScript and TypeScript, no equivalent
turned up that was both well maintained and coherent, so numerics-js was written
to fill the gap.

It aims to be a small, dependable scientific toolbox that runs wherever
JavaScript does: pure TypeScript, no dependencies, and native ESM for the browser
and Node.js. The modules share `Vector` and `Matrix` and are designed to compose;
the example below combines an ODE solver, adaptive quadrature, and a root finder
in a few lines of ordinary TypeScript. Names and conventions follow NumPy and
SciPy where that makes sense (`linspace`, `curveFit`, ODE events), and key
algorithms are validated against SciPy.

## Install

```bash
npm install numerics-js
```

## Example

The period of a pendulum is a classic: it is constant for small swings, but grows with
the amplitude through an elliptic integral. This example computes it in two independent ways,
by integrating the equation of motion and by evaluating the elliptic integral with adaptive
quadrature, and then uses a root finder to invert the relation.

```ts
import { integrate, linalg, ode, roots } from 'numerics-js';

const g = 9.81; // m/s²
const L = 1; // m

// Period from the equation of motion θ'' = -(g/L) sin θ, for a pendulum released from rest at θ0.
// The solver locates the first crossing of θ = 0, a quarter period, as a terminal event.
function periodFromOde(theta0: number): number {
  const result = ode.rungeKuttaAdaptive(
    'rk45',
    (_t, y, dydt) => dydt.set([y.data[1], -(g / L) * Math.sin(y.data[0])]),
    0,
    10,
    new linalg.Vector([theta0, 0]),
    {
      rtol: 1e-10,
      atol: 1e-12,
      events: [{ fn: (_t, y) => y.data[0], direction: 'falling', terminal: true }]
    }
  );
  return 4 * result.events![0].t;
}

// Exact period, 4 √(L/g) K(k), where K(k) = ∫₀^{π/2} dφ / √(1 − k² sin²φ) is the complete
// elliptic integral of the first kind and k = sin(θ0/2).
function periodFromIntegral(theta0: number): number {
  const k = Math.sin(theta0 / 2);
  const K = integrate.quad((phi) => 1 / Math.sqrt(1 - (k * Math.sin(phi)) ** 2), 0, Math.PI / 2);
  return 4 * Math.sqrt(L / g) * K.value;
}

const T0 = 2 * Math.PI * Math.sqrt(L / g); // small-angle period
const theta0 = Math.PI / 3; // 60°

console.log('small-angle period:', T0);
console.log('60° period, ODE:   ', periodFromOde(theta0));
console.log('60° period, quad:  ', periodFromIntegral(theta0));

// Which amplitude makes the period 10 % longer than the small-angle one?
const amplitude = roots.brent((theta) => periodFromIntegral(theta) / T0 - 1.1, 0.1, 3);
const degrees = (amplitude.x * 180) / Math.PI;
console.log('amplitude for +10 % (deg):', degrees.toFixed(2));
```

Output:

```text
small-angle period: 2.0060666807106475
60° period, ODE:    2.152874666881198
60° period, quad:   2.1528746668805154
amplitude for +10 % (deg): 69.33
```

## What’s included

- `linalg`: `Vector`, `Matrix`, and `Vec3`, with solvers, factorizations, and eigenvalue methods.
- `integrate`: sampled-data rules and adaptive quadrature, including infinite intervals.
- `interpolate`: linear and shape-preserving interpolation for scalar and vector-valued data.
- `numdiff`: finite-difference derivatives and Jacobians.
- `ode`: fixed-step and adaptive Runge-Kutta solvers, an adaptive implicit solver for stiff problems, plus velocity Verlet integration.
- `optimize`: scalar and multidimensional derivative-free optimization, Levenberg-Marquardt nonlinear least squares, and nonlinear curve fitting with parameter uncertainties.
- `roots`: scalar bisection, Brent, and secant methods, plus quasi-Newton system solving.
- `special`: Bessel functions of the first kind and their positive zeros, and the error function and its complement.
- `math`: clipping, signed magnitude, approximate equality, and seeded random numbers.

The package is native ESM and exposes both the package root and focused module
subpaths such as `numerics-js/linalg`, `numerics-js/ode`, and
`numerics-js/special/bessel`.

## Documentation

Read the [API reference](https://hugomvale.github.io/numerics-js/) for complete
signatures, options, guarantees, and examples. Every example there has a **Run**
button that opens it in the [playground](https://hugomvale.github.io/numerics-js/playground/),
where you can edit and run numerics-js code directly in your browser.

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
