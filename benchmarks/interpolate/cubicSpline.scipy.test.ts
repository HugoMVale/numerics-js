import { describe, expect, it } from 'vitest';
import {
  CubicSplineInterpolator1D,
  CubicSplineInterpolatorND
} from '../../src/interpolate/cubic.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import fixtures from './cubicSpline.scipy.json' with { type: 'json' };

// Absolute + relative tolerance. numerics-js agrees with scipy.interpolate.CubicSpline to
// round-off (see the benchmark report); this leaves headroom for platform differences in the
// floating-point evaluation order only. The cases are data-driven: each fixture carries its own
// knots, values, end derivatives and query points, so there is no case list to keep in sync with
// generate_cubic_spline_fixtures.py.
const ATOL = 1e-11;
const RTOL = 1e-11;

function expectClose(actual: number, reference: number, what: string): void {
  expect(
    Math.abs(actual - reference),
    `${what}: got ${actual}, SciPy ${reference}`
  ).toBeLessThanOrEqual(ATOL + RTOL * Math.abs(reference));
}

describe('CubicSplineInterpolator1D/ND vs scipy.interpolate.CubicSpline reference values', () => {
  for (const fx of fixtures) {
    it(`matches SciPy for ${fx.id}: ${fx.description}`, () => {
      const options = {
        bc: fx.bc as 'not-a-knot' | 'natural' | 'clamped',
        ...(fx.leftDerivative !== null && {
          leftDerivative: fx.leftDerivative as number | number[],
          rightDerivative: fx.rightDerivative as number | number[]
        })
      };

      if (fx.dim === null) {
        // Scalar-valued
        const f = new CubicSplineInterpolator1D(fx.x, fx.y as number[], {
          ...options,
          leftDerivative: options.leftDerivative as number | undefined,
          rightDerivative: options.rightDerivative as number | undefined
        });
        const ev = fx.eval as number[];
        const dv = fx.derivative as number[];
        fx.queries.forEach((q, i) => {
          expectClose(f.eval(q), ev[i], `eval(${q})`);
          expectClose(f.derivative(q), dv[i], `derivative(${q})`);
        });
        for (const { a, b, value } of fx.integrals) {
          expectClose(f.integrate(a, b), value as number, `integrate(${a}, ${b})`);
        }
      } else {
        // Vector-valued: every component must match its SciPy column
        const f = new CubicSplineInterpolatorND(fx.x, Matrix.from(fx.y as number[][]), options);
        expect(f.dim).toBe(fx.dim);
        const ev = fx.eval as number[][];
        const dv = fx.derivative as number[][];
        fx.queries.forEach((q, i) => {
          const e = f.eval(q);
          const d = f.derivative(q);
          for (let j = 0; j < fx.dim!; j++) {
            expectClose(e.get(j), ev[i][j], `eval(${q})[${j}]`);
            expectClose(d.get(j), dv[i][j], `derivative(${q})[${j}]`);
          }
        });
        for (const { a, b, value } of fx.integrals) {
          const I = f.integrate(a, b);
          for (let j = 0; j < fx.dim; j++) {
            expectClose(I.get(j), (value as number[])[j], `integrate(${a}, ${b})[${j}]`);
          }
        }
      }
    });
  }
});
