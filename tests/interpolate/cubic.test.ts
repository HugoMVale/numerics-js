import { describe, it, expect, expectTypeOf } from 'vitest';
import {
  CubicSplineInterpolator1D,
  CubicSplineInterpolatorND,
  type CubicSplineBoundary
} from '../../src/interpolate/cubic.js';
import { PchipInterpolator1D } from '../../src/interpolate/pchip.js';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { clobber, mutableVectorLikeForms, vectorLikeForms } from '../vectorLikeForms.js';

const BOUNDARIES: CubicSplineBoundary[] = ['not-a-knot', 'natural', 'clamped'];

// Non-uniform knots, so that spacing-dependent mistakes cannot cancel out
const xp = [0, 0.7, 1.1, 2.4, 3, 4.6, 5.1, 7];

/** Second derivative of a spline at `x`, from first derivatives sampled strictly inside one segment. */
function secondDerivative(f: { derivative(x: number): number }, x: number, side: -1 | 1): number {
  const e = 1e-3;
  // f' is quadratic within a segment, so these differences are exact; f'' is then linear in x
  const g1 = (f.derivative(x + side * 2 * e) - f.derivative(x + side * e)) / (side * e);
  const g2 = (f.derivative(x + side * 3 * e) - f.derivative(x + side * 2 * e)) / (side * e);
  // g1 = f''(x + side * 1.5e), g2 = f''(x + side * 2.5e): extrapolate to x
  return g1 - 1.5 * (g2 - g1);
}

describe('CubicSplineInterpolator1D', () => {
  it('produces the values shown in its documentation example', () => {
    const f = new CubicSplineInterpolator1D([0, 1, 2, 3, 4], [0, 1, 8, 27, 64]);

    expect(f.eval(2.5)).toBeCloseTo(15.625, 12);
    expect(f.derivative(2.5)).toBeCloseTo(18.75, 12);
    expect(f.integrate(0, 4)).toBeCloseTo(64, 12);
  });

  it('evaluates exactly at the knots for every boundary type', () => {
    const fp = xp.map((x) => Math.sin(x) + x * x);
    for (const bc of BOUNDARIES) {
      const f = new CubicSplineInterpolator1D(xp, fp, { bc });
      for (let i = 0; i < xp.length; i++) expect(f.eval(xp[i])).toBe(fp[i]);
    }
  });

  describe('polynomial reproduction', () => {
    const cubic = (x: number) => 2 - 3 * x + 0.5 * x * x + 0.25 * x * x * x;
    const cubicD = (x: number) => -3 + x + 0.75 * x * x;
    const cubicI = (x: number) => 2 * x - 1.5 * x * x + (0.5 / 3) * x ** 3 + (0.25 / 4) * x ** 4;
    const xq = [0.05, 0.7, 0.9, 1.7, 2.9, 3.3, 4.55, 5.0, 6.2, 6.99];

    it('not-a-knot reproduces any cubic polynomial (value, derivative, integral)', () => {
      for (const n of [4, 5, xp.length]) {
        const x = xp.slice(0, n);
        const f = new CubicSplineInterpolator1D(x, x.map(cubic));
        for (const q of xq.filter((v) => v <= x[n - 1])) {
          expect(f.eval(q)).toBeCloseTo(cubic(q), 11);
          expect(f.derivative(q)).toBeCloseTo(cubicD(q), 10);
        }
        const [a, b] = [0.3, x[n - 1] - 0.1];
        expect(f.integrate(a, b)).toBeCloseTo(cubicI(b) - cubicI(a), 10);
      }
    });

    it('not-a-knot reproduces parabolas with 3 points and lines with 2 points', () => {
      const parabola = (x: number) => 1 - 2 * x + 3 * x * x;
      const f3 = new CubicSplineInterpolator1D([0, 0.4, 2], [0, 0.4, 2].map(parabola));
      for (const q of [0, 0.1, 0.4, 1.3, 2]) expect(f3.eval(q)).toBeCloseTo(parabola(q), 13);
      expect(f3.derivative(1.3)).toBeCloseTo(-2 + 6 * 1.3, 12);

      for (const bc of ['not-a-knot', 'natural'] as const) {
        const f2 = new CubicSplineInterpolator1D([1, 3], [5, 1], { bc });
        expect(f2.eval(2)).toBeCloseTo(3, 14);
        expect(f2.derivative(2)).toBeCloseTo(-2, 14);
        expect(f2.integrate(1, 3)).toBeCloseTo(6, 14);
      }
    });

    it('natural reproduces lines; clamped with exact end slopes reproduces cubics', () => {
      const line = (x: number) => 4 - 1.5 * x;
      const fn = new CubicSplineInterpolator1D(xp, xp.map(line), { bc: 'natural' });
      for (const q of xq) expect(fn.eval(q)).toBeCloseTo(line(q), 12);

      const fc = new CubicSplineInterpolator1D(xp, xp.map(cubic), {
        bc: 'clamped',
        leftDerivative: cubicD(xp[0]),
        rightDerivative: cubicD(xp[xp.length - 1])
      });
      for (const q of xq) expect(fc.eval(q)).toBeCloseTo(cubic(q), 10);
    });
  });

  describe('boundary conditions', () => {
    it('natural: matches the hand-derived spline through (0,0), (1,1), (2,0)', () => {
      // Slopes from the natural system: s = [1.5, 0, -1.5]
      const f = new CubicSplineInterpolator1D([0, 1, 2], [0, 1, 0], { bc: 'natural' });
      expect(f.derivative(0)).toBeCloseTo(1.5, 14);
      expect(f.derivative(1)).toBeCloseTo(0, 14);
      expect(f.derivative(1.999999999)).toBeCloseTo(-1.5, 8);
      expect(f.eval(0.5)).toBeCloseTo(0.6875, 14);
      expect(f.eval(1.5)).toBeCloseTo(0.6875, 14);
    });

    it('natural: second derivative vanishes at both ends', () => {
      const f = new CubicSplineInterpolator1D(
        xp,
        xp.map((x) => Math.cos(x)),
        { bc: 'natural' }
      );
      expect(secondDerivative(f, xp[0], 1)).toBeCloseTo(0, 6);
      expect(secondDerivative(f, xp[xp.length - 1], -1)).toBeCloseTo(0, 6);
    });

    it('clamped: honors the prescribed end derivatives and defaults them to 0', () => {
      const fp = xp.map((x) => Math.sin(x));
      const f = new CubicSplineInterpolator1D(xp, fp, {
        bc: 'clamped',
        leftDerivative: 2.5,
        rightDerivative: -1
      });
      expect(f.derivative(xp[0])).toBeCloseTo(2.5, 12);
      expect(f.derivative(xp[xp.length - 1] - 1e-12)).toBeCloseTo(-1, 9);

      const flat = new CubicSplineInterpolator1D(xp, fp, { bc: 'clamped' });
      expect(flat.derivative(xp[0])).toBeCloseTo(0, 12);
      expect(flat.derivative(xp[xp.length - 1] - 1e-12)).toBeCloseTo(0, 9);
    });

    it('clamped with two points is the Hermite cubic with the given end slopes', () => {
      const smoothstep = new CubicSplineInterpolator1D([0, 1], [0, 1], { bc: 'clamped' });
      expect(smoothstep.eval(0.5)).toBeCloseTo(0.5, 14);
      expect(smoothstep.derivative(0.5)).toBeCloseTo(1.5, 14);

      // slopes 2 and -1: y(t) = 0*h00 + 2*h10 + 1*h01 - 1*h11 at t = 0.5 -> 0.5 + 0.25 + 0.125 = 0.875
      const f = new CubicSplineInterpolator1D([0, 1], [0, 1], {
        bc: 'clamped',
        leftDerivative: 2,
        rightDerivative: -1
      });
      expect(f.eval(0.5)).toBeCloseTo(0.5 + 2 * 0.125 + 1 * 0.125, 14);
    });

    it('the boundary types give different splines on generic data', () => {
      const fp = xp.map((x) => Math.exp(-x) * Math.sin(3 * x));
      const [a, n, c] = BOUNDARIES.map((bc) =>
        new CubicSplineInterpolator1D(xp, fp, { bc }).eval(xp[0] + 0.1)
      );
      expect(Math.abs(a - n)).toBeGreaterThan(1e-6);
      expect(Math.abs(a - c)).toBeGreaterThan(1e-6);
      expect(Math.abs(n - c)).toBeGreaterThan(1e-6);
    });
  });

  it('is C2: first and second derivatives are continuous at interior knots', () => {
    const fp = xp.map((x) => Math.sin(2 * x) + 0.3 * x);
    for (const bc of BOUNDARIES) {
      const f = new CubicSplineInterpolator1D(xp, fp, { bc });
      for (let i = 1; i < xp.length - 1; i++) {
        const e = 1e-9;
        expect(f.derivative(xp[i] - e)).toBeCloseTo(f.derivative(xp[i] + e), 6);
        expect(secondDerivative(f, xp[i], -1)).toBeCloseTo(secondDerivative(f, xp[i], 1), 5);
      }
    }
  });

  it('is accurate for smooth functions and converges at fourth order', () => {
    const err = (n: number) => {
      const x = Array.from({ length: n }, (_, i) => (i * Math.PI) / (n - 1));
      const f = new CubicSplineInterpolator1D(
        x,
        x.map((v) => Math.sin(v))
      );
      let worst = 0;
      for (let i = 0; i < 200; i++) {
        const q = 0.05 + (i * (Math.PI - 0.1)) / 199;
        worst = Math.max(worst, Math.abs(f.eval(q) - Math.sin(q)));
      }
      return worst;
    };
    // Fourth order: halving the spacing divides the error by 2^4 = 16 once in the asymptotic
    // regime (coarser not-a-knot grids are not there yet, with ratios near 30)
    const ratio = err(81) / err(161);
    expect(ratio).toBeGreaterThan(15);
    expect(ratio).toBeLessThan(17);
  });

  it('is smoother than, but not shape-preserving like, PCHIP', () => {
    const x = [0, 1, 2, 3, 4];
    const fp = [0, 0, 1, 1, 1]; // monotonic step
    const spline = new CubicSplineInterpolator1D(x, fp);
    const pchip = new PchipInterpolator1D(x, fp);
    const q = Array.from({ length: 81 }, (_, i) => i * 0.05);

    expect(Math.min(...pchip.eval(q).toArray())).toBeGreaterThanOrEqual(0);
    expect(Math.max(...pchip.eval(q).toArray())).toBeLessThanOrEqual(1);
    const s = spline.eval(q).toArray();
    expect(Math.min(...s) < -1e-3 || Math.max(...s) > 1 + 1e-3).toBe(true);
  });

  describe('degenerate sizes', () => {
    it('a single point is a constant', () => {
      const f = new CubicSplineInterpolator1D([2], [7]);
      expect(f.size).toBe(1);
      expect(f.eval(2)).toBe(7);
      expect(f.derivative(2)).toBe(0);
      expect(f.integrate(0, 4)).toBe(28);
    });

    it('works for the minimum number of points of each boundary type', () => {
      for (const bc of BOUNDARIES) {
        for (const n of [1, 2, 3, 4]) {
          const x = Array.from({ length: n }, (_, i) => i * 1.5);
          const f = new CubicSplineInterpolator1D(
            x,
            x.map((v) => v * v),
            { bc }
          );
          for (const v of x) expect(Number.isFinite(f.eval(v))).toBe(true);
        }
      }
    });
  });

  describe('clamping outside the data range', () => {
    const f = new CubicSplineInterpolator1D([0, 1, 2, 3], [1, 3, 2, 5]);

    it('clamps to the end values and has zero derivative there', () => {
      expect(f.eval(-1)).toBe(1);
      expect(f.eval(4)).toBe(5);
      expect(f.derivative(-1)).toBe(0);
      expect(f.derivative(4)).toBe(0);
    });

    it('honors left and right options', () => {
      const g = new CubicSplineInterpolator1D([0, 1, 2, 3], [1, 3, 2, 5], { left: -9, right: 9 });
      expect(g.eval(-1)).toBe(-9);
      expect(g.eval(4)).toBe(9);
    });

    it('propagates NaN', () => {
      expect(f.eval(NaN)).toBeNaN();
      expect(f.derivative(NaN)).toBeNaN();
    });
  });

  describe('integrate', () => {
    const f = new CubicSplineInterpolator1D(
      xp,
      xp.map((x) => Math.sin(x))
    );

    it('is additive and antisymmetric', () => {
      expect(f.integrate(0.4, 2.2) + f.integrate(2.2, 6)).toBeCloseTo(f.integrate(0.4, 6), 12);
      expect(f.integrate(6, 0.4)).toBeCloseTo(-f.integrate(0.4, 6), 14);
      expect(f.integrate(2, 2)).toBe(0);
    });

    it('is consistent with the derivative (fundamental theorem of calculus)', () => {
      // d/db integral(a, b) = f(b); check with a central difference
      const e = 1e-5;
      const b = 3.3;
      const dI = (f.integrate(0.4, b + e) - f.integrate(0.4, b - e)) / (2 * e);
      expect(dI).toBeCloseTo(f.eval(b), 8);
    });

    it('integrates constant clamp regions as rectangles', () => {
      const g = new CubicSplineInterpolator1D([0, 1], [2, 2], { left: 10, right: -3 });
      expect(g.integrate(-2, 0)).toBeCloseTo(20, 14);
      expect(g.integrate(1, 3)).toBeCloseTo(-6, 14);
      expect(g.integrate(-1, 2)).toBeCloseTo(10 + 2 - 3, 14);
    });

    it('throws for NaN bounds', () => {
      expect(() => f.integrate(NaN, 1)).toThrow(RangeError);
      expect(() => f.integrate(0, NaN)).toThrow(RangeError);
    });
  });

  describe('input validation', () => {
    it('rejects empty, mismatched, non-finite and unsorted knots', () => {
      expect(() => new CubicSplineInterpolator1D([], [])).toThrow(RangeError);
      expect(() => new CubicSplineInterpolator1D([0, 1], [0])).toThrow(RangeError);
      expect(() => new CubicSplineInterpolator1D([0, NaN], [0, 1])).toThrow(RangeError);
      expect(() => new CubicSplineInterpolator1D([0, 2, 1], [0, 1, 2])).toThrow(RangeError);
    });

    it('rejects duplicate knots, which make the slope system singular', () => {
      expect(() => new CubicSplineInterpolator1D([0, 1, 1, 2], [0, 1, 2, 3])).toThrow(RangeError);
    });

    it('rejects non-finite knots even when checkSorted is false', () => {
      for (const knot of [NaN, Infinity, -Infinity]) {
        expect(
          () => new CubicSplineInterpolator1D([0, knot], [0, 1], { checkSorted: false })
        ).toThrow(RangeError);
      }
    });

    it('rejects an unknown bc', () => {
      expect(
        () => new CubicSplineInterpolator1D([0, 1, 2], [0, 1, 2], { bc: 'cubic' as never })
      ).toThrow(/bc must be one of/);
    });

    it('requires bc to be clamped for end derivatives, and finite values', () => {
      expect(
        () => new CubicSplineInterpolator1D([0, 1, 2], [0, 1, 2], { leftDerivative: 1 })
      ).toThrow(RangeError);
      expect(
        () =>
          new CubicSplineInterpolator1D([0, 1, 2], [0, 1, 2], { bc: 'natural', rightDerivative: 0 })
      ).toThrow(RangeError);
      for (const bad of [NaN, Infinity]) {
        expect(
          () =>
            new CubicSplineInterpolator1D([0, 1, 2], [0, 1, 2], {
              bc: 'clamped',
              leftDerivative: bad
            })
        ).toThrow(RangeError);
      }
    });
  });
});

describe('CubicSplineInterpolatorND', () => {
  const x = [0, 0.5, 1.7, 2, 3.3, 4];
  const cols = [x.map((v) => Math.sin(v)), x.map((v) => v * v - v), x.map((v) => Math.exp(-v))];
  const fp = Matrix.from(x.map((_, i) => cols.map((c) => c[i])));

  it('produces the values shown in its documentation example', () => {
    const m = Matrix.from([
      [0, 0],
      [1, 1],
      [8, 0],
      [27, -1],
      [64, 0]
    ]);
    const f = new CubicSplineInterpolatorND([0, 1, 2, 3, 4], m);
    const v = f.eval(2.5);
    expect(v.get(0)).toBeCloseTo(15.625, 12);
    expect(v.get(1)).toBeCloseTo(-0.625, 12);
  });

  it('exposes size and dim', () => {
    const f = new CubicSplineInterpolatorND(x, fp);
    expect(f.size).toBe(6);
    expect(f.dim).toBe(3);
  });

  for (const bc of BOUNDARIES) {
    it(`matches independent 1D splines column by column (bc: ${bc})`, () => {
      const opts =
        bc === 'clamped'
          ? { bc, leftDerivative: [1, -2, 0.5], rightDerivative: 0.25 }
          : { bc, left: [1, 2, 3], right: [4, 5, 6] };
      const nd = new CubicSplineInterpolatorND(x, fp, opts);
      const queries = [-1, 0, 0.3, 1.7, 2.6, 3.99, 4, 5];

      for (let j = 0; j < 3; j++) {
        const opts1 =
          bc === 'clamped'
            ? { bc, leftDerivative: [1, -2, 0.5][j], rightDerivative: 0.25 }
            : { bc, left: [1, 2, 3][j], right: [4, 5, 6][j] };
        const f1 = new CubicSplineInterpolator1D(x, cols[j], opts1);
        const ev = nd.eval(queries);
        const dv = nd.derivative(queries);
        for (let i = 0; i < queries.length; i++) {
          expect(ev.get(i, j)).toBe(f1.eval(queries[i]));
          expect(dv.get(i, j)).toBe(f1.derivative(queries[i]));
        }
        expect(nd.integrate(-0.5, 4.5).get(j)).toBe(f1.integrate(-0.5, 4.5));
      }
    });
  }

  it('broadcasts a scalar end derivative to every component', () => {
    const a = new CubicSplineInterpolatorND(x, fp, {
      bc: 'clamped',
      leftDerivative: 2,
      rightDerivative: -1
    });
    const b = new CubicSplineInterpolatorND(x, fp, {
      bc: 'clamped',
      leftDerivative: [2, 2, 2],
      rightDerivative: [-1, -1, -1]
    });
    expect(a.eval(1.1).toArray()).toEqual(b.eval(1.1).toArray());
    for (let j = 0; j < 3; j++) expect(a.derivative(0).get(j)).toBeCloseTo(2, 12);
  });

  it('handles a single point, two points and one component', () => {
    const one = new CubicSplineInterpolatorND([2], Matrix.from([[7, -3]]));
    expect(one.eval(2).toArray()).toEqual([7, -3]);
    expect(one.integrate(0, 4).toArray()).toEqual([28, -12]);

    const two = new CubicSplineInterpolatorND(
      [0, 2],
      Matrix.from([
        [0, 4],
        [2, 0]
      ])
    );
    expect(two.eval(1).toArray()).toEqual([1, 2]);

    const col = new CubicSplineInterpolatorND([0, 1, 2, 3], Matrix.from([[0], [1], [8], [27]]));
    expect(col.dim).toBe(1);
    expect(col.eval(1.5).get(0)).toBeCloseTo(3.375, 12);
  });

  it('returns shaped results and rejects empty evaluation sets', () => {
    const f = new CubicSplineInterpolatorND(x, fp);
    expect(f.eval(1).size).toBe(3);
    const m = f.eval([0, 1, 2, 3]);
    expect([m.rows, m.cols]).toEqual([4, 3]);
    expect(() => f.eval([])).toThrow(RangeError);
    expect(() => f.derivative([])).toThrow(RangeError);
    expect(f.eval(NaN).toArray().every(Number.isNaN)).toBe(true);
    expect(f.derivative(-5).toArray()).toEqual([0, 0, 0]);
  });

  it('integrates with swapped bounds and NaN checks', () => {
    const f = new CubicSplineInterpolatorND(x, fp);
    const fwd = f.integrate(0.2, 3.9);
    const rev = f.integrate(3.9, 0.2);
    for (let j = 0; j < 3; j++) expect(rev.get(j)).toBeCloseTo(-fwd.get(j), 14);
    expect(f.integrate(1, 1).toArray()).toEqual([0, 0, 0]);
    expect(() => f.integrate(NaN, 1)).toThrow(RangeError);
    expect(() => f.integrate(0, NaN)).toThrow(RangeError);
  });

  it('validates knots, options and end-derivative lengths', () => {
    expect(() => new CubicSplineInterpolatorND([], Matrix.from([[0]]))).toThrow(RangeError);
    expect(() => new CubicSplineInterpolatorND([0, 1], Matrix.from([[0], [1], [2]]))).toThrow(
      RangeError
    );
    expect(() => new CubicSplineInterpolatorND([0, 1, 1], Matrix.from([[0], [1], [2]]))).toThrow(
      RangeError
    );
    expect(() => new CubicSplineInterpolatorND(x, fp, { bc: 'foo' as never })).toThrow(RangeError);
    expect(() => new CubicSplineInterpolatorND(x, fp, { leftDerivative: 1 })).toThrow(RangeError);
    expect(
      () => new CubicSplineInterpolatorND(x, fp, { bc: 'clamped', leftDerivative: [1, 2] })
    ).toThrow(/length 3/);
    expect(
      () => new CubicSplineInterpolatorND(x, fp, { bc: 'clamped', rightDerivative: NaN })
    ).toThrow(RangeError);
    expect(
      () =>
        new CubicSplineInterpolatorND(x, fp, { bc: 'clamped', rightDerivative: [0, 0, Infinity] })
    ).toThrow(RangeError);
  });
});

describe('Cubic spline interpolators with VectorLike inputs', () => {
  const xpRaw = [1, 2, 3, 4, 5];
  const fpRaw = [3, 2, 0, 1, 4];
  const xRaw = [0, 1.5, 2.5, 3.9, 6];
  const fpMat = Matrix.from([
    [0, 10],
    [1, 20],
    [4, 40],
    [5, 30],
    [2, 25]
  ]);
  const ref1D = new CubicSplineInterpolator1D(Vector.from(xpRaw), Vector.from(fpRaw));
  const refND = new CubicSplineInterpolatorND(Vector.from(xpRaw), fpMat);

  for (const [name, make] of vectorLikeForms) {
    it(`CubicSplineInterpolator1D accepts xp, fp, and eval/derivative inputs as ${name}`, () => {
      const f = new CubicSplineInterpolator1D(make(xpRaw), make(fpRaw));
      expect(f.eval(make(xRaw)).toArray()).toEqual(ref1D.eval(Vector.from(xRaw)).toArray());
      expect(f.derivative(make(xRaw)).toArray()).toEqual(
        ref1D.derivative(Vector.from(xRaw)).toArray()
      );
      expect(f.eval(2.5)).toBe(ref1D.eval(2.5));
    });

    it(`CubicSplineInterpolatorND accepts xp, left, right and end derivatives as ${name}`, () => {
      const opts = {
        bc: 'clamped' as const,
        left: make([-1, -2]),
        right: make([7, 8]),
        leftDerivative: make([1, 2]),
        rightDerivative: make([3, 4])
      };
      const f = new CubicSplineInterpolatorND(make(xpRaw), fpMat, opts);
      const ref = new CubicSplineInterpolatorND(Vector.from(xpRaw), fpMat, {
        bc: 'clamped',
        left: Vector.from([-1, -2]),
        right: Vector.from([7, 8]),
        leftDerivative: Vector.from([1, 2]),
        rightDerivative: Vector.from([3, 4])
      });
      expect(f.eval(make(xRaw)).toArray()).toEqual(ref.eval(Vector.from(xRaw)).toArray());
      expect(f.eval(0).toArray()).toEqual([-1, -2]);
      expect(f.eval(6).toArray()).toEqual([7, 8]);
    });
  }

  for (const [name, make] of mutableVectorLikeForms) {
    it(`CubicSplineInterpolator1D copies ${name} data: mutation after construction has no effect`, () => {
      const xp = make(xpRaw);
      const fp = make(fpRaw);
      const f = new CubicSplineInterpolator1D(xp, fp);
      const before = f.eval(Vector.from(xRaw)).toArray();
      clobber(xp);
      clobber(fp);
      expect(f.eval(Vector.from(xRaw)).toArray()).toEqual(before);
    });

    it(`CubicSplineInterpolatorND copies ${name} knots, clamp values and end derivatives`, () => {
      const xp = make(xpRaw);
      const left = make([-1, -2]);
      const right = make([7, 8]);
      const leftDerivative = make([1, 2]);
      const f = new CubicSplineInterpolatorND(xp, fpMat, {
        bc: 'clamped',
        left,
        right,
        leftDerivative
      });
      const before = f.eval(Vector.from(xRaw)).toArray();
      const dBefore = f.derivative(Vector.from(xRaw)).toArray();
      clobber(xp);
      clobber(left);
      clobber(right);
      clobber(leftDerivative);
      expect(f.eval(Vector.from(xRaw)).toArray()).toEqual(before);
      expect(f.derivative(Vector.from(xRaw)).toArray()).toEqual(dBefore);
    });
  }

  it('keeps the overloads: number -> number/Vector, VectorLike -> Vector/Matrix', () => {
    expectTypeOf(ref1D.eval(1)).toEqualTypeOf<number>();
    expectTypeOf(ref1D.eval([1, 2])).toEqualTypeOf<Vector>();
    expectTypeOf(ref1D.derivative(1)).toEqualTypeOf<number>();
    expectTypeOf(ref1D.derivative(Vector.from([1]))).toEqualTypeOf<Vector>();
    expectTypeOf(refND.eval(1)).toEqualTypeOf<Vector>();
    expectTypeOf(refND.eval([1, 2])).toEqualTypeOf<Matrix>();
    expectTypeOf(refND.derivative(1)).toEqualTypeOf<Vector>();
    expectTypeOf(refND.derivative(new Float64Array(2))).toEqualTypeOf<Matrix>();
  });

  it('throws a TypeError for unsupported inputs', () => {
    expect(() => new CubicSplineInterpolator1D('abc' as never, fpRaw)).toThrowError(TypeError);
    expect(() => new CubicSplineInterpolator1D(xpRaw, null as never)).toThrowError(TypeError);
    expect(() => new CubicSplineInterpolatorND({} as never, fpMat)).toThrowError(TypeError);
    expect(
      () =>
        new CubicSplineInterpolatorND(xpRaw, fpMat, { bc: 'clamped', leftDerivative: 'a' as never })
    ).toThrowError(TypeError);
    expect(() => ref1D.eval(null as never)).toThrowError(/Vector\.asVector/);
    expect(() => refND.derivative('x' as never)).toThrowError(TypeError);
  });
});
