import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { quasiNewton, type QuasiNewtonOptions } from '../../src/optimize/quasiNewton.js';
import { GRADIENT_TEST_FUNCTIONS } from './gradientTestFunctions.js';

const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;

/** Calls the solver on the Rosenbrock function with an arbitrary (possibly invalid) `x0`. */
const call = (x0: unknown, options: QuasiNewtonOptions = {}) =>
  quasiNewton(t.f, x0 as number[], options);

describe('quasiNewton input validation', () => {
  it('rejects an empty or non-finite x0', () => {
    expect(() => call([])).toThrow(RangeError);
    expect(() => call([])).toThrow('x0 must have at least one component');
    for (const bad of [NaN, Infinity, -Infinity]) {
      expect(() => call([1, bad])).toThrow(RangeError);
      expect(() => call([1, bad])).toThrow('x0 must contain only finite values');
    }
  });

  it('rejects an x0 that is not vector-like', () => {
    expect(() => call('abc')).toThrow(TypeError);
    expect(() => call(null)).toThrow(TypeError);
  });

  it.each([0, -1, 1.5, NaN, Infinity])('rejects maxIter = %s', (maxIter) => {
    expect(() => call(t.x0, { maxIter })).toThrow(RangeError);
  });

  it.each(['tolx', 'sclf'] as const)('%s must be finite and strictly positive', (key) => {
    for (const bad of [0, -1e-3, NaN, Infinity]) {
      expect(() => call(t.x0, { [key]: bad })).toThrow(RangeError);
      expect(() => call(t.x0, { [key]: bad })).toThrow(key);
    }
  });

  it.each(['tolg', 'epsf'] as const)('%s must be finite and non-negative', (key) => {
    for (const bad of [-1e-3, NaN, Infinity]) {
      expect(() => call(t.x0, { [key]: bad })).toThrow(RangeError);
      expect(() => call(t.x0, { [key]: bad })).toThrow(key);
    }
  });

  it.each([0, -1, NaN])('rejects maxLenFactor = %s', (maxLenFactor) => {
    expect(() => call(t.x0, { maxLenFactor })).toThrow('maxLenFactor must be strictly positive');
  });

  it('rejects trustLen = NaN', () => {
    expect(() => call(t.x0, { trustLen: NaN })).toThrow('trustLen must not be NaN');
  });

  it('validates sclx', () => {
    expect(() => call(t.x0, { sclx: [1] })).toThrow('sclx must have 2 components, got 1');
    expect(() => call(t.x0, { sclx: [1, 2, 3] })).toThrow(RangeError);
    for (const bad of [0, -1, NaN]) {
      expect(() => call(t.x0, { sclx: [1, bad] })).toThrow(
        'sclx must contain strictly positive values'
      );
    }
    expect(() => call(t.x0, { sclx: [1, 0] })).toThrow(RangeError);
    expect(() => call(t.x0, { sclx: 'ab' as unknown as number[] })).toThrow(TypeError);
  });

  it('validates the method names', () => {
    expect(() => call(t.x0, { globalMethod: 'newton' as never })).toThrow(
      "unknown globalMethod 'newton'"
    );
    expect(() => call(t.x0, { bfgsMethod: 'dfp' as never })).toThrow("unknown bfgsMethod 'dfp'");
    expect(() => call(t.x0, { globalMethod: 'newton' as never })).toThrow(RangeError);
    expect(() => call(t.x0, { bfgsMethod: 'dfp' as never })).toThrow(RangeError);
    // the names are checked even if they would be ignored
    expect(() => call(t.x0, { bfgsMethod: 'dfp' as never, bfgsUpdate: false })).toThrow(
      "unknown bfgsMethod 'dfp'"
    );
  });

  it('validates the shape of H0', () => {
    expect(() => call(t.x0, { H0: Matrix.identity(3) })).toThrow(
      'H0 must have dimensions 2x2, got 3x3'
    );
    expect(() => call(t.x0, { H0: new Matrix(2, 3) })).toThrow(RangeError);
  });

  it('validates before evaluating the function', () => {
    let calls = 0;
    const f = (x: Vector) => (calls++, t.f(x));
    const invalid: QuasiNewtonOptions[] = [
      { tolx: 0 },
      { tolg: -1 },
      { maxIter: 0 },
      { sclx: [1] },
      { H0: Matrix.identity(3) },
      { globalMethod: 'x' as never },
      { bfgsMethod: 'x' as never }
    ];
    for (const options of invalid) {
      expect(() => quasiNewton(f, t.x0, options)).toThrow();
    }
    expect(calls).toBe(0);
  });

  it('does not modify the options object or the arrays inside it', () => {
    const sclx = [1, 2];
    const H0 = Matrix.identity(2);
    const options: QuasiNewtonOptions = { sclx, H0, grad: t.grad, maxIter: 3 };
    quasiNewton(t.f, t.x0, options);
    expect(sclx).toEqual([1, 2]);
    expect(H0.allClose(Matrix.identity(2), { atol: 0, rtol: 0 })).toBe(true);
    expect(Object.keys(options).sort()).toEqual(['H0', 'grad', 'maxIter', 'sclx']);
  });

  it('accepts boundary values', () => {
    const res = (options: QuasiNewtonOptions) => quasiNewton(t.f, t.x0, { maxIter: 2, ...options });
    expect(() => res({ tolg: 0 })).not.toThrow();
    expect(() => res({ epsf: 0 })).not.toThrow();
    expect(() => res({ maxLenFactor: Infinity })).not.toThrow();
    expect(() => res({ trustLen: -1 })).not.toThrow();
    expect(() => res({ trustLen: Infinity })).not.toThrow();
    expect(() => res({ tolx: 1e-300 })).not.toThrow();
    expect(() => res({ sclf: 1e-6 })).not.toThrow();
    expect(() => res({ bfgsUpdate: true, hess: t.hess })).not.toThrow();
  });

  it('honors `sclf` instead of silently altering it', () => {
    // A very small sclf means that the function scale is huge, so the scaled gradient is tiny
    // and the algorithm stops earlier (as documented, too small a value stops prematurely).
    const small = quasiNewton(t.f, t.x0, { grad: t.grad, sclf: 1e-6, gradCheck: false });
    const dflt = quasiNewton(t.f, t.x0, { grad: t.grad });
    expect(small.iterations).toBeLessThan(dflt.iterations);
    expect(small.success).toBe(true);
  });

  it('a larger `epsf` coarsens the finite-difference steps', () => {
    const steps: number[] = [];
    quasiNewton((x) => (steps.push(x.get(0)), t.f(x)), [1.5, 1], {
      epsf: 1e-8,
      maxIter: 1,
      globalMethod: null
    });
    // x0, then the perturbed point of the first gradient component
    expect(steps[1] - 1.5).toBeCloseTo(1e-4 * 1.5, 8);
  });

  it('a user-supplied `sclx` is used for the finite-difference steps', () => {
    const steps: number[] = [];
    quasiNewton((x) => (steps.push(x.get(0)), t.f(x)), [0, 1], {
      sclx: [1e-2, 1],
      maxIter: 1,
      globalMethod: null
    });
    // h = sqrt(eps) * max(|x0|, 1/sclx0) = sqrt(eps) * 100
    expect(steps[1]).toBeCloseTo(Math.sqrt(Number.EPSILON) * 100, 12);
  });

  it('uses the scaling of x0 by default', () => {
    // x0 = [0, 5] -> scaleVector gives sclx0 = 1e1/5 = 2, so h0 = sqrt(eps)/2.
    const steps: number[] = [];
    quasiNewton((x) => (steps.push(x.get(0)), t.f(x)), [0, 5], { maxIter: 1, globalMethod: null });
    expect(steps[1]).toBeCloseTo(Math.sqrt(Number.EPSILON) / 2, 12);
  });
});
