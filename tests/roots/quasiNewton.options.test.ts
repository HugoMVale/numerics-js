import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { quasiNewton } from '../../src/roots/quasiNewton.js';

// Root at [sqrt(2), 1].
const f = (x: Vector): Vector => Vector.from([x.get(0) ** 2 - 2, x.get(1) - 1]);

describe('quasiNewton option ranges', () => {
  // Each of these used to make the dogleg strategy loop forever or return garbage silently.
  it.each([
    ['tolx', 0, 'tolx must be a finite strictly positive number'],
    ['tolx', -1e-10, 'tolx must be a finite strictly positive number'],
    ['tolx', NaN, 'tolx must be a finite strictly positive number'],
    ['tolx', Infinity, 'tolx must be a finite strictly positive number'],
    ['tolf', -1, 'tolf must be a finite non-negative number'],
    ['tolf', NaN, 'tolf must be a finite non-negative number'],
    ['epsf', -1, 'epsf must be a finite non-negative number'],
    ['epsf', NaN, 'epsf must be a finite non-negative number'],
    ['maxLenFactor', 0, 'maxLenFactor must be strictly positive'],
    ['maxLenFactor', -5, 'maxLenFactor must be strictly positive'],
    ['maxLenFactor', NaN, 'maxLenFactor must be strictly positive'],
    ['trustLen', NaN, 'trustLen must not be NaN']
  ] as const)('rejects %s = %s', (name, value, message) => {
    let evaluated = false;
    const spy = (x: Vector) => {
      evaluated = true;
      return f(x);
    };
    expect(() => quasiNewton(spy, [1, 0.5], { [name]: value })).toThrowError(RangeError);
    expect(() => quasiNewton(spy, [1, 0.5], { [name]: value })).toThrow(`quasiNewton: ${message}`);
    expect(evaluated).toBe(false);
  });

  it.each(['dogleg', 'line-search', null] as const)(
    'accepts boundary values (globalMethod = %s)',
    (globalMethod) => {
      const res = quasiNewton(f, [1, 0.5], {
        globalMethod,
        tolf: 0,
        maxLenFactor: Infinity,
        trustLen: -1,
        epsf: 0,
        maxIter: 30
      });
      expect(res.x.allClose(Vector.from([Math.SQRT2, 1]), 1e-6, 1e-6)).toBe(true);
    }
  );

  it('terminates when every trial step is rejected, even for a tiny tolx', () => {
    // f is finite only at the starting point, so no trial step can make progress. The dogleg
    // radius used to shrink forever for a non-positive `tolx`.
    const poisoned = (x: Vector): Vector =>
      x.get(0) === 3 ? Vector.from([1, 1]) : Vector.from([NaN, NaN]);
    for (const globalMethod of ['dogleg', 'line-search'] as const) {
      const res = quasiNewton(poisoned, [3, 3], {
        globalMethod,
        jac: () => Matrix.identity(2),
        jacCheck: false,
        tolx: Number.MIN_VALUE
      });
      expect(res.iterations).toBeLessThanOrEqual(2);
      expect(res.x.allClose(Vector.from([3, 3]), 0, 0)).toBe(true);
    }
  });
});

describe('quasiNewton user Jacobian shape', () => {
  const F = (x: Vector): Vector =>
    Vector.from([x.get(0) ** 2 + x.get(1) - 3, x.get(0) - x.get(1) ** 2 + 1]);
  const J = (x: Vector): Matrix =>
    Matrix.from([
      [2 * x.get(0), 1],
      [1, -2 * x.get(1)]
    ]);

  it.each([false, true])(
    'reports a wrongly shaped jac(x) at a later iteration (broydenUpdate = %s)',
    (broydenUpdate) => {
      // Correct at x0 (so the initial check passes), wrong afterwards. With Broyden updates, `jac`
      // is only called again after a failed global step, so force that with `globalMethod: null`
      // and a Jacobian that is never re-evaluated: only the non-Broyden path calls it.
      let calls = 0;
      const jac = (x: Vector): Matrix => (calls++ === 0 ? J(x) : Matrix.from([[1]]));
      const run = () =>
        quasiNewton(F, [1, 1], { jac, broydenUpdate, jacCheck: false, globalMethod: null });

      if (broydenUpdate) {
        expect(run).not.toThrow();
      } else {
        expect(run).toThrowError(RangeError);
        calls = 0;
        expect(run).toThrow('quasiNewton: jac(x) must have dimensions 2x2, got 1x1');
      }
    }
  );
});
