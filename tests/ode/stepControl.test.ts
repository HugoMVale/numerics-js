import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { estimateInitialStep } from '../../src/ode/stepControl.js';
import type { DerivativeFunction } from '../../src/ode/types.js';

describe('estimateInitialStep', () => {
  const decay: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);

  it('leaves f(t0, y0) in f0 and uses exactly two derivative evaluations', () => {
    let calls = 0;
    const f: DerivativeFunction = (t, y, out) => {
      calls++;
      return decay(t, y, out);
    };
    const y0 = new Vector([2, -3]);
    const f0 = new Vector(2);

    estimateInitialStep(f, 0, y0, 1, 1e-6, 1e-3, 2, f0, new Vector(2), new Vector(2));

    expect(calls).toBe(2);
    expect(Array.from(f0.data)).toEqual([-2, 3]);
    expect(Array.from(y0.data)).toEqual([2, -3]); // not mutated
  });

  it('returns a step with the sign of the integration direction', () => {
    const run = (dir: 1 | -1) =>
      estimateInitialStep(
        decay,
        0,
        new Vector([1]),
        dir,
        1e-6,
        1e-3,
        2,
        new Vector(1),
        new Vector(1),
        new Vector(1)
      );

    const forward = run(1);
    const backward = run(-1);

    expect(forward).toBeGreaterThan(0);
    expect(backward).toBeLessThan(0);
    expect(Number.isFinite(forward)).toBe(true);
  });

  it('falls back to a tiny step when the solution and its derivative are negligible', () => {
    const f: DerivativeFunction = (_t, _y, out) => out.fill(0);
    const h = estimateInitialStep(
      f,
      0,
      new Vector([0]),
      1,
      1e-6,
      1e-3,
      2,
      new Vector(1),
      new Vector(1),
      new Vector(1)
    );

    expect(h).toBeGreaterThan(0);
    expect(h).toBeLessThan(1e-3);
  });
});
