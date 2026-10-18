import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { rungeKuttaAdaptive } from '../../src/ode/rungeKuttaAdaptive.js';
import type { RungeKuttaAdaptiveOptions } from '../../src/ode/rungeKuttaAdaptive.js';
import { trBdf2Adaptive } from '../../src/ode/trBdf2Adaptive.js';
import type { DerivativeFunction, OdeEvent, OdeResult } from '../../src/ode/types.js';

// Both adaptive solvers share this subset of options, so every test below runs against each of them.
type CommonOptions = Pick<
  RungeKuttaAdaptiveOptions,
  'atol' | 'rtol' | 'h0' | 'hMax' | 'maxSteps' | 'events' | 'denseOutput'
>;
type Solve = (
  f: DerivativeFunction,
  t0: number,
  tEnd: number,
  y0: Vector,
  options?: CommonOptions
) => OdeResult;

const solvers: [string, Solve][] = [
  ['rk23', (f, t0, tEnd, y0, o) => rungeKuttaAdaptive('rk23', f, t0, tEnd, y0, o)],
  ['rk45', (f, t0, tEnd, y0, o) => rungeKuttaAdaptive('rk45', f, t0, tEnd, y0, o)],
  ['trbdf2', (f, t0, tEnd, y0, o) => trBdf2Adaptive(f, t0, tEnd, y0, o)]
];

// Harmonic oscillator: y = [cos t, -sin t] for y(0) = [1, 0].
const oscillator: DerivativeFunction = (_t, y, out) => {
  out.data[0] = y.data[1];
  out.data[1] = -y.data[0];
  return out;
};
const oscillatorExact = (t: number): number[] => [Math.cos(t), -Math.sin(t)];

// Stiff relaxation y' = -1000 (y - cos t), y(0) = 0.
const LAMBDA = 1000;
const stiff: DerivativeFunction = (t, y, out) => {
  out.data[0] = -LAMBDA * (y.data[0] - Math.cos(t));
  return out;
};
const stiffExact = (t: number): number[] => [
  (LAMBDA ** 2 * Math.cos(t) + LAMBDA * Math.sin(t) - LAMBDA ** 2 * Math.exp(-LAMBDA * t)) /
    (LAMBDA ** 2 + 1)
];

const tol = { atol: 1e-8, rtol: 1e-8 };
const grid = (a: number, b: number, n: number): number[] =>
  Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);

/** Largest component error of `eval` against `exact` on the given times. */
function maxError(
  eval_: (t: Vector) => { get(i: number, j: number): number },
  exact: (t: number) => number[],
  times: number[]
): number {
  const y = eval_(Vector.from(times));
  let error = 0;
  times.forEach((t, i) =>
    exact(t).forEach((e, j) => (error = Math.max(error, Math.abs(y.get(i, j) - e))))
  );
  return error;
}

/** Largest component error of the accepted points of `result` against `exact`. */
function maxNodeError(result: OdeResult, exact: (t: number) => number[]): number {
  let error = 0;
  for (let i = 0; i < result.t.size; i++) {
    exact(result.t.get(i)).forEach(
      (e, j) => (error = Math.max(error, Math.abs(result.y.get(i, j) - e)))
    );
  }
  return error;
}

describe.each(solvers)('denseOutput of %s', (_name, solve) => {
  const y0 = (): Vector => new Vector([1, 0]);

  it('is present only when requested', () => {
    expect(solve(oscillator, 0, 2, y0()).denseOutput).toBeUndefined();
    expect(solve(oscillator, 0, 2, y0(), { denseOutput: false }).denseOutput).toBeUndefined();
    expect(solve(oscillator, 0, 2, y0(), { denseOutput: true }).denseOutput).toBeDefined();
  });

  it('does not change the integration', () => {
    const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] }];
    const plain = solve(oscillator, 0, 7, y0(), { ...tol, events });
    const dense = solve(oscillator, 0, 7, y0(), { ...tol, events, denseOutput: true });

    expect(dense.t).toEqual(plain.t);
    expect(dense.y).toEqual(plain.y);
    expect(dense.evaluations).toBe(plain.evaluations);
    expect(dense.evaluationsJacobian).toBe(plain.evaluationsJacobian);
    expect(dense.decompositions).toBe(plain.decompositions);
    expect(dense.message).toBe(plain.message);
    expect(dense.events).toEqual(plain.events);
  });

  it('covers the integration interval', () => {
    const dense = solve(oscillator, 0.5, 6, y0(), { denseOutput: true }).denseOutput!;
    expect(dense.dim).toBe(2);
    expect(dense.tStart).toBe(0.5);
    expect(dense.tEnd).toBe(6);
  });

  it('reproduces the accepted points: exactly at every step start, to rounding at the end', () => {
    const result = solve(oscillator, 0, 5, y0(), { atol: 1e-6, rtol: 1e-6, denseOutput: true });
    const dense = result.denseOutput!;
    const last = result.t.size - 1;
    expect(last).toBeGreaterThan(3);

    for (let i = 0; i < last; i++) {
      expect(Array.from(dense.eval(result.t.get(i)).data)).toEqual(
        Array.from(result.y.row(i).data)
      );
    }
    const end = dense.eval(result.t.get(last));
    expect(Math.abs(end.get(0) - result.y.get(last, 0))).toBeLessThan(1e-14);
    expect(Math.abs(end.get(1) - result.y.get(last, 1))).toBeLessThan(1e-14);
  });

  it('is as accurate between the steps as at them', () => {
    for (const tolerance of [1e-4, 1e-6, 1e-8]) {
      const result = solve(oscillator, 0, 10, y0(), {
        atol: tolerance,
        rtol: tolerance,
        denseOutput: true
      });
      const dense = result.denseOutput!;
      const nodeError = maxNodeError(result, oscillatorExact);
      const denseError = maxError((t) => dense.eval(t), oscillatorExact, grid(0, 10, 1000));

      // The interpolation error is not controlled, but must not be much larger than the error of the points.
      expect(denseError).toBeLessThan(3 * nodeError);
    }
  });

  it('gets more accurate with tighter tolerances', () => {
    const error = (tolerance: number): number => {
      const dense = solve(oscillator, 0, 10, y0(), {
        atol: tolerance,
        rtol: tolerance,
        denseOutput: true
      }).denseOutput!;
      return maxError((t) => dense.eval(t), oscillatorExact, grid(0, 10, 500));
    };
    expect(error(1e-8)).toBeLessThan(error(1e-5) / 100);
  });

  it('evaluates at several times at once, one row per time, in the given order', () => {
    const dense = solve(oscillator, 0, 5, y0(), { ...tol, denseOutput: true }).denseOutput!;
    const times = [3.3, 0, 5, 1.7];

    for (const argument of [Vector.from(times), new Vector(times)]) {
      const y = dense.eval(argument);
      expect(y.rows).toBe(4);
      expect(y.cols).toBe(2);
      times.forEach((t, i) =>
        expect(Array.from(y.row(i).data)).toEqual(Array.from(dense.eval(t).data))
      );
    }
  });

  it('rejects times outside the solved interval', () => {
    const dense = solve(oscillator, 0, 5, y0(), { denseOutput: true }).denseOutput!;
    expect(() => dense.eval(-1e-9)).toThrow(RangeError);
    expect(() => dense.eval(5 + 1e-9)).toThrow(RangeError);
    expect(() => dense.eval(Vector.from([1, 6]))).toThrow(RangeError);
  });

  it('works for backward integration', () => {
    const start = 4;
    const result = solve(oscillator, start, 0, new Vector(oscillatorExact(start)), {
      ...tol,
      denseOutput: true
    });
    const dense = result.denseOutput!;

    expect(dense.tStart).toBe(4);
    expect(dense.tEnd).toBe(0);
    expect(maxError((t) => dense.eval(t), oscillatorExact, grid(4, 0, 500))).toBeLessThan(
      3 * maxNodeError(result, oscillatorExact)
    );
    expect(Array.from(dense.eval(start).data)).toEqual(Array.from(oscillatorExact(start)));
    expect(() => dense.eval(4.1)).toThrow(RangeError);
    expect(() => dense.eval(-0.1)).toThrow(RangeError);
  });

  it('ends at the event time when a terminal event stops the integration', () => {
    const zero: OdeEvent = { fn: (_t, y) => y.data[0], direction: 'falling', terminal: true };
    const result = solve(oscillator, 0, 10, y0(), {
      ...tol,
      events: [zero],
      denseOutput: true
    });
    const dense = result.denseOutput!;
    const [hit] = result.events!;

    expect(result.message).toBe('Integration stopped by a terminal event.');
    expect(dense.tEnd).toBe(hit.t);
    expect(result.t.get(result.t.size - 1)).toBe(hit.t);
    for (let j = 0; j < 2; j++) expect(dense.eval(hit.t).get(j)).toBeCloseTo(hit.y.get(j), 13);
    expect(() => dense.eval(hit.t + 1e-6)).toThrow(RangeError);
    expect(maxError((t) => dense.eval(t), oscillatorExact, grid(0, hit.t, 300))).toBeLessThan(
      3 * maxNodeError(result, oscillatorExact)
    );
  });

  it('covers the part that was solved when the integration fails', () => {
    const result = solve(oscillator, 0, 10, y0(), { maxSteps: 4, denseOutput: true });
    const dense = result.denseOutput!;
    const last = result.t.size - 1;

    expect(result.success).toBe(false);
    expect(last).toBeGreaterThan(0);
    expect(dense.tEnd).toBe(result.t.get(last));
    for (let j = 0; j < 2; j++)
      expect(dense.eval(dense.tEnd).get(j)).toBeCloseTo(result.y.get(last, j), 14);
    expect(() => dense.eval(10)).toThrow(RangeError);
  });

  it('returns the initial state when there is nothing to integrate', () => {
    const result = solve(oscillator, 1, 1, y0(), { denseOutput: true });
    const dense = result.denseOutput!;

    expect(dense.tStart).toBe(1);
    expect(dense.tEnd).toBe(1);
    expect(Array.from(dense.eval(1).data)).toEqual([1, 0]);
    expect(() => dense.eval(1.5)).toThrow(RangeError);
  });

  it('is not affected by later changes to the initial state', () => {
    const initial = y0();
    const dense = solve(oscillator, 0, 2, initial, { denseOutput: true }).denseOutput!;
    initial.data[0] = 99;
    expect(Array.from(dense.eval(0).data)).toEqual([1, 0]);
  });
});

describe('denseOutput of trbdf2 on a stiff problem', () => {
  it.each([1e-4, 1e-6, 1e-8])(
    'is as accurate between the steps as at them (tolerance %g)',
    (tolerance) => {
      const result = trBdf2Adaptive(stiff, 0, 1, new Vector([0]), {
        atol: tolerance,
        rtol: tolerance,
        denseOutput: true
      });
      const dense = result.denseOutput!;

      const denseError = maxError((t) => dense.eval(t), stiffExact, grid(0, 1, 2000));
      expect(denseError).toBeLessThan(3 * maxNodeError(result, stiffExact));
    }
  );
});

describe.each(['rk23', 'rk45'] as const)('denseOutput of %s', (method) => {
  it('reproduces exactly a solution that is a cubic polynomial in t', () => {
    // y' = 3 t^2 + 1 has y = t^3 + t, which both methods integrate exactly, and so does their interpolant.
    const f: DerivativeFunction = (t, _y, out) => {
      out.data[0] = 3 * t * t + 1;
      return out;
    };
    const result = rungeKuttaAdaptive(method, f, 0, 3, new Vector([0]), {
      atol: 1e-3,
      rtol: 1e-3,
      denseOutput: true
    });
    const dense = result.denseOutput!;

    expect(result.t.size).toBeGreaterThan(2);
    for (const t of grid(0, 3, 100)) {
      expect(Math.abs(dense.eval(t).get(0) - (t ** 3 + t))).toBeLessThan(1e-12);
    }
  });
});
