import { describe, expect, it, vi } from 'vitest';
import { rungeKuttaAdaptive } from '../../src/ode/rungeKuttaAdaptive.js';
import { trBdf2Adaptive } from '../../src/ode/trBdf2Adaptive.js';
import type { DerivativeFunction, OdeResult } from '../../src/ode/types.js';
import type { VectorLike } from '../../src/linalg/Vector.js';

type Options = { h0?: number };
type Solver = (
  f: DerivativeFunction,
  t0: number,
  tEnd: number,
  y0: VectorLike,
  o?: Options
) => OdeResult;

const solvers: [string, Solver][] = [
  [
    'rungeKuttaAdaptive (rk23)',
    (f, t0, tEnd, y0, o) => rungeKuttaAdaptive('rk23', f, t0, tEnd, y0, o)
  ],
  [
    'rungeKuttaAdaptive (rk45)',
    (f, t0, tEnd, y0, o) => rungeKuttaAdaptive('rk45', f, t0, tEnd, y0, o)
  ],
  ['trBdf2Adaptive', (f, t0, tEnd, y0, o) => trBdf2Adaptive(f, t0, tEnd, y0, o)]
];

describe.each(solvers)('%s: non-finite input', (_name, solve) => {
  const makeF = () => vi.fn<DerivativeFunction>((_t, y, out) => out.set(y.data).multSelf(-1));

  it.each<[string, number, number, number[], Options, RegExp]>([
    ['NaN t0', NaN, 1, [1], {}, /t0 \(NaN\) and tEnd \(1\) must be finite/],
    ['NaN tEnd', 0, NaN, [1], {}, /t0 \(0\) and tEnd \(NaN\) must be finite/],
    ['NaN t0 and tEnd', NaN, NaN, [1], {}, /must be finite/],
    ['infinite tEnd', 0, Infinity, [1], {}, /tEnd \(Infinity\) must be finite/],
    ['infinite t0', -Infinity, 0, [1], {}, /t0 \(-Infinity\) and tEnd \(0\) must be finite/],
    ['NaN in y0', 0, 1, [1, NaN], {}, /y0 must be finite, got y0\[1\] = NaN/],
    ['infinite y0', 0, 1, [Infinity], {}, /y0\[0\] = Infinity/],
    ['NaN h0', 0, 1, [1], { h0: NaN }, /h0 \(NaN\) must be finite and nonzero/],
    ['infinite h0', 0, 1, [1], { h0: Infinity }, /h0 \(Infinity\) must be finite and nonzero/]
  ])(
    'throws a RangeError for %s, without evaluating f',
    (_label, t0, tEnd, y0, options, message) => {
      const f = makeF();
      const call = () => solve(f, t0, tEnd, y0, options);
      expect(call).toThrow(RangeError);
      expect(call).toThrow(message);
      expect(f).not.toHaveBeenCalled();
    }
  );

  it('still integrates normally with finite input, in both directions', () => {
    const forward = solve(makeF(), 0, 1, [1]);
    const backward = solve(makeF(), 1, 0, [1]);
    expect(forward.success).toBe(true);
    expect(backward.success).toBe(true);
    expect(forward.y.get(forward.y.rows - 1, 0)).toBeCloseTo(Math.exp(-1), 2);
    expect(backward.y.get(backward.y.rows - 1, 0)).toBeCloseTo(Math.exp(1), 1);
  });
});
