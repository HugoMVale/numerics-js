import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  trBdf2Adaptive,
  type OdeMethod,
  type RungeKuttaAdaptiveMethod,
  type RungeKuttaFixedMethod,
  type TrBdf2AdaptiveMethod
} from '../../src/ode.js';

describe('method types', () => {
  it('are named after their solver and exported from the module entry point', () => {
    expectTypeOf<RungeKuttaFixedMethod>().toEqualTypeOf<
      'euler' | 'midpoint' | 'trapezoid' | 'rk4'
    >();
    expectTypeOf<RungeKuttaAdaptiveMethod>().toEqualTypeOf<'rk23' | 'rk45'>();
    expectTypeOf<TrBdf2AdaptiveMethod>().toEqualTypeOf<'trbdf2'>();
  });

  it('make up OdeMethod', () => {
    expectTypeOf<OdeMethod>().toEqualTypeOf<
      RungeKuttaFixedMethod | RungeKuttaAdaptiveMethod | TrBdf2AdaptiveMethod
    >();
  });

  it('describe the method reported by trBdf2Adaptive', () => {
    const result = trBdf2Adaptive((_t, y, out) => out.set(y.data).multSelf(-1), 0, 1, [1]);
    const method: TrBdf2AdaptiveMethod = result.method as TrBdf2AdaptiveMethod;
    expect(method).toBe('trbdf2');
  });
});
