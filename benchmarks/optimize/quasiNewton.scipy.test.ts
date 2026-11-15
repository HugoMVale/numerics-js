import { describe, expect, it } from 'vitest';
import { quasiNewton, type QuasiNewtonOptions } from '../../src/optimize/quasiNewton.js';
import { GRADIENT_TEST_FUNCTIONS } from '../../tests/optimize/gradientTestFunctions.js';
import fixtures from './quasiNewton.scipy.json' with { type: 'json' };

// Must mirror CASES in generate_quasi_newton_fixtures.py (same id, objective, gradient, Hessian and
// starting point); the test problems are shared with tests/optimize/gradientTestFunctions.ts.
//
// Each SciPy algorithm is compared with the closest solver configuration:
// - BFGS, analytic gradient: line search with inverse BFGS update (the same algorithm), and the
//   default dogleg strategy.
// - BFGS, finite-difference gradient: the same two strategies, without `grad`.
// - dogleg, analytic gradient and Hessian: dogleg strategy with the exact Hessian.

// Tolerance on the scaled gradient for analytic derivatives. SciPy's `gtol` (1e-10) acts on the
// unscaled infinity norm of the gradient; this is the strictest value at which every strategy
// terminates successfully (the line search cannot always resolve a decrease of f at 1e-10).
const TOLG_ANALYTIC = 1e-8;
// With finite-difference gradients, the solver default is used.
const TOLG_FINITE_DIFFERENCE = 1e-5;

const X_TOLERANCE = { analytic: 1e-7, 'finite-difference': 1e-4 };
const F_TOLERANCE = { analytic: 1e-12, 'finite-difference': 1e-8 };
const EVALUATION_FACTOR = 1.5;

type Strategy = [name: string, options: QuasiNewtonOptions];

const BFGS_STRATEGIES: Strategy[] = [
  ['dogleg, factored BFGS (default)', {}],
  ['line search, inverse BFGS', { globalMethod: 'line-search', bfgsMethod: 'inverse' }]
];
const EXACT_HESSIAN_STRATEGIES: Strategy[] = [
  ['dogleg, exact Hessian', { globalMethod: 'dogleg' }]
];

describe('quasiNewton vs scipy.optimize.minimize reference values', () => {
  for (const fixture of fixtures) {
    const data = GRADIENT_TEST_FUNCTIONS[fixture.id];
    const derivatives = fixture.derivatives as 'analytic' | 'finite-difference';
    const strategies = fixture.method === 'dogleg' ? EXACT_HESSIAN_STRATEGIES : BFGS_STRATEGIES;

    it.each(strategies)(
      `matches SciPy ${fixture.method} (${derivatives} derivatives) for ${fixture.id}: %s`,
      (_name, strategy) => {
        expect(data, `no matching test function registered for id "${fixture.id}"`).toBeDefined();
        expect(data.x0).toEqual(fixture.x0);

        const result = quasiNewton(data.f, data.x0, {
          grad: derivatives === 'analytic' ? data.grad : undefined,
          hess: fixture.method === 'dogleg' ? data.hess : undefined,
          gradCheck: false,
          tolg: derivatives === 'analytic' ? TOLG_ANALYTIC : TOLG_FINITE_DIFFERENCE,
          ...strategy
        });

        expect(result.success, result.message).toBe(true);
        for (let i = 0; i < data.x0.length; i++) {
          expect(Math.abs(result.x.get(i) - fixture.x[i])).toBeLessThanOrEqual(
            X_TOLERANCE[derivatives]
          );
        }
        expect(Math.abs(result.fx - fixture.fx)).toBeLessThanOrEqual(F_TOLERANCE[derivatives]);
        expect(result.evaluations).toBeLessThanOrEqual(fixture.evaluations * EVALUATION_FACTOR);
        expect(result.evaluationsGradient).toBeLessThanOrEqual(
          fixture.evaluationsGradient * EVALUATION_FACTOR
        );
        expect(result.evaluationsHessian).toBeLessThanOrEqual(
          fixture.evaluationsHessian * EVALUATION_FACTOR
        );
      }
    );
  }
});
