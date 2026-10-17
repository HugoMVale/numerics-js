import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { levenbergMarquardt } from '../../src/optimize/levenbergMarquardt.js';
import { LEAST_SQUARES_PROBLEMS } from '../../tests/optimize/leastSquaresFunctions.js';
import fixtures from './levenbergMarquardt.scipy.json' with { type: 'json' };

// The problems (and their Jacobians) must mirror PROBLEMS in generate_levenberg_marquardt_fixtures.py;
// they are defined in tests/optimize/leastSquaresFunctions.ts, indexed by the fixture's `problem` field.

const MIRROR_TOLERANCE = 1e-12;
const X_TOLERANCE = 1e-5;
// Allowed ratio of numerics-js to SciPy evaluations of f (finite-difference evaluations included). Both
// implementations follow MINPACK, and the observed ratio is at most 1.0 on all cases.
const EVALUATION_FACTOR = 1.1;
// SciPy stops early on these, at a worse point than numerics-js (see the benchmark report).
const SCIPY_STOPS_EARLY = new Set(['watson']);
// The solution of Powell's problem is a singular point, so `x` is only determined to about 1e-8 (the
// sum of squares is then ~1e-16 or smaller): the minimizer is compared with an absolute tolerance
// `X_TOLERANCE` instead of a relative one.
const X_SCALE_FLOOR: Record<string, number> = { powellSingular: 1 };

describe('test problems mirror the SciPy reference functions', () => {
  for (const fixture of fixtures) {
    it(`${fixture.id}: f agrees with the Python implementation at x0 and at generic points`, () => {
      const problem = LEAST_SQUARES_PROBLEMS[fixture.problem];
      expect(problem, `no matching test problem registered for "${fixture.problem}"`).toBeDefined();

      expect(
        problem
          .f(Vector.from(fixture.x0))
          .allClose(Vector.from(fixture.fx0), MIRROR_TOLERANCE, MIRROR_TOLERANCE)
      ).toBe(true);
      for (const point of fixture.mirrorPoints) {
        expect(
          problem
            .f(Vector.from(point.x))
            .allClose(Vector.from(point.fx), MIRROR_TOLERANCE, MIRROR_TOLERANCE)
        ).toBe(true);
      }
    });
  }
});

describe('levenbergMarquardt vs scipy.optimize.leastsq (MINPACK) reference values', () => {
  for (const fixture of fixtures) {
    it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
      const problem = LEAST_SQUARES_PROBLEMS[fixture.problem];
      expect(problem, `no matching test problem registered for "${fixture.problem}"`).toBeDefined();
      expect(
        !fixture.useJac || problem.jac,
        `problem "${fixture.problem}" has no Jacobian`
      ).toBeTruthy();

      const result = levenbergMarquardt(problem.f, Vector.from(fixture.x0), {
        tolf: fixture.tolf,
        tolx: fixture.tolx,
        tolg: fixture.tolg,
        maxFunEvals: fixture.maxFunEvals,
        // SciPy does not check the Jacobian either, and its evaluation counts exclude the check.
        jacCheck: false,
        ...(fixture.useJac ? { jac: problem.jac } : {})
      });

      expect(result.success, result.message).toBe(true);

      // Same (or better) minimum.
      const sumSquares = 2 * result.cost;
      expect(sumSquares).toBeLessThanOrEqual(fixture.scipySumSquares * (1 + 1e-6) + 1e-12);
      expect(Math.abs(sumSquares - problem.minSumSquares)).toBeLessThanOrEqual(
        1e-6 * Math.max(problem.minSumSquares, 1e-3) + 1e-12
      );

      // Same minimizer, if it is unique and SciPy did not stop early.
      if (fixture.uniqueMinimizer && !SCIPY_STOPS_EARLY.has(fixture.problem)) {
        for (let i = 0; i < fixture.scipyX.length; i++) {
          const floor = X_SCALE_FLOOR[fixture.problem] ?? 1e-5;
          const tol = X_TOLERANCE * Math.max(Math.abs(fixture.scipyX[i]), floor);
          expect(Math.abs(result.x.get(i) - fixture.scipyX[i])).toBeLessThanOrEqual(tol);
        }
      }

      // Comparable cost.
      expect(result.evaluationsFunction).toBeLessThanOrEqual(
        fixture.scipyEvaluations * EVALUATION_FACTOR
      );
      if (fixture.useJac) {
        expect(result.evaluationsJacobian).toBeLessThanOrEqual(
          Math.ceil(fixture.scipyJacobianEvaluations * EVALUATION_FACTOR)
        );
      }
    });
  }
});
