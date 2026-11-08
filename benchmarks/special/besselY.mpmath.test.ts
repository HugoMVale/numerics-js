import { describe, expect, it } from 'vitest';
import { besselY } from '../../src/special/bessel.js';
import fixtures from './besselY.mpmath.json' with { type: 'json' };

// Must mirror the case ids in generate_besselY_mpmath_fixtures.py.
// A point passes when |actual - mpmath| <= atol + rtol * |mpmath|.
// atol covers the absolute round-off near the zeros of Y_n, where the relative
// error is not meaningful; rtol covers the large values (n > x) and the
// accumulated recurrence error.
const RTOL = 2e-14;
const TOLERANCES: Record<string, { atol: number; rtol: number }> = {
  small_x: { atol: 1e-15, rtol: RTOL },
  mid_x: { atol: 3e-15, rtol: RTOL },
  large_x: { atol: 1e-15, rtol: RTOL },
  growing_n_ge_x: { atol: 2e-15, rtol: RTOL },
  oscillatory_n_lt_x: { atol: 1e-15, rtol: RTOL },
  turning_point: { atol: 4e-15, rtol: RTOL },
  dense_n01: { atol: 1e-15, rtol: RTOL }
};

// Accuracy documented in the besselY docstring (src/special/bessel.ts): absolute error where
// |Y_n| <= 1, relative error where |Y_n| > 1; n <= 1 and n >= 2 (up to n = 1000) differ.
function documentedLimit(n: number, reference: number): number {
  const [abs, rel] = n <= 1 ? [4e-16, 4e-16] : [2e-15, 1e-14];
  return Math.abs(reference) <= 1 ? abs : rel * Math.abs(reference);
}

describe('besselY vs mpmath reference values', () => {
  for (const fixture of fixtures) {
    it(`matches mpmath for ${fixture.id}: ${fixture.description}`, () => {
      const tol = TOLERANCES[fixture.id];
      expect(tol, `no tolerance registered for id "${fixture.id}"`).toBeDefined();
      expect(fixture.points.length).toBeGreaterThan(0);

      for (const { n, x, mpmath } of fixture.points) {
        const actual = besselY(n, x);
        const reference = mpmath.hi + mpmath.lo;
        const allowed = tol.atol + tol.rtol * Math.abs(reference);
        expect(
          Math.abs(actual - reference),
          `Y_${n}(${x}): got ${actual}, mpmath ${mpmath.hi}`
        ).toBeLessThanOrEqual(allowed);
      }
    });
  }

  it('meets the accuracy documented in the besselY docstring at every point', () => {
    for (const fixture of fixtures) {
      for (const { n, x, mpmath } of fixture.points) {
        const reference = mpmath.hi + mpmath.lo;
        expect(
          Math.abs(besselY(n, x) - reference),
          `${fixture.id}: Y_${n}(${x}), mpmath ${mpmath.hi}`
        ).toBeLessThanOrEqual(documentedLimit(n, reference));
      }
    }
  });
});
