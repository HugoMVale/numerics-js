import { describe, expect, it } from 'vitest';
import { besselJ } from '../../src/special/bessel.js';
import fixtures from './besselJ.mpmath.json' with { type: 'json' };

// Must mirror the case ids in generate_besselJ_mpmath_fixtures.py.
// A point passes when |actual - mpmath| <= atol + rtol * |mpmath|.
const RTOL = 2e-14;
const TOLERANCES: Record<string, { atol: number; rtol: number }> = {
  small_x: { atol: 0, rtol: RTOL },
  decaying_n_ge_x: { atol: 0, rtol: RTOL },
  oscillatory_n_lt_x: { atol: 1e-14, rtol: RTOL },
  turning_point: { atol: 1e-14, rtol: RTOL },
  large_x: { atol: 1e-13, rtol: RTOL },
  negative_x: { atol: 0, rtol: RTOL }
};

// Accuracy documented in the besselJ docstring (src/special/bessel.ts): relative error for
// |x| < 1 and for n >= |x| >= 1, absolute error for n < |x| (looser beyond |x| = 5e3).
// Number.MIN_VALUE keeps the relative limits meaningful for subnormal references.
function documentedLimit(n: number, x: number, reference: number): number {
  const ax = Math.abs(x);
  if (ax < 1) return Math.max(1e-15 * Math.abs(reference), Number.MIN_VALUE);
  if (n >= ax) return Math.max(1e-14 * Math.abs(reference), Number.MIN_VALUE);
  return ax <= 5e3 ? 1e-14 : 1e-13;
}

describe('besselJ vs mpmath reference values', () => {
  for (const fixture of fixtures) {
    it(`matches mpmath for ${fixture.id}: ${fixture.description}`, () => {
      const tol = TOLERANCES[fixture.id];
      expect(tol, `no tolerance registered for id "${fixture.id}"`).toBeDefined();
      expect(fixture.points.length).toBeGreaterThan(0);

      for (const { n, x, mpmath } of fixture.points) {
        const actual = besselJ(n, x);
        const reference = mpmath.hi + mpmath.lo;
        const allowed = tol.atol + tol.rtol * Math.abs(reference);
        expect(
          Math.abs(actual - reference),
          `J_${n}(${x}): got ${actual}, mpmath ${mpmath.hi}`
        ).toBeLessThanOrEqual(allowed);
      }
    });
  }

  it('meets the accuracy documented in the besselJ docstring at every point', () => {
    for (const fixture of fixtures) {
      for (const { n, x, mpmath } of fixture.points) {
        const reference = mpmath.hi + mpmath.lo;
        expect(
          Math.abs(besselJ(n, x) - reference),
          `${fixture.id}: J_${n}(${x}), mpmath ${mpmath.hi}`
        ).toBeLessThanOrEqual(documentedLimit(n, x, reference));
      }
    }
  });
});
