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
});
