import { describe, expect, it } from 'vitest';
import { besselK, besselKScaled } from '../../src/special/bessel.js';
import fixtures from './besselK.mpmath.json' with { type: 'json' };

// Must mirror the case ids in generate_besselK_mpmath_fixtures.py.
// A point passes when |actual - mpmath| <= rtol * |mpmath|. K_n has no zeros for
// x > 0, so a pure relative tolerance is meaningful everywhere. The forward
// recurrence adds round-off that grows slowly with n, hence the looser
// tolerances for the high-order groups.
const RTOL_SCALED: Record<string, number> = {
  small_x: 1e-14,
  moderate_x: 1e-14,
  large_x: 1e-14,
  turning_point: 1e-14,
  high_order: 1e-14,
  huge_x_scaled: 1e-14
};
const RTOL_PLAIN: Record<string, number> = {
  small_x: 1e-14,
  moderate_x: 1e-14,
  large_x: 1e-14,
  turning_point: 1e-14,
  high_order: 1e-14,
  huge_x_scaled: 1e-14
};

describe('besselK / besselKScaled vs mpmath reference values', () => {
  for (const fixture of fixtures) {
    it(`matches mpmath for ${fixture.id}: ${fixture.description}`, () => {
      const rtolScaled = RTOL_SCALED[fixture.id];
      const rtolPlain = RTOL_PLAIN[fixture.id];
      expect(rtolScaled, `no tolerance registered for id "${fixture.id}"`).toBeDefined();
      expect(fixture.points.length).toBeGreaterThan(0);

      for (const { n, x, mpmath, mpmathScaled, log10K } of fixture.points) {
        if (mpmathScaled === null) {
          // e^x K_n(x) exceeds the double range.
          expect(besselKScaled(n, x), `scaled K_${n}(${x}) should overflow`).toBe(Infinity);
        } else {
          const ref = mpmathScaled.hi + mpmathScaled.lo;
          const actual = besselKScaled(n, x);
          expect(
            Math.abs(actual - ref),
            `scaled K_${n}(${x}): got ${actual}, mpmath ${mpmathScaled.hi}`
          ).toBeLessThanOrEqual(rtolScaled * Math.abs(ref));
        }

        if (mpmath !== null) {
          const ref = mpmath.hi + mpmath.lo;
          const actual = besselK(n, x);
          expect(
            Math.abs(actual - ref),
            `K_${n}(${x}): got ${actual}, mpmath ${mpmath.hi}`
          ).toBeLessThanOrEqual(rtolPlain * Math.abs(ref));
        } else if (log10K > 0) {
          expect(besselK(n, x), `K_${n}(${x}) should overflow`).toBe(Infinity);
        } else {
          // Underflow to zero or into the subnormal range: no relative accuracy is expected.
          const actual = besselK(n, x);
          expect(actual, `K_${n}(${x}) should underflow`).toBeGreaterThanOrEqual(0);
          expect(actual, `K_${n}(${x}) should underflow`).toBeLessThan(2.3e-308);
        }
      }
    });
  }
});
