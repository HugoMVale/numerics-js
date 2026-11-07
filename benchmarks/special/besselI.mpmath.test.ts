import { describe, expect, it } from 'vitest';
import { besselI, besselIScaled } from '../../src/special/bessel.js';
import fixtures from './besselI.mpmath.json' with { type: 'json' };

// Must mirror the case ids in generate_besselI_mpmath_fixtures.py.
// A point passes when |actual - mpmath| <= rtol * |mpmath|. I_n has no zeros for
// x > 0, so a pure relative tolerance is meaningful everywhere (including
// underflow tails). `rtol` for the plain function is looser only where the
// result is evaluated through the log domain (underflowing scaled value).
const RTOL_SCALED = 1e-14;
const RTOL_PLAIN: Record<string, number> = {
  small_x: 1e-14,
  moderate_x: 2e-14,
  large_x: 1e-14,
  turning_point: 1e-14,
  high_order: 5e-13,
  huge_x_scaled: 1e-14,
  negative_x: 1e-14
};

describe('besselI / besselIScaled vs mpmath reference values', () => {
  for (const fixture of fixtures) {
    it(`matches mpmath for ${fixture.id}: ${fixture.description}`, () => {
      const rtol = RTOL_PLAIN[fixture.id];
      expect(rtol, `no tolerance registered for id "${fixture.id}"`).toBeDefined();
      expect(fixture.points.length).toBeGreaterThan(0);

      for (const { n, x, mpmath, mpmathScaled } of fixture.points) {
        const scaledRef = mpmathScaled.hi + mpmathScaled.lo;
        const scaled = besselIScaled(n, x);
        expect(
          Math.abs(scaled - scaledRef),
          `scaled I_${n}(${x}): got ${scaled}, mpmath ${mpmathScaled.hi}`
        ).toBeLessThanOrEqual(RTOL_SCALED * Math.abs(scaledRef));

        if (mpmath === null) {
          // I_n(x) exceeds the double range.
          expect(besselI(n, x), `I_${n}(${x}) should overflow`).toBe(Infinity);
        } else {
          const ref = mpmath.hi + mpmath.lo;
          const actual = besselI(n, x);
          expect(
            Math.abs(actual - ref),
            `I_${n}(${x}): got ${actual}, mpmath ${mpmath.hi}`
          ).toBeLessThanOrEqual(rtol * Math.abs(ref));
        }
      }
    });
  }
});
