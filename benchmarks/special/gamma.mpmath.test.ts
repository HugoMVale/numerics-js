import { describe, expect, it } from 'vitest';
import { gamma } from '../../src/special/gamma.js';
import { lgamma } from '../../src/special/lgamma.js';
import fixtures from './gamma.mpmath.json' with { type: 'json' };

// Must mirror the case ids in generate_gamma_fixtures.py.
// Documented accuracy in src/special/gamma.ts and src/special/lgamma.ts:
// gamma within 6 ulp; lgamma within 2 ulp of the result for x > 0 and within 5 ulp of the
// larger reflection term for x < 0, which is what the `scale` field of those points holds.
const MAX_ULP = { gamma: 6, lgamma: 2, lgammaReflection: 5 };

// Unit in the last place of a double; 2^-1074 for subnormal values.
const ulp = (value: number): number => {
  const a = Math.abs(value);
  if (a < 2.2250738585072014e-308) return 2 ** -1074;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--; // Math.log2 may round up just below a power of two
  else if (2 ** (e + 1) <= a) e++;
  return 2 ** (e - 52);
};

const boundOf = (group: { id: string; function: string }): number =>
  group.function === 'gamma'
    ? MAX_ULP.gamma
    : group.id.startsWith('lgamma_negative')
      ? MAX_ULP.lgammaReflection
      : MAX_ULP.lgamma;

describe('gamma / lgamma vs mpmath reference values', () => {
  for (const group of fixtures) {
    const bound = boundOf(group);
    it(`is within ${bound} ulp of mpmath for ${group.id}: ${group.description}`, () => {
      const fn = group.function === 'gamma' ? gamma : lgamma;
      expect(group.points.length).toBeGreaterThan(0);

      for (const point of group.points) {
        const { x, mpmath } = point;
        const actual = fn(x);
        // Error against the double-double reference hi + lo, in ulps of the result, or of the
        // larger reflection term where the result itself may be small through cancellation.
        const unit = ulp('scale' in point && point.scale !== undefined ? point.scale : mpmath.hi);
        const error = Math.abs(actual - mpmath.hi - mpmath.lo) / unit;
        expect(
          error,
          `${group.function}(${x}): got ${actual}, mpmath ${mpmath.hi}`
        ).toBeLessThanOrEqual(bound);
      }
    });
  }
});
