import { describe, expect, it } from 'vitest';
import { erf, erfc } from '../../src/special/erf.js';
import fixtures from './erf.mpmath.json' with { type: 'json' };

// Must mirror the case ids in generate_erf_fixtures.py.
// Documented accuracy in src/special/erf.ts: erf within 1 ulp, erfc within 3 ulp.
const MAX_ULP: Record<string, number> = { erf: 1, erfc: 3 };

// Unit in the last place of a double; 2^-1074 for subnormal values.
const ulp = (value: number): number =>
  Math.max(2 ** (Math.floor(Math.log2(Math.abs(value))) - 52), 2 ** -1074);

describe('erf / erfc vs mpmath reference values', () => {
  for (const group of fixtures) {
    it(`is within ${MAX_ULP[group.function]} ulp of mpmath for ${group.id}: ${group.description}`, () => {
      const fn = group.function === 'erfc' ? erfc : erf;
      const bound = MAX_ULP[group.function];
      expect(bound, `no ulp bound for "${group.function}"`).toBeDefined();
      expect(group.points.length).toBeGreaterThan(0);

      for (const { x, mpmath } of group.points) {
        const actual = fn(x);
        // Error against the double-double reference hi + lo.
        const error = Math.abs(actual - mpmath.hi - mpmath.lo) / ulp(mpmath.hi);
        expect(
          error,
          `${group.function}(${x}): got ${actual}, mpmath ${mpmath.hi}`
        ).toBeLessThanOrEqual(bound);
      }
    });
  }
});
