import { describe, expect, it } from 'vitest';
import { erf, erfc } from '../../src/special/erf.js';
import fixtures from './erf.scipy.json' with { type: 'json' };

// Must mirror the case ids in generate_erf_fixtures.py; the id prefix selects the function.
// A point passes when |actual - scipy| <= atol + rtol * |scipy|.
//
// SciPy (Cephes) is itself the less accurate side in the erfc tail: against 60-digit
// mpmath values its relative error reaches ~3.6e-15 for 1.25 <= x < 6 and ~5.5e-14 for
// 6 <= x <= 26, while numerics-js stays within ~4e-16 everywhere. The `rtol` values there
// are therefore set by SciPy's error, not by numerics-js.
const TOLERANCES: Record<string, { atol: number; rtol: number }> = {
  erf_small: { atol: 0, rtol: 1e-15 },
  erf_mid: { atol: 0, rtol: 1e-15 },
  erf_tail: { atol: 0, rtol: 1e-15 },
  erf_saturated: { atol: 0, rtol: 0 },
  erfc_small: { atol: 0, rtol: 2e-15 },
  erfc_mid: { atol: 0, rtol: 4e-15 },
  erfc_tail: { atol: 0, rtol: 1e-14 },
  erfc_far_tail: { atol: 0, rtol: 2e-13 },
  erfc_negative: { atol: 0, rtol: 1e-15 }
};

describe('erf / erfc vs scipy.special reference values', () => {
  for (const fixture of fixtures) {
    it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
      const tol = TOLERANCES[fixture.id];
      expect(tol, `no tolerance registered for id "${fixture.id}"`).toBeDefined();
      expect(fixture.points.length).toBeGreaterThan(0);

      const fn = fixture.id.startsWith('erfc') ? erfc : erf;
      const name = fn.name;

      for (const { x, scipy } of fixture.points) {
        const actual = fn(x);
        const allowed = tol.atol + tol.rtol * Math.abs(scipy);
        expect(
          Math.abs(actual - scipy),
          `${name}(${x}): got ${actual}, SciPy ${scipy}`
        ).toBeLessThanOrEqual(allowed);
      }
    });
  }
});
