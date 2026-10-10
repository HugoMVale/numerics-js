import { describe, expect, it } from 'vitest';
import { besselJ } from '../../src/special/bessel.js';
import fixtures from './besselJ.scipy.json' with { type: 'json' };

// Must mirror the case ids in generate_besselJ_fixtures.py.
// A point passes when |actual - scipy| <= atol + rtol * |scipy|.
//
// SciPy's own relative error is up to ~3e-13 for large n (observed against
// 40-digit mpmath values), so `rtol` is deliberately loose where J_n decays;
// numerics-js is itself accurate to ~1e-15 relative there. In oscillatory
// regions J_n is O(n^(-1/2)) and an absolute tolerance applies; its growth with
// x reflects round-off in the phase x*sin(t) of the integral representation.
const TOLERANCES: Record<string, { atol: number; rtol: number }> = {
    small_x: { atol: 0, rtol: 1e-12 },
    decaying_n_ge_x: { atol: 0, rtol: 1e-12 },
    oscillatory_n_lt_x: { atol: 1e-13, rtol: 1e-12 },
    turning_point: { atol: 5e-14, rtol: 1e-12 },
    large_x: { atol: 3e-13, rtol: 1e-12 },
    negative_x: { atol: 5e-14, rtol: 1e-12 },
};

describe('besselJ vs scipy.special.jv reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
            const tol = TOLERANCES[fixture.id];
            expect(tol, `no tolerance registered for id "${fixture.id}"`).toBeDefined();
            expect(fixture.points.length).toBeGreaterThan(0);

            for (const { n, x, scipy } of fixture.points) {
                const actual = besselJ(n, x);
                const allowed = tol.atol + tol.rtol * Math.abs(scipy);
                expect(
                    Math.abs(actual - scipy),
                    `J_${n}(${x}): got ${actual}, SciPy ${scipy}`
                ).toBeLessThanOrEqual(allowed);
            }
        });
    }
});
