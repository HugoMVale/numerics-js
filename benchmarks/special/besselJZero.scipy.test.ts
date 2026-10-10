import { beforeEach, describe, expect, it } from 'vitest';
import { besselJZero, clearBesselCache } from '../../src/special/bessel.js';
import fixtures from './besselJZero.scipy.json' with { type: 'json' };

// Relative tolerance on each zero. numerics-js agrees with scipy.special.jn_zeros to
// ~3e-16 for every listed case, so this leaves room for platform differences in
// the trigonometric functions only.
const RTOL = 1e-13;

describe('besselJZero vs scipy.special.jn_zeros reference values', () => {
    beforeEach(() => {
        clearBesselCache();
    });

    for (const fixture of fixtures) {
        it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
            // Largest index first, then smaller ones, so both the scan and the cache are exercised.
            const ordered = [...fixture.zeros].sort((a, b) => b.m - a.m);
            for (const { m, scipy } of ordered) {
                const actual = besselJZero(fixture.n, m);
                expect(
                    Math.abs(actual - scipy) / scipy,
                    `j_{${fixture.n},${m}}: got ${actual}, SciPy ${scipy}`
                ).toBeLessThanOrEqual(RTOL);
            }
        });

        it(`returns every consecutive zero of J_${fixture.n} up to m = 60 in order`, () => {
            const reference = fixture.zeros.filter((z) => z.m <= 60);
            let previous = 0;
            for (let m = 1; m <= 60; m++) {
                const z = besselJZero(fixture.n, m);
                expect(z).toBeGreaterThan(previous);
                previous = z;
                const ref = reference.find((r) => r.m === m);
                if (ref) expect(Math.abs(z - ref.scipy) / ref.scipy).toBeLessThanOrEqual(RTOL);
            }
        });
    }
});
