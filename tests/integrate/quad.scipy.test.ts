import { describe, expect, it } from 'vitest';
import { quad } from '../../src/integrate/quad.js';
import fixtures from './fixtures/quad.scipy.json' with { type: 'json' };

// Must mirror the CASES list in scripts/benchmarks/generate_quad_fixtures.py (same id, math, bounds).
const INTEGRANDS: Record<string, (x: number) => number> = {
    sine: Math.sin,
    rational_pi: (x) => 4 / (1 + x * x),
    oscillatory: (x) => Math.sin(100 * x),
    sqrt_singularity: (x) => 1 / Math.sqrt(x),
    log_singularity: (x) => -Math.log(x),
    gaussian: (x) => Math.exp(-x * x),
    runge: (x) => 1 / (1 + 25 * x * x),
    high_freq_cos: (x) => Math.cos(50 * x),
    lorentzian_peak: (x) => 1 / ((x - 0.5) ** 2 + 1e-4),
    exp_growth: Math.exp,
    exp_decay_positive_infinite: (x) => Math.exp(-x),
    exp_decay_negative_infinite: Math.exp,
    gaussian_fully_infinite: (x) => Math.exp(-x * x),
    piecewise_breakpoint: (x) => (x < 1 ? 1 : 2),
};

// Allowed slack for evaluations vs SciPy's QUADPACK reference, applied only to smooth
// integrands (see EVALUATION_FACTOR use below): observed ratios there are ~0.7x-1.2x.
const EVALUATION_FACTOR = 1.2;
const decodeBound = (bound: number | string): number =>
    bound === 'Infinity' ? Infinity : bound === '-Infinity' ? -Infinity : Number(bound);

describe('quad vs SciPy reference values', () => {
    for (const fixture of fixtures) {
        it(`matches scipy.integrate.quad for ${fixture.id}: ${fixture.description}`, () => {
            const f = INTEGRANDS[fixture.id];
            expect(f, `no matching integrand registered for id "${fixture.id}"`).toBeDefined();

            const a = decodeBound(fixture.a);
            const b = decodeBound(fixture.b);

            // Uses gaussKronrod's own default tol, matching TOL in generate_quad_fixtures.py,
            // so evaluation counts reflect the library's representative use case.
            const result = quad(f, a, b, { breakpoints: fixture.breakpoints });
            expect(result.converged).toBe(true);

            // Tolerance accounts for both SciPy's and our own reported error estimates.
            const tolerance = Math.max(fixture.scipyAbsError, result.error) * 10 + 1e-10;
            expect(Math.abs(result.value - fixture.scipyValue)).toBeLessThan(tolerance);

            // gaussKronrod has no singularity-aware subdivision (unlike QUADPACK's QAGS), so
            // evaluation counts are only meaningfully comparable outside its domain of validity.
            if (!fixture.singular) {
                expect(result.evaluations).toBeLessThan(fixture.scipyEvaluations * EVALUATION_FACTOR);
            }
        });
    }
});