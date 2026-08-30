import { describe, expect, it } from 'vitest';
import { gaussKronrod } from '../../src/integrate/gaussKronrod.js';
import fixtures from './fixtures/gaussKronrod.scipy.json' with { type: 'json' };

// Must mirror the CASES list in scripts/benchmarks/generate_gauss_kronrod_fixtures.py (same id, math, bounds).
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
};

// Allowed slack for evaluations vs SciPy's QUADPACK reference. Both are globally adaptive,
// but ours always bisects the worst panel at its midpoint rather than using QUADPACK's
// singularity-aware subdivision, so endpoint-singular integrands need more panels here
// (observed up to ~11x for sqrt_singularity/log_singularity).
const EVALUATION_FACTOR = 15;

describe('gaussKronrod vs SciPy quad reference values', () => {
    for (const fixture of fixtures) {
        it(`matches scipy.integrate.quad for ${fixture.id}: ${fixture.description}`, () => {
            const f = INTEGRANDS[fixture.id];
            expect(f, `no matching integrand registered for id "${fixture.id}"`).toBeDefined();

            // Same tol as TOL in generate_gauss_kronrod_fixtures.py, so evaluation counts are comparable.
            const result = gaussKronrod(f, fixture.a, fixture.b, 1e-12);
            expect(result.converged).toBe(true);

            // Tolerance accounts for both SciPy's and our own reported error estimates.
            const tolerance = Math.max(fixture.scipyAbsError, result.error) * 10 + 1e-10;
            expect(Math.abs(result.value - fixture.scipyValue)).toBeLessThan(tolerance);

            // Both are globally adaptive Gauss-Kronrod solvers (ours G7-K15, QUADPACK's G10-K21),
            // so panel counts should be within the same order of magnitude, not just "converged".
            expect(result.evaluations).toBeLessThan(fixture.scipyEvaluations * EVALUATION_FACTOR);
        });
    }
});
