import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { levenbergMarquardt } from '../../src/optimize/levenbergMarquardt.js';
import {
    LEAST_SQUARES_PROBLEMS,
    beale,
    brownBadlyScaled,
    jennrichSampson,
    powellSingular,
    rankDeficient,
    rosenbrock,
} from './leastSquaresFunctions.js';

/** Relative agreement of the sum of squares with the known minimum. */
function expectMinimum(sumSquares: number, expected: number): void {
    expect(Math.abs(sumSquares - expected)).toBeLessThanOrEqual(1e-6 * Math.max(expected, 1e-3) + 1e-12);
}

describe.each(Object.entries(LEAST_SQUARES_PROBLEMS))('levenbergMarquardt — %s', (_name, problem) => {
    it('converges from the standard starting point with finite-difference Jacobian', () => {
        const res = levenbergMarquardt(problem.f, problem.x0);

        expect(res.success, res.message).toBe(true);
        expectMinimum(2 * res.cost, problem.minSumSquares);
        if (problem.xs) {
            expect(res.x.allClose(problem.xs, 1e-5, 1e-5), res.x.toString()).toBe(true);
        }
        expect(res.evaluationsJacobian).toBe(0);
    });

    it.runIf(problem.jac !== undefined)('converges with the analytic Jacobian, using fewer evaluations of f', () => {
        const fd = levenbergMarquardt(problem.f, problem.x0);
        const res = levenbergMarquardt(problem.f, problem.x0, { jac: problem.jac });

        expect(res.success, res.message).toBe(true);
        expectMinimum(2 * res.cost, problem.minSumSquares);
        if (problem.xs) {
            expect(res.x.allClose(problem.xs, 1e-5, 1e-5), res.x.toString()).toBe(true);
        }
        expect(res.evaluationsJacobian).toBeGreaterThan(0);
        expect(res.evaluationsFunction).toBeLessThan(fd.evaluationsFunction);
    });
});

describe('levenbergMarquardt — result consistency', () => {
    it('returns the residual, cost and Jacobian that belong to the solution', () => {
        const res = levenbergMarquardt(beale.f, beale.x0);

        expect(res.fx.allClose(beale.f(res.x), 0, 0)).toBe(true);
        expect(res.cost).toBeCloseTo(0.5 * res.fx.normSq(), 14);
        expect(res.Jx).not.toBeNull();
        expect(res.Jx!.rows).toBe(3);
        expect(res.Jx!.cols).toBe(2);
        expect(res.method).toContain('Levenberg-Marquardt');
        expect(res.iterations).toBeGreaterThan(0);
    });

    it('counts every evaluation of f, including those of the finite-difference Jacobian', () => {
        let calls = 0;
        const res = levenbergMarquardt((x) => { calls++; return rosenbrock.f(x); }, rosenbrock.x0);
        expect(res.evaluationsFunction).toBe(calls);

        calls = 0;
        const resJac = levenbergMarquardt((x) => { calls++; return rosenbrock.f(x); }, rosenbrock.x0, { jac: rosenbrock.jac });
        // jacCheck (default) adds n = 2 evaluations at x0.
        expect(resJac.evaluationsFunction).toBe(calls);
        expect(resJac.evaluationsJacobian).toBeGreaterThan(0);
    });

    it('does not modify x0, and accepts a plain array', () => {
        const x0 = Vector.from([-1.2, 1]);
        const before = x0.copy();
        const fromVector = levenbergMarquardt(rosenbrock.f, x0);
        const fromArray = levenbergMarquardt(rosenbrock.f, [-1.2, 1]);

        expect(x0.allClose(before, 0, 0)).toBe(true);
        expect(fromArray.x.allClose(fromVector.x, 0, 0)).toBe(true);
        expect(fromArray.evaluationsFunction).toBe(fromVector.evaluationsFunction);
    });

    it('stops immediately at an exact solution', () => {
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.xs!);
        expect(res.success).toBe(true);
        expect(res.iterations).toBe(0);
        expect(res.cost).toBe(0);
    });
});

describe('levenbergMarquardt — rank deficiency and machine precision', () => {
    it('handles a rank-deficient Jacobian and returns a least-squares point', () => {
        const res = levenbergMarquardt(rankDeficient.f, rankDeficient.x0);
        expect(res.success, res.message).toBe(true);
        expect(res.x.get(0) + res.x.get(1)).toBeCloseTo(2, 8);
    });

    it('reports success when it stops at machine precision with a negligible residual', () => {
        // The Jacobian is singular at the solution; MINPACK stops with `info = 8` at a cost of ~1e-67.
        const res = levenbergMarquardt(powellSingular.f, powellSingular.x0, { jac: powellSingular.jac });
        expect(res.success, res.message).toBe(true);
        expect(res.cost).toBeLessThan(1e-20);
    });

    it('reports failure when it stops at machine precision with a non-negligible residual', () => {
        const res = levenbergMarquardt(jennrichSampson.f, jennrichSampson.x0, { tolf: 0, tolx: 0, tolg: 0, maxFunEvals: 10000 });
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/machine precision with a non-negligible residual/);
        expectMinimum(2 * res.cost, jennrichSampson.minSumSquares);
    });
});

describe('levenbergMarquardt — options', () => {
    it('stops with failure when maxFunEvals is exhausted', () => {
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { maxFunEvals: 6 });
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/Maximum number of function evaluations \(6\)/);
        expect(res.evaluationsFunction).toBeLessThanOrEqual(6 + 2); // may exceed by one Jacobian
    });

    it('stops with failure when maxIter is exhausted', () => {
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { maxIter: 2 });
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/Maximum number of iterations \(2\)/);
        expect(res.iterations).toBe(2);
    });

    it('converges with fixed scaling on a badly scaled problem, and reports it in the method name', () => {
        const res = levenbergMarquardt(brownBadlyScaled.f, brownBadlyScaled.x0, {
            sclx: [1e-6, 1e6],
            jac: brownBadlyScaled.jac,
        });
        expect(res.success, res.message).toBe(true);
        expect(res.method).toContain('Fixed');
        expect(res.x.allClose(brownBadlyScaled.xs!, 1e-6, 1e-12)).toBe(true);
        expect(levenbergMarquardt(brownBadlyScaled.f, brownBadlyScaled.x0).method).toContain('Adaptive');
    });

    it('accepts sclx given as a Vector', () => {
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { sclx: Vector.ones(2) });
        expect(res.success, res.message).toBe(true);
    });

    it('a smaller initial trust region still converges', () => {
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { trustLenFactor: 0.1 });
        expect(res.success, res.message).toBe(true);
        expect(res.x.allClose(rosenbrock.xs!, 1e-6, 1e-6)).toBe(true);
    });

    it('a looser tolf/tolx terminates earlier', () => {
        const strict = levenbergMarquardt(jennrichSampson.f, jennrichSampson.x0);
        const loose = levenbergMarquardt(jennrichSampson.f, jennrichSampson.x0, { tolf: 1e-2, tolx: 1e-2 });
        expect(loose.success).toBe(true);
        expect(loose.evaluationsFunction).toBeLessThan(strict.evaluationsFunction);
    });

    it('tolg terminates at a point where the residual is orthogonal to the Jacobian', () => {
        const res = levenbergMarquardt(jennrichSampson.f, jennrichSampson.x0, { tolf: 0, tolx: 0, tolg: 1e-6 });
        expect(res.success, res.message).toBe(true);
        expect(res.message).toMatch(/cosine/);
    });
});

describe('levenbergMarquardt — user-supplied Jacobian', () => {
    it('fails if the Jacobian is inconsistent with f, unless jacCheck is disabled', () => {
        const wrongJac = (x: Vector) => rosenbrock.jac!(x).mult(1.5);

        const checked = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { jac: wrongJac });
        expect(checked.success).toBe(false);
        expect(checked.message).toMatch(/does not match finite-difference/);
        expect(checked.iterations).toBe(0);
        expect(checked.Jx).not.toBeNull();

        const unchecked = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { jac: wrongJac, jacCheck: false });
        expect(unchecked.message).not.toMatch(/does not match/);
    });

    it('loosens the jacCheck tolerance when the function values are less precise', () => {
        const slightlyOff = (x: Vector) => rosenbrock.jac!(x).mult(1.0001);
        expect(levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { jac: slightlyOff }).message).toMatch(/does not match/);
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, { jac: slightlyOff, epsf: 1e-6 });
        expect(res.message).not.toMatch(/does not match/);
    });

    it('throws if jac returns a matrix of the wrong shape', () => {
        expect(() => levenbergMarquardt(beale.f, beale.x0, { jac: () => new Matrix(2, 2) })).toThrow(RangeError);
    });

    it('returns a failure result if the Jacobian at x0 is not finite', () => {
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, {
            jac: () => Matrix.from([[NaN, 0], [0, 0]]),
            jacCheck: false,
        });
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/Jacobian at x0/);
    });

    it('returns a failure result if a later Jacobian is not finite', () => {
        let calls = 0;
        const res = levenbergMarquardt(rosenbrock.f, rosenbrock.x0, {
            jac: (x) => (++calls > 2 ? Matrix.from([[NaN, 0], [0, 0]]) : rosenbrock.jac!(x)),
            jacCheck: false,
        });
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/Jacobian at the current iterate/);
        expect(res.iterations).toBeGreaterThan(0);
    });
});

describe('levenbergMarquardt — non-finite function values', () => {
    it('returns a failure result if f(x0) is not finite', () => {
        const res = levenbergMarquardt(() => Vector.from([NaN, 1]), [1, 1]);
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/f\(x0\)/);
        expect(res.evaluationsFunction).toBe(1);
    });

    it('rejects trial points where f is not finite and shrinks the trust region', () => {
        // Residuals r = (sqrt(x) - 2, x - 4): undefined (NaN) for x < 0, minimum at x = 4.
        const f = (x: Vector) => Vector.from([Math.sqrt(x.get(0)) - 2, 0.1 * (x.get(0) - 4)]);
        const res = levenbergMarquardt(f, [0.01]);
        expect(res.success, res.message).toBe(true);
        expect(res.x.get(0)).toBeCloseTo(4, 6);
    });
});

describe('levenbergMarquardt — input validation', () => {
    const f = rosenbrock.f;

    it('throws for an empty x0', () => {
        expect(() => levenbergMarquardt(() => Vector.from([1]), [])).toThrow(RangeError);
    });

    it('throws if there are fewer residuals than variables', () => {
        expect(() => levenbergMarquardt(() => Vector.from([1]), [1, 2])).toThrow(RangeError);
    });

    it('throws if f changes the number of residuals', () => {
        let n = 2;
        expect(() => levenbergMarquardt(() => Vector.zero(n++).add(1), [1, 1])).toThrow(RangeError);
    });

    it.each([
        ['maxFunEvals', { maxFunEvals: 0 }],
        ['maxFunEvals', { maxFunEvals: 1.5 }],
        ['maxFunEvals', { maxFunEvals: Infinity }],
        ['maxIter', { maxIter: 0 }],
        ['maxIter', { maxIter: 2.5 }],
        ['tolf', { tolf: -1 }],
        ['tolx', { tolx: NaN }],
        ['tolg', { tolg: -1e-3 }],
        ['trustLenFactor', { trustLenFactor: 0 }],
        ['trustLenFactor', { trustLenFactor: Infinity }],
    ])('throws a RangeError for invalid %s', (_name, options) => {
        expect(() => levenbergMarquardt(f, rosenbrock.x0, options)).toThrow(RangeError);
    });

    it('throws for an sclx of the wrong size or with non-positive entries', () => {
        expect(() => levenbergMarquardt(f, rosenbrock.x0, { sclx: [1, 1, 1] })).toThrow(RangeError);
        expect(() => levenbergMarquardt(f, rosenbrock.x0, { sclx: [1, 0] })).toThrow(Error);
        expect(() => levenbergMarquardt(f, rosenbrock.x0, { sclx: [1, -1] })).toThrow(Error);
    });
});
