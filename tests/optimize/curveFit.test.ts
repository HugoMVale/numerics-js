import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { curveFit } from '../../src/optimize/curveFit.js';

// Deterministic "noise" without a random generator.
const wobble = (i: number) => 0.05 * Math.sin(12.9898 * (i + 1)) + 0.03 * Math.cos(4.1414 * (i + 1));

/** Straight line `y = p0 + p1*x`, whose least-squares problem has a closed-form solution. */
const line = (x: number, p: Vector) => p.get(0) + p.get(1) * x;
const lineJac = (x: number) => [1, x];

/** Closed-form weighted linear least squares: `p`, `inv(X^T W X)` and the weighted rss. */
function linearReference(x: Vector, y: Vector, sigma: Vector) {
    const invSigma = Vector.ones(x.size).div(sigma);
    const X = new Matrix(x.size, 2).setCol(0, invSigma).setCol(1, x.div(sigma));
    const yw = y.div(sigma);
    const A = X.transpose().matmul(X);
    const p = A.solve(X.transpose().mulVec(yw));
    const Ainv = A.inverse();
    const rss = X.mulVec(p).sub(yw).normSq();
    return { p, Ainv, rss };
}

describe('curveFit — linear model against the closed-form solution', () => {
    const x = Vector.linspace(0, 5, 12);
    const y = x.map((xi, i) => 1 + 2 * xi + wobble(i));
    const m = x.size;

    it('recovers parameters, covariance, residuals and rss (unweighted)', () => {
        const ref = linearReference(x, y, Vector.ones(m));
        const res = curveFit(line, x, y, [0, 0]);

        expect(res.success, res.message).toBe(true);
        expect(res.p.allClose(ref.p, 1e-7, 1e-9)).toBe(true);
        const s2 = ref.rss / (m - 2);
        expect(res.cov.allClose(ref.Ainv.mult(s2), 1e-5, 1e-12)).toBe(true);
        expect(res.stderr.allClose(res.cov.diag().sqrt(), 0, 0)).toBe(true);
        expect(res.rss).toBeCloseTo(ref.rss, 10);
        expect(res.dof).toBe(m - 2);
        // residuals = y - f(x, p)
        const expected = y.sub(x.map((xi) => line(xi, res.p)));
        expect(res.residuals.allClose(expected, 1e-9, 1e-12)).toBe(true);
        expect(res.residuals.normSq()).toBeCloseTo(res.rss, 10);
        expect(res.cov.isSymmetric(1e-12 * res.cov.normInf())).toBe(true);
    });

    it('weights the residuals by sigma, and scales the covariance by the residual variance', () => {
        const sigma = x.map((xi) => 0.5 + 0.4 * xi);
        const ref = linearReference(x, y, sigma);
        const res = curveFit(line, x, y, [0, 0], { sigma });

        expect(res.success, res.message).toBe(true);
        expect(res.p.allClose(ref.p, 1e-7, 1e-9)).toBe(true);
        expect(res.rss).toBeCloseTo(ref.rss, 10);
        expect(res.cov.allClose(ref.Ainv.mult(ref.rss / (m - 2)), 1e-5, 1e-12)).toBe(true);
        // residuals stay unweighted
        expect(res.residuals.div(sigma).normSq()).toBeCloseTo(res.rss, 10);
    });

    it('with relative sigma, scaling sigma does not change cov; with absoluteSigma, cov scales with sigma²', () => {
        const sigma = x.map((xi) => 0.5 + 0.4 * xi);
        const scaled = sigma.mult(10);

        const rel1 = curveFit(line, x, y, [0, 0], { sigma });
        const rel2 = curveFit(line, x, y, [0, 0], { sigma: scaled });
        expect(rel2.cov.allClose(rel1.cov, 1e-6, 1e-14)).toBe(true);

        const abs1 = curveFit(line, x, y, [0, 0], { sigma, absoluteSigma: true });
        const abs2 = curveFit(line, x, y, [0, 0], { sigma: scaled, absoluteSigma: true });
        expect(abs2.cov.allClose(abs1.cov.mult(100), 1e-6, 1e-14)).toBe(true);
        expect(abs1.cov.allClose(linearReference(x, y, sigma).Ainv, 1e-5, 1e-12)).toBe(true);
    });

    it('accepts plain arrays as well as Vectors, without modifying its inputs', () => {
        const xv = x.copy();
        const yv = y.copy();
        const p0 = Vector.from([0, 0]);
        const sigma = Vector.full(m, 0.1);
        const a = curveFit(line, xv, yv, p0, { sigma });
        const xArray = x.toArray();
        const yArray = y.toArray();
        const sigmaArray = Array(m).fill(0.1);
        const b = curveFit(line, xArray, yArray, [0, 0], { sigma: sigmaArray });

        expect(a.p.allClose(b.p, 0, 0)).toBe(true);
        expect(xv.toArray()).toEqual(x.toArray());
        expect(yv.toArray()).toEqual(y.toArray());
        expect(p0.toArray()).toEqual([0, 0]);
        expect(sigma.toArray()).toEqual(sigmaArray);
        expect(xArray).toEqual(x.toArray());
        expect(yArray).toEqual(y.toArray());
        expect(sigmaArray).toEqual(Array(m).fill(0.1));
    });
});

describe('curveFit — nonlinear models', () => {
    it('fits an exponential decay to exact data, with a tiny residual', () => {
        const t = Vector.linspace(0, 4, 15);
        const y = t.map((ti) => 2.5 * Math.exp(-1.3 * ti));
        const res = curveFit((ti, p) => p.get(0) * Math.exp(-p.get(1) * ti), t, y, [1, 1]);

        expect(res.success, res.message).toBe(true);
        expect(res.p.allClose(Vector.from([2.5, 1.3]), 1e-6, 1e-8)).toBe(true);
        expect(res.rss).toBeLessThan(1e-15);
        expect(res.stderr.max()).toBeLessThan(1e-6);
    });

    const gauss = (x: number, p: Vector) => p.get(0) * Math.exp(-(((x - p.get(1)) / p.get(2)) ** 2) / 2);
    const gaussJac = (x: number, p: Vector) => {
        const [a, mu, s] = [p.get(0), p.get(1), p.get(2)];
        const e = Math.exp(-(((x - mu) / s) ** 2) / 2);
        return [e, (a * e * (x - mu)) / s ** 2, (a * e * (x - mu) ** 2) / s ** 3];
    };

    it('fits a noisy Gaussian peak; analytic and finite-difference Jacobians agree and use fewer evaluations', () => {
        const x = Vector.linspace(-3, 5, 40);
        const y = x.map((xi, i) => 4 * Math.exp(-(((xi - 1) / 0.8) ** 2) / 2) + wobble(i));
        const fd = curveFit(gauss, x, y, [3, 0.5, 1]);
        const an = curveFit(gauss, x, y, [3, 0.5, 1], { jac: gaussJac });

        expect(fd.success, fd.message).toBe(true);
        expect(an.success, an.message).toBe(true);
        expect(an.p.allClose(fd.p, 1e-6, 1e-8)).toBe(true);
        // Entries that vanish by symmetry (~1e-12 vs ~1e-16) need a tolerance relative to the largest entry.
        expect(an.cov.allClose(fd.cov, 1e-5, 1e-6 * an.cov.normInf())).toBe(true);
        expect(Math.abs(an.p.get(0) - 4)).toBeLessThan(0.1);
        expect(Math.abs(an.p.get(1) - 1)).toBeLessThan(0.05);
        expect(an.evaluationsFunction).toBeLessThan(fd.evaluationsFunction);
        expect(an.evaluationsJacobian).toBeGreaterThan(0);
        expect(fd.evaluationsJacobian).toBe(0);
    });

    it('fits a logistic curve with fixed parameter scaling', () => {
        const x = Vector.linspace(0, 20, 30);
        const y = x.map((xi, i) => 1000 / (1 + Math.exp(-0.6 * (xi - 9))) + 5 * wobble(i) * 10);
        const res = curveFit(
            (xi, p) => p.get(0) / (1 + Math.exp(-p.get(1) * (xi - p.get(2)))),
            x,
            y,
            [800, 0.3, 7],
            { sclx: [1e-3, 1, 0.1] }
        );
        expect(res.success, res.message).toBe(true);
        expect(res.method).toContain('Fixed');
        expect(res.p.allClose(Vector.from([1000, 0.6, 9]), 0.05, 0.1)).toBe(true);
    });

    it('counts evaluations of the whole residual vector, including the final Jacobian', () => {
        const x = Vector.linspace(0, 4, 15);
        const y = x.map((xi, i) => 2 * Math.exp(-xi) + wobble(i));
        let calls = 0;
        const res = curveFit((xi, p) => { calls++; return p.get(0) * Math.exp(-p.get(1) * xi); }, x, y, [1, 0.5]);
        expect(res.evaluationsFunction * x.size).toBe(calls);

        calls = 0;
        const model = (xi: number, p: Vector) => { calls++; return p.get(0) * Math.exp(-p.get(1) * xi); };
        const resJac = curveFit(model, x, y, [1, 0.5], {
            jac: (xi, p) => [Math.exp(-p.get(1) * xi), -p.get(0) * xi * Math.exp(-p.get(1) * xi)],
            jacCheck: false,
        });
        expect(resJac.evaluationsFunction * x.size).toBe(calls);
        expect(resJac.evaluationsJacobian).toBeGreaterThan(1);
    });
});

describe('curveFit — covariance edge cases', () => {
    const x = Vector.linspace(0, 4, 8);
    const y = x.map((xi, i) => 1 + 2 * xi + wobble(i));

    it('reports an infinite covariance for redundant parameters', () => {
        // p0 + p1 enter only as their sum.
        const res = curveFit((xi, p) => p.get(0) + p.get(1) + p.get(2) * xi, x, y, [0, 0, 0]);
        expect(res.p.allFinite()).toBe(true);
        expect(res.p.get(0) + res.p.get(1)).toBeCloseTo(
            linearReference(x, y, Vector.ones(8)).p.get(0),
            6
        );
        expect(res.cov.data.every((v) => v === Infinity)).toBe(true);
        expect(res.stderr.data.every((v) => v === Infinity)).toBe(true);
    });

    it('reports an infinite covariance without degrees of freedom, unless sigma is absolute', () => {
        const x2 = Vector.from([0, 1]);
        const y2 = Vector.from([1, 3]);
        const rel = curveFit(line, x2, y2, [0, 0]);
        expect(rel.dof).toBe(0);
        expect(rel.p.allClose(Vector.from([1, 2]), 1e-6, 1e-8)).toBe(true);
        expect(rel.cov.data.every((v) => v === Infinity)).toBe(true);

        const abs = curveFit(line, x2, y2, [0, 0], { sigma: [0.1, 0.1], absoluteSigma: true });
        expect(abs.cov.allFinite()).toBe(true);
        // inv(X^T X) * sigma², with X = [[1, 0], [1, 1]], i.e. X^T X = [[2, 1], [1, 1]]
        expect(abs.cov.allClose(Matrix.from([[1, -1], [-1, 2]]).mult(0.01), 1e-5, 1e-12)).toBe(true);
    });

    it('returns a failure result with NaN covariance if the data are not finite', () => {
        const res = curveFit(line, Vector.from([0, 1, 2]), Vector.from([1, NaN, 5]), [0, 0]);
        expect(res.success).toBe(false);
        expect(res.message).toMatch(/non-finite/);
        expect(res.cov.data.every(Number.isNaN)).toBe(true);
    });
});

describe('curveFit — jac option', () => {
    const x = Vector.linspace(0, 4, 10);
    const y = x.map((xi, i) => 1 + 2 * xi + wobble(i));

    it('accepts a gradient returned as a Vector or as an array', () => {
        const a = curveFit(line, x, y, [0, 0], { jac: (xi) => Vector.from(lineJac(xi)) });
        const b = curveFit(line, x, y, [0, 0], { jac: lineJac });
        expect(a.p.allClose(b.p, 0, 0)).toBe(true);
        expect(a.cov.allClose(b.cov, 0, 0)).toBe(true);
    });

    it('fails if the gradient is inconsistent with the model, unless jacCheck is disabled', () => {
        const wrong = (xi: number) => [1, 1.5 * xi];
        const checked = curveFit(line, x, y, [0, 0], { jac: wrong });
        expect(checked.success).toBe(false);
        expect(checked.message).toMatch(/does not match finite-difference/);

        const unchecked = curveFit(line, x, y, [0, 0], { jac: wrong, jacCheck: false });
        expect(unchecked.message).not.toMatch(/does not match/);
    });

    it('throws if the gradient has the wrong number of components', () => {
        expect(() => curveFit(line, x, y, [0, 0], { jac: () => [1] })).toThrow(RangeError);
    });
});

describe('curveFit — limits and validation', () => {
    const x = Vector.linspace(0, 4, 10);
    const y = x.map((xi) => 1 + 2 * xi);

    it('passes the solver limits through, and reports non-convergence', () => {
        const res = curveFit((xi, p) => p.get(0) * Math.exp(-p.get(1) * xi), x, x.map((xi) => 3 * Math.exp(-2 * xi)), [1, 0.1], { maxIter: 2 });
        expect(res.success).toBe(false);
        expect(res.iterations).toBe(2);
        expect(res.message).toMatch(/Maximum number of iterations/);
        expect(res.cov.rows).toBe(2);
    });

    it('throws for inconsistent or insufficient data', () => {
        expect(() => curveFit(line, Vector.from([0, 1, 2]), Vector.from([1, 2]), [0, 0])).toThrow(RangeError);
        expect(() => curveFit(line, Vector.from([0]), Vector.from([1]), [0, 0])).toThrow(RangeError);
        expect(() => curveFit(line, x, y, [])).toThrow(RangeError);
    });

    it('throws for an invalid sigma', () => {
        expect(() => curveFit(line, x, y, [0, 0], { sigma: Vector.from([1, 2]) })).toThrow(RangeError);
        expect(() => curveFit(line, x, y, [0, 0], { sigma: Vector.zero(10) })).toThrow(Error);
        expect(() => curveFit(line, x, y, [0, 0], { sigma: Vector.full(10, -1) })).toThrow(Error);
        expect(() => curveFit(line, x, y, [0, 0], { sigma: Vector.full(10, NaN) })).toThrow(Error);
    });

    it('throws for invalid solver options', () => {
        expect(() => curveFit(line, x, y, [0, 0], { maxFunEvals: 0 })).toThrow(RangeError);
        expect(() => curveFit(line, x, y, [0, 0], { sclx: [1, -1] })).toThrow(Error);
    });
});
