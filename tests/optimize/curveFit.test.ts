import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { curveFit } from '../../src/optimize/curveFit.js';
import { vectorLikeForms, mutableVectorLikeForms } from '../vectorLikeForms.js';

// Deterministic "noise" without a random generator.
const wobble = (i: number) =>
  0.05 * Math.sin(12.9898 * (i + 1)) + 0.03 * Math.cos(4.1414 * (i + 1));

/** Straight line `y = p0 + p1*x`, whose least-squares problem has a closed-form solution. */
const line = (x: Vector, p: Vector) => x.map((xi) => p.get(0) + p.get(1) * xi);
const lineJac = (x: Vector) => new Matrix(x.size, 2).setCol(0, Vector.ones(x.size)).setCol(1, x);

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
    const res = curveFit(line, x, y, Vector.from([0, 0]));

    expect(res.success, res.message).toBe(true);
    expect(res.p.allClose(ref.p, 1e-7, 1e-9)).toBe(true);
    const s2 = ref.rss / (m - 2);
    expect(res.cov.allClose(ref.Ainv.mult(s2), 1e-5, 1e-12)).toBe(true);
    expect(res.stderr.allClose(res.cov.diag().sqrt(), 0, 0)).toBe(true);
    expect(res.rss).toBeCloseTo(ref.rss, 10);
    expect(res.dof).toBe(m - 2);
    // residuals = y - f(x, p)
    const expected = y.sub(line(x, res.p));
    expect(res.residuals.allClose(expected, 1e-9, 1e-12)).toBe(true);
    expect(res.residuals.normSq()).toBeCloseTo(res.rss, 10);
    expect(res.cov.isSymmetric(1e-12 * res.cov.normInf())).toBe(true);
  });

  it('weights the residuals by sigma, and scales the covariance by the residual variance', () => {
    const sigma = x.map((xi) => 0.5 + 0.4 * xi);
    const ref = linearReference(x, y, sigma);
    const res = curveFit(line, x, y, Vector.from([0, 0]), { sigma });

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

    const rel1 = curveFit(line, x, y, Vector.from([0, 0]), { sigma });
    const rel2 = curveFit(line, x, y, Vector.from([0, 0]), { sigma: scaled });
    expect(rel2.cov.allClose(rel1.cov, 1e-6, 1e-14)).toBe(true);

    const abs1 = curveFit(line, x, y, Vector.from([0, 0]), { sigma, absoluteSigma: true });
    const abs2 = curveFit(line, x, y, Vector.from([0, 0]), {
      sigma: scaled,
      absoluteSigma: true
    });
    expect(abs2.cov.allClose(abs1.cov.mult(100), 1e-6, 1e-14)).toBe(true);
    expect(abs1.cov.allClose(linearReference(x, y, sigma).Ainv, 1e-5, 1e-12)).toBe(true);
  });

  it('does not modify its input Vectors', () => {
    const xv = x.copy();
    const yv = y.copy();
    const p0 = Vector.from([0, 0]);
    const sigma = Vector.full(m, 0.1);
    const a = curveFit(line, xv, yv, p0, { sigma });
    expect(a.success, a.message).toBe(true);
    expect(xv.toArray()).toEqual(x.toArray());
    expect(yv.toArray()).toEqual(y.toArray());
    expect(p0.toArray()).toEqual([0, 0]);
    expect(sigma.toArray()).toEqual(Array(m).fill(0.1));
  });
});

describe('curveFit — nonlinear models', () => {
  it('fits an exponential decay to exact data, with a tiny residual', () => {
    const t = Vector.linspace(0, 4, 15);
    const y = t.map((ti) => 2.5 * Math.exp(-1.3 * ti));
    const res = curveFit(
      (t, p) => t.map((ti) => p.get(0) * Math.exp(-p.get(1) * ti)),
      t,
      y,
      Vector.from([1, 1])
    );

    expect(res.success, res.message).toBe(true);
    expect(res.p.allClose(Vector.from([2.5, 1.3]), 1e-6, 1e-8)).toBe(true);
    expect(res.rss).toBeLessThan(1e-15);
    expect(res.stderr.max()).toBeLessThan(1e-6);
  });

  const gauss = (x: Vector, p: Vector) =>
    x.map((xi) => p.get(0) * Math.exp(-(((xi - p.get(1)) / p.get(2)) ** 2) / 2));
  const gaussJac = (x: Vector, p: Vector) => {
    const [a, mu, s] = [p.get(0), p.get(1), p.get(2)];
    const J = new Matrix(x.size, 3);
    for (let i = 0; i < x.size; i++) {
      const xi = x.get(i);
      const e = Math.exp(-(((xi - mu) / s) ** 2) / 2);
      J.set(i, 0, e);
      J.set(i, 1, (a * e * (xi - mu)) / s ** 2);
      J.set(i, 2, (a * e * (xi - mu) ** 2) / s ** 3);
    }
    return J;
  };

  it('fits a noisy Gaussian peak; analytic and finite-difference Jacobians agree and use fewer evaluations', () => {
    const x = Vector.linspace(-3, 5, 40);
    const y = x.map((xi, i) => 4 * Math.exp(-(((xi - 1) / 0.8) ** 2) / 2) + wobble(i));
    const fd = curveFit(gauss, x, y, Vector.from([3, 0.5, 1]));
    const an = curveFit(gauss, x, y, Vector.from([3, 0.5, 1]), { jac: gaussJac });

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

  it('by default, computes the covariance from the last Jacobian of the iterations; `jacFinal: true` evaluates it at the solution', () => {
    const x = Vector.linspace(-3, 5, 40);
    const y = x.map((xi, i) => 4 * Math.exp(-(((xi - 1) / 0.8) ** 2) / 2) + wobble(i));
    const p0 = Vector.from([3, 0.5, 1]);

    // With the analytic Jacobian, the exact covariance is available as a reference.
    const stale = curveFit(gauss, x, y, p0, { jac: gaussJac });
    const exact = curveFit(gauss, x, y, p0, { jac: gaussJac, jacFinal: true });
    const reference = curveFit(gauss, x, y, exact.p, {
      jac: gaussJac,
      maxIter: 1,
      jacCheck: false
    });
    expect(exact.success, exact.message).toBe(true);
    expect(stale.success, stale.message).toBe(true);
    expect(stale.p.allClose(exact.p, 0, 0)).toBe(true);
    expect(exact.cov.allClose(reference.cov, 1e-8, 1e-8 * exact.cov.normInf())).toBe(true);
    // The last Jacobian is one step behind: slightly different covariance, one evaluation less.
    expect(stale.cov.allClose(exact.cov, 1e-3, 1e-3 * exact.cov.normInf())).toBe(true);
    expect(stale.evaluationsJacobian).toBe(exact.evaluationsJacobian - 1);

    // Same with finite differences, where the saving is `n` evaluations of the residuals.
    const fdStale = curveFit(gauss, x, y, p0);
    const fdExact = curveFit(gauss, x, y, p0, { jacFinal: true });
    expect(fdStale.evaluationsFunction).toBe(fdExact.evaluationsFunction - p0.size);
    expect(fdStale.cov.allClose(fdExact.cov, 1e-3, 1e-3 * fdExact.cov.normInf())).toBe(true);
  });

  it('fits a logistic curve with fixed parameter scaling', () => {
    const x = Vector.linspace(0, 20, 30);
    const y = x.map((xi, i) => 1000 / (1 + Math.exp(-0.6 * (xi - 9))) + 5 * wobble(i) * 10);
    const res = curveFit(
      (xv, p) => xv.map((xi) => p.get(0) / (1 + Math.exp(-p.get(1) * (xi - p.get(2))))),
      x,
      y,
      Vector.from([800, 0.3, 7]),
      { sclx: Vector.from([1e-3, 1, 0.1]) }
    );
    expect(res.success, res.message).toBe(true);
    expect(res.method).toContain('Fixed');
    expect(res.p.allClose(Vector.from([1000, 0.6, 9]), 0.05, 0.1)).toBe(true);
  });

  it('counts evaluations of the whole residual vector, including those of the finite-difference Jacobian', () => {
    const x = Vector.linspace(0, 4, 15);
    const y = x.map((xi, i) => 2 * Math.exp(-xi) + wobble(i));
    let calls = 0;
    const model = (xv: Vector, p: Vector) => {
      calls++;
      // The model is called with all the data points at once, and with one parameter per `p0`.
      expect(xv.size).toBe(x.size);
      expect(p.size).toBe(2);
      return xv.map((xi) => p.get(0) * Math.exp(-p.get(1) * xi));
    };
    const res = curveFit(model, x, y, Vector.from([1, 0.5]));
    expect(res.evaluationsFunction).toBe(calls);

    calls = 0;
    const resJac = curveFit(model, x, y, Vector.from([1, 0.5]), {
      jac: (xv, p) => {
        const J = new Matrix(xv.size, 2);
        for (let i = 0; i < xv.size; i++) {
          const xi = xv.get(i);
          J.set(i, 0, Math.exp(-p.get(1) * xi));
          J.set(i, 1, -p.get(0) * xi * Math.exp(-p.get(1) * xi));
        }
        return J;
      },
      jacCheck: false
    });
    expect(resJac.evaluationsFunction).toBe(calls);
    expect(resJac.evaluationsJacobian).toBeGreaterThan(1);
  });
});

describe('curveFit — covariance edge cases', () => {
  const x = Vector.linspace(0, 4, 8);
  const y = x.map((xi, i) => 1 + 2 * xi + wobble(i));

  it('reports an infinite covariance for redundant parameters', () => {
    // p0 + p1 enter only as their sum.
    const res = curveFit(
      (xv, p) => xv.map((xi) => p.get(0) + p.get(1) + p.get(2) * xi),
      x,
      y,
      Vector.from([0, 0, 0])
    );
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
    const rel = curveFit(line, x2, y2, Vector.from([0, 0]));
    expect(rel.dof).toBe(0);
    expect(rel.p.allClose(Vector.from([1, 2]), 1e-6, 1e-8)).toBe(true);
    expect(rel.cov.data.every((v) => v === Infinity)).toBe(true);

    const abs = curveFit(line, x2, y2, Vector.from([0, 0]), {
      sigma: Vector.from([0.1, 0.1]),
      absoluteSigma: true
    });
    expect(abs.cov.allFinite()).toBe(true);
    // inv(X^T X) * sigma², with X = [[1, 0], [1, 1]], i.e. X^T X = [[2, 1], [1, 1]]
    expect(
      abs.cov.allClose(
        Matrix.from([
          [1, -1],
          [-1, 2]
        ]).mult(0.01),
        1e-5,
        1e-12
      )
    ).toBe(true);
  });

  it('returns a failure result with NaN covariance if the data are not finite', () => {
    const res = curveFit(
      line,
      Vector.from([0, 1, 2]),
      Vector.from([1, NaN, 5]),
      Vector.from([0, 0])
    );
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/non-finite/);
    expect(res.cov.data.every(Number.isNaN)).toBe(true);
  });
});

describe('curveFit — jac option', () => {
  const x = Vector.linspace(0, 4, 10);
  const y = x.map((xi, i) => 1 + 2 * xi + wobble(i));

  it('uses a Matrix-valued Jacobian of the model, ∂f/∂p', () => {
    const fd = curveFit(line, x, y, Vector.from([0, 0]));
    const analytic = curveFit(line, x, y, Vector.from([0, 0]), { jac: lineJac });
    expect(analytic.success, analytic.message).toBe(true);
    expect(analytic.p.allClose(fd.p, 1e-7, 1e-9)).toBe(true);
    expect(analytic.cov.allClose(fd.cov, 1e-7, 1e-9)).toBe(true);
  });

  it('applies sigma to the Jacobian rows', () => {
    const sigma = x.map((xi) => 0.5 + 0.4 * xi);
    const fd = curveFit(line, x, y, Vector.from([0, 0]), { sigma });
    const analytic = curveFit(line, x, y, Vector.from([0, 0]), { sigma, jac: lineJac });
    expect(analytic.success, analytic.message).toBe(true);
    expect(analytic.p.allClose(fd.p, 1e-7, 1e-9)).toBe(true);
    expect(analytic.cov.allClose(fd.cov, 1e-6, 1e-9)).toBe(true);
  });

  it('fails if the Jacobian is inconsistent with the model, unless jacCheck is disabled', () => {
    const wrong = (xv: Vector) =>
      new Matrix(xv.size, 2).setCol(0, Vector.ones(xv.size)).setCol(1, xv.mult(1.5));
    const checked = curveFit(line, x, y, Vector.from([0, 0]), { jac: wrong });
    expect(checked.success).toBe(false);
    expect(checked.message).toMatch(/does not match finite-difference/);

    const unchecked = curveFit(line, x, y, Vector.from([0, 0]), {
      jac: wrong,
      jacCheck: false
    });
    expect(unchecked.message).not.toMatch(/does not match/);
  });

  it('expects the Jacobian of the model, not of the residuals (sign convention)', () => {
    // The Jacobian of the residuals y - f is -∂f/∂p; passing it must be rejected by the check.
    const residualJac = (xv: Vector) => lineJac(xv).mult(-1);
    const res = curveFit(line, x, y, Vector.from([0, 0]), { jac: residualJac });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/does not match finite-difference/);
  });

  it('throws if the Jacobian has the wrong shape', () => {
    // Wrong number of columns.
    expect(() =>
      curveFit(line, x, y, Vector.from([0, 0]), { jac: (xv) => new Matrix(xv.size, 1) })
    ).toThrow(RangeError);
    // Wrong number of rows, e.g. the gradient at a single point.
    expect(() =>
      curveFit(line, x, y, Vector.from([0, 0]), { jac: () => new Matrix(1, 2) })
    ).toThrow(RangeError);
  });

  it('throws if the model does not return one value per data point', () => {
    expect(() => curveFit((xv) => Vector.ones(xv.size + 1), x, y, Vector.from([0, 0]))).toThrow(
      RangeError
    );
    expect(() => curveFit(() => Vector.from([1]), x, y, Vector.from([0, 0]))).toThrow(RangeError);
  });
});

describe('curveFit — limits and validation', () => {
  const x = Vector.linspace(0, 4, 10);
  const y = x.map((xi) => 1 + 2 * xi);

  it('passes the solver limits through, and reports non-convergence', () => {
    const res = curveFit(
      (xv, p) => xv.map((xi) => p.get(0) * Math.exp(-p.get(1) * xi)),
      x,
      x.map((xi) => 3 * Math.exp(-2 * xi)),
      Vector.from([1, 0.1]),
      { maxIter: 2 }
    );
    expect(res.success).toBe(false);
    expect(res.iterations).toBe(2);
    expect(res.message).toMatch(/Maximum number of iterations/);
    expect(res.cov.rows).toBe(2);
  });

  it('throws for inconsistent or insufficient data', () => {
    expect(() =>
      curveFit(line, Vector.from([0, 1, 2]), Vector.from([1, 2]), Vector.from([0, 0]))
    ).toThrow(RangeError);
    expect(() => curveFit(line, Vector.from([0]), Vector.from([1]), Vector.from([0, 0]))).toThrow(
      RangeError
    );
    expect(() => curveFit(line, x, y, Vector.zero(0))).toThrow(RangeError);
  });

  it('throws for an invalid sigma', () => {
    expect(() => curveFit(line, x, y, Vector.from([0, 0]), { sigma: Vector.from([1, 2]) })).toThrow(
      RangeError
    );
    expect(() => curveFit(line, x, y, Vector.from([0, 0]), { sigma: Vector.zero(10) })).toThrow(
      Error
    );
    expect(() => curveFit(line, x, y, Vector.from([0, 0]), { sigma: Vector.full(10, -1) })).toThrow(
      Error
    );
    expect(() =>
      curveFit(line, x, y, Vector.from([0, 0]), { sigma: Vector.full(10, NaN) })
    ).toThrow(Error);
  });

  it('throws for invalid solver options', () => {
    expect(() => curveFit(line, x, y, Vector.from([0, 0]), { maxFunEvals: 0 })).toThrow(RangeError);
    expect(() => curveFit(line, x, y, Vector.from([0, 0]), { sclx: Vector.from([1, -1]) })).toThrow(
      Error
    );
  });
});

describe('curveFit with VectorLike data', () => {
  const xRaw = Array.from({ length: 12 }, (_, i) => (5 * i) / 11);
  const yRaw = xRaw.map((xi, i) => 1 + 2 * xi + wobble(i));
  const sigmaRaw = xRaw.map((_, i) => 0.5 + 0.1 * i);
  const p0 = Vector.from([0, 0]);
  const ref = curveFit(line, Vector.from(xRaw), Vector.from(yRaw), p0, {
    sigma: Vector.from(sigmaRaw)
  });

  for (const [name, make] of vectorLikeForms) {
    it(`accepts xdata, ydata and sigma as ${name}`, () => {
      const res = curveFit(line, make(xRaw), make(yRaw), p0, { sigma: make(sigmaRaw) });
      expect(res.success, res.message).toBe(true);
      expect(res.p.toArray()).toEqual(ref.p.toArray());
      expect(res.cov.toArray()).toEqual(ref.cov.toArray());
      expect(res.rss).toBe(ref.rss);
    });

    it(`accepts xdata and ydata as ${name} without sigma`, () => {
      const unweighted = curveFit(line, Vector.from(xRaw), Vector.from(yRaw), p0);
      const res = curveFit(line, make(xRaw), make(yRaw), p0);
      expect(res.p.toArray()).toEqual(unweighted.p.toArray());
    });
  }

  it('passes the independent variable to the model and the Jacobian as a Vector', () => {
    const seen: unknown[] = [];
    const model = (x: Vector, p: Vector) => {
      seen.push(x);
      return line(x, p);
    };
    const jac = (x: Vector) => {
      seen.push(x);
      return lineJac(x);
    };
    curveFit(model, xRaw, yRaw, p0, { jac });
    expect(seen.length).toBeGreaterThan(0);
    for (const x of seen) expect(x).toBeInstanceOf(Vector);
  });

  it('does not copy a Vector xdata passed to the model', () => {
    const x = Vector.from(xRaw);
    let received: Vector | undefined;
    curveFit(
      (xi, p) => {
        received = xi;
        return line(xi, p);
      },
      x,
      yRaw,
      p0
    );
    expect(received).toBe(x);
  });

  it('does not mutate Vector inputs', () => {
    const x = Vector.from(xRaw);
    const y = Vector.from(yRaw);
    const sigma = Vector.from(sigmaRaw);
    curveFit(line, x, y, p0, { sigma });
    expect(x.toArray()).toEqual(xRaw);
    expect(y.toArray()).toEqual(yRaw);
    expect(sigma.toArray()).toEqual(sigmaRaw);
  });

  it('throws a RangeError for mismatched lengths, whatever the form', () => {
    expect(() => curveFit(line, xRaw, yRaw.slice(1), p0)).toThrowError(RangeError);
    expect(() => curveFit(line, xRaw, yRaw, p0, { sigma: new Float64Array(3) })).toThrowError(
      RangeError
    );
  });

  it('throws a TypeError for unsupported inputs', () => {
    expect(() => curveFit(line, 'abc' as never, yRaw, p0)).toThrowError(TypeError);
    expect(() => curveFit(line, xRaw, null as never, p0)).toThrowError(/Vector\.asVector/);
    expect(() => curveFit(line, xRaw, yRaw, p0, { sigma: {} as never })).toThrowError(TypeError);
  });
});

describe('curveFit with a VectorLike initial guess and scaling', () => {
  const xRaw = Array.from({ length: 12 }, (_, i) => (5 * i) / 11);
  const yRaw = xRaw.map((xi, i) => 1 + 2 * xi + wobble(i));
  const p0Raw = [0, 0];
  const sclxRaw = [1, 0.5];
  const x = Vector.from(xRaw);
  const y = Vector.from(yRaw);
  const refPlain = curveFit(line, x, y, Vector.from(p0Raw));
  const refScaled = curveFit(line, x, y, Vector.from(p0Raw), { sclx: Vector.from(sclxRaw) });

  for (const [name, make] of vectorLikeForms) {
    it(`accepts p0 as ${name}`, () => {
      const res = curveFit(line, x, y, make(p0Raw));
      expect(res.success, res.message).toBe(true);
      expect(res.p.toArray()).toEqual(refPlain.p.toArray());
      expect(res.cov.toArray()).toEqual(refPlain.cov.toArray());
    });

    it(`accepts sclx as ${name}`, () => {
      const res = curveFit(line, xRaw, yRaw, p0Raw, { sclx: make(sclxRaw) });
      expect(res.success, res.message).toBe(true);
      expect(res.method).toContain('Fixed');
      expect(res.p.toArray()).toEqual(refScaled.p.toArray());
      expect(res.evaluationsFunction).toBe(refScaled.evaluationsFunction);
    });
  }

  for (const [name, make] of mutableVectorLikeForms) {
    it(`does not mutate p0 or sclx given as ${name}`, () => {
      const p0 = make(p0Raw);
      const sclx = make(sclxRaw);
      curveFit(line, xRaw, yRaw, p0, { sclx });
      expect(Array.from(p0)).toEqual(p0Raw);
      expect(Array.from(sclx)).toEqual(sclxRaw);
    });
  }

  it('accepts every data input as a plain array at once', () => {
    const res = curveFit(line, xRaw, yRaw, p0Raw, { sigma: xRaw.map(() => 1) });
    expect(res.p.toArray()).toEqual(refPlain.p.toArray());
  });

  it('throws a RangeError for an empty p0, whatever the form', () => {
    for (const [, make] of vectorLikeForms) {
      expect(() => curveFit(line, xRaw, yRaw, make([]))).toThrowError(RangeError);
    }
  });

  it('throws a TypeError for an unsupported p0 or sclx', () => {
    expect(() => curveFit(line, xRaw, yRaw, 'abc' as never)).toThrowError(TypeError);
    expect(() => curveFit(line, xRaw, yRaw, null as never)).toThrowError(/Vector\.asVector/);
    expect(() => curveFit(line, xRaw, yRaw, p0Raw, { sclx: {} as never })).toThrowError(TypeError);
  });
});
