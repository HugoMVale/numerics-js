import { describe, expect, it, vi } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import {
  gradientCondition,
  makeHessianSpd,
  updateBfgsFactored,
  updateBfgsInverse,
  updateBfgsUnfactored
} from '../../src/optimize/quasiNewton.js';

const A = Matrix.from([
  [10, 2, 0],
  [2, 5, 1],
  [0, 1, 1]
]);
const grad = (x: Vector) => A.mulVec(x); // gradient of 0.5*xᵀAx

const H0 = Matrix.from([
  [4, 1, 0],
  [1, 3, 0.5],
  [0, 0.5, 2]
]);
const xc = Vector.from([1, -2, 0.5]);
const xp = Vector.from([0.6, -1.5, 0.2]);
const gc = grad(xc);
const gp = grad(xp);
const s = xp.sub(xc);
const y = gp.sub(gc);

const factorOf = (H: Matrix) => H.cholesky();
const LLt = (L: Matrix) => L.matmul(L.transpose());

describe('BFGS updates', () => {
  it('unfactored: satisfies the secant equation, symmetry and positive definiteness', () => {
    const H = updateBfgsUnfactored(xc, xp, gc, gp, H0.copy(), true);
    expect(H.mulVec(s).allClose(y, { atol: 1e-12, rtol: 1e-12 })).toBe(true);
    expect(H.isSymmetric(0)).toBe(true);
    expect(() => H.cholesky()).not.toThrow();
  });

  it('factored: satisfies the secant equation and returns a lower-triangular factor', () => {
    const L = updateBfgsFactored(xc, xp, gc, gp, factorOf(H0), true);
    expect(L.isLowerTriangular()).toBe(true);
    expect(LLt(L).mulVec(s).allClose(y, { atol: 1e-12, rtol: 1e-12 })).toBe(true);
  });

  it('inverse: satisfies the inverse secant equation and is symmetric', () => {
    const B = updateBfgsInverse(xc, xp, gc, gp, H0.inverse(), true);
    expect(B.mulVec(y).allClose(s, { atol: 1e-12, rtol: 1e-12 })).toBe(true);
    expect(B.isSymmetric(0)).toBe(true);
  });

  it('the three forms are equivalent', () => {
    const H = updateBfgsUnfactored(xc, xp, gc, gp, H0.copy(), true);
    const L = updateBfgsFactored(xc, xp, gc, gp, factorOf(H0), true);
    const B = updateBfgsInverse(xc, xp, gc, gp, H0.inverse(), true);
    expect(LLt(L).allClose(H, { atol: 1e-12, rtol: 1e-12 })).toBe(true);
    expect(B.matmul(H).allClose(Matrix.identity(3), { atol: 1e-10, rtol: 1e-10 })).toBe(true);
  });

  it('the forms stay equivalent over a sequence of updates', () => {
    let H = H0.copy();
    let L = factorOf(H0);
    let B = H0.inverse();
    let x = xc;
    let g = gc;
    for (const next of [xp, Vector.from([0.1, -0.9, 0.1]), Vector.from([0.05, -0.2, -0.1])]) {
      const gn = grad(next);
      H = updateBfgsUnfactored(x, next, g, gn, H, true);
      L = updateBfgsFactored(x, next, g, gn, L, true);
      B = updateBfgsInverse(x, next, g, gn, B, true);
      x = next;
      g = gn;
    }
    expect(LLt(L).allClose(H, { atol: 1e-10, rtol: 1e-10 })).toBe(true);
    expect(B.matmul(H).allClose(Matrix.identity(3), { atol: 1e-9, rtol: 1e-9 })).toBe(true);
  });

  it('unfactored and inverse updates act in place; the factored update does not', () => {
    const H = H0.copy();
    expect(updateBfgsUnfactored(xc, xp, gc, gp, H, true)).toBe(H);
    const B = H0.inverse();
    expect(updateBfgsInverse(xc, xp, gc, gp, B, true)).toBe(B);
    const L = factorOf(H0);
    const Lcopy = L.copy();
    const Lp = updateBfgsFactored(xc, xp, gc, gp, L, true);
    expect(Lp).not.toBe(L);
    expect(L.allClose(Lcopy, { atol: 0, rtol: 0 })).toBe(true);
  });

  describe('skips the update', () => {
    const unchanged = (update: (c: Matrix) => Matrix, base: Matrix) => {
      const copy = base.copy();
      const result = update(base);
      expect(result.allClose(copy, { atol: 0, rtol: 0 })).toBe(true);
    };

    it('if the curvature condition is not met (yᵀs <= 0)', () => {
      const gBad = gc.sub(y); // reverses y
      unchanged((H) => updateBfgsUnfactored(xc, xp, gc, gBad, H, true), H0.copy());
      unchanged((H) => updateBfgsInverse(xc, xp, gc, gBad, H, true), H0.inverse());
      unchanged((L) => updateBfgsFactored(xc, xp, gc, gBad, L, true), factorOf(H0));
    });

    it('if yᵀs is zero, even with a zero step (no NaN contamination)', () => {
      unchanged((H) => updateBfgsUnfactored(xc, xc, gc, gc, H, true), H0.copy());
      unchanged((H) => updateBfgsInverse(xc, xc, gc, gc, H, true), H0.inverse());
      unchanged((L) => updateBfgsFactored(xc, xc, gc, gc, L, true), factorOf(H0));
    });

    it('if the update is negligible (the model already matches the gradient change)', () => {
      const yExact = H0.mulVec(s);
      const gExact = gc.add(yExact);
      unchanged((H) => updateBfgsUnfactored(xc, xp, gc, gExact, H, true), H0.copy());
      unchanged((L) => updateBfgsFactored(xc, xp, gc, gExact, L, true), factorOf(H0));
      unchanged((B) => updateBfgsInverse(xc, xp, gc, gExact, B, true), H0.inverse());
    });
  });

  it('uses a larger threshold for the negligible update if the gradient is approximate', () => {
    // y differs from H0 s by about 1e-12 relative: negligible for finite-difference gradients,
    // but not for analytic ones.
    const yClose = H0.mulVec(s).map((v) => v * (1 + 1e-12));
    const g = gc.add(yClose);
    const analytic = updateBfgsUnfactored(xc, xp, gc, g, H0.copy(), true);
    const approximate = updateBfgsUnfactored(xc, xp, gc, g, H0.copy(), false);
    expect(analytic.allClose(H0, { atol: 0, rtol: 0 })).toBe(false);
    expect(approximate.allClose(H0, { atol: 0, rtol: 0 })).toBe(true);
  });
});

describe('makeHessianSpd', () => {
  it('leaves a well-conditioned SPD matrix unchanged', () => {
    const { H, L, perturbation } = makeHessianSpd(A);
    expect(perturbation).toBe(0);
    expect(H.allClose(A, { atol: 0, rtol: 0 })).toBe(true);
    expect(LLt(L).allClose(A, { atol: 1e-14, rtol: 1e-14 })).toBe(true);
    expect(L.isLowerTriangular()).toBe(true);
  });

  it('does not modify its input and returns an independent matrix', () => {
    const input = A.copy();
    const { H } = makeHessianSpd(input);
    expect(input.allClose(A, { atol: 0, rtol: 0 })).toBe(true);
    expect(H).not.toBe(input);
  });

  it('symmetrizes the input', () => {
    const asym = Matrix.from([
      [10, 4, 0],
      [0, 5, 2],
      [0, 0, 1]
    ]);
    const { H } = makeHessianSpd(asym);
    expect(H.isSymmetric(0)).toBe(true);
    expect(H.get(0, 1)).toBe(2);
    expect(H.get(1, 2)).toBe(1);
  });

  it('makes a matrix with negative or tiny diagonal elements positive definite', () => {
    for (const m of [
      [
        [-1, 0],
        [0, 1]
      ],
      [
        [1e-3, 5],
        [5, 1e-3]
      ],
      [
        [1, 1.9, 1.9],
        [1.9, 1, 1.9],
        [1.9, 1.9, 1]
      ],
      [
        [-1e3, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ]
    ]) {
      const { H, L, perturbation } = makeHessianSpd(Matrix.from(m));
      expect(perturbation).toBeGreaterThan(0);
      expect(LLt(L).allClose(H, { atol: 1e-12, rtol: 1e-12 })).toBe(true);
      expect(H.allFinite()).toBe(true);
    }
  });

  it('turns the zero matrix into the identity', () => {
    const { H, L, perturbation } = makeHessianSpd(new Matrix(3, 3));
    expect(perturbation).toBe(1);
    expect(H.allClose(Matrix.identity(3), { atol: 0, rtol: 0 })).toBe(true);
    expect(L.allClose(Matrix.identity(3), { atol: 0, rtol: 0 })).toBe(true);
  });

  it('handles a 1x1 matrix', () => {
    expect(makeHessianSpd(Matrix.from([[4]])).L.get(0, 0)).toBe(2);
    expect(makeHessianSpd(Matrix.from([[-4]])).H.get(0, 0)).toBeGreaterThan(0);
  });

  it('throws if the matrix cannot be repaired', () => {
    const indefinite = Matrix.from([
      [1, -3],
      [-3, -2]
    ]);
    expect(() => makeHessianSpd(indefinite)).toThrow(/Perturbed Cholesky failed/);
  });

  it('applies the diagonal adjustment of exactly singular matrices before factorizing', () => {
    const singular = Matrix.from([
      [1, 1],
      [1, 1]
    ]);
    const { H, perturbation } = makeHessianSpd(singular, 0);
    expect(perturbation).toBeGreaterThan(0);
    expect(perturbation).toBeLessThan(1e-7);
    expect(() => H.cholesky()).not.toThrow();
  });

  it('reports a non-finite Hessian as such, without retrying', () => {
    // An infinite entry used to be swallowed as a failed Cholesky factorization, retried 11
    // times and reported as "Perturbed Cholesky failed".
    for (const bad of [Infinity, -Infinity, NaN]) {
      const H = Matrix.from([
        [2, bad],
        [bad, 3]
      ]);
      expect(() => makeHessianSpd(H)).toThrow(RangeError);
      expect(() => makeHessianSpd(H)).toThrow(
        'makeHessianSpd: the Hessian must contain only finite values.'
      );
    }
    const diagonal = Matrix.from([
      [Infinity, 0],
      [0, 1]
    ]);
    expect(() => makeHessianSpd(diagonal)).toThrow(/only finite values/);
  });

  it('retries only for a matrix that is not positive definite', () => {
    // Not positive definite: retried, and finally reported as such.
    const indefinite = Matrix.from([
      [1, -3],
      [-3, -2]
    ]);
    expect(() => makeHessianSpd(indefinite)).toThrow(/Perturbed Cholesky failed/);
    // Any other error from cholesky is rethrown as it is, not retried.
    const spy = vi.spyOn(Matrix.prototype, 'cholesky').mockImplementation(() => {
      throw new TypeError('unexpected');
    });
    try {
      expect(() => makeHessianSpd(Matrix.identity(2))).toThrow(new TypeError('unexpected'));
      expect(spy).toHaveBeenCalledTimes(1);
    } finally {
      spy.mockRestore();
    }
  });

  it('does not exceed the allowed number of perturbation attempts', () => {
    const indefinite = Matrix.from([
      [1, -3],
      [-3, -2]
    ]);
    expect(() => makeHessianSpd(indefinite, 0)).toThrow(/Perturbed Cholesky failed/);
  });
});

describe('gradientCondition', () => {
  const sclx = Vector.from([1, 1]);

  it('scales the gradient by max(|x|, 1/sclx) / max(|f|, 1/sclf)', () => {
    const x = Vector.from([10, 0.1]);
    const g = Vector.from([1e-6, 1e-6]);
    // scaled: [1e-6*10/1, 1e-6*1/1] = [1e-5, 1e-6]
    expect(gradientCondition(x, 0, g, sclx, 1, 1.1e-5)).toBe(true);
    expect(gradientCondition(x, 0, g, sclx, 1, 0.9e-5)).toBe(false);
    // |f| = 100 dominates 1/sclf: scaled gradient is 100x smaller
    expect(gradientCondition(x, 100, g, sclx, 1, 0.9e-7)).toBe(false);
    expect(gradientCondition(x, 100, g, sclx, 1, 1.1e-7)).toBe(true);
    // 1/sclf = 1e3 dominates |f|
    expect(gradientCondition(x, 100, g, sclx, 1e-3, 1.1e-8)).toBe(true);
    expect(gradientCondition(x, 100, g, sclx, 1e-3, 0.9e-8)).toBe(false);
  });

  it('is inclusive, and exact zero satisfies a zero tolerance', () => {
    const x = Vector.from([1, 1]);
    expect(gradientCondition(x, 0, Vector.from([2e-5, 0]), sclx, 1, 2e-5)).toBe(true);
    expect(gradientCondition(x, 0, Vector.from([0, 0]), sclx, 1, 0)).toBe(true);
    expect(gradientCondition(x, 0, Vector.from([1e-300, 0]), sclx, 1, 0)).toBe(false);
  });

  it('is never satisfied by NaN', () => {
    const x = Vector.from([1, 1]);
    expect(gradientCondition(x, 0, Vector.from([0, NaN]), sclx, 1, 1)).toBe(false);
    expect(gradientCondition(x, NaN, Vector.from([0, 0]), sclx, 1, 1)).toBe(false);
  });
});

describe('non-finite Hessians in the solver', () => {
  it('reports them accurately, not as a failed Cholesky factorization', async () => {
    const { quasiNewton } = await import('../../src/optimize/quasiNewton.js');
    const f = (x: Vector) => x.get(0) ** 2 + x.get(1) ** 2;
    const grad = (x: Vector) => Vector.from([2 * x.get(0), 2 * x.get(1)]);
    for (const bfgsUpdate of [true, false]) {
      const res = quasiNewton(f, [1, 1], {
        grad,
        hess: () =>
          Matrix.from([
            [Infinity, 0],
            [0, 1]
          ]),
        bfgsUpdate
      });
      expect(res.success).toBe(false);
      expect(res.message).toMatch(/non-finite/);
      expect(res.message).not.toMatch(/Perturbed Cholesky/);
    }
  });

  it('cholesky throws an Error whose message says "not positive definite"', () => {
    expect(() =>
      Matrix.from([
        [1, 2],
        [2, 1]
      ]).cholesky()
    ).toThrow(Error);
    expect(() =>
      Matrix.from([
        [1, 2],
        [2, 1]
      ]).cholesky()
    ).toThrow(/not positive definite/);
  });
});
