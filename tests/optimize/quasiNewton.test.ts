import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import {
  quasiNewton,
  type BfgsMethod,
  type QuasiNewtonOptions
} from '../../src/optimize/quasiNewton.js';
import type { GlobalMethod } from '../../src/roots/types.js';
import { vectorLikeForms } from '../vectorLikeForms.js';
import { GRADIENT_TEST_FUNCTIONS, type GradientTestFunction } from './gradientTestFunctions.js';

const maxError = (x: Vector, xs: number[]) => Math.max(...xs.map((v, i) => Math.abs(v - x.get(i))));

type Derivatives = 'fd' | 'grad' | 'full';

/** Options selecting how derivatives are obtained. */
function derivativeOptions(t: GradientTestFunction, deriv: Derivatives): QuasiNewtonOptions {
  return {
    grad: deriv === 'fd' ? undefined : t.grad,
    hess: deriv === 'full' ? t.hess : undefined
  };
}

describe('quasiNewton convergence', () => {
  const derivs: Derivatives[] = ['fd', 'grad', 'full'];

  describe.each(Object.entries(GRADIENT_TEST_FUNCTIONS))('%s', (_id, t) => {
    const available = derivs.filter((d) => d !== 'full' || t.hess);

    // The dogleg strategy always uses the factored BFGS form.
    it.each(available)('dogleg, BFGS (factored), derivatives: %s', (deriv) => {
      const res = quasiNewton(t.f, t.x0, {
        globalMethod: 'dogleg',
        ...derivativeOptions(t, deriv)
      });
      expect(res.success, res.message).toBe(true);
      expect(maxError(res.x, t.xs)).toBeLessThan(1e-4);
      expect(Math.abs(res.fx - t.fs)).toBeLessThan(1e-8);
    });

    const forms: BfgsMethod[] = ['factored', 'unfactored', 'inverse'];
    const cases = forms.flatMap((form) => available.map((deriv) => [form, deriv] as const));
    it.each(cases)('line-search, BFGS (%s), derivatives: %s', (bfgsMethod, deriv) => {
      const res = quasiNewton(t.f, t.x0, {
        globalMethod: 'line-search',
        bfgsMethod,
        ...derivativeOptions(t, deriv)
      });
      expect(res.success, res.message).toBe(true);
      expect(maxError(res.x, t.xs)).toBeLessThan(1e-4);
    });
  });

  // Newton with a Hessian re-evaluated at each iteration: only the convex problems, because the
  // perturbed Cholesky factorization cannot repair a strongly indefinite Hessian.
  describe.each(['rosenbrock', 'quadratic'])('no BFGS update, %s', (id) => {
    const t = GRADIENT_TEST_FUNCTIONS[id];
    const available = derivs.filter((d) => d !== 'full' || t.hess);
    const cases = (['dogleg', 'line-search'] as const).flatMap((g) =>
      available.map((deriv) => [g, deriv] as const)
    );
    it.each(cases)('%s, derivatives: %s', (globalMethod, deriv) => {
      const res = quasiNewton(t.f, t.x0, {
        globalMethod,
        bfgsUpdate: false,
        ...derivativeOptions(t, deriv)
      });
      expect(res.success, res.message).toBe(true);
      expect(maxError(res.x, t.xs)).toBeLessThan(1e-4);
    });
  });

  it.each(['factored', 'unfactored', 'inverse'] as const)(
    'converges without global strategy near the minimum, BFGS (%s)',
    (bfgsMethod) => {
      const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
      const res = quasiNewton(t.f, [0.9, 0.85], {
        globalMethod: null,
        bfgsMethod,
        grad: t.grad
      });
      expect(res.success, res.message).toBe(true);
      expect(maxError(res.x, t.xs)).toBeLessThan(1e-4);
    }
  );

  it('solves a quadratic in one iteration with the exact Hessian and a full Newton step', () => {
    const t = GRADIENT_TEST_FUNCTIONS.quadratic;
    const res = quasiNewton(t.f, t.x0, { globalMethod: null, grad: t.grad, hess: t.hess });
    expect(res.success, res.message).toBe(true);
    expect(res.iterations).toBe(1);
    expect(res.evaluationsHessian).toBe(1);
    expect(maxError(res.x, t.xs)).toBeLessThan(1e-12);
  });

  it('reproduces the reference iteration and evaluation counts of the Rosenbrock example', () => {
    // Same problem as the docstring example of the original implementation (`fmin_qnewton`).
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const res = quasiNewton(t.f, [-1, -1]);
    expect(res.success, res.message).toBe(true);
    expect(res.method).toBe('Quasi-Newton (Global: Dogleg, BFGS: Factored)');
    expect(res.iterations).toBe(28);
    expect(res.evaluations).toBe(98);
    expect(res.evaluationsGradient).toBe(0);
    expect(res.evaluationsHessian).toBe(0);
    expect(maxError(res.x, t.xs)).toBeLessThan(1e-4);
  });
});

describe('quasiNewton method options', () => {
  const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;

  it.each([
    [{}, 'Quasi-Newton (Global: Dogleg, BFGS: Factored)'],
    [{ globalMethod: 'line-search' }, 'Quasi-Newton (Global: Line-Search, BFGS: Inverse)'],
    [{ globalMethod: null }, 'Quasi-Newton (Global: None, BFGS: Inverse)'],
    [
      { globalMethod: 'line-search', bfgsMethod: 'unfactored' },
      'Quasi-Newton (Global: Line-Search, BFGS: Unfactored)'
    ],
    [{ bfgsUpdate: false }, 'Quasi-Newton (Global: Dogleg)'],
    // BFGS is not used by default if the Hessian is provided
    [{ hess: t.hess }, 'Quasi-Newton (Global: Dogleg)'],
    [{ hess: t.hess, bfgsUpdate: true }, 'Quasi-Newton (Global: Dogleg, BFGS: Factored)'],
    // the dogleg strategy needs the Cholesky factor, whatever the requested form
    [{ bfgsMethod: 'inverse' }, 'Quasi-Newton (Global: Dogleg, BFGS: Factored)'],
    [{ bfgsMethod: 'unfactored' }, 'Quasi-Newton (Global: Dogleg, BFGS: Factored)'],
    // `bfgsMethod` is ignored if there is no BFGS update
    [{ bfgsUpdate: false, bfgsMethod: 'inverse' }, 'Quasi-Newton (Global: Dogleg)']
  ] as [QuasiNewtonOptions, string][])('reports the method for %j', (options, method) => {
    expect(quasiNewton(t.f, [-1.2, 1], { ...options, maxIter: 1 }).method).toBe(method);
  });

  it('uses the Hessian function at each iteration by default, and not with BFGS', () => {
    const noBfgs = quasiNewton(t.f, t.x0, { grad: t.grad, hess: t.hess });
    expect(noBfgs.success, noBfgs.message).toBe(true);
    // x0 and one evaluation after each completed iteration but the last
    expect(noBfgs.evaluationsHessian).toBe(noBfgs.iterations);

    const bfgs = quasiNewton(t.f, t.x0, { grad: t.grad, hess: t.hess, bfgsUpdate: true });
    expect(bfgs.success, bfgs.message).toBe(true);
    expect(bfgs.evaluationsHessian).toBe(1);
  });

  describe('result matrices', () => {
    const q = GRADIENT_TEST_FUNCTIONS.quadratic;
    const A = q.hess!(Vector.from(q.x0));

    it('factored: L is lower triangular and Hx = L*Lᵀ', () => {
      const res = quasiNewton(t.f, t.x0, { grad: t.grad, bfgsMethod: 'factored' });
      expect(res.L).not.toBeNull();
      expect(res.L!.isLowerTriangular()).toBe(true);
      expect(res.Hinv).toBeNull();
      expect(
        res.Hx!.allClose(res.L!.matmul(res.L!.transpose()), { atol: 1e-12, rtol: 1e-12 })
      ).toBe(true);
      expect(res.Hx!.isSymmetric(1e-9)).toBe(true);
    });

    it('unfactored: Hx is symmetric and the other matrices are null', () => {
      const res = quasiNewton(t.f, t.x0, {
        grad: t.grad,
        globalMethod: 'line-search',
        bfgsMethod: 'unfactored'
      });
      expect(res.Hx).not.toBeNull();
      expect(res.Hx!.isSymmetric(0)).toBe(true);
      expect(res.L).toBeNull();
      expect(res.Hinv).toBeNull();
    });

    it('inverse: Hinv is symmetric and the other matrices are null', () => {
      const res = quasiNewton(t.f, t.x0, {
        grad: t.grad,
        globalMethod: 'line-search',
        bfgsMethod: 'inverse'
      });
      expect(res.Hinv).not.toBeNull();
      expect(res.Hinv!.isSymmetric(0)).toBe(true);
      expect(res.Hx).toBeNull();
      expect(res.L).toBeNull();
    });

    it('no update: Hx is the last evaluated Hessian', () => {
      const res = quasiNewton(q.f, q.x0, { grad: q.grad, hess: q.hess });
      expect(res.Hx!.allClose(A, { atol: 1e-12, rtol: 1e-12 })).toBe(true);
      expect(res.L).toBeNull();
      expect(res.Hinv).toBeNull();
    });
  });
});

describe('quasiNewton bookkeeping', () => {
  type Counts = { f: number; g: number; h: number };

  const counted = (t: GradientTestFunction) => {
    const counts: Counts = { f: 0, g: 0, h: 0 };
    return {
      counts,
      f: (x: Vector) => (counts.f++, t.f(x)),
      grad: (x: Vector) => (counts.g++, t.grad(x)),
      hess: t.hess ? (x: Vector) => (counts.h++, t.hess!(x)) : undefined
    };
  };

  const globals: GlobalMethod[] = ['dogleg', 'line-search', null];
  const cases = globals.flatMap((g) =>
    (['fd', 'grad', 'full'] as const).flatMap((d) => [true, false].map((u) => [g, d, u] as const))
  );

  // The counts reported in the result must equal the number of calls actually made.
  it.each(cases)('counts evaluations exactly: global %s, derivatives %s, BFGS %s', (g, d, u) => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const c = counted(t);
    const res = quasiNewton(c.f, [-1, -1], {
      globalMethod: g,
      bfgsUpdate: u,
      grad: d === 'fd' ? undefined : c.grad,
      hess: d === 'full' ? c.hess : undefined,
      maxIter: 15
    });
    expect(res.evaluations).toBe(c.counts.f);
    expect(res.evaluationsGradient).toBe(c.counts.g);
    expect(res.evaluationsHessian).toBe(c.counts.h);
    if (d === 'fd') expect(res.evaluationsGradient).toBe(0);
    if (d !== 'full') expect(res.evaluationsHessian).toBe(0);
  });

  it('the gradient check costs n function evaluations and can be disabled', () => {
    const t = GRADIENT_TEST_FUNCTIONS.wood;
    const on = counted(t);
    const resOn = quasiNewton(on.f, t.x0, { grad: on.grad, maxIter: 1 });
    const off = counted(t);
    const resOff = quasiNewton(off.f, t.x0, { grad: off.grad, gradCheck: false, maxIter: 1 });
    expect(on.counts.f - off.counts.f).toBe(4);
    expect(resOn.evaluations).toBe(on.counts.f);
    expect(resOff.evaluations).toBe(off.counts.f);
  });

  it('does not modify x0 and does not alias it in the result', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const x0 = new Float64Array(t.x0);
    const res = quasiNewton(t.f, x0, { maxIter: 3 });
    expect(Array.from(x0)).toEqual(t.x0);
    res.x.set(0, 123);
    expect(Array.from(x0)).toEqual(t.x0);

    // Iteration 0: the initial guess itself is returned
    const solved = quasiNewton(t.f, new Float64Array(t.xs), { grad: t.grad });
    expect(solved.iterations).toBe(0);
    solved.x.set(0, 123);
  });

  it.each(vectorLikeForms)('accepts x0 and sclx given as %s', (_name, make) => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const ref = quasiNewton(t.f, t.x0, { sclx: [1, 1], grad: t.grad });
    const res = quasiNewton(t.f, make(t.x0), { sclx: make([1, 1]), grad: t.grad });
    expect(res.x.toArray()).toEqual(ref.x.toArray());
    expect(res.evaluations).toBe(ref.evaluations);
  });

  it('returns immediately if the initial guess is a minimizer', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    let called = false;
    const res = quasiNewton(t.f, t.xs, {
      grad: t.grad,
      callback: () => ((called = true), { stop: false, success: false })
    });
    expect(res.success).toBe(true);
    expect(res.iterations).toBe(0);
    expect(res.message).toContain('initial guess');
    expect(res.fx).toBe(0);
    expect(res.gx!.toArray().map(Math.abs)).toEqual([0, 0]);
    expect(res.Hx).toBeNull();
    expect(called).toBe(false);
  });

  it('reports non-convergence when maxIter is reached', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const res = quasiNewton(t.f, t.x0, { grad: t.grad, maxIter: 3 });
    expect(res.success).toBe(false);
    expect(res.iterations).toBe(3);
    expect(res.message).toBe('Maximum number of iterations (3) reached.');
  });

  it('the returned gradient and function value correspond to the returned x', () => {
    const t = GRADIENT_TEST_FUNCTIONS.wood;
    for (const maxIter of [1, 5, 100]) {
      const res = quasiNewton(t.f, t.x0, { grad: t.grad, maxIter });
      expect(res.fx).toBe(t.f(res.x));
      expect(res.gx!.toArray()).toEqual(t.grad(res.x).toArray());
    }
  });

  it('does not depend on a gradient buffer being fresh at each call', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const buffer = new Vector(2);
    const reusing = (x: Vector) => buffer.set(t.grad(x).toArray());
    const fresh = quasiNewton(t.f, t.x0, { grad: t.grad });
    const reused = quasiNewton(t.f, t.x0, { grad: reusing });
    expect(reused.success).toBe(true);
    expect(reused.x.toArray()).toEqual(fresh.x.toArray());
    expect(reused.iterations).toBe(fresh.iterations);
  });
});

describe('quasiNewton callback', () => {
  const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;

  it('is called after each iteration with consistent arguments', () => {
    const seen: { n: number; x: number[]; fx: number; gx: number[] }[] = [];
    const res = quasiNewton(t.f, t.x0, {
      grad: t.grad,
      callback: (n, x, fx, gx) => {
        seen.push({ n, x: x.toArray(), fx, gx: gx.toArray() });
        return { stop: false, success: false };
      }
    });
    expect(res.success, res.message).toBe(true);
    expect(seen.map((s) => s.n)).toEqual(Array.from({ length: res.iterations }, (_, i) => i + 1));
    for (const s of seen) {
      expect(s.fx).toBe(t.f(Vector.from(s.x)));
      expect(s.gx).toEqual(t.grad(Vector.from(s.x)).toArray());
    }
    // the last call is the final iterate
    expect(seen.at(-1)!.x).toEqual(res.x.toArray());
    expect(seen.at(-1)!.fx).toBe(res.fx);
  });

  it.each([true, false])(
    'stops the iteration with success = %s, before the built-in checks',
    (ok) => {
      const q = GRADIENT_TEST_FUNCTIONS.quadratic;
      // A full Newton step solves the quadratic in the first iteration: the callback has precedence.
      const res = quasiNewton(q.f, q.x0, {
        globalMethod: null,
        grad: q.grad,
        hess: q.hess,
        callback: () => ({ stop: true, success: ok })
      });
      expect(res.iterations).toBe(1);
      expect(res.success).toBe(ok);
      expect(res.message).toBe('Terminated by user callback.');
      expect(maxError(res.x, q.xs)).toBeLessThan(1e-12);
    }
  );

  it('receives the unchanged iterate if the global step is rejected', () => {
    // The gradient has the wrong sign, so no step along the quasi-Newton direction decreases f.
    const calls: [number, number[]][] = [];
    const res = quasiNewton(t.f, [-1, 1], {
      grad: (x) => t.grad(x).mult(-1),
      gradCheck: false,
      globalMethod: 'line-search',
      callback: (n, x) => {
        calls.push([n, x.toArray()]);
        return { stop: false, success: false };
      }
    });
    expect(res.success).toBe(false);
    expect(res.message).toContain('failed to decrease');
    expect(calls).toEqual([[1, [-1, 1]]]);
    expect(res.x.toArray()).toEqual([-1, 1]);
  });
});

describe('quasiNewton termination', () => {
  it('stops on `tolx`, without reporting success', () => {
    // Newton's method converges only linearly to the quadruple root of the gradient.
    const f = (x: Vector) => (x.get(0) - 1) ** 4;
    const grad = (x: Vector) => Vector.from([4 * (x.get(0) - 1) ** 3]);
    const hess = (x: Vector) => Matrix.from([[12 * (x.get(0) - 1) ** 2]]);
    const res = quasiNewton(f, [3], { grad, hess, tolg: 0, globalMethod: 'line-search' });
    expect(res.success).toBe(false);
    expect(res.message).toContain('`tolx`');
    expect(Math.abs(res.x.get(0) - 1)).toBeLessThan(1e-5);
  });

  it('stops after 5 consecutive steps of maximum length', () => {
    // f is unbounded below: the quasi-Newton step is always cut to `maxLen`.
    const res = quasiNewton((x) => -x.get(0), [1], {
      grad: () => Vector.from([-1]),
      maxLenFactor: 0.5,
      globalMethod: 'line-search'
    });
    expect(res.success).toBe(false);
    expect(res.iterations).toBe(5);
    expect(res.message).toContain('Maximum number (5) of consecutive steps');
    expect(res.x.get(0)).toBeCloseTo(1 + 5 * 0.5, 12);
  });

  it('does not apply the maximum-length criterion without a global strategy', () => {
    const res = quasiNewton((x) => -x.get(0), [1], {
      grad: () => Vector.from([-1]),
      maxLenFactor: 0.5,
      globalMethod: null,
      maxIter: 8
    });
    expect(res.message).toBe('Maximum number of iterations (8) reached.');
  });

  it('reports a failed global step', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    for (const globalMethod of ['line-search', 'dogleg'] as const) {
      const res = quasiNewton(t.f, [-1, 1], {
        grad: (x) => t.grad(x).mult(-1),
        gradCheck: false,
        globalMethod
      });
      expect(res.success).toBe(false);
      expect(res.message).toContain('Last global step failed');
      expect(res.iterations).toBe(1);
    }
  });

  it('reports failure if the Hessian cannot be made positive definite', () => {
    const t = GRADIENT_TEST_FUNCTIONS.beale;
    const res = quasiNewton(t.f, t.x0, { grad: t.grad, bfgsUpdate: false });
    expect(res.success).toBe(false);
    expect(res.message).toContain('Perturbed Cholesky factorization of Hessian failed');
    expect(res.iterations).toBe(1);
    expect(res.x.toArray()).toEqual(t.x0);
  });
});

describe('quasiNewton initial Hessian', () => {
  const q = GRADIENT_TEST_FUNCTIONS.quadratic;
  const A = q.hess!(Vector.from(q.x0));

  it('an exact H0 speeds up convergence relative to the identity', () => {
    for (const bfgsMethod of ['factored', 'unfactored', 'inverse'] as const) {
      const globalMethod = bfgsMethod === 'factored' ? 'dogleg' : 'line-search';
      const base = { grad: q.grad, bfgsMethod, globalMethod } as const;
      const identity = quasiNewton(q.f, q.x0, base);
      const exact = quasiNewton(q.f, q.x0, { ...base, H0: A });
      expect(exact.success, exact.message).toBe(true);
      expect(exact.iterations).toBeLessThan(identity.iterations);
      expect(exact.iterations).toBeLessThanOrEqual(2);
    }
  });

  it('H0 takes precedence over `hess` at x0, and is not modified', () => {
    const H0 = Matrix.identity(3);
    const res = quasiNewton(q.f, q.x0, { grad: q.grad, hess: q.hess, H0, bfgsUpdate: true });
    expect(res.evaluationsHessian).toBe(0);
    expect(H0.allClose(Matrix.identity(3), { atol: 0, rtol: 0 })).toBe(true);
  });

  it('H0 replaces the Hessian evaluation at x0 if the Hessian is re-evaluated', () => {
    const r = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const calls: number[][] = [];
    const res = quasiNewton(r.f, r.x0, {
      grad: r.grad,
      hess: (x) => (calls.push(x.toArray()), r.hess!(x)),
      H0: r.hess!(Vector.from(r.x0)),
      globalMethod: 'line-search'
    });
    expect(res.success, res.message).toBe(true);
    // the first Hessian evaluation is at the first iterate, not at x0
    expect(calls[0]).not.toEqual(r.x0);
    expect(res.evaluationsHessian).toBe(calls.length);
  });

  it('an asymmetric H0 is symmetrized', () => {
    const H0 = Matrix.from([
      [10, 4, 0],
      [0, 5, 2],
      [0, 0, 1]
    ]);
    for (const bfgsMethod of ['unfactored', 'inverse'] as const) {
      const res = quasiNewton(q.f, q.x0, {
        grad: q.grad,
        H0,
        bfgsMethod,
        globalMethod: 'line-search'
      });
      expect(res.success, res.message).toBe(true);
    }
  });

  it('reports failure if the initial Hessian cannot be factorized or inverted', () => {
    const r = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const indefinite = Matrix.from([
      [1, -3],
      [-3, -2]
    ]);
    for (const bfgsMethod of ['factored', 'unfactored'] as const) {
      const res = quasiNewton(r.f, r.x0, {
        grad: r.grad,
        H0: indefinite,
        bfgsMethod,
        globalMethod: 'line-search'
      });
      expect(res.success).toBe(false);
      expect(res.message).toContain('Cholesky factorization of initial Hessian failed');
      expect(res.message.endsWith('..')).toBe(false);
      expect(res.iterations).toBe(0);
    }

    const singular = Matrix.from([
      [1, 1],
      [1, 1]
    ]);
    const res = quasiNewton(r.f, r.x0, {
      grad: r.grad,
      H0: singular,
      bfgsMethod: 'inverse',
      globalMethod: 'line-search'
    });
    expect(res.success).toBe(false);
    expect(res.message).toContain('Inverse of initial Hessian failed');
    expect(res.message).toContain('Matrix.inverse: matrix is singular.');
    expect(res.message.endsWith('.')).toBe(true);
    expect(res.message.endsWith('..')).toBe(false);
    expect(res.iterations).toBe(0);
  });

  it('reports failure if H0 or a Hessian has non-finite values', () => {
    const res = quasiNewton(q.f, q.x0, {
      grad: q.grad,
      H0: Matrix.from([
        [1, 0, 0],
        [0, NaN, 0],
        [0, 0, 1]
      ])
    });
    expect(res.success).toBe(false);
    expect(res.message).toBe('H0 contains non-finite values.');
    expect(res.iterations).toBe(0);
  });
});

describe('quasiNewton non-finite values', () => {
  const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;

  it('fails if f(x0) is not finite', () => {
    for (const bad of [NaN, Infinity, -Infinity]) {
      let calls = 0;
      const res = quasiNewton(() => (calls++, bad), [1, 2]);
      expect(res.success).toBe(false);
      expect(res.message).toBe('f(x0) is not finite.');
      expect(res.iterations).toBe(0);
      expect(res.gx).toBeNull();
      expect(calls).toBe(1);
    }
  });

  it('fails if the gradient at x0 is not finite', () => {
    const res = quasiNewton(t.f, t.x0, { grad: () => Vector.from([NaN, 1]) });
    expect(res.success).toBe(false);
    expect(res.message).toBe('Gradient at x0 contains non-finite values.');
    expect(res.iterations).toBe(0);
  });

  it('fails if the finite-difference gradient at x0 is not finite', () => {
    // f is finite at x0 but not at the perturbed point
    const f = (x: Vector) => (x.get(0) === 1 ? 0.5 : NaN);
    const res = quasiNewton(f, [1]);
    expect(res.success).toBe(false);
    expect(res.message).toBe('Gradient at x0 contains non-finite values.');
  });

  it('fails if the Hessian at x0 is not finite', () => {
    const res = quasiNewton(t.f, t.x0, {
      grad: t.grad,
      hess: () =>
        Matrix.from([
          [1, Infinity],
          [Infinity, 1]
        ])
    });
    expect(res.success).toBe(false);
    expect(res.message).toBe('Hessian at x0 contains non-finite values.');
    expect(res.iterations).toBe(0);
  });

  // f is not defined for x < 0; the first quasi-Newton step from x0 = 4 jumps to x = -2.
  const f = (x: Vector) => (x.get(0) < 0 ? NaN : (x.get(0) - 1) ** 2);
  const grad = (x: Vector) => Vector.from([2 * (x.get(0) - 1)]);

  it.each(['dogleg', 'line-search'] as const)(
    'backtracks from a non-finite trial point with %s',
    (globalMethod) => {
      const res = quasiNewton(f, [4], { grad, globalMethod });
      expect(res.success, res.message).toBe(true);
      expect(res.x.get(0)).toBeCloseTo(1, 4);
    }
  );

  it('fails if f is not finite at the step and there is no global strategy', () => {
    const res = quasiNewton(f, [4], { grad, globalMethod: null });
    expect(res.success).toBe(false);
    expect(res.message).toContain('f returned non-finite values at the proposed step');
    expect(res.x.get(0)).toBe(4);
    expect(res.fx).toBe(9);
  });

  it('fails if the gradient is not finite at a later iterate', () => {
    const res = quasiNewton((x) => x.get(0) ** 2, [4], {
      grad: (x) => (x.get(0) < 1 ? Vector.from([NaN]) : Vector.from([2 * x.get(0)])),
      globalMethod: null,
      gradCheck: false
    });
    expect(res.success).toBe(false);
    expect(res.message).toBe('Gradient at the current iterate contains non-finite values.');
    expect(res.gx!.allFinite()).toBe(false);
  });

  it('fails if the Hessian is not finite at a later iterate', () => {
    const res = quasiNewton((x) => x.get(0) ** 4, [4], {
      grad: (x) => Vector.from([4 * x.get(0) ** 3]),
      hess: (x) => Matrix.from([[x.get(0) < 3 ? NaN : 12 * x.get(0) ** 2]]),
      globalMethod: 'line-search'
    });
    expect(res.success).toBe(false);
    expect(res.message).toBe('Hessian at the current iterate contains non-finite values.');
    expect(res.iterations).toBeGreaterThanOrEqual(1);
  });
});

describe('quasiNewton gradient check', () => {
  it.each(Object.entries(GRADIENT_TEST_FUNCTIONS))(
    'accepts the correct gradient of %s at x0 and next to the minimizer',
    (_id, t) => {
      for (const x0 of [t.x0, t.xs.map((v) => v + 1e-3), t.xs]) {
        const res = quasiNewton(t.f, x0, { grad: t.grad, maxIter: 1 });
        expect(res.message).not.toContain('does not match');
      }
    }
  );

  it.each([
    ['wrong factor', (g: Vector) => g.mult(1.1)],
    ['wrong sign', (g: Vector) => g.mult(-1)],
    ['wrong component', (g: Vector) => g.map((v, i) => (i === 0 ? 0.5 * v : v))],
    ['missing term', (g: Vector) => g.map((v, i) => (i === 1 ? 0 : v))]
  ])('rejects a gradient with %s', (_label, corrupt) => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const res = quasiNewton(t.f, t.x0, { grad: (x) => corrupt(t.grad(x)) });
    expect(res.success).toBe(false);
    expect(res.message).toBe(
      'User-provided gradient `grad` does not match finite-difference approximation.'
    );
    expect(res.iterations).toBe(0);
    expect(res.evaluationsGradient).toBe(1);
    expect(res.x.toArray()).toEqual(t.x0);
  });

  it('is skipped with `gradCheck: false`', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    const res = quasiNewton(t.f, t.x0, {
      grad: (x) => t.grad(x).mult(1.1),
      gradCheck: false,
      maxIter: 2
    });
    expect(res.message).not.toContain('does not match');
    expect(res.iterations).toBeGreaterThan(0);
  });

  it('does not apply without `grad`', () => {
    const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;
    let calls = 0;
    quasiNewton((x) => (calls++, t.f(x)), t.x0, { maxIter: 1, gradCheck: true });
    const callsOff = calls;
    calls = 0;
    quasiNewton((x) => (calls++, t.f(x)), t.x0, { maxIter: 1, gradCheck: false });
    expect(callsOff).toBe(calls);
  });
});

describe('quasiNewton user-supplied derivative shapes', () => {
  const t = GRADIENT_TEST_FUNCTIONS.rosenbrock;

  it('throws if grad returns a vector of the wrong size', () => {
    expect(() => quasiNewton(t.f, t.x0, { grad: () => Vector.from([1, 2, 3]) })).toThrow(
      'quasiNewton: grad(x0) must have 2 components, got 3'
    );
    expect(() =>
      quasiNewton(t.f, t.x0, {
        grad: (x) => (x.get(0) === t.x0[0] ? t.grad(x) : Vector.from([1])),
        gradCheck: false
      })
    ).toThrow('quasiNewton: grad(x) must have 2 components, got 1.');
  });

  it('throws if hess returns a matrix of the wrong size', () => {
    expect(() => quasiNewton(t.f, t.x0, { hess: () => Matrix.identity(3) })).toThrow(
      'quasiNewton: hess(x0) must have dimensions 2x2, got 3x3'
    );
    expect(() => quasiNewton(t.f, t.x0, { hess: () => new Matrix(2, 3) })).toThrow(RangeError);
  });

  it('does not modify the matrix returned by hess', () => {
    const H = Matrix.from([
      [10, 2],
      [2, 5]
    ]);
    const copy = H.copy();
    quasiNewton((x) => 0.5 * x.dot(H.mulVec(x)), [1, 1], {
      hess: () => H,
      bfgsUpdate: true,
      bfgsMethod: 'unfactored',
      globalMethod: 'line-search',
      maxIter: 5
    });
    expect(H.allClose(copy, { atol: 0, rtol: 0 })).toBe(true);
  });
});
