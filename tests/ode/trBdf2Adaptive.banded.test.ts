import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { BandedMatrix } from '../../src/linalg/BandedMatrix.js';
import { trBdf2Adaptive } from '../../src/ode/trBdf2Adaptive.js';
import type {
  BandedJacobianFunction,
  DerivativeFunction,
  JacobianFunction,
  OdeResult
} from '../../src/ode/types.js';

// Heat equation u_t = nu * (u_{i-1} - 2 u_i + u_{i+1}) with homogeneous Dirichlet conditions: the
// Jacobian is constant and tridiagonal (kl = ku = 1).
const nu = 40;
const heat: DerivativeFunction = (_t, y, out) => {
  const n = y.size;
  for (let i = 0; i < n; i++) {
    const left = i > 0 ? y.data[i - 1] : 0;
    const right = i < n - 1 ? y.data[i + 1] : 0;
    out.data[i] = nu * (left - 2 * y.data[i] + right);
  }
  return out;
};
const heatBanded: BandedJacobianFunction = (_t, y, J) => {
  for (let i = 0; i < y.size; i++) {
    J.set(i, i, -2 * nu);
    if (i > 0) J.set(i, i - 1, nu);
    if (i < y.size - 1) J.set(i, i + 1, nu);
  }
  return J;
};
const heatDense: JacobianFunction = (_t, y, J) => {
  for (let i = 0; i < y.size; i++) {
    J.set(i, i, -2 * nu);
    if (i > 0) J.set(i, i - 1, nu);
    if (i < y.size - 1) J.set(i, i + 1, nu);
  }
  return J;
};
const mode = (n: number) =>
  Vector.from(Array.from({ length: n }, (_, i) => Math.sin((Math.PI * (i + 1)) / (n + 1))));
// The first discrete sine mode decays exactly as exp(-rate * t).
const rate = (n: number) => 2 * nu * (1 - Math.cos(Math.PI / (n + 1)));

// Nonlinear reaction-diffusion with a pentadiagonal Jacobian (fourth-order stencil, kl = ku = 2).
const pentaCoefficients = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12];
const pentaK = 20;
const penta: DerivativeFunction = (_t, y, out) => {
  const n = y.size;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let q = -2; q <= 2; q++) {
      const j = i + q;
      if (j >= 0 && j < n) s += pentaCoefficients[q + 2] * y.data[j];
    }
    out.data[i] = pentaK * s + y.data[i] * (1 - y.data[i]);
  }
  return out;
};
const pentaEntries = (y: Vector, set: (i: number, j: number, v: number) => void): void => {
  for (let i = 0; i < y.size; i++) {
    for (let q = -2; q <= 2; q++) {
      const j = i + q;
      if (j >= 0 && j < y.size) {
        set(i, j, pentaK * pentaCoefficients[q + 2] + (q === 0 ? 1 - 2 * y.data[i] : 0));
      }
    }
  }
};
const pentaBanded: BandedJacobianFunction = (_t, y, J) => {
  pentaEntries(y, (i, j, v) => J.set(i, j, v));
  return J;
};
const pentaDense: JacobianFunction = (_t, y, J) => {
  pentaEntries(y, (i, j, v) => J.set(i, j, v));
  return J;
};
const bump = (n: number) =>
  Vector.from(
    Array.from({ length: n }, (_, i) => Math.exp(-(((i + 1) / (n + 1) - 0.3) ** 2) / 0.01))
  );

// Same number of steps, at times equal to roundoff: a banded and a dense elimination of the same
// matrix differ in the last bit, which moves a step time by ~1e-13 on the longest runs.
const sameSteps = (a: OdeResult, b: OdeResult): boolean =>
  a.t.size === b.t.size &&
  Array.from(a.t.data).every(
    (v, i) => Math.abs(v - b.t.data[i]) <= 1e-10 * Math.max(1, Math.abs(v))
  );
const maxRelDiff = (a: OdeResult, b: OdeResult): number => {
  let m = 0;
  for (let i = 0; i < a.y.data.length; i++) {
    m = Math.max(m, Math.abs(a.y.data[i] - b.y.data[i]) / (1e-9 + Math.abs(a.y.data[i])));
  }
  return m;
};

describe('trBdf2Adaptive with a banded Jacobian', () => {
  describe('agreement with the dense mode', () => {
    const cases = [
      { name: 'tridiagonal heat equation', f: heat, band: 1, banded: heatBanded, dense: heatDense },
      {
        name: 'pentadiagonal reaction-diffusion',
        f: penta,
        band: 2,
        banded: pentaBanded,
        dense: pentaDense
      }
    ];
    for (const { name, f, band, banded, dense } of cases) {
      const n = 40;
      const y0 = name.startsWith('tri') ? mode(n) : bump(n);
      const options = { atol: 1e-9, rtol: 1e-6 };

      it(`${name}, analytic Jacobian: same steps, counters and solution to roundoff`, () => {
        const d = trBdf2Adaptive(f, 0, 2, y0, { ...options, jac: dense });
        const b = trBdf2Adaptive(f, 0, 2, y0, { ...options, jac: banded, kl: band, ku: band });

        expect(b.success).toBe(true);
        expect(sameSteps(b, d)).toBe(true);
        expect(b.evaluations).toBe(d.evaluations);
        expect(b.evaluationsJacobian).toBe(d.evaluationsJacobian);
        expect(b.decompositions).toBe(d.decompositions);
        expect(maxRelDiff(b, d)).toBeLessThan(1e-10);
      });

      it(`${name}, finite-difference Jacobian: same steps, fewer evaluations`, () => {
        const d = trBdf2Adaptive(f, 0, 2, y0, options);
        const b = trBdf2Adaptive(f, 0, 2, y0, { ...options, kl: band, ku: band });

        expect(b.success).toBe(true);
        expect(sameSteps(b, d)).toBe(true);
        expect(b.evaluationsJacobian).toBe(d.evaluationsJacobian);
        expect(b.decompositions).toBe(d.decompositions);
        expect(maxRelDiff(b, d)).toBeLessThan(1e-8);
        // Each refresh costs n + 1 evaluations of f in dense mode and kl + ku + 2 in banded mode.
        const saved = d.evaluationsJacobian * (n + 1 - (2 * band + 2));
        expect(d.evaluations - b.evaluations).toBe(saved);
      });
    }
  });

  it('is accurate on the discrete heat equation, whose decay rate is known exactly', () => {
    const n = 100;
    const result = trBdf2Adaptive(heat, 0, 0.05, mode(n), {
      atol: 1e-10,
      rtol: 1e-8,
      jac: heatBanded,
      kl: 1,
      ku: 1
    });
    const last = result.y.rows - 1;
    const exact = Math.exp(-rate(n) * 0.05);

    expect(result.success).toBe(true);
    for (const i of [0, 25, 49, 99]) {
      const m = Math.sin((Math.PI * (i + 1)) / (n + 1));
      expect(result.y.get(last, i)).toBeCloseTo(m * exact, 6);
    }
  });

  describe('counters', () => {
    it('counts the finite-difference evaluations: kl + ku + 2 per refresh, all in `evaluations`', () => {
      let calls = 0;
      const counted: DerivativeFunction = (t, y, out) => {
        calls++;
        return heat(t, y, out);
      };
      const n = 30;
      const result = trBdf2Adaptive(counted, 0, 1, mode(n), {
        h0: 0.1,
        hMax: 0.1,
        atol: 1e-6,
        rtol: 1e-4,
        kl: 1,
        ku: 1
      });
      const analytic = trBdf2Adaptive(heat, 0, 1, mode(n), {
        h0: 0.1,
        hMax: 0.1,
        atol: 1e-6,
        rtol: 1e-4,
        kl: 1,
        ku: 1,
        jac: heatBanded
      });

      expect(result.success).toBe(true);
      expect(result.evaluations).toBe(calls);
      expect(sameSteps(result, analytic)).toBe(true);
      expect(result.evaluationsJacobian).toBe(analytic.evaluationsJacobian);
      expect(result.evaluations - analytic.evaluations).toBe(result.evaluationsJacobian * (3 + 1));
    });

    it('calls an analytic banded Jacobian once per refresh and never evaluates f for it', () => {
      let jacCalls = 0;
      const jac: BandedJacobianFunction = (t, y, J) => {
        jacCalls++;
        return heatBanded(t, y, J);
      };
      const result = trBdf2Adaptive(heat, 0, 1, mode(20), { jac, kl: 1, ku: 1 });
      expect(result.success).toBe(true);
      expect(jacCalls).toBe(result.evaluationsJacobian);
    });
  });

  describe('Jacobian buffer', () => {
    it('hands the callback a zero-filled BandedMatrix with the declared (clamped) band', () => {
      const seen: { kl: number; ku: number; n: number; zero: boolean; isBanded: boolean }[] = [];
      const jac: BandedJacobianFunction = (t, y, J) => {
        seen.push({
          kl: J.kl,
          ku: J.ku,
          n: J.n,
          zero: J.data.every((v) => v === 0),
          isBanded: J instanceof BandedMatrix
        });
        return heatBanded(t, y, J);
      };
      // Force several refreshes: a stiff nonlinear problem with a stale Jacobian.
      const result = trBdf2Adaptive(heat, 0, 1, mode(8), { jac, kl: 1, ku: 100 });
      expect(result.success).toBe(true);
      expect(seen.length).toBeGreaterThanOrEqual(1);
      for (const s of seen) expect(s).toEqual({ kl: 1, ku: 7, n: 8, zero: true, isBanded: true });
    });

    it('throws a RangeError, even for an explicit 0, if the callback writes outside the band', () => {
      const outside: BandedJacobianFunction = (t, y, J) => {
        heatBanded(t, y, J);
        return J.set(0, 5, 0);
      };
      expect(() => trBdf2Adaptive(heat, 0, 1, mode(8), { jac: outside, kl: 1, ku: 1 })).toThrow(
        /outside the band/
      );
    });

    it('does not allocate any n x n array (a 20000-state problem would need 3.2 GB dense)', () => {
      const n = 20000;
      const result = trBdf2Adaptive(heat, 0, 1e-3, mode(n), { jac: heatBanded, kl: 1, ku: 1 });
      expect(result.success).toBe(true);
      expect(result.y.cols).toBe(n);
    });
  });

  describe('band covering the whole matrix (kl + ku >= n - 1)', () => {
    it('runs banded on the full band and agrees with the dense mode', () => {
      const n = 4;
      const options = { atol: 1e-9, rtol: 1e-6 };
      const d = trBdf2Adaptive(heat, 0, 1, mode(n), { ...options, jac: heatDense });
      // Every pair contains the tridiagonal band; most of them cover or exceed the whole 4x4 matrix.
      for (const [kl, ku] of [
        [3, 3],
        [10, 1],
        [1, 10],
        [2, 1],
        [50, 50]
      ]) {
        const band = trBdf2Adaptive(heat, 0, 1, mode(n), { ...options, kl, ku, jac: heatBanded });
        expect(sameSteps(band, d)).toBe(true);
        expect(maxRelDiff(band, d)).toBeLessThan(1e-10);
        expect(band.decompositions).toBe(d.decompositions);
      }
    });

    it('costs n + 1 evaluations per finite-difference refresh, like the dense mode', () => {
      const n = 4;
      const options = { atol: 1e-9, rtol: 1e-6 };
      const d = trBdf2Adaptive(heat, 0, 1, mode(n), options);
      const b = trBdf2Adaptive(heat, 0, 1, mode(n), { ...options, kl: 3, ku: 3 });
      expect(b.evaluations).toBe(d.evaluations);
    });

    it('handles a single state component', () => {
      const f: DerivativeFunction = (_t, y, out) => out.set([-5 * y.data[0]]);
      const result = trBdf2Adaptive(f, 0, 1, [1], {
        atol: 1e-10,
        rtol: 1e-8,
        kl: 0,
        ku: 0,
        jac: ((_t, _y, J) => J.set(0, 0, -5)) satisfies BandedJacobianFunction
      });
      expect(result.success).toBe(true);
      expect(result.y.get(result.y.rows - 1, 0)).toBeCloseTo(Math.exp(-5), 6);
    });
  });

  describe('events and dense output', () => {
    const n = 30;
    const options = { atol: 1e-9, rtol: 1e-7 };
    const events = [
      { fn: (_t: number, y: Vector) => y.data[14] - 0.5, direction: 'falling' as const },
      { fn: (_t: number, y: Vector) => y.data[3] - 0.1, terminal: true }
    ];

    it('locates the same events and ends at the same terminal event as the dense mode', () => {
      const d = trBdf2Adaptive(heat, 0, 5, mode(n), { ...options, events, jac: heatDense });
      const b = trBdf2Adaptive(heat, 0, 5, mode(n), {
        ...options,
        events,
        jac: heatBanded,
        kl: 1,
        ku: 1
      });

      expect(b.success).toBe(true);
      expect(b.events!.length).toBe(2);
      expect(b.message).toBe('Integration stopped by a terminal event.');
      expect(b.events!.map((e) => e.t)).toEqual(d.events!.map((e) => e.t));
      expect(b.t.size).toBe(d.t.size);
      expect(b.message).toBe(d.message);
    });

    it('evaluates dense output identically to the dense mode and close to the exact decay', () => {
      const d = trBdf2Adaptive(heat, 0, 0.02, mode(n), {
        ...options,
        denseOutput: true,
        jac: heatDense
      });
      const b = trBdf2Adaptive(heat, 0, 0.02, mode(n), {
        ...options,
        denseOutput: true,
        kl: 1,
        ku: 1,
        jac: heatBanded
      });
      for (const time of [0, 0.003, 0.0111, 0.02]) {
        const yb = b.denseOutput!.eval(time) as Vector;
        const yd = d.denseOutput!.eval(time) as Vector;
        const m = Math.sin((Math.PI * 15) / (n + 1));
        expect(Math.abs(yb.get(14) - yd.get(14))).toBeLessThan(1e-12);
        expect(yb.get(14)).toBeCloseTo(m * Math.exp(-rate(n) * time), 4);
      }
    });
  });

  describe('failure paths keep their retry behavior', () => {
    it('returns an unsuccessful result when the Newton iteration cannot converge', () => {
      const f: DerivativeFunction = (t, y, out) => {
        for (let i = 0; i < y.size; i++) out.data[i] = t > 0.5 ? Number.NaN : -y.data[i];
        return out;
      };
      for (const useJac of [false, true]) {
        const result = trBdf2Adaptive(f, 0, 1, [1, 1, 1], {
          hMin: 1e-6,
          kl: 0,
          ku: 0,
          ...(useJac ? { jac: ((_t, _y, J) => J) as BandedJacobianFunction } : {})
        });
        expect(result.success).toBe(false);
        expect(result.message).toMatch(/underflowed below hMin/);
        expect(result.message).toMatch(/Newton/);
        expect(result.t.data[result.t.size - 1]).toBeLessThanOrEqual(0.5);
      }
    });

    it('treats NaN in the iteration matrix as a failed step, without throwing', () => {
      const nanJac: BandedJacobianFunction = (_t, _y, J) => {
        J.data.fill(Number.NaN);
        return J;
      };
      const result = trBdf2Adaptive(heat, 0, 1, mode(6), {
        jac: nanJac,
        kl: 1,
        ku: 1,
        hMin: 1e-9
      });
      expect(result.success).toBe(false);
      expect(result.message).toMatch(/underflowed below hMin/);
      expect(result.decompositions).toBe(0);
    });

    it('treats an infinite entry in the iteration matrix as a failed step, without throwing', () => {
      const infJac: BandedJacobianFunction = (_t, _y, J) => {
        J.data.fill(Number.POSITIVE_INFINITY);
        return J;
      };
      const result = trBdf2Adaptive(heat, 0, 1, mode(6), {
        jac: infJac,
        kl: 1,
        ku: 1,
        hMin: 1e-9
      });
      expect(result.success).toBe(false);
      expect(result.message).toMatch(/underflowed below hMin/);
      expect(result.decompositions).toBe(0);
    });

    it('retries with a smaller step when the iteration matrix has an exactly zero pivot', () => {
      // With h = 0.25 the matrix is I - c J with c = D h; J(0, 0) = 1 / c makes its first pivot
      // exactly 0 (a zero column of a 3x3 tridiagonal matrix), so the plain-Error path of lu() runs.
      const D = (2 - Math.SQRT2) / 2;
      const c = D * 0.25;
      const lambda = 1 / c;
      expect(1 - c * lambda).toBe(0); // the precondition of this test
      const zero: DerivativeFunction = (_t, _y, out) => out.fill(0);
      const jac: BandedJacobianFunction = (_t, _y, J) => J.set(0, 0, lambda);
      const result = trBdf2Adaptive(zero, 0, 1, [1, 1, 1], {
        h0: 0.25,
        hMax: 0.25,
        jac,
        kl: 1,
        ku: 1
      });

      expect(result.success).toBe(true);
      expect(result.t.data[1]).toBe(0.125); // the singular step was halved, not accepted
      expect(result.decompositions).toBeGreaterThanOrEqual(1);
    });

    it('refreshes a poor banded Jacobian when the Newton iteration struggles', () => {
      let calls = 0;
      const lambdaJac: BandedJacobianFunction = (t, y, J) => {
        calls++;
        return calls > 1 ? heatBanded(t, y, J) : J; // the first call leaves J = 0
      };
      const result = trBdf2Adaptive(heat, 0, 0.5, mode(10), {
        atol: 1e-8,
        rtol: 1e-5,
        jac: lambdaJac,
        kl: 1,
        ku: 1
      });
      expect(result.success).toBe(true);
      expect(result.evaluationsJacobian).toBeGreaterThanOrEqual(2);
    });
  });

  describe('validation', () => {
    const y0 = [1, 2, 3];
    it('requires kl and ku together', () => {
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: 1 })).toThrow(
        /kl and ku must be given together\./
      );
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { ku: 1 })).toThrow(RangeError);
    });

    it('rejects a bandwidth that is not a non-negative integer', () => {
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: -1, ku: 1 })).toThrow(RangeError);
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: 1, ku: 1.5 })).toThrow(
        /ku must be a non-negative integer, got 1\.5\./
      );
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: Number.NaN, ku: 1 })).toThrow(RangeError);
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: 1, ku: Infinity })).toThrow(RangeError);
    });

    it('rejects a bandwidth that is not a number', () => {
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: '1' as never, ku: 1 })).toThrow(
        /kl must be a number, got string\./
      );
      expect(() => trBdf2Adaptive(heat, 0, 1, y0, { kl: 1, ku: null as never })).toThrow(TypeError);
    });

    it('validates the band even when tEnd === t0', () => {
      expect(() => trBdf2Adaptive(heat, 0, 0, y0, { kl: 1 })).toThrow(RangeError);
    });
  });
});
