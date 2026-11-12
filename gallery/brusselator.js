// Numerics of chamber VI: the Brusselator reaction–diffusion system on a ring of cells, kept free
// of any DOM so it can also be run (and checked) in Node.
//
//   du/dt = A − (B+1) u + u² v + Du ∂²u/∂x²
//   dv/dt = B u − u² v          + Dv ∂²v/∂x²
//
// Discretising x (the method of lines) turns the PDE into 2N coupled ODEs. The reaction terms are
// nonlinear and the diffusion terms are stiff on a fine grid, so the system is advanced with the
// implicit ode.trBdf2Adaptive and an analytic Jacobian.
import { ode } from 'numerics-js';

export const N = 72;

/**
 * Parameter sets. Spatially uniform steady state: (u*, v*) = (A, B/A).
 *  - Hopf (oscillation) when B > 1 + A².
 *  - Turing (stationary spots) when B > (1 + A·√(Du/Dv))² while B < 1 + A², needs Dv ≫ Du.
 */
export const REGIMES = {
  // rate: simulated time units per second of wall clock; row: simulated time per kymograph row.
  waves: { label: 'Waves', A: 1, B: 3, Du: 0.6, Dv: 0.6, L: 40, rate: 2.6, row: 0.12 },
  spots: { label: 'Spots', A: 2, B: 4.2, Du: 1, Dv: 8, L: 50, rate: 9, row: 0.35 }
};

export const hopfB = (A) => 1 + A * A;
export const turingB = (A, Du, Dv) => (1 + A * Math.sqrt(Du / Dv)) ** 2;

/**
 * The limit cycle of a single, well-stirred cell (no diffusion). It is found by letting
 * rungeKuttaAdaptive settle, then asking it for upward crossings of u = A (an event), which give the
 * period, and for the dense output over one period, which lets us sample the cycle at any phase.
 */
export function limitCycle(A, B, samples = 160) {
  const f = (_t, y, d) => {
    const u = y.data[0];
    const v = y.data[1];
    d.data[0] = A - (B + 1) * u + u * u * v;
    d.data[1] = B * u - u * u * v;
    return d;
  };
  const tol = { rtol: 1e-9, atol: 1e-11 };
  const settle = ode.rungeKuttaAdaptive('rk45', f, 0, 120, [A + 0.5, B / A], tol);
  const k = settle.t.size - 1;
  const y1 = [settle.y.get(k, 0), settle.y.get(k, 1)];
  const run = ode.rungeKuttaAdaptive('rk45', f, 0, 60, y1, {
    ...tol,
    denseOutput: true,
    events: [{ fn: (_t, y) => y.data[0] - A, direction: 'rising' }]
  });
  const hits = run.events.map((e) => e.t);
  let amplitude = 0;
  for (let i = 0; i < run.t.size; i++)
    amplitude = Math.max(amplitude, Math.abs(run.y.get(i, 0) - A));
  // no oscillation: the uniform state is stable (and "crossings" would only be round-off noise)
  if (hits.length < 2 || amplitude < 1e-3) return null;
  const t0 = hits[0];
  const period = hits[1] - hits[0];
  const grid = Array.from({ length: samples + 1 }, (_, i) => t0 + (period * i) / samples);
  const pts = run.denseOutput.eval(grid);
  const path = new Float64Array(2 * (samples + 1));
  let umax = -Infinity;
  let vmax = -Infinity;
  for (let i = 0; i <= samples; i++) {
    path[2 * i] = pts.get(i, 0);
    path[2 * i + 1] = pts.get(i, 1);
    umax = Math.max(umax, path[2 * i]);
    vmax = Math.max(vmax, path[2 * i + 1]);
  }
  return { period, path, samples, umax, vmax };
}

/** A ring of N cells advanced with TR-BDF2. State layout: [u_0…u_{N−1}, v_0…v_{N−1}]. */
export function createField(regime = 'waves') {
  const p = { ...REGIMES[regime] };
  const y = new Float64Array(2 * N);
  let cycle = null;
  let dx2 = (p.L / N) ** 2;

  const rhs = (_t, s, d) => {
    const a = p.Du / dx2;
    const b = p.Dv / dx2;
    const sd = s.data;
    const dd = d.data;
    for (let i = 0; i < N; i++) {
      const l = i === 0 ? N - 1 : i - 1;
      const r = i === N - 1 ? 0 : i + 1;
      const u = sd[i];
      const v = sd[N + i];
      const q = u * u * v;
      dd[i] = p.A - (p.B + 1) * u + q + a * (sd[l] - 2 * u + sd[r]);
      dd[N + i] = p.B * u - q + b * (sd[N + l] - 2 * v + sd[N + r]);
    }
    return d;
  };
  const jac = (_t, s, J) => {
    const a = p.Du / dx2;
    const b = p.Dv / dx2;
    const sd = s.data;
    for (let i = 0; i < N; i++) {
      const l = i === 0 ? N - 1 : i - 1;
      const r = i === N - 1 ? 0 : i + 1;
      const u = sd[i];
      const v = sd[N + i];
      J.set(i, i, -(p.B + 1) + 2 * u * v - 2 * a);
      J.set(i, N + i, u * u);
      J.set(i, l, a);
      J.set(i, r, a);
      J.set(N + i, i, p.B - 2 * u * v);
      J.set(N + i, N + i, -u * u - 2 * b);
      J.set(N + i, N + l, b);
      J.set(N + i, N + r, b);
    }
    return J;
  };

  const field = {
    N,
    y,
    params: p,
    get cycle() {
      return cycle;
    },
    /** Switch regime (or tweak a parameter) and re-seed the field. */
    configure(name, overrides = {}) {
      Object.assign(p, REGIMES[name], overrides);
      dx2 = (p.L / N) ** 2;
      cycle = limitCycle(p.A, p.B);
    },
    /** Retune B on the fly without re-seeding. */
    setB(B) {
      p.B = B;
      cycle = limitCycle(p.A, p.B);
    },
    /**
     * One ring winding once around the limit cycle: cell i sits at phase i/N of the cycle, so a
     * travelling wave circulates indefinitely (the winding number cannot change smoothly).
     */
    seedWaves() {
      if (!cycle) {
        for (let i = 0; i < N; i++) {
          y[i] = p.A;
          y[N + i] = p.B / p.A;
        }
        return;
      }
      for (let i = 0; i < N; i++) {
        const k = Math.round((i / N) * cycle.samples) % cycle.samples;
        y[i] = cycle.path[2 * k];
        y[N + i] = cycle.path[2 * k + 1];
      }
    },
    /** The uniform steady state with small seeded noise: Turing spots grow out of it. */
    seedNoise(rnd, amp = 0.05) {
      for (let i = 0; i < N; i++) {
        y[i] = p.A + amp * (rnd() - 0.5);
        y[N + i] = p.B / p.A + amp * (rnd() - 0.5);
      }
    },
    /** Add a Gaussian bump of width w (in cells) to u around cell x0. */
    kick(x0, amp, w = 2.2) {
      for (let i = 0; i < N; i++) {
        let d = Math.abs(i - x0);
        d = Math.min(d, N - d);
        y[i] = Math.max(0, y[i] + amp * Math.exp(-(d * d) / (2 * w * w)));
      }
    },
    /** Advance the whole ring by dt and report what the solver did. */
    step(dt) {
      const res = ode.trBdf2Adaptive(rhs, 0, dt, y, {
        rtol: 1e-4,
        atol: 1e-6,
        jac
      });
      const last = res.t.size - 1;
      for (let k = 0; k < 2 * N; k++) y[k] = res.y.get(last, k);
      return {
        success: res.success,
        steps: last,
        evaluations: res.evaluations,
        decompositions: res.decompositions
      };
    }
  };
  field.configure(regime);
  return field;
}
