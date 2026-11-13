import type { Vector } from '../linalg/Vector.js';

/**
 * Function that computes the accelerations for the current positions `u` and velocities `v`.
 * It must write the result into the supplied `aOut` buffer, rather than allocating, and return
 * it. It must not mutate `u` or `v`.
 */
export type AccelerationFunction = (u: Vector, v: Vector, aOut: Vector) => Vector;

/**
 * Mutable state of a velocity Verlet simulation, advanced in place by a
 * {@link VerletStepFunction}.
 *
 * All four vectors must have the same size.
 */
export interface VerletState {
  /** Positions `u(t)`. */
  u: Vector;
  /** Velocities `v(t)`. */
  v: Vector;
  /**
   * Accelerations `a(t)` consistent with the current `u` and `v`. The caller must initialize
   * them before the first step (e.g. by calling the acceleration function once); afterwards the
   * step function keeps them up to date.
   */
  a: Vector;
  /** Scratch space for `a(t + dt)`; its content between steps is meaningless. */
  aNext: Vector;
}

/** Advances a {@link VerletState} in place by one time step `dt`. */
export type VerletStepFunction = (state: VerletState, dt: number) => void;

/**
 * Creates a velocity Verlet integrator step function.
 *
 * Velocity Verlet is a second-order symplectic integrator commonly used for particle/physics
 * simulations (e.g. N-body, molecular dynamics, cloth/soft-body sims). Each call to the returned
 * `step` function advances the simulation state by one time step `dt` using the standard
 * three-part update:
 *
 *   1. u(t+dt)  = u(t) + v(t)*dt + 0.5*a(t)*dt^2
 *   2. a(t+dt)  = computeAcceleration(u(t+dt), v(t), ...)
 *   3. v(t+dt)  = v(t) + 0.5*(a(t) + a(t+dt))*dt
 *
 * Note: the acceleration in part 2 is computed using the *old* velocity v(t), not a predicted
 * v(t+dt). This is exact for velocity-independent forces (gravity, springs, etc.). If the force
 * model depends on velocity (e.g. drag/damping), this reduces the accuracy to first order, and a
 * predictor-corrector variant would be needed to restore second-order accuracy.
 *
 * The state must be initialized consistently: `state.a` has to hold the acceleration at the
 * initial `u` and `v`, because the first step uses it. The time step `dt` may vary between
 * calls, but the method is symplectic only for a constant `dt`. No validation is performed.
 *
 * @param computeAcceleration Function that computes the accelerations for the positions `u` and
 * velocities `v`, writing the result into `aOut` and returning it.
 * @returns A `step(state, dt)` function that mutates `state.u`, `state.v` and `state.a` in place
 * to advance the simulation by `dt`. `state.aNext` is used as scratch space, so it must be a
 * pre-allocated vector of the same size as the others; it is reused on every call, which avoids
 * allocations.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { createVelocityVerlet, type VerletState } from 'numerics-js/ode/verlet';
 *
 * // Free fall under constant gravity, starting from rest at u = 0.
 * const g = -9.81;
 * const step = createVelocityVerlet((u, v, aOut) => aOut.set(0, g));
 * // The initial acceleration a(0) must be provided.
 * const state: VerletState = {
 *   u: new Vector([0]),
 *   v: new Vector([0]),
 *   a: new Vector([g]),
 *   aNext: new Vector(1)
 * };
 * step(state, 0.016);
 * console.log(state.u.get(0), state.v.get(0));
 * ```
 *
 * Output:
 * ```text
 * -0.0012556800000000001 -0.15696000000000002
 * ```
 */
export function createVelocityVerlet(
  computeAcceleration: AccelerationFunction
): VerletStepFunction {
  return function step(state: VerletState, dt: number): void {
    const { u, v, a, aNext } = state;
    const ud = u.data;
    const vd = v.data;
    const ad = a.data;
    const aNextd = aNext.data;
    const n = ud.length;

    for (let i = 0; i < n; i++) {
      ud[i] += vd[i] * dt + 0.5 * ad[i] * dt * dt;
    }

    computeAcceleration(u, v, aNext);

    for (let i = 0; i < n; i++) {
      vd[i] += 0.5 * (ad[i] + aNextd[i]) * dt;
      ad[i] = aNextd[i];
    }
  };
}
