/** Plain `number[]` or `Float64Array` holding a vector of positions, velocities or accelerations. */
export type VectorArray = number[] | Float64Array;

/**
 * Function that computes the accelerations for the current positions `u` and velocities `v`,
 * writing them in place into `aOut`. It must not mutate `u` or `v`.
 */
export type ComputeAccelerationFn = (u: VectorArray, v: VectorArray, aOut: VectorArray) => void;

/**
 * Mutable state of a velocity Verlet simulation, advanced in place by a {@link StepFn}.
 *
 * All four arrays must have the same length.
 */
export interface VerletState {
  /** Positions `u(t)`. */
  u: VectorArray;
  /** Velocities `v(t)`. */
  v: VectorArray;
  /**
   * Accelerations `a(t)` consistent with the current `u` and `v`. The caller must initialize
   * them before the first step (e.g. by calling the acceleration function once); afterwards the
   * step function keeps them up to date.
   */
  a: VectorArray;
  /** Scratch space for `a(t + dt)`; its content between steps is meaningless. */
  aNext: VectorArray;
}

/** Advances a {@link VerletState} in place by one time step `dt`. */
export type StepFn = (state: VerletState, dt: number) => void;

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
 * velocities `v`, writing the result in place into `aOut`.
 * @returns A `step(state, dt)` function that mutates `state.u`, `state.v` and `state.a` in place
 * to advance the simulation by `dt`. `state.aNext` is used as scratch space, so it must be a
 * pre-allocated array of the same length as the others; it is reused on every call, which avoids
 * allocations.
 *
 * @example
 * ```ts
 * import { createVelocityVerlet, type VerletState } from 'numerics-js/ode/verlet';
 *
 * // Free fall under constant gravity, starting from rest at u = 0.
 * const g = -9.81;
 * const step = createVelocityVerlet((u, v, aOut) => {
 *   for (let i = 0; i < aOut.length; i++) aOut[i] = g;
 * });
 * // The initial acceleration a(0) must be provided.
 * const state: VerletState = { u: [0], v: [0], a: [g], aNext: [0] };
 * step(state, 0.016);
 * console.log(state.u[0], state.v[0]);
 * ```
 *
 * Output:
 * ```text
 * -0.0012556800000000001 -0.15696000000000002
 * ```
 */
export function createVelocityVerlet(computeAcceleration: ComputeAccelerationFn): StepFn {
  return function step(state: VerletState, dt: number): void {
    const { u, v, a, aNext } = state;
    const n = u.length;

    for (let i = 0; i < n; i++) {
      u[i] += v[i] * dt + 0.5 * a[i] * dt * dt;
    }

    computeAcceleration(u, v, aNext);

    for (let i = 0; i < n; i++) {
      v[i] += 0.5 * (a[i] + aNext[i]) * dt;
      a[i] = aNext[i];
    }
  };
}
