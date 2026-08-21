/**
 * Creates a velocity Verlet integrator step function.
 *
 * Velocity Verlet is a second-order symplectic integrator commonly used
 * for particle/physics simulations (e.g. N-body, molecular dynamics,
 * cloth/soft-body sims). Each call to the returned `step` function advances
 * the simulation state by one timestep `dt` using the standard 3-stage update:
 *
 *   1. u(t+dt)  = u(t) + v(t)*dt + 0.5*a(t)*dt^2
 *   2. a(t+dt)  = computeAcceleration(u(t+dt), v(t), ...)
 *   3. v(t+dt)  = v(t) + 0.5*(a(t) + a(t+dt))*dt
 *
 * Note: acceleration in step 2 is computed using the *old* velocity v(t),
 * not a predicted v(t+dt). This is exact for velocity-independent forces
 * (gravity, springs, etc.). If your force model depends on velocity
 * (e.g. drag/damping), this introduces a first-order error and you'd need
 * a predictor-corrector variant to restore second-order accuracy.
 *
 * @param {(u: number[]|Float64Array, v: number[]|Float64Array, aOut: number[]|Float64Array) => void} computeAcceleration
 *   Function that computes acceleration given current positions `u` and
 *   velocities `v`, writing the result in-place into `aOut`.
 *
 * @returns {(state: {u: number[]|Float64Array, v: number[]|Float64Array, a: number[]|Float64Array, aNext: number[]|Float64Array}, dt: number) => void}
 *   A `step(state, dt)` function that mutates `state.u`, `state.v`, and
 *   `state.a` in place to advance the simulation by `dt`. `state.aNext` is
 *   used as scratch space and is expected to be a pre-allocated array of
 *   the same length as the others (reused every call to avoid allocations).
 *
 * @example
 * const step = createVelocityVerlet((u, v, aOut) => {
 *   // e.g. constant gravity
 *   for (let i = 0; i < aOut.length; i++) aOut[i] = -9.81;
 * });
 * const state = { u: [0], v: [0], a: [0], aNext: [0] };
 * step(state, 0.016);
 */
export function createVelocityVerlet(computeAcceleration) {
    return function step(state, dt) {
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