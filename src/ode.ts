/**
 * Ordinary differential equation solvers for non-stiff and stiff problems.
 *
 * The Runge-Kutta solvers are explicit and are therefore suitable for non-stiff
 * problems only. {@link trBdf2Adaptive} is implicit and handles stiff problems.
 *
 * Choose a method based on the problem and step-size requirements:
 *
 * - Use {@link rungeKuttaAdaptive} for general-purpose integration of non-stiff
 *   problems with automatic step-size and error control.
 * - Use {@link trBdf2Adaptive} for stiff problems at moderate tolerances. It is
 *   second-order accurate and L-stable, and offers the same automatic step-size
 *   and error control.
 * - Locate zero crossings of user-defined functions along the solution, and optionally stop at
 *   them, with the `events` option of {@link rungeKuttaAdaptive} or {@link trBdf2Adaptive}; see
 *   {@link OdeEvent}.
 * - Use {@link rungeKuttaFixed} when the time grid is prescribed or a fixed
 *   step size is required.
 * - Use {@link createVelocityVerlet} for second-order mechanical systems,
 *   especially long-running conservative simulations.
 *
 * @module ode
 */
export { rungeKuttaFixed } from './ode/rungeKuttaFixed.js';

export { rungeKuttaAdaptive } from './ode/rungeKuttaAdaptive.js';
export type { RungeKuttaAdaptiveOptions } from './ode/rungeKuttaAdaptive.js';

export { trBdf2Adaptive } from './ode/trBdf2Adaptive.js';
export type { TrBdf2AdaptiveOptions } from './ode/trBdf2Adaptive.js';

export { wrapAllocatingDerivative } from './ode/adapters.js';

export type {
    DerivativeFunction,
    AllocatingDerivativeFunction,
    JacobianFunction,
    OdeEvent,
    OdeEventDirection,
    OdeEventOccurrence,
    OdeResult,
    RungeKuttaFixedMethod,
    RungeKuttaAdaptiveMethod,
    TrBdf2Method,
    OdeMethod,
} from './ode/types.js';

export { createVelocityVerlet } from './ode/verlet.js';
