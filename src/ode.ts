/**
 * Ordinary differential equation solvers for non-stiff problems.
 *
 * All current solvers are explicit and are therefore suitable for non-stiff problems only.
 * Implicit methods for stiff problems are planned but not yet available.
 *
 * Choose a method based on the problem and step-size requirements:
 *
 * - Use {@link rungeKuttaAdaptive} for general-purpose integration with
 *   automatic step-size and error control.
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

export { wrapAllocatingDerivative } from './ode/adapters.js';

export type {
    DerivativeFunction,
    AllocatingDerivativeFunction,
    OdeResult,
    RungeKuttaFixedMethod,
    RungeKuttaAdaptiveMethod,
    OdeMethod,
} from './ode/types.js';

export { createVelocityVerlet } from './ode/verlet.js';
