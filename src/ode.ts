/**
 * Ordinary differential equation solvers for non-stiff and stiff problems.
 *
 * The Runge-Kutta solvers are explicit and are therefore suitable for non-stiff problems only.
 * {@link trBdf2Adaptive} is implicit and handles stiff problems.
 *
 * Choose a method based on the problem and step-size requirements:
 *
 * - Use {@link rungeKuttaAdaptive} for general-purpose integration of non-stiff problems with
 *   automatic step-size and error control.
 * - Use {@link trBdf2Adaptive} for stiff problems at moderate tolerances. It is second-order
 *   accurate and L-stable, and offers the same automatic step-size and error control.
 * - Use {@link rungeKuttaFixed} when the time grid is prescribed or a fixed step size is required.
 * - Use {@link createVelocityVerlet} for second-order mechanical systems, especially long-running
 *   conservative simulations.
 *
 * The adaptive solvers, {@link rungeKuttaAdaptive} and {@link trBdf2Adaptive}, also offer:
 *
 * - Event location: their `events` option locates zero crossings of user-defined functions along
 *   the solution, and optionally stops the integration at them; see {@link OdeEvent}.
 * - Dense output: their `denseOutput` option makes the solution evaluable at any time inside the
 *   integration interval, not only at the steps taken; see {@link OdeDenseOutput}.
 *
 * Derivative functions write their result into a solver-provided buffer, to avoid allocations;
 * see {@link DerivativeFunction}. Use {@link wrapAllocatingDerivative} to adapt a function that
 * returns a new `Vector` instead.
 *
 * @module ode
 */
export { rungeKuttaFixed } from './ode/rungeKuttaFixed.js';
export type { RungeKuttaFixedOptions } from './ode/rungeKuttaFixed.js';

export { rungeKuttaAdaptive } from './ode/rungeKuttaAdaptive.js';
export type { RungeKuttaAdaptiveOptions } from './ode/rungeKuttaAdaptive.js';

export { trBdf2Adaptive } from './ode/trBdf2Adaptive.js';
export type { TrBdf2AdaptiveOptions } from './ode/trBdf2Adaptive.js';

export { wrapAllocatingDerivative } from './ode/adapters.js';

export type { OdeDenseOutput } from './ode/denseOutput.js';

export type {
  DerivativeFunction,
  AllocatingDerivativeFunction,
  JacobianFunction,
  OdeAdaptiveOptions,
  OdeEvent,
  OdeEventDirection,
  OdeEventOccurrence,
  OdeResult,
  RungeKuttaFixedMethod,
  RungeKuttaAdaptiveMethod,
  TrBdf2Method,
  OdeMethod
} from './ode/types.js';

export { createVelocityVerlet } from './ode/verlet.js';
