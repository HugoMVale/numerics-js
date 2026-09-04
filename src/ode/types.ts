import type { Vector } from '../array/Vector.js';
import type { Matrix } from '../array/Matrix.js';

/** Derivative function that writes its result into the provided `dydt` buffer to avoid allocation. */
export type DerivativeFunction = (t: number, y: Vector, dydt: Vector) => Vector;

/** Derivative function that returns a newly allocated result. */
export type AllocatingDerivativeFunction = (t: number, y: Vector) => Vector;

/** Fixed-step Runge-Kutta scheme used by `rungeKuttaFixed`. */
export type RungeKuttaFixedMethod = 'euler' | 'midpoint' | 'trapezoid' | 'rk4';

/** Fixed-step Runge-Kutta scheme used by `rungeKuttaAdaptive`. */
export type RungeKuttaAdaptiveMethod = 'rk23' | 'rk45';

/** Identifies which ODE solver produced an {@link OdeResult}. */
export type OdeMethod = RungeKuttaFixedMethod | RungeKuttaAdaptiveMethod;

/** Result of integrating an ODE over a span of time. */
export interface OdeResult {
    /** Which solver method produced this result. */
    method: OdeMethod;
    /** Whether the solver successfully reached the end of the integration interval. */
    success: boolean;
    /** Message providing additional information about the solver's success or failure. */
    message: string;
    /** Number of times the solver evaluated the derivative function. */
    evaluations: number;
    /** Time points at which the solution was evaluated. */
    t: Vector;
    /** Solution values, one row per time point and one column per state variable. */
    y: Matrix;
}