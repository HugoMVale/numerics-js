import type { Array1D } from '../array/array1d';
import type { Array2D } from '../array/array2d';

/** Derivative function that writes its result into the provided `dydt` buffer to avoid allocation. */
export type DerivativeFunction = (t: number, y: Array1D, dydt: Array1D) => Array1D;

/** Derivative function that returns a newly allocated result. */
export type AllocatingDerivativeFunction = (t: number, y: Array1D) => Array1D;

/** Fixed-step Runge-Kutta scheme used by `rungeKuttaFixed`. */
export type RungeKuttaFixedMethod = 'euler' | 'midpoint' | 'trapezoid' | 'rk4';

/** Fixed-step Runge-Kutta scheme used by `rungeKuttaAdaptive`. */
export type RungeKuttaAdaptiveMethod = 'rk45' | 'rk23';

/** Identifies which ODE solver produced an {@link OdeResult}. */
export type OdeMethod = RungeKuttaFixedMethod | RungeKuttaAdaptiveMethod;

/** Result of integrating an ODE over a span of time. */
export interface OdeResult {
    /** Time points at which the solution was evaluated. */
    t: Array1D;
    /** Solution values, one row per time point and one column per state variable. */
    y: Array2D;
    /** Number of times the solver evaluated the derivative function. */
    evaluations: number;
    /** Which solver method produced this result. */
    method: OdeMethod;
}