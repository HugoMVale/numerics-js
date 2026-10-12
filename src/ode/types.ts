import type { Vector } from '../linalg/Vector.js';
import type { Matrix } from '../linalg/Matrix.js';

/** Derivative function that writes its result into the provided `dydt` buffer to avoid allocation. */
export type DerivativeFunction = (t: number, y: Vector, dydt: Vector) => Vector;

/** Derivative function that returns a newly allocated result. */
export type AllocatingDerivativeFunction = (t: number, y: Vector) => Vector;

/** Fixed-step Runge-Kutta scheme used by `rungeKuttaFixed`. */
export type RungeKuttaFixedMethod = 'euler' | 'midpoint' | 'trapezoid' | 'rk4';

/** Fixed-step Runge-Kutta scheme used by `rungeKuttaAdaptive`. */
export type RungeKuttaAdaptiveMethod = 'rk23' | 'rk45';

/** Implicit scheme used by `trBdf2Adaptive`: the L-stable, 2nd-order TR-BDF2 method. */
export type TrBdf2Method = 'trbdf2';

/** Identifies which ODE solver produced an {@link OdeResult}. */
export type OdeMethod = RungeKuttaFixedMethod | RungeKuttaAdaptiveMethod | TrBdf2Method;

/**
 * Jacobian function `J = ∂f/∂y` of an ODE right-hand side `dy/dt = f(t, y)`.
 *
 * It must write the Jacobian evaluated at `(t, y)` into the provided `J` buffer,
 * with `J.get(i, j) = ∂f_i/∂y_j`. The buffer is an `n x n` matrix, where `n` is the
 * dimension of `y`, and is zero-filled before each call, so only non-zero entries
 * need to be written. Neither `t` nor `y` may be mutated.
 */
export type JacobianFunction = (t: number, y: Vector, J: Matrix) => Matrix;

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
    /**
     * Number of Jacobian refreshes, whether through a user-supplied {@link JacobianFunction}
     * or finite differences. Derivative evaluations spent on finite-difference Jacobians
     * are included in `evaluations`. Always `0` for explicit solvers.
     */
    evaluationsJacobian: number;
    /** Number of LU factorizations of the Newton iteration matrix. Always `0` for explicit solvers. */
    decompositions: number;
}
