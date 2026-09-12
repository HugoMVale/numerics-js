import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';

/**
 * Result of a scalar root-finding routine.
 */
export interface ScalarRootResult {
    /** Name of the method that produced this result (e.g. `'bisection'`). */
    method: string;
    /** Whether the root-finding was successful. */
    success: boolean;
    /** Message describing the result or error. */
    message: string;
    /** Number of function evaluations performed. */
    evaluations: number;
    /** Approximate root. */
    x: number;
    /** Function value at `x`. */
    fx: number;
}

/**
 * Result of a vector root-finding routine.
 */
export interface VectorRootResult {
    /** Descriptive method name, including the global strategy and update rule used. */
    method: string;
    /** Whether the root-finding was successful. */
    success: boolean;
    /** Message describing the result or error. */
    message: string;
    /** Number of function evaluations. */
    nfEval: number;
    /** Number of user-supplied Jacobian evaluations. */
    njEval: number;
    /** Number of outer quasi-Newton iterations performed. */
    niter: number;
    /** The solution vector (or best estimate, if not converged). */
    x: Vector;
    /** Function (residual) at the returned solution. */
    fx: Vector;
    /** Last evaluated or estimated Jacobian matrix. */
    Jx: Matrix | null;
}

