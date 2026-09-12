import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';

// -----------------------------------------------------------------
// Results section.
// -----------------------------------------------------------------

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
    evaluationsFunction: number;
    /** Number of user-supplied Jacobian evaluations. */
    evaluationsJacobian: number;
    /** Number of outer quasi-Newton iterations performed. */
    iterations: number;
    /** The solution vector (or best estimate, if not converged). */
    x: Vector;
    /** Function (residual) at the returned solution. */
    fx: Vector;
    /** Last evaluated or estimated Jacobian matrix. */
    Jx: Matrix | null;
}

// -----------------------------------------------------------------
// Global methods section.
// -----------------------------------------------------------------

/**
 * Global strategy used to improve convergence from remote starting points.
 *
 * - `'line-search'`: the search direction is the quasi-Newton step; the step
 *   length is determined by backtracking until the Armijo condition holds.
 * - `null`: no global strategy; the full quasi-Newton step is taken as-is.
 */
export type GlobalMethod = 'line-search' | null;

/** 
 * Everything a global-step strategy needs to compute the next iterate. 
 * 
 * @internal
 */
export interface GlobalStepContext {
    /** Objective for global methods: `x ↦ [f(x)=0.5*||sclf*F(x)||², F(x)]`. */
    fN: (x: Vector) => [number, Vector];
    /** Quasi-Newton step direction (unit step; strategies may rescale it). */
    p: Vector;
    /** Current iterate. */
    xc: Vector;
    /** Current objective function value, `f(xc)`. */
    fc: number;
    /** Current gradient of the objective function, `∇f(xc)`. */
    gc: Vector;
    /**
     * Upper-triangular factor associated with the current step (from the QR
     * decomposition of the scaled Jacobian, or — if that was ill-conditioned
     * — from the regularized Cholesky solve). Unused by `line-search`, but
     * available for other global-step strategies that may need it.
     */
    R: Matrix;
    /** Step-size tolerance. */
    tolx: number;
    /** Scaling factors for `x`. */
    sclx: Vector;
    /** Maximum allowed step length. */
    maxLen: number;
}


/** Outcome of a global-step strategy. */
export interface GlobalStepResult {
    /** Whether a step satisfying the strategy's acceptance condition was found. */
    success: boolean;
    /** Whether the accepted step was (on the first sub-iteration) the maximum allowed length. */
    wasMaxStep: boolean;
    /** Number of extra `f` evaluations performed by the strategy. */
    nFev: number;
    /** The proposed next iterate. */
    xp: Vector;
    /** The objective function value evaluated at the new vector `xp`. */
    fp: number;
    /** The vector root function value at `xp` (only populated if the objective function returns a tuple). */
    Fp: Vector;
}

export type GlobalStepStrategy = (ctx: GlobalStepContext) => GlobalStepResult;