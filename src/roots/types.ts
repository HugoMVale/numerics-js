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
 * - `'dogleg'`: a trust-region strategy that interpolates between the
 *   Cauchy (steepest-descent) point and the quasi-Newton step, subject to
 *   a trust-region radius that is adapted from iteration to iteration.
 * - `null`: no global strategy; the full quasi-Newton step is taken as-is.
 */
export type GlobalMethod = 'line-search' | 'dogleg' | null;

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
    /**
     * Current trust-region radius. Only used by `dogleg`; ignored by
     * `line-search`. `dogleg` is stateful across outer iterations: the
     * caller must persist this value between calls, seeding each call's
     * `trustLen` with the previous call's `GlobalStepResult.trustLen`.
     * Omit (or pass `0`/a non-positive value) on the very first call —
     * `dogleg` treats that as "not yet initialized" and picks an initial
     * radius itself.
     */
    trustLen?: number;
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
    /**
     * The (possibly updated) trust-region radius. For `line-search` this
     * simply echoes back `ctx.trustLen` unchanged (or `0` if it wasn't
     * set); for `dogleg` this is the radius to feed into `trustLen` on the
     * next call.
     */
    trustLen: number;
}

export type GlobalStepStrategy = (ctx: GlobalStepContext) => GlobalStepResult;