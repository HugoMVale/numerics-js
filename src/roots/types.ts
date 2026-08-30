/**
 * Result of a scalar root-finding routine.
 */
export interface RootResult {
    /** Name of the method that produced this result (e.g. `'bisection'`). */
    method: string;
    /** Number of function evaluations performed. */
    evaluations: number;
    /** Approximate root. */
    x: number;
    /** Function value at `x`. */
    fx: number;
}

/**
 * Options for the bisection method.
 */
export interface BisectionOptions {
    /** Stop when the interval half-width is below this. Defaults to `1e-8`. */
    tolX?: number;
    /** Maximum number of iterations. Defaults to `100`. */
    maxIter?: number;
}

/**
 * Options for the secant method.
 */
export interface SecantOptions {
    /** Stop when |x1 - x0| (the step size) is below this. Defaults to `1e-8`. */
    tolX?: number;
    /** Maximum number of iterations. Defaults to `100`. */
    maxIter?: number;
}

/**
 * Options for Brent's method.
 */
export interface BrentOptions {
    /** Stop when the bracket half-width is below this (absolute x tolerance). Defaults to `1e-8`. */
    tolX?: number;
    /** Stop when |fn(x)| is below this (absolute function-value tolerance). Defaults to `1e-8`. */
    tolF?: number;
    /** Maximum number of iterations. Defaults to `100`. */
    maxIter?: number;
}