/**
 * Result of a scalar root-finding routine.
 */
export interface RootResult {
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

