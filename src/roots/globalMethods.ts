import { Vector } from "../linalg/Vector.js";
import { Matrix } from "../linalg/Matrix.js";

/**
 * Represents the results of a line search iteration.
 */
export interface LineSearchResult {
    /** Indicates whether the line search successfully found a step satisfying the Armijo condition. */
    success: boolean;
    /** True if the step taken was the maximum allowed step size (`maxLen`) on the first iteration. */
    isMaxStep: boolean;
    /** The total number of objective function evaluations performed during the search. */
    nfEval: number;
    /** The updated variable vector after taking the step. */
    xp: Vector;
    /** The objective function value evaluated at the new vector `xp`. */
    fp: number;
    /** The vector root function value at `xp` (only populated if the objective function returns a tuple). */
    Fp: Vector;
}

/**
 * Perform a line search.
 *
 * This function performs a line search along the quasi-Newton direction to find a step
 * size that satisfies the Armijo condition.
 *
 * **References**
 *
 * *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *     Optimization and Nonlinear Equations", SIAM, 1996.
 *
 * @param f Objective function. For compatibility with optimization and root-finding algorithms, `f` can return either a scalar objective function value or a tuple including the scaled norm of the vector root function and the vector root function itself.
 * @param p Quasi-Newton step.
 * @param xc Current value of the variable vector.
 * @param fc Current objective function value, `f(xc)`.
 * @param gc Current gradient of the objective function, `∇f(xc)`.
 * @param tolx Tolerance for the step size.
 * @param sclx Scaling factors for `x`.
 * @param maxLen Maximum step length.
 * @returns A `LineSearchResult` object containing the updated state and evaluation metrics.
 */
export function lineSearch(
    f: (x: Vector) => number | [number, Vector],
    p: Vector,
    xc: Vector,
    fc: number,
    gc: Vector,
    tolx: number,
    sclx: Vector,
    maxLen: number,
): LineSearchResult {
    let nfEval = 0;
    let success = false;
    let isMaxStep = false;
    const sqrtEps = Math.sqrt(Number.EPSILON);

    let newtLen = sclx.mult(p).norm();
    if (newtLen > maxLen) {
        p = p.mult(maxLen / newtLen);
        newtLen = maxLen;
    }

    const slope = gc.dot(p);
    const alpha = 1e-4;
    let lambda = 1.0;

    const maxDenominator = p.abs().div(
        xc.abs().map((val, idx) => Math.max(val, 1 / sclx.get(idx)))
    ).max();
    const lambdaMin = tolx / maxDenominator;

    const A = new Matrix(2, 2);
    const B = new Vector(2);
    let lambdaPrev = NaN;
    let fpPrev = NaN;
    let Fp = new Vector(0);

    let first = true;
    let xp = xc;
    let fp = fc;
    let lambdaTemp = 0;

    while (true) {
        xp = xc.add(p.mult(lambda));

        const res = f(xp);
        nfEval += 1;

        if (Array.isArray(res)) {
            fp = res[0];
            Fp = res[1];
        } else {
            fp = res;
        }

        if (fp <= fc + alpha * lambda * slope) {
            success = true;
            if (first && newtLen > 0.99 * maxLen) {
                isMaxStep = true;
            }
            break;
        } else if (lambda < lambdaMin) {
            success = false;
            xp = xc;
            break;
        } else {
            if (first) {
                lambdaTemp = -slope / (2 * (fp - fc - slope));
                first = false;
            } else {
                const l2 = lambda * lambda;
                const lp2 = lambdaPrev * lambdaPrev;

                A.set(0, 0, 1 / l2);
                A.set(0, 1, -1 / lp2);
                A.set(1, 0, -lambdaPrev / l2);
                A.set(1, 1, lambda / lp2);

                B.set(0, fp - fc - lambda * slope);
                B.set(1, fpPrev - fc - lambdaPrev * slope);

                const coef = A.mulVec(B).mult(1 / (lambda - lambdaPrev));
                const a = coef.get(0);
                const b = coef.get(1);

                if (Math.abs(a) < sqrtEps) {
                    lambdaTemp = -slope / (2 * b);
                } else {
                    lambdaTemp = (-b + Math.sqrt(b * b - 3 * a * slope)) / (3 * a);
                }
                lambdaTemp = Math.min(lambdaTemp, 0.5 * lambda);
            }
            lambdaPrev = lambda;
            fpPrev = fp;
            lambda = Math.max(0.1 * lambda, lambdaTemp);
        }
    }

    return { success, isMaxStep, nfEval, xp, fp, Fp };
}