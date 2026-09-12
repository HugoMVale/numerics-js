import { Vector } from "../linalg/Vector.js";
import { Matrix } from "../linalg/Matrix.js";
import type { GlobalStepContext, GlobalStepResult } from "./types.js";

const SQRT_EPS = Math.sqrt(Number.EPSILON);

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
 * @param ctx Global-step context containing the objective, current iterate, search direction, and scaling information.
 * @returns A `GlobalStepResult` object containing the updated state and evaluation metrics.
 */
export function lineSearch(ctx: GlobalStepContext): GlobalStepResult {
    const { fN, p, xc, fc, gc, tolx, sclx, maxLen } = ctx;
    let nFev = 0;
    let success = false;
    let wasMaxStep = false;

    let newtLen = sclx.mult(p).norm();
    if (newtLen > maxLen) {
        p.multSelf(maxLen / newtLen);
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

        const res = fN(xp);
        nFev += 1;

        if (Array.isArray(res)) {
            fp = res[0];
            Fp = res[1];
        } else {
            fp = res;
        }

        if (fp <= fc + alpha * lambda * slope) {
            success = true;
            if (first && newtLen > 0.99 * maxLen) {
                wasMaxStep = true;
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

                if (Math.abs(a) < SQRT_EPS) {
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

    return { success, wasMaxStep, nFev, xp, fp, Fp };
}