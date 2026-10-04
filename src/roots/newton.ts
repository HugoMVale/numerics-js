import { Vector } from '../linalg/Vector.js';
import { Matrix, type QRDecomposition } from '../linalg/Matrix.js';
import { jacobianForward, scaleVector } from '../numdiff.js';
import { lineSearch } from './lineSearch.js';
import { dogleg } from './dogleg.js';
import type {
    VectorRootResult,
    GlobalMethod,
    GlobalStepContext,
    GlobalStepResult
} from './types.js';

const EPS = Number.EPSILON;
const SQRT_EPS = Math.sqrt(EPS);


/** Options for {@link quasiNewton}. */
export interface QuasiNewtonOptions {
    /**
     * Tolerance for the scaled step size. The algorithm terminates when the
     * scaled distance between two successive iterates
     * `||Δx/max(x, 1/sclx)||∞` is below this threshold. If too large, the
     * algorithm may terminate prematurely. A value on the order of
     * `eps^(2/3)` is typically recommended.
     * @default 1e-10
     */
    tolx?: number;
    /**
     * Tolerance for the scaled residual norm. This is the main convergence
     * criterion: the algorithm terminates when `||sclf*f(x)||∞` is below
     * this threshold. A value on the order of `eps^(1/3)` is typically
     * recommended.
     * @default 1e-5
     */
    tolf?: number;
    /**
     * Positive scaling factors for the components of `x`. Ideally chosen so
     * that `sclx*x` is of order 1 near the solution for all components. By
     * default, scaling is determined from `x0` via {@link scaleVector}.
     */
    sclx?: Vector;
    /**
     * Positive scaling factors for the components of `f`. Ideally chosen so
     * that `sclf*f` is of order 1 near the root for all components. By
     * default, scaling is determined from the initial Jacobian.
     */
    sclf?: Vector;
    /**
     * Maximum number of outer quasi-Newton iterations. Must be a finite,
     * strictly positive integer.
     * @default 100
     */
    maxIter?: number;
    /**
     * Factor determining the maximum allowable scaled step length
     * `||sclx*Δx||₂` for global methods. Prevents steps that would overflow,
     * leave the domain of interest, or diverge.
     * @default 1e3
     */
    maxLenFactor?: number;
    /**
     * Initial trust-region radius for the dogleg global method. The value is
     * capped at the maximum step length; if omitted or non-positive, the
     * length of the initial scaled gradient is used.
     */
    trustLen?: number;
    /**
     * Machine precision of the function values. If `undefined`, machine
     * precision of the 64-bit floating-point type is assumed. If the number
     * of reliable base-10 digits in `f`'s results is `n`, then `epsf` is
     * approximately `10^(-n)`.
     */
    epsf?: number;
    /**
     * Global strategy to improve convergence from remote starting points.
     * @default 'line-search'
     */
    globalMethod?: GlobalMethod;
    /**
     * If `true`, the Jacobian is updated at each iteration using Broyden's
     * rank-1 update formula instead of being recomputed. Significantly
     * reduces the number of function/Jacobian evaluations, but may lead to
     * inaccurate Jacobian approximations and poor convergence far from the
     * root or for highly nonlinear functions.
     * @default false
     */
    broydenUpdate?: boolean;
    /**
     * Function to compute the Jacobian of `f`. By default, the Jacobian is
     * approximated using forward finite differences ({@link jacobianForward}).
     * In that case, setting `epsf` appropriately is essential.
     */
    jac?: (x: Vector) => Matrix;
    /**
     * If `true` and `jac` is provided, `jac`'s result at `x0` is checked
     * against a forward finite-difference approximation, to help catch
     * errors in the user-provided Jacobian. The check is independent of `J0`:
     * if both are provided, `jac(x0)` is still evaluated for the check (at the
     * cost of one extra `jac` call and `n` function calls), but the first
     * iteration starts from `J0`.
     * @default true
     */
    jacCheck?: boolean;
    /**
     * Initial Jacobian approximation at `x0`. If provided, it is used
     * instead of computing the Jacobian at the first iteration. Useful for
     * restarts, or when a cheap initial approximation (e.g. the identity)
     * is sufficient and reduces function calls. `J0` is an approximation by
     * design and is never checked against finite differences; see `jacCheck`.
     */
    J0?: Matrix;
}

/**
 * Finds the root of a system of nonlinear equations using a quasi-Newton
 * method with optional global strategies.
 *
 * This solver follows Dennis and Schnabel (1996). The user can choose the
 * approach to calculate and update the Jacobian approximation, as well as
 * the global strategy used to improve convergence from remote starting
 * points.
 *
 * The default settings favor the likelihood of convergence over
 * computational efficiency. For situations where maximum efficiency is
 * desired and the initial guess is known to be close to the root, consider
 * disabling the global method and using Broyden's update for the Jacobian.
 *
 * Solving systems of nonlinear equations is a surprisingly complex task —
 * often more difficult than solving systems of differential equations or
 * even multivariate optimization problems. Convergence is guaranteed only
 * when the initial guess is sufficiently close to the root, which is rarely
 * true in practice. The choice of a good initial guess, appropriate scaling
 * factors, and a suitable global strategy is an essential part of solving
 * the problem.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996.
 *
 * @param f Function whose root is to be found.
 * @param x0 Initial guess for the root. If no user-defined `sclx` is
 * provided, the scaling factors are determined from this value.
 * @param options Configuration options for the solver.
 * @returns A `VectorRootResult` with the root solution and diagnostics.
 *
 * Non-finite values (`NaN` or `±Infinity`) are handled as follows: if `f(x0)`
 * or a Jacobian is not finite, the solver returns a failure result
 * (`success: false`). If `f` is not finite at a trial point, the global
 * strategies reject the step and backtrack, whereas with `globalMethod: null`
 * the solver returns a failure result.
 * @throws {RangeError} If `x0` has no components, if `maxIter` is not a finite
 * strictly positive integer, if `f(x0)` has a different dimension than `x0`,
 * or if `J0` or `jac(x0)` is not an `n`×`n` matrix.
 * @throws {Error} If `sclx` or `sclf` contains a value that is not strictly
 * positive, or if `globalMethod` is not `'line-search'`, `'dogleg'`, or
 * `null`.
 *
 * @example
 * ```ts
 * // Steady-state concentrations of A, B, C at the outlet of a CSTR with a
 * // consecutive reaction scheme A+B->C, C+B->D.
 * const A0 = 1.0, B0 = 2.0, C0 = 0.0, k1 = 1e-3, k2 = 5e-4, tau = 1e3;
 * const f = (x: Vector) => {
 *     const [A, B, C] = x;
 *     return Vector.from([
 *         (A0 - A) / tau - k1 * A * B,
 *         (B0 - B) / tau - k1 * A * B,
 *         (C0 - C) / tau + k1 * A * B - k2 * C * B,
 *     ]);
 * };
 * const sol = quasiNewton(f, Vector.from([0.5, 1.0, 0.5]));
 * console.log(sol.x.toString());
 * ```
 *
 * Output:
 * ```text
 * Vector(0.4142135642138721, 1.4142135642135265, 0.34314574906501627)
 * ```
 */
export function quasiNewton(
    f: (x: Vector) => Vector,
    x0: Vector,
    options: QuasiNewtonOptions = {}
): VectorRootResult {

    const n = x0.size;
    if (n === 0) {
        throw new RangeError('quasiNewton: x0 must have at least one component');
    }

    if (options.maxIter !== undefined && !(Number.isInteger(options.maxIter) && options.maxIter >= 1)) {
        throw new RangeError(`quasiNewton: maxIter must be a finite strictly positive integer, got ${options.maxIter}`);
    }

    options.sclx?.map((value) => {
        if (!(value > 0)) throw new Error('quasiNewton: sclx must contain strictly positive values');
        return value;
    });
    options.sclf?.map((value) => {
        if (!(value > 0)) throw new Error('quasiNewton: sclf must contain strictly positive values');
        return value;
    });

    const sclx = options.sclx ?? scaleVector(x0);
    const tolx = options.tolx ?? 1e-10;
    const tolf = options.tolf ?? 1e-5;
    const maxIter = options.maxIter ?? 100;
    const maxLenFactor = options.maxLenFactor ?? 1e3;
    const broydenUpdate = options.broydenUpdate ?? false;
    const jacCheck = options.jacCheck ?? true;
    const globalMethod: GlobalMethod = options.globalMethod === undefined ? 'line-search' : options.globalMethod;
    if (globalMethod !== null && globalMethod !== 'line-search' && globalMethod !== 'dogleg') {
        throw new Error(`quasiNewton: unknown globalMethod '${String(globalMethod)}'`);
    }

    let method = 'Quasi-Newton';
    const methodOptions: string[] = [`Global: ${titleCase(globalMethod ?? 'none')}`];
    if (broydenUpdate) methodOptions.push('Broyden update');
    method += ' (' + methodOptions.join(', ') + ')';

    let nFev = 0;
    let nJev = 0;

    // All evaluations of `f` (including those made by finite differences and by the
    // global strategies) go through this wrapper, which counts them.
    const countedF = (x: Vector): Vector => {
        nFev += 1;
        return f(x);
    };

    const failure = (
        message: string,
        iterations: number,
        x: Vector,
        fx: Vector,
        Jx: Matrix | null
    ): VectorRootResult => ({
        method,
        success: false,
        message,
        evaluationsFunction: nFev,
        evaluationsJacobian: nJev,
        iterations,
        x,
        fx,
        Jx,
    });

    // Evaluates the Jacobian at `x` (with `fx = f(x)`), either with the user-supplied
    // `jac` or by forward finite differences, and counts the evaluations.
    const evaluateJacobian = (x: Vector, fx: Vector): { J: Matrix; finite: boolean } => {
        let J: Matrix;
        if (options.jac) {
            J = options.jac(x);
            nJev += 1;
        } else {
            J = jacobianForward(countedF, x, { fx, sclx, epsf: options.epsf });
        }
        return { J, finite: J.allFinite() };
    };

    let xc = x0.copy();
    let fc = countedF(xc);

    if (fc.size !== n) {
        throw new RangeError(`quasiNewton: f(x0) must have dimension ${n}, got ${fc.size}`);
    }

    if (!fc.allFinite()) {
        return failure('f(x0) contains non-finite values.', 0, xc, fc, null);
    }

    // Jacobian at x0.
    const nonFiniteJacobianX0 = (J: Matrix) =>
        failure('Jacobian at x0 contains non-finite values.', 0, xc, fc, J);

    // Initial Jacobian approximation provided by the user (never checked).
    const J0 = options.J0?.copy();
    if (J0) {
        if (J0.rows !== n || J0.cols !== n) {
            throw new RangeError(`quasiNewton: J0 must have dimensions ${n}x${n}, got ${J0.rows}x${J0.cols}`);
        }
        if (!J0.allFinite()) {
            return nonFiniteJacobianX0(J0);
        }
    }

    // User-supplied Jacobian at x0: needed as the starting Jacobian (if `J0` is not
    // provided) or to check it against a finite-difference approximation.
    let Jjac: Matrix | undefined;
    if (options.jac && (!J0 || jacCheck)) {
        const { J, finite } = evaluateJacobian(xc, fc);
        if (J.rows !== n || J.cols !== n) {
            throw new RangeError(`quasiNewton: jac(x0) must have dimensions ${n}x${n}, got ${J.rows}x${J.cols}`);
        }
        if (!finite) {
            return nonFiniteJacobianX0(J);
        }
        Jjac = J;
    }

    // Check `jac(x0)` against a finite-difference approximation.
    if (jacCheck && Jjac) {
        const Jfd = jacobianForward(countedF, xc, { fx: fc, sclx: sclx, epsf: options.epsf });
        const epsfEff = options.epsf !== undefined ? Math.max(options.epsf, EPS) : EPS;
        const tol = 1e2 * Math.sqrt(epsfEff);
        if (!Jjac.allClose(Jfd, tol, tol)) {
            return failure(
                'User-provided Jacobian `jac` does not match finite-difference approximation.',
                0,
                xc,
                fc,
                Jjac
            );
        }
    }

    // Starting Jacobian: `J0`, else `jac(x0)`, else a finite-difference approximation.
    let Jc: Matrix;
    if (J0) {
        Jc = J0;
    } else if (Jjac) {
        Jc = Jjac;
    } else {
        const { J, finite } = evaluateJacobian(xc, fc);
        if (!finite) {
            return nonFiniteJacobianX0(J);
        }
        Jc = J;
    }

    // Set f scaling factors (default: inverse of the per-equation max |J_ij|).
    const sclf = options.sclf ?? Jc.abs().max(1).map((v) => 1 / (v === 0 ? 1 : v));

    // Check initial solution with a tight tolerance.
    if (sclf.mult(fc).normInf() <= 1e-2 * tolf) {
        return {
            method,
            success: true,
            message: '||sclf*f(x0)||∞ ≤ 1e-2*tolf',
            evaluationsFunction: nFev,
            evaluationsJacobian: nJev,
            iterations: 0,
            x: xc,
            fx: fc,
            Jx: Jc,
        };
    }

    // Set maximum step length for global methods.
    const maxLen = Math.max(0, maxLenFactor) * Math.max(sclx.mult(x0).norm(), sclx.norm());

    // Objective function for global methods: 1/2*||sclf*f(x)||².
    const fN = (x: Vector): [number, Vector] => {
        const fx = countedF(x);
        const fNx = 0.5 * sclf.mult(fx).normSq();
        return [fNx, fx];
    };

    let consecutiveMaxSteps = 0;
    let trustLen = options.trustLen === undefined ? 0 : Math.min(options.trustLen, maxLen);
    let restart = true;
    let nIter = 0;
    let Q!: Matrix;
    let R!: Matrix;
    let gc = new Vector(0);
    let fNc = NaN;
    let message = '';
    let success = false;
    let xp = xc;
    let fp = fc;
    let broke = false;

    for (nIter = 1; nIter <= maxIter; nIter++) {
        // QR decomposition of the scaled Jacobian.
        if (!broydenUpdate || restart) {
            const scaledJ = Jc.copy();
            for (let i = 0; i < sclf.size; i++) scaledJ.scaleRow(i, sclf.get(i));
            ({ Q, R } = scaledJ.qr());
        }

        // Condition number of R (column-scaled by sclx).
        const Rcond = divideColumnsBy(R, sclx).cond1Upper();

        // Solve (Q*R)*p = -sclf*fc.
        let p: Vector;
        let Rstep = R;
        const sclf_fc = sclf.mult(fc);
        if (Rcond < 1 / SQRT_EPS) {
            const rhs = Q.transpose().mulVec(sclf_fc);
            p = R.solveUpper(rhs).mult(-1);
            if (globalMethod) {
                gc = R.transpose().mulVec(rhs);
            }
        } else {
            const H = R.transpose().matmul(R);
            const Hnorm = H.div(sclx.outer(sclx)).norm1(); // could be made more efficient, but probably not worth it.
            for (let i = 0; i < n; i++) {
                H.set(i, i, H.get(i, i) + Math.sqrt(n * EPS) * Hnorm * sclx.get(i) ** 2);
            }
            gc = R.transpose().mulVec(Q.transpose().mulVec(sclf_fc));
            const L = H.cholesky();
            p = L.choleskySolve(gc).mult(-1);
            Rstep = L.transpose();
        }

        // Current value of the global-method objective, 1/2*||sclf*f(xc)||².
        fNc = 0.5 * sclf_fc.normSq();

        // Compute the actual x step.
        const ctx: GlobalStepContext = { fN, p, xc, fc: fNc, gc, R: Rstep, tolx, sclx, maxLen, trustLen };
        let step: GlobalStepResult;
        if (globalMethod === null) {
            // No global strategy: take the full quasi-Newton step as-is.
            const xpFull = ctx.xc.add(ctx.p);
            const [fpFull, FpFull] = ctx.fN(xpFull);
            step = { success: true, wasMaxStep: true, xp: xpFull, fp: fpFull, Fp: FpFull, trustLen };
        } else if (globalMethod === 'line-search') {
            step = lineSearch(ctx);
        } else {
            step = dogleg(ctx);
        }

        consecutiveMaxSteps = step.wasMaxStep ? consecutiveMaxSteps + 1 : 0;
        trustLen = step.trustLen;

        // A step is rejected if the strategy failed or if it returned a non-finite
        // objective value. In both cases the current iterate and its function value
        // are kept (strategies do not guarantee meaningful values on failure).
        const nonFiniteStep = step.success && !Number.isFinite(step.fp);
        if (!step.success || nonFiniteStep) {
            xp = xc;
            fp = fc;
        } else {
            xp = step.xp;
            fp = step.Fp;
        }

        // If the global-method step failed, restart once with a fresh Jacobian.
        if (broydenUpdate && !step.success && !restart) {
            const { J, finite } = evaluateJacobian(xc, fc);
            Jc = J;
            if (!finite) {
                return failure('Jacobian at the current iterate contains non-finite values.', nIter, xc, fc, Jc);
            }
            restart = true;
            continue;
        }

        // Check termination and convergence conditions.
        let stop: boolean;
        if (nonFiniteStep) {
            message = 'f returned non-finite values at the proposed step. Consider using a global method (`globalMethod`).';
            stop = true;
        } else if (!step.success) {
            message =
                'Last global step failed to decrease ||sclf*f(x)||₂ sufficiently. Either `x` is ' +
                'close to a root and no more accuracy is possible, or the secant approximation to ' +
                'the Jacobian is inaccurate, or `tolx` is too large.';
            stop = true;
        } else if (sclf.mult(fp).normInf() <= tolf) {
            message = '||sclf*f(x)||∞ ≤ tolf';
            success = true;
            stop = true;
        } else if (
            xp.sub(xc).div(xp.abs().map((val, idx) => Math.max(val, 1 / sclx.get(idx)))).normInf() <= tolx
        ) {
            message =
                '||Δx/max(x, 1/sclx)||∞ ≤ tolx: `x` may be an approximate root, but it is also ' +
                'possible that the algorithm is making slow progress and is not near a root, or ' +
                'that `tolx` is too large.';
            success = true;
            stop = true;
        } else if (globalMethod && consecutiveMaxSteps >= 5) {
            message =
                'Maximum number (5) of consecutive steps of length `maxLen` reached. Perhaps stuck ' +
                'in a flat region or `maxLen` is too small.';
            stop = true;
        } else {
            stop = false;
        }

        if (stop) {
            xc = xp;
            fc = fp;
            broke = true;
            break;
        }

        // Update Jacobian.
        if (broydenUpdate) {
            ({ Q, R } = updateBroyden(xc, xp, fc, fp, Q, R, sclx, sclf));
        } else {
            const { J, finite } = evaluateJacobian(xp, fp);
            Jc = J;
            if (!finite) {
                return failure('Jacobian at the current iterate contains non-finite values.', nIter, xp, fp, Jc);
            }
        }

        // Next iteration.
        xc = xp;
        fc = fp;
        restart = false;
    }

    if (!broke) {
        message = `Maximum number of iterations (${maxIter}) reached.`;
    }

    if (broydenUpdate) {
        Jc = Q.matmul(R);
        for (let i = 0; i < n; i++) Jc.setRow(i, Jc.row(i).mult(1 / sclf.get(i)));
    }

    return {
        method,
        success,
        message,
        evaluationsFunction: nFev,
        evaluationsJacobian: nJev,
        iterations: Math.min(nIter, maxIter),
        x: xc,
        fx: fc,
        Jx: Jc,
    };
}

/**
 * Performs a Broyden rank-1 update of the QR decomposition of a Jacobian
 * approximation, using scaling factors for both `x` and `f` to improve
 * numerical conditioning. Mutates `Qc`/`Rc` in place (via `Matrix.qrUpdateSelf`)
 * and returns them.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996.
 *
 * @param xc Current value of the variable vector.
 * @param xp Next value of the variable vector.
 * @param fc Current function value, `f(xc)`.
 * @param fp Next function value, `f(xp)`.
 * @param Qc Orthogonal factor of the current Jacobian QR decomposition. Mutated in place.
 * @param Rc Upper-triangular factor of the current Jacobian QR decomposition. Mutated in place.
 * @param sclx Scaling factors for the components of `x`.
 * @param sclf Scaling factors for the components of `f`.
 * @returns The updated `{ Q, R }` factorization (the same, mutated, objects).
 */
function updateBroyden(
    xc: Vector,
    xp: Vector,
    fc: Vector,
    fp: Vector,
    Qc: Matrix,
    Rc: Matrix,
    sclx: Vector,
    sclf: Vector
): QRDecomposition {
    const s = xp.sub(xc);
    const y = fp.sub(fc);

    const w0 = sclf.mult(y).sub(Qc.mulVec(Rc.mulVec(s)));
    const threshold = sclf.mult(fp.abs().add(fc.abs())).mult(EPS);
    const w = w0.map((wi, i) => (Math.abs(wi) < threshold.get(i) ? 0 : wi));

    const t = s.mult(sclx.mult(sclx));
    const v = t.mult(1 / s.dot(t));

    return Matrix.qrUpdateSelf({ Q: Qc, R: Rc }, w, v);
}

// -----------------------------------------------------------------
// Small helpers.
// -----------------------------------------------------------------

/** Returns a copy of `M` with column `j` divided by `v.get(j)`, for every column. */
function divideColumnsBy(M: Matrix, v: Vector): Matrix {
    const res = M.copy();
    for (let j = 0; j < res.cols; j++) {
        for (let i = 0; i < res.rows; i++) {
            res.set(i, j, res.get(i, j) / v.get(j));
        }
    }
    return res;
}

/** Title-cases a hyphen/space-separated label, e.g. `'line-search' -> 'Line-Search'`. */
function titleCase(s: string): string {
    return s.replace(/(^|[\s-])([a-z])/g, (_m, sep: string, ch: string) => sep + ch.toUpperCase());
}