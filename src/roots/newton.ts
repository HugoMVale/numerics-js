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
     * Maximum number of outer quasi-Newton iterations.
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
     * errors in the user-provided Jacobian.
     * @default true
     */
    jacCheck?: boolean;
    /**
     * Initial Jacobian approximation at `x0`. If provided, it is used
     * instead of computing the Jacobian at the first iteration. Useful for
     * restarts, or when a cheap initial approximation (e.g. the identity)
     * is sufficient and reduces function calls.
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
 */
export function quasiNewton(
    f: (x: Vector) => Vector,
    x0: Vector,
    options: QuasiNewtonOptions = {}
): VectorRootResult {

    const tolx = options.tolx ?? 1e-10;
    const tolf = options.tolf ?? 1e-5;
    const maxIter = options.maxIter ?? 100;
    const maxLenFactor = options.maxLenFactor ?? 1e3;
    const broydenUpdate = options.broydenUpdate ?? false;
    const jacCheck = options.jacCheck ?? true;
    const globalMethod: GlobalMethod = options.globalMethod === undefined ? 'line-search' : options.globalMethod;

    let method = 'Quasi-Newton';
    const methodOptions: string[] = [`Global: ${titleCase(globalMethod ?? 'none')}`];
    if (broydenUpdate) methodOptions.push('Broyden update');
    method += ' (' + methodOptions.join(', ') + ')';

    options.sclx?.map((value) => {
        if (!(value > 0)) throw new Error('quasiNewton: sclx must contain strictly positive values');
        return value;
    });
    options.sclf?.map((value) => {
        if (!(value > 0)) throw new Error('quasiNewton: sclf must contain strictly positive values');
        return value;
    });
    const sclx = options.sclx ? options.sclx.abs() : scaleVector(x0);

    let nFev = 0;
    let nJev = 0;

    const n = x0.size;
    let xc = x0.copy();
    let fc = f(xc);
    nFev += 1;

    // Evaluate Jacobian at x0.
    let Jc: Matrix;
    if (options.J0) {
        Jc = options.J0.copy();
    } else if (options.jac) {
        Jc = options.jac(xc);
        nJev += 1;
    } else {
        Jc = jacobianForward(f, xc, { fx: fc, sclx, epsf: options.epsf });
        nFev += n;
    }

    // Check user-provided Jacobian against a finite-difference approximation.
    if (jacCheck && options.jac) {
        const Jfd = jacobianForward(f, xc, { fx: fc, sclx: sclx, epsf: options.epsf });
        nFev += n;
        const epsfEff = options.epsf !== undefined ? Math.max(options.epsf, EPS) : EPS;
        const tol = 1e2 * Math.sqrt(epsfEff);
        if (!Jc.allClose(Jfd, tol, tol)) {
            return {
                method,
                success: false,
                message: 'User-provided Jacobian `jac` does not match finite-difference approximation.',
                evaluationsFunction: nFev,
                evaluationsJacobian: nJev,
                iterations: 0,
                x: x0.copy(),
                fx: fc,
                Jx: Jc,
            };
        }
    }

    // Set f scaling factors.
    let sclf: Vector;
    if (options.sclf) {
        sclf = options.sclf.abs();
    } else {
        const rowMax = Jc.abs().max(1); // per-equation max |J_ij|
        sclf = rowMax.map((v) => 1 / (v === 0 ? 1 : v));
    }

    // Check initial solution with a tight tolerance.
    if (sclf.mult(fc).normInf() <= 1e-2 * tolf) {
        return {
            method,
            success: true,
            message: '||sclf*f(x0)||∞ ≤ 1e-2*tolf',
            evaluationsFunction: nFev,
            evaluationsJacobian: nJev,
            iterations: 0,
            x: x0.copy(),
            fx: fc,
            Jx: Jc,
        };
    }

    // Set maximum step length for global methods.
    const maxLen = Math.max(0, maxLenFactor) * Math.max(sclx.mult(x0).norm(), sclx.norm());

    // Objective function for global methods: 1/2*||sclf*f(x)||².
    const fN = (x: Vector): [number, Vector] => {
        const fx = f(x);
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
            const scaledJ = Matrix.diag(sclf).matmul(Jc);
            ({ Q, R } = scaledJ.qr());
        }

        // Condition number of R (column-scaled by sclx).
        const Rcond = divideColumnsBy(R, sclx).cond1Upper();

        // Solve (Q*R)*p = -sclf*fc.
        let p: Vector;
        let Rstep = R;
        if (Rcond < 1 / SQRT_EPS) {
            const rhs = Q.transpose().mulVec(sclf.mult(fc));
            p = R.solveUpper(rhs).mult(-1);
            if (globalMethod) {
                gc = R.transpose().mulVec(Q.transpose().mulVec(sclf.mult(fc)));
            }
        } else {
            const H = R.transpose().matmul(R);
            const Hnorm = H.div(sclx.outer(sclx)).norm1();
            for (let i = 0; i < n; i++) {
                H.set(i, i, H.get(i, i) + Math.sqrt(n * EPS) * Hnorm * sclx.get(i) ** 2);
            }
            gc = R.transpose().mulVec(Q.transpose().mulVec(sclf.mult(fc)));
            const L = H.cholesky();
            p = L.choleskySolve(gc).mult(-1);
            Rstep = L.transpose();
        }

        // Current value of the global-method objective, 1/2*||sclf*f(xc)||².
        fNc = 0.5 * sclf.mult(fc).normSq();

        // Compute the actual x step.
        const ctx: GlobalStepContext = { fN, p, xc, fc: fNc, gc, R: Rstep, tolx, sclx, maxLen, trustLen };
        let step: GlobalStepResult;
        if (globalMethod === null) {
            // No global strategy: take the full quasi-Newton step as-is.
            const xpFull = ctx.xc.add(ctx.p);
            const res = ctx.fN(xpFull);
            const [fpFull, FpFull] = Array.isArray(res) ? res : [res, new Vector(0)];
            step = { success: true, wasMaxStep: true, nFev: 1, xp: xpFull, fp: fpFull, Fp: FpFull, trustLen };
        } else if (globalMethod === 'line-search') {
            step = lineSearch(ctx);
        } else if (globalMethod === 'dogleg') {
            step = dogleg(ctx);
        } else {
            throw new Error(`quasiNewton: unknown globalMethod '${globalMethod}'`);
        }

        nFev += step.nFev;
        consecutiveMaxSteps = step.wasMaxStep ? consecutiveMaxSteps + 1 : 0;
        trustLen = step.trustLen;

        xp = step.xp;
        fp = step.Fp;

        // If the global-method step failed, restart once with a fresh Jacobian.
        if (!step.success && !restart) {
            if (options.jac) {
                Jc = options.jac(xc);
                nJev += 1;
            } else {
                Jc = jacobianForward(f, xc, { fx: fc, sclx, epsf: options.epsf });
                nFev += n;
            }
            restart = true;
            continue;
        }

        // Check termination and convergence conditions.
        let stop: boolean;
        if (!step.success) {
            message =
                'Last global step failed to decrease ||sclx*f(x)||₂ sufficiently. Either `x` is ' +
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
        } else if (options.jac) {
            Jc = options.jac(xp);
            nJev += 1;
        } else {
            Jc = jacobianForward(f, xp, { fx: fp, sclx: sclx, epsf: options.epsf });
            nFev += n;
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