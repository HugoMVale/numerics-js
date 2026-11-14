import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { ARMIJO_ALPHA, relativeStepLength } from './common.js';
import type { GlobalStepContext, GlobalStepResult } from './types.js';

/** Codes for the status of the trust region step and update. */
enum TrustState {
  Accepted,
  Convergence,
  Rejected,
  ExploratorySuccess,
  Start
}

/**
 * Performs a trust-region step with the double-dogleg strategy.
 *
 * The step follows a path that interpolates between the Cauchy (steepest-descent) point and the
 * quasi-Newton step, subject to the trust-region radius `ctx.trustLen`, which is adapted from
 * iteration to iteration. The result has `success: false` if no acceptable step was found, which
 * includes the case where the step becomes shorter than `tolx` (relative).
 *
 * A trial point at which the objective is not finite (`NaN` or `±Infinity`)
 * is treated as a rejected step and the trust-region radius is reduced. After
 * an exploratory (radius-doubling) step, it instead falls back to the
 * previous finite point.
 *
 * **References**
 *
 * *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *     Optimization and Nonlinear Equations", SIAM, 1996.
 *
 * @internal
 *
 * @param ctx Global-step context containing the objective, current iterate, search direction, and scaling information. `ctx.trustLen` carries the trust-region radius across calls — see `GlobalStepContext.trustLen`.
 * @returns A `GlobalStepResult` with the accepted iterate and its objective values, and the updated trust-region radius (`trustLen`).
 */
export function dogleg(ctx: GlobalStepContext): GlobalStepResult {
  const { fN, p, xc, fc, gc, R, tolx, sclx, maxLen } = ctx;
  let trustLen = ctx.trustLen ?? 0;

  let state = TrustState.Start;
  let wasMaxStep = false;

  let cauchyLen = NaN;
  let eta = NaN;
  let v = new Vector(0);
  let sSD = new Vector(0);

  let xp = xc;
  let xpPrev = xc;
  let fp = NaN;
  let fpPrev = NaN;
  let Fp = new Vector(0);
  let FpPrev = new Vector(0);

  const newtLen = sclx.mult(p).norm();

  let first = true;
  let isNewtStep: boolean;
  let s: Vector;

  while (state !== TrustState.Accepted && state !== TrustState.Convergence) {
    // Perform dogleg step to determine s
    if (newtLen <= trustLen) {
      isNewtStep = true;
      s = p;
      trustLen = newtLen;
    } else {
      isNewtStep = false;
      if (first) {
        first = false;
        // Cauchy point: minimizer of the quadratic model along the scaled steepest descent.
        const gNormSq = gc.div(sclx).norm() ** 2;
        const RgNormSq = R.mulVec(gc.div(sclx.mult(sclx))).norm() ** 2;
        sSD = gc.div(sclx).mult(-(gNormSq / RgNormSq));
        cauchyLen = (gNormSq * Math.sqrt(gNormSq)) / RgNormSq;
        eta = 0.2 + (0.8 * gNormSq ** 2) / (RgNormSq * Math.abs(gc.dot(p)));
        v = p.mult(sclx).mult(eta).sub(sSD);
        if (trustLen <= 0.0) {
          trustLen = Math.min(cauchyLen, maxLen);
        }
      }

      if (eta * newtLen <= trustLen) {
        s = p.mult(trustLen / newtLen);
      } else if (cauchyLen >= trustLen) {
        s = sSD.div(sclx).mult(trustLen / cauchyLen);
      } else {
        const a = v.dot(v);
        const b = v.dot(sSD);
        const lambda = (-b + Math.sqrt(b * b - a * (cauchyLen ** 2 - trustLen ** 2))) / a;
        s = sSD.add(v.mult(lambda)).div(sclx);
      }
    }

    // Update trust region
    const upd = updateTrustRegion(
      fN,
      xc,
      fc,
      gc,
      R,
      s,
      tolx,
      sclx,
      maxLen,
      trustLen,
      isNewtStep,
      state,
      xpPrev,
      fpPrev,
      FpPrev
    );
    state = upd.state;
    wasMaxStep = upd.wasMaxStep;
    trustLen = upd.trustLen;
    xp = upd.xp;
    fp = upd.fp;
    Fp = upd.Fp;
    xpPrev = upd.xpPrev;
    fpPrev = upd.fpPrev;
    FpPrev = upd.FpPrev;
  }

  return {
    success: state === TrustState.Accepted,
    wasMaxStep,
    xp,
    fp,
    Fp,
    trustLen
  };
}

/** Result of a single trust-region update sub-step (internal bookkeeping of {@link dogleg}). */
interface TrustRegionUpdate {
  state: TrustState;
  wasMaxStep: boolean;
  trustLen: number;
  xp: Vector;
  fp: number;
  Fp: Vector;
  xpPrev: Vector;
  fpPrev: number;
  FpPrev: Vector;
}

/**
 * Evaluates the trial step `s`, decides whether to accept, reject or expand it, and updates the
 * trust-region radius accordingly.
 *
 * **References**
 *
 * *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *     Optimization and Nonlinear Equations", SIAM, 1996.
 *
 * @param fN Objective function.
 * @param xc Current value of the variable vector.
 * @param fc Current objective function value, `f(xc)`.
 * @param gc Current gradient of the objective function, `∇f(xc)`.
 * @param R Upper-triangular factor of the current Jacobian QR decomposition.
 * @param s Step vector.
 * @param tolx Tolerance for the step size.
 * @param sclx Scaling factors for `x`.
 * @param maxLen Maximum step length.
 * @param trustLen Current trust region radius.
 * @param isNewtStep Flag indicating if the step is a Newton step.
 * @param state Code indicating the current state of the algorithm.
 * @param xpPrev Previous value of `xp`.
 * @param fpPrev Previous value of `fp`.
 * @param FpPrev Previous value of `Fp`.
 * @returns The updated state, trust-region radius, iterate, and bookkeeping needed for the next sub-iteration.
 */
function updateTrustRegion(
  fN: GlobalStepContext['fN'],
  xc: Vector,
  fc: number,
  gc: Vector,
  R: Matrix,
  s: Vector,
  tolx: number,
  sclx: Vector,
  maxLen: number,
  trustLen: number,
  isNewtStep: boolean,
  state: TrustState,
  xpPrev: Vector,
  fpPrev: number,
  FpPrev: Vector
): TrustRegionUpdate {
  let wasMaxStep = false;

  const stepLen = sclx.mult(s).norm();
  const slope = gc.dot(s);

  let xp = xc.add(s);

  let [fp, Fp] = fN(xp);

  const df = fp - fc;
  const finite = Number.isFinite(fp);

  if (
    state === TrustState.ExploratorySuccess &&
    (!finite || fp >= fpPrev || df > ARMIJO_ALPHA * slope)
  ) {
    state = TrustState.Accepted;
    xp = xpPrev;
    fp = fpPrev;
    Fp = FpPrev;
    trustLen *= 0.5;
  } else if (!finite || df >= ARMIJO_ALPHA * slope) {
    if (relativeStepLength(s, xp, sclx) < tolx) {
      state = TrustState.Convergence;
      xp = xc;
    } else {
      state = TrustState.Rejected;
      if (finite) {
        const raw = (-slope * stepLen) / (2 * (df - slope));
        trustLen = Math.min(Math.max(raw, 0.1 * trustLen), 0.5 * trustLen);
      } else {
        // Non-finite objective value: no model to interpolate, shrink strongly.
        trustLen *= 0.1;
      }
    }
  } else {
    const dfPred = slope + 0.5 * R.mulVec(s).norm() ** 2;
    if (
      state !== TrustState.Rejected &&
      !isNewtStep &&
      trustLen <= 0.99 * maxLen &&
      (Math.abs(dfPred - df) <= 0.1 * Math.abs(df) || df <= slope)
    ) {
      state = TrustState.ExploratorySuccess;
      xpPrev = xp;
      fpPrev = fp;
      FpPrev = Fp;
      trustLen = Math.min(2 * trustLen, maxLen);
    } else {
      state = TrustState.Accepted;
      if (stepLen >= 0.99 * maxLen) {
        wasMaxStep = true;
      }
      if (df >= 0.1 * dfPred) {
        trustLen *= 0.5;
      } else if (df <= 0.75 * dfPred) {
        trustLen = Math.min(2 * trustLen, maxLen);
      }
      // else: trustLen unchanged
    }
  }

  return { state, wasMaxStep, trustLen, xp, fp, Fp, xpPrev, fpPrev, FpPrev };
}
