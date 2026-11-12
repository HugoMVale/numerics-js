import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { findBracket, integrateConstant, addConstantIntegral } from './common.js';

// -----------------------------------------------------------------
// Shared piecewise cubic Hermite kernels.
//
// A piecewise cubic Hermite interpolant is fully determined by the knots
// `xp`, the values `fp` and the slopes `d` at the knots. Different methods
// differ only in how they choose `d` (PCHIP: monotonicity-preserving local
// formulas; cubic spline: C2 continuity through a tridiagonal system), so
// evaluation, differentiation and integration are shared here.
//
// Out-of-range behavior follows the rest of the interpolate module: `x`
// outside `[xp[0], xp[n-1]]` is clamped to `leftVal`/`rightVal` (derivative
// `0`), and `NaN` propagates.
// -----------------------------------------------------------------

/**
 * Evaluates the integral of the Hermite basis functions from `0` to `t`,
 * i.e. `[intH00, intH10, intH01, intH11](t)`.
 */
function hermiteBasisIntegral(t: number): { i00: number; i10: number; i01: number; i11: number } {
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  return {
    i00: 0.5 * t4 - t3 + t,
    i10: 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2,
    i01: -0.5 * t4 + t3,
    i11: 0.25 * t4 - (1.0 / 3.0) * t3
  };
}

/**
 * Evaluates the cubic Hermite polynomial or its derivative at a single point.
 */
export function hermiteOne1D(
  xi: number,
  xp: Vector,
  fp: Vector,
  d: Float64Array,
  leftVal: number,
  rightVal: number,
  computeDerivative: boolean
): number {
  const n = xp.size;
  const xpd = xp.data;
  const fpd = fp.data;

  if (Number.isNaN(xi)) return NaN;
  if (xi < xpd[0]) return computeDerivative ? 0 : leftVal; // Derivative of flat clamp is 0
  if (xi > xpd[n - 1]) return computeDerivative ? 0 : rightVal;

  const [lo, hi] = findBracket(xi, xpd);

  const x0 = xpd[lo];
  const x1 = xpd[hi];
  if (x1 === x0) return computeDerivative ? 0 : fpd[lo];

  const h = x1 - x0;
  const t = (xi - x0) / h;

  const y0 = fpd[lo];
  const y1 = fpd[hi];
  const d0 = d[lo];
  const d1 = d[hi];

  if (computeDerivative) {
    // H'00(t), H'10(t), H'01(t), H'11(t) divided by h (chain rule for dt/dx)
    const dp00 = (6 * t * t - 6 * t) / h;
    const dp10 = 3 * t * t - 4 * t + 1; // h * H'10 / h
    const dp01 = (-6 * t * t + 6 * t) / h;
    const dp11 = 3 * t * t - 2 * t; // h * H'11 / h

    return y0 * dp00 + d0 * dp10 + y1 * dp01 + d1 * dp11;
  } else {
    // H00(t), H10(t), H01(t), H11(t)
    const t2 = t * t;
    const t3 = t2 * t;
    const p00 = 2 * t3 - 3 * t2 + 1;
    const p10 = t3 - 2 * t2 + t;
    const p01 = -2 * t3 + 3 * t2;
    const p11 = t3 - t2;

    return y0 * p00 + h * d0 * p10 + y1 * p01 + h * d1 * p11;
  }
}

/** Vectorized {@link hermiteOne1D}: evaluates at every entry of `x`. */
export function hermiteMany1D(
  x: VectorLike,
  xp: Vector,
  fp: Vector,
  d: Float64Array,
  leftVal: number,
  rightVal: number,
  computeDerivative: boolean
): Vector {
  const xv = Vector.asVector(x);
  const res = new Vector(xv.size);
  const xdata = xv.data;
  const rdata = res.data;
  for (let i = 0; i < xv.size; i++) {
    rdata[i] = hermiteOne1D(xdata[i], xp, fp, d, leftVal, rightVal, computeDerivative);
  }
  return res;
}

// -----------------------------------------------------------------
// Vector-valued (ND) kernels. Same structure as hermiteOne1D/hermiteMany1D, but
// `fp` and `d` are Matrices (row i = knot i's value/slope vector); the
// bracket/t/basis-function computation is identical across components,
// so it's done once and then looped over fp's columns against the flat
// buffers directly, same style as linear.ts's *ND kernels.
// -----------------------------------------------------------------

export function hermiteOneND(
  xi: number,
  xp: Vector,
  fp: Matrix,
  d: Matrix,
  leftVal: Vector,
  rightVal: Vector,
  computeDerivative: boolean
): Vector {
  const n = xp.size;
  const xpd = xp.data;
  const m = fp.cols;

  if (Number.isNaN(xi)) return Vector.full(m, NaN);
  if (xi < xpd[0]) return computeDerivative ? new Vector(m) : leftVal.copy();
  if (xi > xpd[n - 1]) return computeDerivative ? new Vector(m) : rightVal.copy();

  const [lo, hi] = findBracket(xi, xpd);
  const x0 = xpd[lo];
  const x1 = xpd[hi];

  const loOff = fp.flatIndex(lo, 0);
  const hiOff = fp.flatIndex(hi, 0);

  if (x1 === x0) {
    const res = new Vector(m);
    if (computeDerivative) return res; // zeros
    for (let j = 0; j < m; j++) res.data[j] = fp.data[loOff + j];
    return res;
  }

  const h = x1 - x0;
  const t = (xi - x0) / h;
  const dLoOff = d.flatIndex(lo, 0);
  const dHiOff = d.flatIndex(hi, 0);

  const res = new Vector(m);
  if (computeDerivative) {
    const dp00 = (6 * t * t - 6 * t) / h;
    const dp10 = 3 * t * t - 4 * t + 1;
    const dp01 = (-6 * t * t + 6 * t) / h;
    const dp11 = 3 * t * t - 2 * t;
    for (let j = 0; j < m; j++) {
      const y0 = fp.data[loOff + j];
      const y1 = fp.data[hiOff + j];
      const d0 = d.data[dLoOff + j];
      const d1 = d.data[dHiOff + j];
      res.data[j] = y0 * dp00 + d0 * dp10 + y1 * dp01 + d1 * dp11;
    }
  } else {
    const t2 = t * t;
    const t3 = t2 * t;
    const p00 = 2 * t3 - 3 * t2 + 1;
    const p10 = t3 - 2 * t2 + t;
    const p01 = -2 * t3 + 3 * t2;
    const p11 = t3 - t2;
    for (let j = 0; j < m; j++) {
      const y0 = fp.data[loOff + j];
      const y1 = fp.data[hiOff + j];
      const d0 = d.data[dLoOff + j];
      const d1 = d.data[dHiOff + j];
      res.data[j] = y0 * p00 + h * d0 * p10 + y1 * p01 + h * d1 * p11;
    }
  }
  return res;
}

/** Vectorized {@link hermiteOneND}: one result row per entry of `x`. */
export function hermiteManyND(
  x: VectorLike,
  xp: Vector,
  fp: Matrix,
  d: Matrix,
  leftVal: Vector,
  rightVal: Vector,
  computeDerivative: boolean,
  caller: string
): Matrix {
  const xv = Vector.asVector(x);
  if (xv.size === 0) {
    throw new RangeError(
      `${caller}: cannot evaluate at zero points; Matrix cannot represent a result with 0 rows`
    );
  }
  const res = new Matrix(xv.size, fp.cols);
  for (let i = 0; i < xv.size; i++) {
    res.setRow(i, hermiteOneND(xv.data[i], xp, fp, d, leftVal, rightVal, computeDerivative));
  }
  return res;
}

/**
 * Integrates the piecewise cubic Hermite interpolant over `[a, b]`
 * analytically, with the `left`/`right` clamp values forming constant
 * rectangular areas outside the `xp` range. Bounds must not be `NaN`
 * (checked by the public callers, which own the error message).
 */
export function hermiteIntegrate1D(
  xp: Vector,
  fp: Vector,
  d: Float64Array,
  leftVal: number,
  rightVal: number,
  a: number,
  b: number
): number {
  if (a === b) return 0;
  if (b < a) return -hermiteIntegrate1D(xp, fp, d, leftVal, rightVal, b, a);

  const xpd = xp.data;
  const fpd = fp.data;
  const n = xpd.length;

  let total = 0;

  // 1. Left clamp region (constant area)
  if (a < xpd[0]) {
    const end = Math.min(b, xpd[0]);
    total += integrateConstant(end - a, leftVal);
  }

  // 2. Right clamp region (constant area)
  if (b > xpd[n - 1]) {
    const start = Math.max(a, xpd[n - 1]);
    total += integrateConstant(b - start, rightVal);
  }

  // 3. Interior cubic regions
  const startX = Math.max(a, xpd[0]);
  const endX = Math.min(b, xpd[n - 1]);

  if (startX < endX) {
    for (let i = 0; i < n - 1; i++) {
      const x0 = xpd[i];
      const x1 = xpd[i + 1];

      if (x1 <= startX) continue;
      if (x0 >= endX) break;

      const current = Math.max(startX, x0);
      const next = Math.min(endX, x1);
      const h = x1 - x0;

      if (h > 0) {
        const ta = (current - x0) / h;
        const tb = (next - x0) / h;

        const y0 = fpd[i];
        const y1 = fpd[i + 1];
        const d0 = d[i];
        const d1 = d[i + 1];

        const Ia = hermiteBasisIntegral(ta);
        const Ib = hermiteBasisIntegral(tb);
        const valA = y0 * Ia.i00 + h * d0 * Ia.i10 + y1 * Ia.i01 + h * d1 * Ia.i11;
        const valB = y0 * Ib.i00 + h * d0 * Ib.i10 + y1 * Ib.i01 + h * d1 * Ib.i11;

        total += h * (valB - valA);
      }
    }
  }

  return total;
}

/**
 * Component-wise {@link hermiteIntegrate1D} for vector-valued data: returns a
 * `Vector` of length `fp.cols`. Bounds must not be `NaN`.
 */
export function hermiteIntegrateND(
  xp: Vector,
  fp: Matrix,
  d: Matrix,
  leftVal: Vector,
  rightVal: Vector,
  a: number,
  b: number
): Vector {
  const m = fp.cols;
  if (a === b) return new Vector(m);
  if (b < a) return hermiteIntegrateND(xp, fp, d, leftVal, rightVal, b, a).mult(-1);

  const xpd = xp.data;
  const n = xpd.length;

  const total = new Vector(m);

  // 1. Left clamp region (constant area)
  if (a < xpd[0]) {
    const end = Math.min(b, xpd[0]);
    addConstantIntegral(total, leftVal, end - a);
  }

  // 2. Right clamp region (constant area)
  if (b > xpd[n - 1]) {
    const start = Math.max(a, xpd[n - 1]);
    addConstantIntegral(total, rightVal, b - start);
  }

  // 3. Interior cubic regions
  const startX = Math.max(a, xpd[0]);
  const endX = Math.min(b, xpd[n - 1]);

  if (startX < endX) {
    for (let i = 0; i < n - 1; i++) {
      const x0 = xpd[i];
      const x1 = xpd[i + 1];

      if (x1 <= startX) continue;
      if (x0 >= endX) break;

      const current = Math.max(startX, x0);
      const next = Math.min(endX, x1);
      const h = x1 - x0;

      if (h > 0) {
        const ta = (current - x0) / h;
        const tb = (next - x0) / h;

        const y0Off = fp.flatIndex(i, 0);
        const y1Off = fp.flatIndex(i + 1, 0);
        const d0Off = d.flatIndex(i, 0);
        const d1Off = d.flatIndex(i + 1, 0);

        const Ia = hermiteBasisIntegral(ta);
        const Ib = hermiteBasisIntegral(tb);

        // A 4-term weighted combination (y0, y1, d0, d1, each with its own
        // basis coefficient) rather than a running "add one scaled vector"
        // accumulation, so looping over the flat buffers keeps this
        // allocation-free per segment (row()/col() would add 4 allocations).
        for (let j = 0; j < m; j++) {
          const y0 = fp.data[y0Off + j];
          const y1 = fp.data[y1Off + j];
          const dd0 = d.data[d0Off + j];
          const dd1 = d.data[d1Off + j];

          const valA = y0 * Ia.i00 + h * dd0 * Ia.i10 + y1 * Ia.i01 + h * dd1 * Ia.i11;
          const valB = y0 * Ib.i00 + h * dd0 * Ib.i10 + y1 * Ib.i01 + h * dd1 * Ib.i11;

          total.data[j] += h * (valB - valA);
        }
      }
    }
  }

  return total;
}
