import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { prepareInterp1D, prepareInterpND, checkStrictlyIncreasing, findBracket, Interp1DOptions, InterpNDOptions } from './common.js';

/**
 * Precomputes PCHIP gradients (`d[i]`, the slope at knot `i`) ensuring
 * monotonic shape preservation, given one scalar y-series `y` over knots
 * `x`. Shared by `PchipInterpolator1D` (called once, over `fp`) and
 * `PchipInterpolatorND` (called once per output component, over each
 * column of `fp`), since the slope-fitting algorithm itself has no notion
 * of "scalar vs vector output" -- it only ever sees one series at a time.
 * @param x The x-coordinates of the data points (length `n`).
 * @param y The y-coordinates of the data points (length `n`).
 * @param n The number of data points (`x.length === y.length === n`).
 * @returns The fitted slope at each knot.
 */
function computeMonotonicDerivatives1D(x: Float64Array, y: Float64Array, n: number): Float64Array {
    const d = new Float64Array(n);

    if (n === 1) return d;

    if (n === 2) {
        d[0] = d[1] = (y[1] - y[0]) / (x[1] - x[0]);
        return d;
    }

    const hk = new Float64Array(n - 1);
    const mk = new Float64Array(n - 1);
    for (let i = 0; i < n - 1; i++) {
        hk[i] = x[i + 1] - x[i];
        mk[i] = (y[i + 1] - y[i]) / hk[i];
    }

    // Interior derivatives (harmonic mean)
    for (let i = 1; i < n - 1; i++) {
        if (mk[i - 1] * mk[i] <= 0) {
            d[i] = 0;
        } else {
            const w1 = 2 * hk[i] + hk[i - 1];
            const w2 = hk[i] + 2 * hk[i - 1];
            d[i] = (w1 + w2) / (w1 / mk[i - 1] + w2 / mk[i]);
        }
    }

    // Endpoint derivative (left)
    d[0] = ((2 * hk[0] + hk[1]) * mk[0] - hk[0] * mk[1]) / (hk[0] + hk[1]);
    if (Math.sign(d[0]) !== Math.sign(mk[0])) {
        d[0] = 0;
    } else if (Math.sign(mk[0]) !== Math.sign(mk[1]) && Math.abs(d[0]) > 3 * Math.abs(mk[0])) {
        d[0] = 3 * mk[0];
    }

    // Endpoint derivative (right)
    const mN1 = mk[n - 2];
    const mN2 = mk[n - 3];
    const hN1 = hk[n - 2];
    const hN2 = hk[n - 3];
    d[n - 1] = ((2 * hN1 + hN2) * mN1 - hN1 * mN2) / (hN1 + hN2);
    if (Math.sign(d[n - 1]) !== Math.sign(mN1)) {
        d[n - 1] = 0;
    } else if (Math.sign(mN1) !== Math.sign(mN2) && Math.abs(d[n - 1]) > 3 * Math.abs(mN1)) {
        d[n - 1] = 3 * mN1;
    }

    return d;
}

/**
 * Evaluates the integral of the Hermite basis functions from `0` to `t`,
 * i.e. `[intH00, intH10, intH01, intH11](t)`. Shared by
 * `PchipInterpolator1D.integrate` and `PchipInterpolatorND.integrate`,
 * which both need it (the latter once per output component per segment).
 */
function hermiteBasisIntegral(t: number): { i00: number; i10: number; i01: number; i11: number } {
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    return {
        i00: 0.5 * t4 - t3 + t,
        i10: 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2,
        i01: -0.5 * t4 + t3,
        i11: 0.25 * t4 - (1.0 / 3.0) * t3,
    };
}

/**
 * Evaluates the cubic Hermite polynomial or its derivative at a single point.
 */
function pchipOne1D(
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
        const dp10 = (3 * t * t - 4 * t + 1); // h * H'10 / h
        const dp01 = (-6 * t * t + 6 * t) / h;
        const dp11 = (3 * t * t - 2 * t);     // h * H'11 / h

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

function pchipMany1D(
    x: number[] | Vector,
    xp: Vector,
    fp: Vector,
    d: Float64Array,
    leftVal: number,
    rightVal: number,
    computeDerivative: boolean
): Vector {
    const xv = x instanceof Vector ? x : Vector.from(x);
    const res = new Vector(xv.size);
    const xdata = xv.data;
    const rdata = res.data;
    for (let i = 0; i < xv.size; i++) {
        rdata[i] = pchipOne1D(xdata[i], xp, fp, d, leftVal, rightVal, computeDerivative);
    }
    return res;
}

// -----------------------------------------------------------------
// Vector-valued (ND) kernels. Same structure as pchipOne/pchipMany, but
// `fp` and `d` are Matrices (row i = knot i's value/slope vector); the
// bracket/t/basis-function computation is identical across components,
// so it's done once and then looped over fp's columns against the flat
// buffers directly, same style as interp1.ts's *ND kernels.
// -----------------------------------------------------------------

function pchipOneND(
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
        const dp10 = (3 * t * t - 4 * t + 1);
        const dp01 = (-6 * t * t + 6 * t) / h;
        const dp11 = (3 * t * t - 2 * t);
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

function pchipManyND(
    x: number[] | Vector,
    xp: Vector,
    fp: Matrix,
    d: Matrix,
    leftVal: Vector,
    rightVal: Vector,
    computeDerivative: boolean,
    caller: string
): Matrix {
    const xv = x instanceof Vector ? x : Vector.from(x);
    if (xv.size === 0) {
        throw new RangeError(`${caller}: cannot evaluate at zero points; Matrix cannot represent a result with 0 rows`);
    }
    const res = new Matrix(xv.size, fp.cols);
    for (let i = 0; i < xv.size; i++) {
        res.setRow(i, pchipOneND(xv.data[i], xp, fp, d, leftVal, rightVal, computeDerivative));
    }
    return res;
}

/**
 * Builds the per-knot, per-component slope Matrix (`n x m`) for
 * `PchipInterpolatorND`, by running `computeMonotonicDerivatives1D`
 * independently on each column of `fp` (each output component is its
 * own scalar PCHIP problem sharing `xp`).
 */
function computeMonotonicDerivativesND(xp: Vector, fp: Matrix): Matrix {
    const n = xp.size;
    const m = fp.cols;
    const d = new Matrix(n, m);

    for (let j = 0; j < m; j++) {
        d.setCol(j, computeMonotonicDerivatives1D(xp.data, fp.col(j).data, n));
    }
    return d;
}

/**
 * A one-dimensional Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
 *
 * Given the discrete data points `(xp[i], fp[i])`, with `xp` strictly increasing,
 * `eval(x)` returns a shape-preserving cubic interpolation at `x`. Unlike a
 * standard cubic spline, PCHIP does not introduce overshoots between monotonic
 * data points and has a continuous first derivative. For `x` outside the range
 * of `xp`, the result is clamped to the boundary value of `fp` unless
 * `options.left`/`options.right` are given.
 *
 * Validation (matching lengths, non-empty, sorted) happens once, in the
 * constructor, rather than on every evaluation. This makes `PchipInterpolator1D`
 * the better choice when the same `(xp, fp)` pair is evaluated repeatedly and
 * smooth, shape-preserving interpolation is needed.
 *
 * For interpolating a vector-valued function (`number -> Vector`) against
 * the same shared `xp` knots, see `PchipInterpolatorND`.
 *
 * @example
 * ```ts
 * const f = new PchipInterpolator1D([1, 2, 3], [3, 2, 0]);
 * f.eval(2.5); // 1.1458333333333333
 * f.eval([0, 1.5, 3.14]); // Vector(3, 3, 2.6041666666666665, 0)
 * ```
 */
export class PchipInterpolator1D {
    private readonly xp: Vector;
    private readonly fp: Vector;
    private readonly leftVal: number;
    private readonly rightVal: number;
    private readonly d: Float64Array; // Precomputed slopes at knots

    /**
    * @param xp The `x`-coordinates of the data points. Must be
     * strictly increasing and non-empty. A single point defines a constant
     * interpolant.
     * @param fp The `y`-coordinates of the data points. Must have the same
     * length as `xp`.
     * @param options Optional settings; see {@link Interp1DOptions}. Validation
     * (matching lengths, non-empty, and, unless `options.checkSorted` is
     * `false`, strictly increasing) happens once, here in the constructor.
     * If `options.checkSorted` is `false` and `xp` is not actually sorted,
     * `eval`, `derivative`, and `integrate` results are unspecified.
     * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
     * lengths, or (when `options.checkSorted` is `true`) if `xp` is not
     * strictly increasing.
     */
    constructor(xp: number[] | Vector, fp: number[] | Vector, options: Interp1DOptions = {}) {
        const { left, right, checkSorted = true } = options;
        const p = prepareInterp1D(xp, fp, left, right, checkSorted, 'PchipInterpolator1D');
        if (checkSorted) {
            checkStrictlyIncreasing(p.xp, 'PchipInterpolator1D');
        }
        this.xp = p.xp;
        this.fp = p.fp;
        this.leftVal = p.leftVal;
        this.rightVal = p.rightVal;
        this.d = computeMonotonicDerivatives1D(this.xp.data, this.fp.data, this.size);
    }

    /**
     * The number of data points backing this interpolant.
     */
    get size(): number {
        return this.xp.size;
    }

    /**
     * Evaluates the shape-preserving cubic interpolant at `x`.
     * @param x The x-coordinate(s) at which to evaluate. A single `number` returns a `number`;
     * a plain array or `Vector` returns an `Vector`.
     * @returns The interpolated or clamped value(s), matching the shape of `x`.
     */
    eval(x: number): number;
    eval(x: number[] | Vector): Vector;
    eval(x: number | number[] | Vector): number | Vector {
        if (typeof x === 'number') {
            return pchipOne1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
        }
        return pchipMany1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
    }

    /**
     * Evaluates the first derivative of the interpolant at `x`.
     * Values outside the data range return `0`, the derivative of the
     * constant clamping regions.
     * @param x The `x`-coordinate(s) at which to evaluate. A single
     * `number` returns a `number`; a plain array or `Vector` returns an `Vector`.
     * @returns The derivative value(s), matching the shape of `x`.
     */
    derivative(x: number): number;
    derivative(x: number[] | Vector): Vector;
    derivative(x: number | number[] | Vector): number | Vector {
        if (typeof x === 'number') {
            return pchipOne1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
        }
        return pchipMany1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
    }

    /**
     * Integrates the interpolant from `a` to `b` analytically.
     *
     * Evaluates the exact piecewise cubic polynomial within the data bounds,
     * with `left`/`right` clamp values forming constant rectangular areas
     * outside the `xp` range.
     *
     * @param a Lower integration bound `a`.
     * @param b Upper integration bound `b`.
     * @returns The integral of the interpolant over `[a, b]`.
     * @throws {RangeError} If either bound is `NaN`.
     */
    integrate(a: number, b: number): number {
        if (Number.isNaN(a) || Number.isNaN(b)) {
            throw new RangeError('PchipInterpolator1D.integrate: bounds must not be NaN');
        }
        if (a === b) return 0;
        if (b < a) return -this.integrate(b, a);

        const xpd = this.xp.data;
        const fpd = this.fp.data;
        const d = this.d;
        const n = xpd.length;

        let total = 0;

        // 1. Left clamp region (constant area)
        if (a < xpd[0]) {
            const end = Math.min(b, xpd[0]);
            total += (end - a) * this.leftVal;
        }

        // 2. Right clamp region (constant area)
        if (b > xpd[n - 1]) {
            const start = Math.max(a, xpd[n - 1]);
            total += (b - start) * this.rightVal;
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
}

/**
 * A shape-preserving cubic Hermite interpolant (PCHIP) for a
 * vector-valued function (`number -> Vector`).
 *
 * Given the discrete data points `(xp[i], fp.row(i))`, with `xp` strictly
 * increasing, `eval(x)` returns the shape-preserving cubic interpolation
 * at `x`, applied independently to each output component (each column of
 * `fp` gets its own PCHIP fit, all sharing the same `xp` knots). For `x`
 * outside the range of `xp`, the result is clamped to the boundary row of
 * `fp` unless `options.left`/`options.right` are given.
 *
 * The vector-valued counterpart of `PchipInterpolator1D`.
 *
 * @example
 * ```ts
 * const f = new PchipInterpolatorND([1, 2, 3], Matrix.from([[3, 0], [2, 1], [0, 4]]));
 * f.eval(2.5); // Vector(1.1458333333333333, 2.5208333333333335)
 * ```
 */
export class PchipInterpolatorND {
    private readonly xp: Vector;
    private readonly fp: Matrix;
    private readonly leftVal: Vector;
    private readonly rightVal: Vector;
    private readonly d: Matrix; // Precomputed slopes at knots, n x dim

    /**
     * @param xp The `x`-coordinates of the data points, shared by every
     * output component. Must be strictly increasing and non-empty. A
     * single point defines a constant interpolant.
     * @param fp The vector-valued data points: row `i` is the value at
     * `xp[i]`. Must have `fp.rows === xp.length`.
     * @param options Optional settings; see {@link InterpNDOptions}.
     * @throws {RangeError} If `xp` is empty, if `xp.length !== fp.rows`,
     * or (when `options.checkSorted` is `true`) if `xp` is not strictly
     * increasing.
     */
    constructor(xp: number[] | Vector, fp: Matrix, options: InterpNDOptions = {}) {
        const { left, right, checkSorted = true } = options;
        const p = prepareInterpND(xp, fp, left, right, checkSorted, 'PchipInterpolatorND');
        if (checkSorted) {
            checkStrictlyIncreasing(p.xp, 'PchipInterpolatorND');
        }
        this.xp = p.xp;
        this.fp = p.fp;
        this.leftVal = p.leftVal;
        this.rightVal = p.rightVal;
        this.d = computeMonotonicDerivativesND(this.xp, this.fp);
    }

    /**
     * The number of data points (knots) backing this interpolant.
     */
    get size(): number {
        return this.xp.size;
    }

    /**
     * The dimension of the interpolated output vector (`fp.cols`).
     */
    get dim(): number {
        return this.fp.cols;
    }

    /**
     * Evaluates the shape-preserving cubic interpolant at `x`.
     * @param x The x-coordinate(s) at which to evaluate. A single `number`
     * returns a `Vector` (length `dim`); a plain array or `Vector` returns
     * a `Matrix` (one row per evaluation point, `dim` columns).
     * @returns The interpolated or clamped value(s), matching the shape of `x`.
     * @throws {RangeError} If `x` is an array/`Vector` of length `0`
     * (`Matrix` cannot represent a result with `0` rows).
     */
    eval(x: number): Vector;
    eval(x: number[] | Vector): Matrix;
    eval(x: number | number[] | Vector): Vector | Matrix {
        if (typeof x === 'number') {
            return pchipOneND(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
        }
        return pchipManyND(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false, 'PchipInterpolatorND.eval');
    }

    /**
     * Evaluates the first derivative of the interpolant at `x`, component-wise.
     * Values outside the data range return the zero vector, the derivative
     * of the constant clamping regions.
     * @param x The `x`-coordinate(s) at which to evaluate. A single
     * `number` returns a `Vector`; a plain array or `Vector` returns a `Matrix`.
     * @returns The derivative value(s), matching the shape of `x`.
     * @throws {RangeError} If `x` is an array/`Vector` of length `0`
     * (`Matrix` cannot represent a result with `0` rows).
     */
    derivative(x: number): Vector;
    derivative(x: number[] | Vector): Matrix;
    derivative(x: number | number[] | Vector): Vector | Matrix {
        if (typeof x === 'number') {
            return pchipOneND(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
        }
        return pchipManyND(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true, 'PchipInterpolatorND.derivative');
    }

    /**
     * Integrates the interpolant from `a` to `b`, component-wise,
     * analytically. See `PchipInterpolator1D.integrate` for the
     * algorithm; this applies it independently to each output component.
     * @param a Lower integration bound `a`.
     * @param b Upper integration bound `b`.
     * @returns A `Vector` of length `dim`, the integral of each component over `[a, b]`.
     * @throws {RangeError} If either bound is `NaN`.
     */
    integrate(a: number, b: number): Vector {
        if (Number.isNaN(a) || Number.isNaN(b)) {
            throw new RangeError('PchipInterpolatorND.integrate: bounds must not be NaN');
        }
        const m = this.dim;
        if (a === b) return new Vector(m);
        if (b < a) return this.integrate(b, a).mult(-1);

        const xpd = this.xp.data;
        const fp = this.fp;
        const d = this.d;
        const n = xpd.length;

        const total = new Vector(m);

        // 1. Left clamp region (constant area)
        if (a < xpd[0]) {
            const end = Math.min(b, xpd[0]);
            total.addScaled(this.leftVal, end - a);
        }

        // 2. Right clamp region (constant area)
        if (b > xpd[n - 1]) {
            const start = Math.max(a, xpd[n - 1]);
            total.addScaled(this.rightVal, b - start);
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

                    // Unlike the clamp regions above, this is a 4-term
                    // weighted combination (y0, y1, d0, d1, each with its
                    // own basis coefficient) rather than a running "add one
                    // scaled vector" accumulation, so addScaled doesn't fit
                    // as directly; looping over the flat buffers keeps this
                    // allocation-free per segment (row()/col() would add 4
                    // allocations here) and avoids re-deriving combined
                    // coefficients by hand.
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
}