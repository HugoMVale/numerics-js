import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { prepareInterp1D, prepareInterpND, findBracket, Interp1DOptions, InterpNDOptions } from './common.js';

/**
 * Interpolates a single scalar value, locating the bracketing interval in
 * `xp` via `findBracket`.
 * @param xi The x-coordinate to evaluate.
 * @param xp The (increasing) x-coordinates of the data points.
 * @param fp The y-coordinates of the data points, same size as `xp`.
 * @param leftVal Value to return for `xi < xp[0]`.
 * @param rightVal Value to return for `xi > xp[xp.size - 1]`.
 * @returns The interpolated (or clamped) value at `xi`.
 */
function interpOne1D(xi: number, xp: Vector, fp: Vector, leftVal: number, rightVal: number): number {
    const n = xp.size;
    const xpd = xp.data;
    const fpd = fp.data;

    // NaN propagates, matching Vector.min/max's NaN-wins convention.
    if (Number.isNaN(xi)) return NaN;
    if (xi < xpd[0]) return leftVal;
    if (xi > xpd[n - 1]) return rightVal;

    const [lo, hi] = findBracket(xi, xpd);

    const x0 = xpd[lo];
    const x1 = xpd[hi];
    if (x1 === x0) return fpd[lo]; // duplicate x-coordinates: avoid 0/0
    const t = (xi - x0) / (x1 - x0);
    return fpd[lo] + t * (fpd[hi] - fpd[lo]);
}

/**
 * Interpolates each component of `x`, delegating to `interpOne1D` per element.
 * @param x The x-coordinates to evaluate.
 * @param xp The (increasing) x-coordinates of the data points.
 * @param fp The y-coordinates of the data points, same size as `xp`.
 * @param leftVal Value to return for components below `xp[0]`.
 * @param rightVal Value to return for components above `xp[xp.size - 1]`.
 * @returns The interpolated (or clamped) values, one per component of `x`.
 */
function interpMany1D(x: number[] | Vector, xp: Vector, fp: Vector, leftVal: number, rightVal: number): Vector {
    const xv = x instanceof Vector ? x : Vector.from(x);
    const res = new Vector(xv.size);
    const xdata = xv.data;
    const rdata = res.data;
    for (let i = 0; i < xv.size; i++) {
        rdata[i] = interpOne1D(xdata[i], xp, fp, leftVal, rightVal);
    }
    return res;
}

/**
 * Evaluates the derivative of the piecewise-linear interpolant at one point.
 */
function linearDerivativeOne1D(xi: number, xp: Vector, fp: Vector): number {
    const n = xp.size;
    const xpd = xp.data;
    const fpd = fp.data;

    if (Number.isNaN(xi)) return NaN;
    if (xi < xpd[0] || xi > xpd[n - 1]) return 0;

    const [lo, hi] = findBracket(xi, xpd);

    const h = xpd[hi] - xpd[lo];
    return h === 0 ? 0 : (fpd[hi] - fpd[lo]) / h;
}

function linearDerivativeMany1D(x: number[] | Vector, xp: Vector, fp: Vector): Vector {
    const xv = x instanceof Vector ? x : Vector.from(x);
    const res = new Vector(xv.size);
    const xdata = xv.data;
    const rdata = res.data;
    for (let i = 0; i < xv.size; i++) {
        rdata[i] = linearDerivativeOne1D(xdata[i], xp, fp);
    }
    return res;
}

// -----------------------------------------------------------------
// Vector-valued (ND) kernels. `fp` is a Matrix (row i = the vector value
// at xp[i]); the bracket/t computation is identical to the scalar case
// (same xp for every output component), so each kernel below does that
// bracket search once and then loops over fp's columns directly against
// the underlying flat buffers -- mirroring Matrix.matmul/mulVec's style
// of avoiding a per-component Vector allocation in the hot loop.
// -----------------------------------------------------------------

function interpOneND(xi: number, xp: Vector, fp: Matrix, leftVal: Vector, rightVal: Vector): Vector {
    const n = xp.size;
    const xpd = xp.data;
    const m = fp.cols;

    if (Number.isNaN(xi)) return Vector.full(m, NaN);
    if (xi < xpd[0]) return leftVal.copy();
    if (xi > xpd[n - 1]) return rightVal.copy();

    const [lo, hi] = findBracket(xi, xpd);
    const x0 = xpd[lo];
    const x1 = xpd[hi];
    const loOff = fp.flatIndex(lo, 0);
    const hiOff = fp.flatIndex(hi, 0);

    const res = new Vector(m);
    if (x1 === x0) {
        for (let j = 0; j < m; j++) res.data[j] = fp.data[loOff + j];
        return res;
    }
    const t = (xi - x0) / (x1 - x0);
    for (let j = 0; j < m; j++) {
        const y0 = fp.data[loOff + j];
        const y1 = fp.data[hiOff + j];
        res.data[j] = y0 + t * (y1 - y0);
    }
    return res;
}

function interpManyND(x: number[] | Vector, xp: Vector, fp: Matrix, leftVal: Vector, rightVal: Vector, caller: string): Matrix {
    const xv = x instanceof Vector ? x : Vector.from(x);
    if (xv.size === 0) {
        throw new RangeError(`${caller}: cannot evaluate at zero points; Matrix cannot represent a result with 0 rows`);
    }
    const m = fp.cols;
    const res = new Matrix(xv.size, m);
    for (let i = 0; i < xv.size; i++) {
        const row = interpOneND(xv.data[i], xp, fp, leftVal, rightVal);
        res.setRow(i, row);
    }
    return res;
}

function linearDerivativeOneND(xi: number, xp: Vector, fp: Matrix): Vector {
    const n = xp.size;
    const xpd = xp.data;
    const m = fp.cols;

    if (Number.isNaN(xi)) return Vector.full(m, NaN);
    if (xi < xpd[0] || xi > xpd[n - 1]) return new Vector(m);

    const [lo, hi] = findBracket(xi, xpd);
    const h = xpd[hi] - xpd[lo];

    const res = new Vector(m);
    if (h === 0) return res;
    const loOff = fp.flatIndex(lo, 0);
    const hiOff = fp.flatIndex(hi, 0);
    for (let j = 0; j < m; j++) {
        res.data[j] = (fp.data[hiOff + j] - fp.data[loOff + j]) / h;
    }
    return res;
}

function linearDerivativeManyND(x: number[] | Vector, xp: Vector, fp: Matrix, caller: string): Matrix {
    const xv = x instanceof Vector ? x : Vector.from(x);
    if (xv.size === 0) {
        throw new RangeError(`${caller}: cannot evaluate at zero points; Matrix cannot represent a result with 0 rows`);
    }
    const m = fp.cols;
    const res = new Matrix(xv.size, m);
    for (let i = 0; i < xv.size; i++) {
        const row = linearDerivativeOneND(xv.data[i], xp, fp);
        res.setRow(i, row);
    }
    return res;
}

/**
 * A reusable one-dimensional linear interpolant.
 *
 * Given the discrete data points `(xp[i], fp[i])`, with `xp` increasing,
 * `eval(x)` returns the linearly interpolated value(s) at `x`. For `x`
 * outside the range of `xp`, the result is clamped to the boundary value of
 * `fp` unless `options.left`/`options.right` are given.
 *
 * Validation (matching lengths, non-empty, sorted) happens once, in the
 * constructor, rather than on every evaluation. This makes `LinearInterpolator1D`
 * the better choice over the standalone `interp1D` function whenever the same
 * `(xp, fp)` pair is evaluated repeatedly, e.g. inside an optimize loop or
 * when resampling many points against one curve.
 *
 * For interpolating a vector-valued function (`number -> Vector`) against
 * the same shared `xp` knots, see {@link LinearInterpolatorND}.
 *
 * @example
 * ```ts
 * const f = new LinearInterpolator1D([1, 2, 3], [3, 2, 0]);
 * f.eval(2.5); // 1
 * f.eval([0, 1.5, 3.14]); // Vector(3, 2.5, 0)
 * ```
 */
export class LinearInterpolator1D {
    private readonly xp: Vector;
    private readonly fp: Vector;
    private readonly leftVal: number;
    private readonly rightVal: number;

    /**
     * @param xp The `x`-coordinates of the data points. Must be
     * monotonically increasing (duplicates allowed) and non-empty.
     * @param fp The `y`-coordinates of the data points. Must have the same
     * length as `xp`.
     * @param options Optional settings; see {@link Interp1DOptions}. Validation
     * (matching lengths, non-empty, and, unless `options.checkSorted` is
     * `false`, sorted) happens once, here in the constructor.
     * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
     * lengths, or (when `options.checkSorted` is `true`) if `xp` is not
     * monotonically increasing.
     */
    constructor(xp: number[] | Vector, fp: number[] | Vector, options: Interp1DOptions = {}) {
        const { left, right, checkSorted = true } = options;
        const p = prepareInterp1D(xp, fp, left, right, checkSorted, 'LinearInterpolator1D');
        this.xp = p.xp;
        this.fp = p.fp;
        this.leftVal = p.leftVal;
        this.rightVal = p.rightVal;
    }

    /**
     * The number of data points backing this interpolant.
     */
    get size(): number {
        return this.xp.size;
    }

    /**
     * Evaluates the first derivative of the interpolant at `x`.
     * Values outside the data range return `0`, the derivative of the
     * constant clamping regions. At a knot, this returns the slope of the
     * segment to its right (or the final segment at the right endpoint).
     * @param x The x-coordinate(s) at which to evaluate. A single
     * `number` returns a `number`; a plain array or `Vector` returns an
     * `Vector`.
     * @returns The derivative value(s), matching the shape of `x`.
     */
    derivative(x: number): number;
    derivative(x: number[] | Vector): Vector;
    derivative(x: number | number[] | Vector): number | Vector {
        if (typeof x === 'number') {
            return linearDerivativeOne1D(x, this.xp, this.fp);
        }
        return linearDerivativeMany1D(x, this.xp, this.fp);
    }

    /**
     * Integrates the interpolant from `a` to `b` using the trapezoid rule.
     *
     * The integration includes every interpolation knot between the bounds,
     * with `left`/`right` clamp values used outside the `xp` range. Since the
     * interpolant is piecewise linear, applying the trapezoid rule at the
     * interpolation knots is exact (up to floating-point rounding). Reversed
     * bounds return the negative of the integral with the bounds swapped.
     *
     * @param a Lower integration bound `a`.
     * @param b Upper integration bound `b`.
     * @returns The integral of the interpolant over `[a, b]`.
     * @throws {RangeError} If either bound is `NaN`.
     */
    integrate(a: number, b: number): number {
        if (Number.isNaN(a) || Number.isNaN(b)) {
            throw new RangeError('LinearInterpolator1D.integrate: bounds must not be NaN');
        }
        if (a === b) return 0;
        if (b < a) return -this.integrate(b, a);

        const xpd = this.xp.data;
        const fpd = this.fp.data;
        const n = xpd.length;

        // Integrate over the intervals formed by the requested bounds and
        // every xp knot inside them. For a piecewise-linear interpolant, the
        // trapezoid rule over these intervals is exact.
        let total = 0;
        let x0 = a;
        let y0 = interpOne1D(a, this.xp, this.fp, this.leftVal, this.rightVal);

        // Skip duplicate knots naturally; zero-width trapezoids contribute 0.
        for (let i = 0; i < n; i++) {
            const x1 = xpd[i];
            if (x1 <= a) continue;
            if (x1 >= b) break;

            const y1 = fpd[i];
            total += (x1 - x0) * (y0 + y1) * 0.5;
            x0 = x1;
            y0 = y1;
        }

        const y1 = interpOne1D(b, this.xp, this.fp, this.leftVal, this.rightVal);
        total += (b - x0) * (y0 + y1) * 0.5;
        return total;
    }

    /**
     * Evaluates the interpolant at `x`.
     * @param x The x-coordinate(s) at which to evaluate. A single
     * `number` returns a `number`; a plain array or `Vector` returns an
     * `Vector`.
     * @returns The interpolated value(s), matching the shape of `x`.
     */
    eval(x: number): number;
    eval(x: number[] | Vector): Vector;
    eval(x: number | number[] | Vector): number | Vector {
        if (typeof x === 'number') {
            return interpOne1D(x, this.xp, this.fp, this.leftVal, this.rightVal);
        }
        return interpMany1D(x, this.xp, this.fp, this.leftVal, this.rightVal);
    }
}

/**
 * A reusable linear interpolant for a vector-valued function
 * (`number -> Vector`).
 *
 * Given the discrete data points `(xp[i], fp.row(i))`, with `xp`
 * increasing, `eval(x)` returns the linearly interpolated vector at `x` --
 * every output component is interpolated against the same shared `xp`
 * knots. For `x` outside the range of `xp`, the result is clamped to the
 * boundary row of `fp` unless `options.left`/`options.right` are given.
 *
 * The vector-valued counterpart of `LinearInterpolator1D`: use this when
 * interpolating trajectories, multi-channel signals, or any `Vector`-valued
 * curve, rather than calling `LinearInterpolator1D` once per component.
 *
 * @example
 * ```ts
 * const f = new LinearInterpolatorND([0, 1, 2], Matrix.from([[0, 0], [1, 2], [4, 2]]));
 * f.eval(0.5); // Vector(0.5, 1)
 * f.eval([0, 1.5]); // Matrix [[0, 0], [2.5, 2]]
 * ```
 */
export class LinearInterpolatorND {
    private readonly xp: Vector;
    private readonly fp: Matrix;
    private readonly leftVal: Vector;
    private readonly rightVal: Vector;

    /**
     * @param xp The `x`-coordinates of the data points, shared by every
     * output component. Must be monotonically increasing (duplicates
     * allowed) and non-empty.
     * @param fp The vector-valued data points: row `i` is the value at
     * `xp[i]`. Must have `fp.rows === xp.length`.
     * @param options Optional settings; see {@link InterpNDOptions}.
     * @throws {RangeError} If `xp` is empty, if `xp.length !== fp.rows`,
     * or (when `options.checkSorted` is `true`) if `xp` is not
     * monotonically increasing.
     */
    constructor(xp: number[] | Vector, fp: Matrix, options: InterpNDOptions = {}) {
        const { left, right, checkSorted = true } = options;
        const p = prepareInterpND(xp, fp, left, right, checkSorted, 'LinearInterpolatorND');
        this.xp = p.xp;
        this.fp = p.fp;
        this.leftVal = p.leftVal;
        this.rightVal = p.rightVal;
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
     * Evaluates the first derivative of the interpolant at `x`, component-wise.
     * Values outside the data range return the zero vector, the derivative
     * of the constant clamping regions.
     * @param x The x-coordinate(s) at which to evaluate. A single `number`
     * returns a `Vector`; a plain array or `Vector` returns a `Matrix`
     * (one row per evaluation point).
     * @returns The derivative value(s), matching the shape of `x`.
     * @throws {RangeError} If `x` is an array/`Vector` of length `0`
     * (`Matrix` cannot represent a result with `0` rows).
     */
    derivative(x: number): Vector;
    derivative(x: number[] | Vector): Matrix;
    derivative(x: number | number[] | Vector): Vector | Matrix {
        if (typeof x === 'number') {
            return linearDerivativeOneND(x, this.xp, this.fp);
        }
        return linearDerivativeManyND(x, this.xp, this.fp, 'LinearInterpolatorND.derivative');
    }

    /**
     * Integrates the interpolant from `a` to `b`, component-wise, using
     * the trapezoid rule. See `LinearInterpolator1D.integrate` for the
     * algorithm; this applies it independently to each output component.
     * @param a Lower integration bound `a`.
     * @param b Upper integration bound `b`.
     * @returns A `Vector` of length `dim`, the integral of each component over `[a, b]`.
     * @throws {RangeError} If either bound is `NaN`.
     */
    integrate(a: number, b: number): Vector {
        if (Number.isNaN(a) || Number.isNaN(b)) {
            throw new RangeError('LinearInterpolatorND.integrate: bounds must not be NaN');
        }
        const m = this.dim;
        if (a === b) return new Vector(m);
        if (b < a) return this.integrate(b, a).mult(-1);

        const xpd = this.xp.data;
        const fp = this.fp;
        const n = xpd.length;

        const total = new Vector(m);
        let x0 = a;
        let y0 = interpOneND(a, this.xp, this.fp, this.leftVal, this.rightVal);

        for (let i = 0; i < n; i++) {
            const x1 = xpd[i];
            if (x1 <= a) continue;
            if (x1 >= b) break;

            const y1 = fp.row(i);
            const w = (x1 - x0) * 0.5;
            total.addScaled(y0, w).addScaled(y1, w);
            x0 = x1;
            y0 = y1;
        }

        const y1 = interpOneND(b, this.xp, this.fp, this.leftVal, this.rightVal);
        const wLast = (b - x0) * 0.5;
        total.addScaled(y0, wLast).addScaled(y1, wLast);
        return total;
    }

    /**
     * Evaluates the interpolant at `x`.
     * @param x The x-coordinate(s) at which to evaluate. A single `number`
     * returns a `Vector` (length `dim`); a plain array or `Vector` returns
     * a `Matrix` (one row per evaluation point, `dim` columns).
     * @returns The interpolated value(s), matching the shape of `x`.
     * @throws {RangeError} If `x` is an array/`Vector` of length `0`
     * (`Matrix` cannot represent a result with `0` rows).
     */
    eval(x: number): Vector;
    eval(x: number[] | Vector): Matrix;
    eval(x: number | number[] | Vector): Vector | Matrix {
        if (typeof x === 'number') {
            return interpOneND(x, this.xp, this.fp, this.leftVal, this.rightVal);
        }
        return interpManyND(x, this.xp, this.fp, this.leftVal, this.rightVal, 'LinearInterpolatorND.eval');
    }
}

/**
 * One-dimensional linear interpolation.
 *
 * The algorithm is similar to, and inspired by, `numpy.interp`.
 *
 * Given the discrete data points `(xp[i], fp[i])`, with `xp` increasing,
 * returns the linearly interpolated value(s) at `x`. For `x` outside the
 * range of `xp`, the result is clamped to the boundary value of `fp`
 * unless `options.left`/`options.right` are given.
 *
 * A one-shot convenience function. If you need to evaluate the same
 * `(xp, fp)` pair more than once, construct a `LinearInterpolator1D` directly to
 * avoid re-validating `xp`/`fp` on every call. For a vector-valued `fp`, see {@link interpND}.
 *
 * @param x The `x`-coordinate(s) at which to evaluate the interpolated
 * value(s). A single `number` returns a `number`; a plain array or
 * `Vector` returns an `Vector`.
 * @param xp The `x`-coordinates of the data points. Must be monotonically
 * increasing (duplicates allowed) and non-empty.
 * @param fp The `y`-coordinates of the data points. Must have the same
 * length as `xp`.
 * @param options Optional settings; see {@link Interp1DOptions}.
 * @returns The interpolated value(s), matching the shape of `x`.
 * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
 * lengths, or (when `options.checkSorted` is `true`) if `xp` is not
 * monotonically increasing.
 *
 * @example
 * ```ts
 * const xp = [1, 2, 3];
 * const fp = [3, 2, 0];
 * interp1D(2.5, xp, fp); // 1
 * interp1D([0, 1, 1.5, 2.72, 3.14], xp, fp); // Vector(3, 3, 2.5, 0.56, 0)
 * interp1D(0, xp, fp, { left: -1 }); // -1
 * ```
 */
export function interp1D(
    x: number,
    xp: number[] | Vector,
    fp: number[] | Vector,
    options?: Interp1DOptions
): number;
export function interp1D(
    x: number[] | Vector,
    xp: number[] | Vector,
    fp: number[] | Vector,
    options?: Interp1DOptions
): Vector;
export function interp1D(
    x: number | number[] | Vector,
    xp: number[] | Vector,
    fp: number[] | Vector,
    options: Interp1DOptions = {}
): number | Vector {
    const { left, right, checkSorted = true } = options;
    const { xp: xpv, fp: fpv, leftVal, rightVal } = prepareInterp1D(xp, fp, left, right, checkSorted, 'interp1D');

    if (typeof x === 'number') {
        return interpOne1D(x, xpv, fpv, leftVal, rightVal);
    }
    return interpMany1D(x, xpv, fpv, leftVal, rightVal);
}

/**
 * One-dimensional linear interpolation of a vector-valued function
 * (`number -> Vector`).
 *
 * The vector-valued counterpart of `interp1D`: given the discrete data
 * points `(xp[i], fp.row(i))`, with `xp` increasing, returns the linearly
 * interpolated vector(s) at `x`. For `x` outside the range of `xp`, the
 * result is clamped to the boundary row of `fp` unless
 * `options.left`/`options.right` are given.
 *
 * A one-shot convenience function. If you need to evaluate the same
 * `(xp, fp)` pair more than once, construct a `LinearInterpolatorND`
 * directly to avoid re-validating `xp`/`fp` on every call.
 *
 * @param x The `x`-coordinate(s) at which to evaluate. A single `number`
 * returns a `Vector`; a plain array or `Vector` returns a `Matrix`.
 * @param xp The `x`-coordinates of the data points, shared by every
 * output component. Must be monotonically increasing (duplicates
 * allowed) and non-empty.
 * @param fp The vector-valued data points: row `i` is the value at `xp[i]`.
 * Must have `fp.rows === xp.length`.
 * @param options Optional settings; see {@link InterpNDOptions}.
 * @returns The interpolated value(s), matching the shape of `x`.
 * @throws {RangeError} If `xp` is empty, if `xp.length !== fp.rows`, if
 * `x` is an array/`Vector` of length `0`, or (when
 * `options.checkSorted` is `true`) if `xp` is not monotonically increasing.
 *
 * @example
 * ```ts
 * const xp = [0, 1, 2];
 * const fp = Matrix.from([[0, 0], [1, 2], [4, 2]]);
 * interpND(0.5, xp, fp); // Vector(0.5, 1)
 * ```
 */
export function interpND(
    x: number,
    xp: number[] | Vector,
    fp: Matrix,
    options?: InterpNDOptions
): Vector;
export function interpND(
    x: number[] | Vector,
    xp: number[] | Vector,
    fp: Matrix,
    options?: InterpNDOptions
): Matrix;
export function interpND(
    x: number | number[] | Vector,
    xp: number[] | Vector,
    fp: Matrix,
    options: InterpNDOptions = {}
): Vector | Matrix {
    const { left, right, checkSorted = true } = options;
    const { xp: xpv, fp: fpv, leftVal, rightVal } = prepareInterpND(xp, fp, left, right, checkSorted, 'interpND');

    if (typeof x === 'number') {
        return interpOneND(x, xpv, fpv, leftVal, rightVal);
    }
    return interpManyND(x, xpv, fpv, leftVal, rightVal, 'interpND');
}