import { Array1D } from '../array/array1d.js';
import { prepareInterp, InterpOptions } from './common.js';

/**
 * Interpolates a single scalar value, locating the bracketing interval in
 * `xp` via binary search.
 * @param xi The x-coordinate to evaluate.
 * @param xp The (increasing) x-coordinates of the data points.
 * @param fp The y-coordinates of the data points, same size as `xp`.
 * @param leftVal Value to return for `xi < xp[0]`.
 * @param rightVal Value to return for `xi > xp[xp.size - 1]`.
 * @returns The interpolated (or clamped) value at `xi`.
 */
function interpOne(xi: number, xp: Array1D, fp: Array1D, leftVal: number, rightVal: number): number {
    const n = xp.size;
    const xpd = xp.data;
    const fpd = fp.data;

    // NaN propagates, matching Array1D.min/max's NaN-wins convention.
    if (Number.isNaN(xi)) return NaN;
    if (xi < xpd[0]) return leftVal;
    if (xi > xpd[n - 1]) return rightVal;

    // Binary search for lo such that xp[lo] <= xi <= xp[lo + 1].
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >>> 1;
        if (xpd[mid] <= xi) lo = mid;
        else hi = mid;
    }

    const x0 = xpd[lo];
    const x1 = xpd[hi];
    if (x1 === x0) return fpd[lo]; // duplicate x-coordinates: avoid 0/0
    const t = (xi - x0) / (x1 - x0);
    return fpd[lo] + t * (fpd[hi] - fpd[lo]);
}

/**
 * Interpolates each component of `x`, delegating to `interpOne` per element.
 * @param x The x-coordinates to evaluate.
 * @param xp The (increasing) x-coordinates of the data points.
 * @param fp The y-coordinates of the data points, same size as `xp`.
 * @param leftVal Value to return for components below `xp[0]`.
 * @param rightVal Value to return for components above `xp[xp.size - 1]`.
 * @returns The interpolated (or clamped) values, one per component of `x`.
 */
function interpMany(x: number[] | Array1D, xp: Array1D, fp: Array1D, leftVal: number, rightVal: number): Array1D {
    const xv = x instanceof Array1D ? x : Array1D.from(x);
    const res = new Array1D(xv.size);
    const xdata = xv.data;
    const rdata = res.data;
    for (let i = 0; i < xv.size; i++) {
        rdata[i] = interpOne(xdata[i], xp, fp, leftVal, rightVal);
    }
    return res;
}

/**
 * Evaluates the derivative of the piecewise-linear interpolant at one point.
 */
function linearDerivativeOne(xi: number, xp: Array1D, fp: Array1D): number {
    const n = xp.size;
    const xpd = xp.data;
    const fpd = fp.data;

    if (Number.isNaN(xi)) return NaN;
    if (xi < xpd[0] || xi > xpd[n - 1]) return 0;

    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >>> 1;
        if (xpd[mid] <= xi) lo = mid;
        else hi = mid;
    }

    const h = xpd[hi] - xpd[lo];
    return h === 0 ? 0 : (fpd[hi] - fpd[lo]) / h;
}

function linearDerivativeMany(x: number[] | Array1D, xp: Array1D, fp: Array1D): Array1D {
    const xv = x instanceof Array1D ? x : Array1D.from(x);
    const res = new Array1D(xv.size);
    const xdata = xv.data;
    const rdata = res.data;
    for (let i = 0; i < xv.size; i++) {
        rdata[i] = linearDerivativeOne(xdata[i], xp, fp);
    }
    return res;
}

/**
 * A reusable one-dimensional linear interpolant.
 *
 * Given the discrete data points `(xp[i], fp[i])`, with `xp` increasing,
 * `eval(x)` returns the linearly interpolated value(s) at `x`. For `x`
 * outside the range of `xp`, the result is clamped to the boundary value of
 * `fp` unless `left`/`right` are given.
 *
 * Validation (matching lengths, non-empty, sorted) happens once, in the
 * constructor, rather than on every evaluation. This makes `LinearInterpolator`
 * the better choice over the standalone `interp` function whenever the same
 * `(xp, fp)` pair is evaluated repeatedly, e.g. inside an optimize loop or
 * when resampling many points against one curve.
 *
 * @example
 * ```ts
 * const f = new LinearInterpolator ([1, 2, 3], [3, 2, 0]);
 * f.eval(2.5); // 1
 * f.eval([0, 1.5, 3.14]); // Array1D(3, 2.5, 0)
 * ```
 */
export class LinearInterpolator {
    private readonly xp: Array1D;
    private readonly fp: Array1D;
    private readonly leftVal: number;
    private readonly rightVal: number;

    /**
     * @param xp The x-coordinates of the data points. Must be
     * monotonically increasing (duplicates allowed) and non-empty.
     * @param fp The y-coordinates of the data points. Must have the same
     * length as `xp`.
     * @param left Value to return for `x < xp[0]`. Defaults to `fp[0]`.
     * @param right Value to return for `x > xp[xp.length - 1]`. Defaults
     * to `fp[fp.length - 1]`.
     * @param checkSorted Whether to verify that `xp` is monotonically
     * increasing. Defaults to `true`. This check is `O(xp.size)` and runs
     * once, here in the constructor; pass `false` to skip it if `xp` is
     * already known to be sorted. If `false` and `xp` is not actually
     * sorted, `eval` results are unspecified, matching `numpy.interp`, which
     * performs no such check at all.
     * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
     * lengths, or (when `checkSorted` is `true`) if `xp` is not
     * monotonically increasing.
     */
    constructor(
        xp: number[] | Array1D,
        fp: number[] | Array1D,
        left?: number,
        right?: number,
        checkSorted: boolean = true
    ) {
        const p = prepareInterp(xp, fp, left, right, checkSorted, 'LinearInterpolator');
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
     * `number` returns a `number`; a plain array or `Array1D` returns an
     * `Array1D`.
     * @returns The derivative value(s), matching the shape of `x`.
     */
    derivative(x: number): number;
    derivative(x: number[] | Array1D): Array1D;
    derivative(x: number | number[] | Array1D): number | Array1D {
        if (typeof x === 'number') {
            return linearDerivativeOne(x, this.xp, this.fp);
        }
        return linearDerivativeMany(x, this.xp, this.fp);
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
     * @param a Lower integration bound.
     * @param b Upper integration bound.
     * @returns The integral of the interpolant over `[a, b]`.
     * @throws {RangeError} If either bound is `NaN`.
     */
    integrate(a: number, b: number): number {
        if (Number.isNaN(a) || Number.isNaN(b)) {
            throw new RangeError('LinearInterpolator.integrate: bounds must not be NaN');
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
        let y0 = interpOne(a, this.xp, this.fp, this.leftVal, this.rightVal);

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

        const y1 = interpOne(b, this.xp, this.fp, this.leftVal, this.rightVal);
        total += (b - x0) * (y0 + y1) * 0.5;
        return total;
    }

    /**
     * Evaluates the interpolant at `x`.
     * @param x The x-coordinate(s) at which to evaluate. A single
     * `number` returns a `number`; a plain array or `Array1D` returns an
     * `Array1D`.
     * @returns The interpolated value(s), matching the shape of `x`.
     */
    eval(x: number): number;
    eval(x: number[] | Array1D): Array1D;
    eval(x: number | number[] | Array1D): number | Array1D {
        if (typeof x === 'number') {
            return interpOne(x, this.xp, this.fp, this.leftVal, this.rightVal);
        }
        return interpMany(x, this.xp, this.fp, this.leftVal, this.rightVal);
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
 * `(xp, fp)` pair more than once, construct a `LinearInterpolator` directly to
 * avoid re-validating `xp`/`fp` on every call.
 *
 * @param x The x-coordinate(s) at which to evaluate the interpolated
 * value(s). A single `number` returns a `number`; a plain array or
 * `Array1D` returns an `Array1D`.
 * @param xp The x-coordinates of the data points. Must be monotonically
 * increasing (duplicates allowed) and non-empty.
 * @param fp The y-coordinates of the data points. Must have the same
 * length as `xp`.
 * @param options Optional settings; see `InterpOptions`.
 * @returns The interpolated value(s), matching the shape of `x`.
 * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
 * lengths, or (when `options.checkSorted` is `true`) if `xp` is not
 * monotonically increasing.
 *
 * @example
 * ```ts
 * const xp = [1, 2, 3];
 * const fp = [3, 2, 0];
 * interp(2.5, xp, fp); // 1
 * interp([0, 1, 1.5, 2.72, 3.14], xp, fp); // Array1D(3, 3, 2.5, 0.56, 0)
 * interp(0, xp, fp, { left: -1 }); // -1
 * ```
 */
export function interp(
    x: number,
    xp: number[] | Array1D,
    fp: number[] | Array1D,
    options?: InterpOptions
): number;
export function interp(
    x: number[] | Array1D,
    xp: number[] | Array1D,
    fp: number[] | Array1D,
    options?: InterpOptions
): Array1D;
export function interp(
    x: number | number[] | Array1D,
    xp: number[] | Array1D,
    fp: number[] | Array1D,
    options: InterpOptions = {}
): number | Array1D {
    const { left, right, checkSorted = true } = options;
    const { xp: xpv, fp: fpv, leftVal, rightVal } = prepareInterp(xp, fp, left, right, checkSorted, 'interp');

    if (typeof x === 'number') {
        return interpOne(x, xpv, fpv, leftVal, rightVal);
    }
    return interpMany(x, xpv, fpv, leftVal, rightVal);
}