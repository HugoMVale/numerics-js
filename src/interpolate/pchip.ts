import { Array1D } from '../array/array1d.js';
import { prepareInterp, InterpOptions } from './common.js';

/**
 * Evaluates the cubic Hermite polynomial or its derivative at a single point.
 */
function pchipOne(
    xi: number,
    xp: Array1D,
    fp: Array1D,
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

    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >>> 1;
        if (xpd[mid] <= xi) lo = mid;
        else hi = mid;
    }

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

function pchipMany(
    x: number[] | Array1D,
    xp: Array1D,
    fp: Array1D,
    d: Float64Array,
    leftVal: number,
    rightVal: number,
    computeDerivative: boolean
): Array1D {
    const xv = x instanceof Array1D ? x : Array1D.from(x);
    const res = new Array1D(xv.size);
    const xdata = xv.data;
    const rdata = res.data;
    for (let i = 0; i < xv.size; i++) {
        rdata[i] = pchipOne(xdata[i], xp, fp, d, leftVal, rightVal, computeDerivative);
    }
    return res;
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
 * constructor, rather than on every evaluation. This makes `PchipInterpolator`
 * the better choice when the same `(xp, fp)` pair is evaluated repeatedly and
 * smooth, shape-preserving interpolation is needed.
 *
 * @example
 * ```ts
 * const f = new PchipInterpolator([1, 2, 3], [3, 2, 0]);
 * f.eval(2.5); // 1.1458333333333333
 * f.eval([0, 1.5, 3.14]); // Array1D(3, 3, 2.6041666666666665, 0)
 * ```
 */
export class PchipInterpolator {
    private readonly xp: Array1D;
    private readonly fp: Array1D;
    private readonly leftVal: number;
    private readonly rightVal: number;
    private readonly d: Float64Array; // Precomputed slopes at knots

    /**
     * @param xp The x-coordinates of the data points. Must be
     * strictly increasing and non-empty. A single point defines a constant
     * interpolant.
     * @param fp The y-coordinates of the data points. Must have the same
     * length as `xp`.
     * @param options Optional settings; see {@link InterpOptions}. Validation
     * (matching lengths, non-empty, and, unless `options.checkSorted` is
     * `false`, strictly increasing) happens once, here in the constructor.
     * If `options.checkSorted` is `false` and `xp` is not actually sorted,
     * `eval`, `derivative`, and `integrate` results are unspecified.
     * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
     * lengths, or (when `options.checkSorted` is `true`) if `xp` is not
     * strictly increasing.
     */
    constructor(xp: number[] | Array1D, fp: number[] | Array1D, options: InterpOptions = {}) {
        const { left, right, checkSorted = true } = options;
        const p = prepareInterp(xp, fp, left, right, checkSorted, 'PchipInterpolator');
        if (checkSorted) {
            for (let i = 1; i < p.xp.size; i++) {
                if (p.xp.data[i] === p.xp.data[i - 1]) {
                    throw new RangeError('PchipInterpolator: xp must be strictly increasing');
                }
            }
        }
        this.xp = p.xp;
        this.fp = p.fp;
        this.leftVal = p.leftVal;
        this.rightVal = p.rightVal;
        this.d = this.computeDerivatives(this.xp.data, this.fp.data);
    }

    /**
     * The number of data points backing this interpolant.
     */
    get size(): number {
        return this.xp.size;
    }

    /**
     * Precomputes PCHIP gradients ensuring monotonic shape preservation.
     */
    private computeDerivatives(x: Float64Array | number[], y: Float64Array | number[]): Float64Array {
        const n = this.size;
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
     * Evaluates the shape-preserving cubic interpolant at `x`.
     * @param x The x-coordinate(s) at which to evaluate. A single `number` returns a `number`;
     * a plain array or `Array1D` returns an `Array1D`.
     * @returns The interpolated or clamped value(s), matching the shape of `x`.
     */
    eval(x: number): number;
    eval(x: number[] | Array1D): Array1D;
    eval(x: number | number[] | Array1D): number | Array1D {
        if (typeof x === 'number') {
            return pchipOne(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
        }
        return pchipMany(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
    }

    /**
     * Evaluates the first derivative of the interpolant at `x`.
     * Values outside the data range return `0`, the derivative of the
     * constant clamping regions.
     * @param x The x-coordinate(s) at which to evaluate. A single
     * `number` returns a `number`; a plain array or `Array1D` returns an `Array1D`.
     * @returns The derivative value(s), matching the shape of `x`.
     */
    derivative(x: number): number;
    derivative(x: number[] | Array1D): Array1D;
    derivative(x: number | number[] | Array1D): number | Array1D {
        if (typeof x === 'number') {
            return pchipOne(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
        }
        return pchipMany(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
    }

    /**
     * Integrates the interpolant from `a` to `b` analytically.
     *
     * Evaluates the exact piecewise cubic polynomial within the data bounds,
     * with `left`/`right` clamp values forming constant rectangular areas
     * outside the `xp` range.
     *
     * @param a Lower integration bound.
     * @param b Upper integration bound.
     * @returns The integral of the interpolant over `[a, b]`.
     * @throws {RangeError} If either bound is `NaN`.
     */
    integrate(a: number, b: number): number {
        if (Number.isNaN(a) || Number.isNaN(b)) {
            throw new RangeError('PchipInterpolator.integrate: bounds must not be NaN');
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

                    // Helper to evaluate the integrated Hermite basis at t
                    const I = (t: number) => {
                        const t2 = t * t;
                        const t3 = t2 * t;
                        const t4 = t3 * t;

                        const i00 = 0.5 * t4 - t3 + t;
                        const i10 = 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2;
                        const i01 = -0.5 * t4 + t3;
                        const i11 = 0.25 * t4 - (1.0 / 3.0) * t3;

                        return y0 * i00 + h * d0 * i10 + y1 * i01 + h * d1 * i11;
                    };

                    total += h * (I(tb) - I(ta));
                }
            }
        }

        return total;
    }
}