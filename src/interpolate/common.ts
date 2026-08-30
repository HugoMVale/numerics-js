import { Array1D } from '../array/array1d.js';

/**
 * Options controlling out-of-range clamping and input validation for `interp`.
 */
export interface InterpOptions {
    /** Value to return for `x < xp[0]`. Defaults to `fp[0]`. */
    left?: number;
    /** Value to return for `x > xp[xp.length - 1]`. Defaults to `fp[fp.length - 1]`. */
    right?: number;
    /**
     * Whether to verify that `xp` is monotonically increasing before
     * interpolating. Defaults to `true`. This check is `O(xp.size)`; pass
     * `false` to skip it (e.g. in a hot loop where `xp` is reused and
     * already known to be sorted). If `false` and `xp` is not actually
     * sorted, results are unspecified, matching `numpy.interp`, which
     * performs no such check at all.
     */
    checkSorted?: boolean;
}

/**
 * Validated, ready-to-evaluate interpolation data: `xp`/`fp` as `Array1D`,
 * with the `left`/`right` clamp values resolved to concrete numbers.
 */
interface PreparedInterp {
    xp: Array1D;
    fp: Array1D;
    leftVal: number;
    rightVal: number;
}

/**
 * Normalizes and validates `xp`/`fp`/`left`/`right` for interpolation.
 * Shared by both `interp` and `LinearInterpolator ` so validation and defaulting
 * logic exists in exactly one place, without either public API depending on
 * the other.
 * @param xp The x-coordinates of the data points.
 * @param fp The y-coordinates of the data points.
 * @param left Value to return for `x < xp[0]`, or `undefined` to default to `fp[0]`.
 * @param right Value to return for `x > xp[last]`, or `undefined` to default to `fp[last]`.
 * @param checkSorted Whether to verify that `xp` is monotonically increasing.
 * @param caller Name of the public entry point invoking this check, used
 * to produce a precise error message (e.g. `"interp"` or `"LinearInterpolator "`).
 * @returns The validated `xp`/`fp` as `Array1D`, plus resolved clamp values.
 * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
 * lengths, or (when `checkSorted` is `true`) if `xp` is not monotonically
 * increasing.
 */
export function prepareInterp(
    xp: number[] | Array1D,
    fp: number[] | Array1D,
    left: number | undefined,
    right: number | undefined,
    checkSorted: boolean,
    caller: string
): PreparedInterp {
    const xpv = xp instanceof Array1D ? xp : Array1D.from(xp);
    const fpv = fp instanceof Array1D ? fp : Array1D.from(fp);

    if (xpv.size === 0) {
        throw new RangeError(`${caller}: xp must have at least one element`);
    }
    if (xpv.size !== fpv.size) {
        throw new RangeError(`${caller}: xp and fp must have the same length: ${xpv.size} vs ${fpv.size}`);
    }
    if (checkSorted) {
        for (let i = 0; i < xpv.size; i++) {
            if (Number.isNaN(xpv.data[i])) {
                throw new RangeError(`${caller}: xp must not contain NaN`);
            }
        }
        for (let i = 1; i < xpv.size; i++) {
            if (xpv.data[i] < xpv.data[i - 1]) {
                throw new RangeError(`${caller}: xp must be monotonically increasing`);
            }
        }
    }

    return {
        xp: xpv,
        fp: fpv,
        leftVal: left ?? fpv.get(0),
        rightVal: right ?? fpv.get(fpv.size - 1),
    };
}