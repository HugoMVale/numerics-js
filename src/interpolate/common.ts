import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';

/**
 * Options controlling out-of-range clamping and input validation for
 * scalar-valued (1D) interpolation methods: `interp1D`, `LinearInterpolator1D`,
 * `PchipInterpolator1D`.
 */
export interface Interp1DOptions {
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
 * Options controlling out-of-range clamping and input validation for
 * vector-valued (ND) interpolation methods: `interpND`, `LinearInterpolatorND`,
 * `PchipInterpolatorND`. The vector counterpart of {@link Interp1DOptions}:
 * `left`/`right` are per-component here (one value per column of `fp`)
 * rather than a single scalar, but a bare `number` is still accepted and
 * broadcasts to every component.
 */
export interface InterpNDOptions {
    /**
     * Value(s) to return for `x < xp[0]`. A single `number` broadcasts to
     * every component; a `number[]`/`Vector` must have length `fp.cols`.
     * Defaults to `fp.row(0)`.
     */
    left?: number | number[] | Vector;
    /**
     * Value(s) to return for `x > xp[xp.length - 1]`. A single `number`
     * broadcasts to every component; a `number[]`/`Vector` must have length
     * `fp.cols`. Defaults to `fp.row(fp.rows - 1)`.
     */
    right?: number | number[] | Vector;
    /** Same meaning as {@link Interp1DOptions.checkSorted}. Defaults to `true`. */
    checkSorted?: boolean;
}

/**
 * Validated, ready-to-evaluate interpolation data: `xp`/`fp` as `Vector`,
 * with the `left`/`right` clamp values resolved to concrete numbers.
 */
interface PreparedInterp1D {
    xp: Vector;
    fp: Vector;
    leftVal: number;
    rightVal: number;
}

/**
 * Validated, ready-to-evaluate vector-interpolation data: `xp` as a
 * `Vector` and `fp` as a `Matrix` (row `i` is the vector value at
 * `xp[i]`), with the `left`/`right` clamp values resolved to concrete
 * `Vector`s of length `fp.cols`.
 */
interface PreparedInterpND {
    xp: Vector;
    fp: Matrix;
    leftVal: Vector;
    rightVal: Vector;
}

/**
 * Checks that `xpv` contains no `NaN` and is monotonically increasing
 * (duplicates allowed). Shared by `prepareInterp` and `prepareInterpND` —
 * the two entry points have different `fp` shapes but identical
 * requirements on `xp` itself.
 * @throws {RangeError} If `xpv` contains `NaN` or is not monotonically increasing.
 */
function checkMonotonic(xpv: Vector, caller: string): void {
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

/**
 * Checks that `xpv` is *strictly* increasing (no duplicate knots), the
 * stricter requirement PCHIP places on top of `checkMonotonic`'s
 * non-decreasing check. Shared by `PchipInterpolator1D` and
 * `PchipInterpolatorND`, which both call `prepareInterp`/`prepareInterpND`
 * (non-strict) first and then this, as a separate step, since PCHIP's
 * stricter requirement is specific to it and not to interpolation in general.
 * @throws {RangeError} If any two consecutive `xp` entries are equal.
 */
export function checkStrictlyIncreasing(xpv: Vector, caller: string): void {
    for (let i = 1; i < xpv.size; i++) {
        if (xpv.data[i] === xpv.data[i - 1]) {
            throw new RangeError(`${caller}: xp must be strictly increasing`);
        }
    }
}

/**
 * Locates the bracketing interval in `xpData` for `xi` via binary search:
 * returns `[lo, hi]` such that `xpData[lo] <= xi <= xpData[hi]`, with
 * `hi === lo + 1` except in the single-point (`xpData.length === 1`) case,
 * where `hi === lo === 0`.
 *
 * Shared by every scalar and vector interpolation kernel (linear and
 * PCHIP alike, both 1D and ND) — the bracket-finding step is identical
 * regardless of what's done with `lo`/`hi` afterwards.
 *
 * Callers must ensure `xi` is not `NaN` and lies within
 * `[xpData[0], xpData[xpData.length - 1]]`; those cases are handled by
 * each caller before reaching the search (their clamp/`NaN` values
 * differ, e.g. `0` for a derivative vs `leftVal`/`rightVal` for a value).
 * @param xi The x-coordinate to bracket.
 * @param xpData The (increasing) x-coordinates of the data points.
 * @returns The `[lo, hi]` bracketing indices.
 */
export function findBracket(xi: number, xpData: Float64Array): [number, number] {
    let lo = 0;
    let hi = xpData.length - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >>> 1;
        if (xpData[mid] <= xi) lo = mid;
        else hi = mid;
    }
    return [lo, hi];
}

/**
 * Normalizes and validates `xp`/`fp`/`left`/`right` for scalar (1D)
 * interpolation. Shared by `interp1D` and `LinearInterpolator1D`/
 * `PchipInterpolator1D` so validation and defaulting logic exists in
 * exactly one place, without either public API depending on the other.
 * @param xp The x-coordinates of the data points.
 * @param fp The y-coordinates of the data points.
 * @param left Value to return for `x < xp[0]`, or `undefined` to default to `fp[0]`.
 * @param right Value to return for `x > xp[last]`, or `undefined` to default to `fp[last]`.
 * @param checkSorted Whether to verify that `xp` is monotonically increasing.
 * @param caller Name of the public entry point invoking this check, used
 * to produce a precise error message (e.g. `"interp1D"` or `"LinearInterpolator1D"`).
 * @returns The validated `xp`/`fp` as `Vector`, plus resolved clamp values.
 * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
 * lengths, or (when `checkSorted` is `true`) if `xp` is not monotonically
 * increasing.
 */
export function prepareInterp1D(
    xp: number[] | Vector,
    fp: number[] | Vector,
    left: number | undefined,
    right: number | undefined,
    checkSorted: boolean,
    caller: string
): PreparedInterp1D {
    const xpv = xp instanceof Vector ? xp : Vector.from(xp);
    const fpv = fp instanceof Vector ? fp : Vector.from(fp);

    if (xpv.size === 0) {
        throw new RangeError(`${caller}: xp must have at least one element`);
    }
    if (xpv.size !== fpv.size) {
        throw new RangeError(`${caller}: xp and fp must have the same length: ${xpv.size} vs ${fpv.size}`);
    }
    if (checkSorted) {
        checkMonotonic(xpv, caller);
    }

    return {
        xp: xpv,
        fp: fpv,
        leftVal: left ?? fpv.get(0),
        rightVal: right ?? fpv.get(fpv.size - 1),
    };
}

/**
 * Resolves an `InterpNDOptions.left`/`right` value against `fp`'s column
 * count: `undefined` falls back to `fallback` (a row of `fp`), a bare
 * `number` broadcasts to every component, and a `number[]`/`Vector` is
 * validated to have exactly `dim` entries.
 * @throws {RangeError} If an explicit `number[]`/`Vector` doesn't have length `dim`.
 */
function resolveClampVector(
    val: number | number[] | Vector | undefined,
    fallback: Vector,
    dim: number,
    paramName: string,
    caller: string
): Vector {
    if (val === undefined) return fallback;
    if (typeof val === 'number') return Vector.full(dim, val);
    const v = val instanceof Vector ? val : Vector.from(val);
    if (v.size !== dim) {
        throw new RangeError(`${caller}: ${paramName} must have length ${dim} (fp.cols), got ${v.size}`);
    }
    return v;
}

/**
 * Normalizes and validates `xp`/`fp`/`left`/`right` for vector-valued
 * (ND) interpolation. The ND counterpart of `prepareInterp`: `fp` is a
 * `Matrix` (row `i` is the vector value at `xp[i]`) rather than a
 * `Vector`, and `left`/`right` resolve to `Vector`s of length `fp.cols`
 * rather than plain numbers. Shared by `interpND` and
 * `LinearInterpolatorND`/`PchipInterpolatorND`.
 * @param xp The x-coordinates of the data points (the shared knots for
 * every output component).
 * @param fp The vector-valued data points, one row per knot in `xp`.
 * @param left Value(s) to return for `x < xp[0]`, or `undefined` to default to `fp.row(0)`.
 * @param right Value(s) to return for `x > xp[last]`, or `undefined` to default to `fp.row(fp.rows - 1)`.
 * @param checkSorted Whether to verify that `xp` is monotonically increasing.
 * @param caller Name of the public entry point invoking this check.
 * @returns The validated `xp` (`Vector`) and `fp` (`Matrix`), plus resolved clamp vectors.
 * @throws {RangeError} If `xp` is empty, if `xp.size !== fp.rows`, if an
 * explicit `left`/`right` doesn't have length `fp.cols`, or (when
 * `checkSorted` is `true`) if `xp` is not monotonically increasing.
 */
export function prepareInterpND(
    xp: number[] | Vector,
    fp: Matrix,
    left: number | number[] | Vector | undefined,
    right: number | number[] | Vector | undefined,
    checkSorted: boolean,
    caller: string
): PreparedInterpND {
    const xpv = xp instanceof Vector ? xp : Vector.from(xp);

    if (xpv.size === 0) {
        throw new RangeError(`${caller}: xp must have at least one element`);
    }
    if (xpv.size !== fp.rows) {
        throw new RangeError(`${caller}: xp and fp must have the same length: ${xpv.size} vs ${fp.rows} (fp.rows)`);
    }
    if (checkSorted) {
        checkMonotonic(xpv, caller);
    }

    return {
        xp: xpv,
        fp,
        leftVal: resolveClampVector(left, fp.row(0), fp.cols, 'left', caller),
        rightVal: resolveClampVector(right, fp.row(fp.rows - 1), fp.cols, 'right', caller),
    };
}