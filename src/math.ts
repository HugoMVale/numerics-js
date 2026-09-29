/**
 * Basic scalar mathematical utility functions.
 *
 * @module math
 */

/**
 * Restricts a number to the inclusive range between `lo` and `hi`.
 *
 * @param x The number to clip.
 * @param low The lower bound.
 * @param high The upper bound.
 * @returns The clipped number.
 *
 * @example
 * ```ts
 * import { clip } from 'numerics-js/math';
 *
 * const value = clip(12, 0, 10);
 * console.log(value);
 * ```
 *
 * Output:
 * ```text
 * 10
 * ```
 */
export function clip(x: number, low: number, high: number): number {
    return x < low ? low : x > high ? high : x;
}


/**
 * Returns a value with the magnitude of `magnitude` and the sign of `sign`.
 *
 * Mirrors the semantics of `Math.copysign` in other languages (e.g. Python's
 * `math.copysign`, NumPy's `np.copysign`): a `sign` of zero is treated as
 * positive.
 *
 * @param magnitude Value whose absolute value is used.
 * @param sign Value whose sign is used.
 * @returns `|magnitude|` with the sign of `sign`.
 *
 * @example
 * ```ts
 * import { copysign } from 'numerics-js/math';
 *
 * const value = copysign(3.5, -1);
 * console.log(value);
 * ```
 *
 * Output:
 * ```text
 * -3.5
 * ```
 */
export function copysign(magnitude: number, sign: number): number {
    return sign < 0 ? -Math.abs(magnitude) : Math.abs(magnitude);
}


/**
 * Checks whether two numbers are approximately equal, mirroring numpy.isclose:
 * |a - b| <= atol + rtol * |b|.
 *
 * @param a First value.
 * @param b Second value (reference value for the relative tolerance term).
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @returns True if a and b are within tolerance of each other.
 *
 * @example
 * ```ts
 * import { isClose } from 'numerics-js/math';
 *
 * const result = isClose(Math.PI, 3.14159, 1e-5);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * true
 * ```
 */
export function isClose(a: number, b: number, atol: number, rtol: number = 1e-5): boolean {
    return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

/** Supported pseudo-random generator algorithms for {@link rand}. */
export type RandMethod = 'mulberry32';

/**
 * Creates a deterministic, seedable pseudo-random number generator.
 *
 * Each call to the returned function yields a uniform value in `[0, 1)`.
 * The same seed and method always reproduce the same sequence. Not suitable
 * for cryptographic use.
 *
 * @param seed Integer seed; non-integers are truncated to a 32-bit integer.
 * @param method Generator algorithm. Defaults to `'mulberry32'`.
 * @returns A function returning the next pseudo-random number in `[0, 1)`.
 * @throws {RangeError} If `seed` is not finite or `method` is unknown.
 *
 * @example
 * ```ts
 * import { rand } from 'numerics-js/math';
 *
 * const next = rand(42);
 * console.log(next(), next());
 * ```
 *
 * Output:
 * ```text
 * 0.6011037519201636 0.44829055899754167
 * ```
 */
export function rand(seed: number, method: RandMethod = 'mulberry32'): () => number {
    if (!Number.isFinite(seed)) {
        throw new RangeError(`rand: seed must be finite, got ${seed}`);
    }
    if (method === 'mulberry32') {
        let a = seed | 0;
        return (): number => {
            a = (a + 0x6d2b79f5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }
    throw new RangeError(`rand: unknown method '${String(method)}'`);
}
