/** Supported pseudo-random generator algorithms for {@link createRng}. */
export type RngMethod = 'mulberry32';

/**
 * Creates a deterministic, seedable pseudo-random number generator.
 *
 * Each call to the returned function yields a uniform value in `[0, 1)`. The same seed and
 * method always reproduce the same sequence, on every platform, and generators are independent
 * of one another. Not suitable for cryptographic use.
 *
 * @param seed Finite number. It is converted to a 32-bit integer (truncated toward zero, then
 * taken modulo 2^32), so seeds that differ by a multiple of 2^32 give the same sequence.
 * @param method Generator algorithm; the default is `'mulberry32'`.
 * @returns A function returning the next pseudo-random number in `[0, 1)`.
 * @throws {TypeError} If `seed` is not a number.
 * @throws {RangeError} If `seed` is not finite or `method` is unknown.
 *
 * @example
 * ```ts
 * import { createRng } from 'numerics-js/math';
 *
 * const next = createRng(42);
 * console.log(next(), next());
 * ```
 *
 * Output:
 * ```text
 * 0.6011037519201636 0.44829055899754167
 * ```
 */
export function createRng(seed: number, method: RngMethod = 'mulberry32'): () => number {
  if (typeof seed !== 'number') {
    throw new TypeError(`createRng: seed must be a number, got ${typeof seed}.`);
  }
  if (!Number.isFinite(seed)) {
    throw new RangeError(`createRng: seed must be finite, got ${seed}.`);
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
  throw new RangeError(`createRng: unknown method '${String(method)}'.`);
}
