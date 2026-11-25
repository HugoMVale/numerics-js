import { Matrix } from './Matrix.js';
import { Vector } from './Vector.js';

/**
 * `scanNonFinite` for the whole matrix, fused with the search of the largest absolute entry, for
 * the routines that need it (the elimination routines scale the matrix by it, see `safeScale` in
 * `Matrix.linalg.ts`): one pass instead of two, and the same errors, raised at the same entry.
 * Internal; not part of the public API.
 * @param A The matrix to scan. Not modified.
 * @param who Name of the public method, for the error message.
 * @returns `NaN` if `A` has a `NaN` entry (and no infinite one), else the largest absolute entry
 * (`0` for a matrix of zeros).
 * @throws {RangeError} If `A` has an infinite entry, even if it also has a `NaN`.
 */
export function scanExtent(A: Matrix, who: string): number {
  const d = A.data;
  let max = 0;
  let nan = false;
  for (let k = 0; k < d.length; k++) {
    const v = d[k];
    // `v - v` is 0 for every finite v and NaN for NaN and ±Infinity.
    if (v - v !== 0) {
      if (v === v) {
        const i = Math.floor(k / A.cols);
        throw new RangeError(
          `Matrix.${who}: matrix contains an infinite entry at (${i}, ${k - i * A.cols}), got ${v}.`
        );
      }
      nan = true;
    } else {
      const a = v < 0 ? -v : v;
      if (a > max) max = a;
    }
  }
  return nan ? NaN : max;
}

/**
 * Applies the library's non-finite input policy to the matrix argument of a
 * factorization, solve or decomposition (see "Linear-algebra error and
 * non-finite policy" in AGENTS.md): any infinite entry is a domain error,
 * while `NaN` is reported to the caller, who then returns an all-`NaN`
 * result instead of running the algorithm.
 *
 * One pass over the data, `O(rows * cols)`: negligible next to the `O(n^3)`
 * work it guards. Internal; not part of the public API.
 * @param A The matrix to scan. Not modified.
 * @param who Name of the public method, for the error message.
 * @param lowerOnly If `true`, only the lower triangle including the diagonal
 * is scanned (for routines that ignore the upper triangle, like `cholesky`).
 * @returns `true` if `A` has a `NaN` entry (and no infinite one), else `false`.
 * @throws {RangeError} If `A` has an infinite entry, even if it also has a `NaN`.
 */
export function scanNonFinite(A: Matrix, who: string, lowerOnly = false): boolean {
  const d = A.data;
  const n = A.cols;
  let nan = false;
  for (let i = 0; i < A.rows; i++) {
    const end = lowerOnly ? Math.min(i + 1, n) : n;
    const off = i * n;
    for (let j = 0; j < end; j++) {
      const v = d[off + j];
      // `v - v` is 0 for every finite v and NaN for NaN and ±Infinity.
      if (v - v !== 0) {
        if (v === v) {
          throw new RangeError(
            `Matrix.${who}: matrix contains an infinite entry at (${i}, ${j}), got ${v}.`
          );
        }
        nan = true;
      }
    }
  }
  return nan;
}

/**
 * Short description of a value's type for error messages (`null`, `undefined`, `Array`, `number`, ...).
 * @param x Any value.
 * @returns `'null'`, the constructor name for an object, else the `typeof` string.
 */
export function typeName(x: unknown): string {
  if (x === null) return 'null';
  if (typeof x === 'object') return (x as object).constructor?.name ?? 'object';
  return typeof x;
}

/**
 * Rejects an `out` argument that is not a `Vector` (a plain array or a typed array
 * would otherwise surface as a misleading size error).
 * @param out The `out` argument as received.
 * @param who Prefix for the message, e.g. `'Matrix.solveLU'`.
 * @throws {TypeError} If `out` is not a `Vector`.
 */
export function checkVectorOut(out: unknown, who: string): void {
  if (!(out instanceof Vector)) {
    throw new TypeError(`${who}: out must be a Vector, got ${typeName(out)}.`);
  }
}

/**
 * Returns the numbers held by a `Vector` or an array-like object (an array, a
 * typed array, ...), rejecting everything else. Strings, numbers, `null`,
 * iterables without a `length` (a `Set`) and other values are refused, since
 * `Float64Array.set` would silently coerce or mis-handle them. Only the
 * container is validated, not the individual elements.
 * @param x The argument as received.
 * @param who Name of the public method, for the message (e.g. `Matrix.setRow`).
 * @param what Name of the argument, for the message.
 * @param expected Description of what was expected, for the message.
 * @returns The underlying values (not copied).
 * @throws {TypeError} If `x` is not a `Vector` or an object with a numeric `length`.
 */
export function asArrayLike(
  x: unknown,
  who: string,
  what: string,
  expected = 'an array-like of numbers or a Vector'
): ArrayLike<number> {
  if (x instanceof Vector) return x.data;
  if (x === null || typeof x !== 'object' || typeof (x as ArrayLike<number>).length !== 'number') {
    throw new TypeError(`${who}: ${what} must be ${expected}, got ${typeName(x)}.`);
  }
  return x as ArrayLike<number>;
}

/**
 * Validates an options argument: `undefined`, `null` or an object.
 * @param options The argument to check.
 * @param who Name of the public method, for the error message.
 * @throws {TypeError} If `options` is anything else (a number, a string, ...).
 */
export function checkOptions(options: unknown, who: string): void {
  if (options !== undefined && options !== null && typeof options !== 'object') {
    throw new TypeError(`Matrix.${who}: options must be an object, got ${typeName(options)}.`);
  }
}

/**
 * Validates a tolerance-like option: a number that is `>= 0` (so not `NaN`).
 * @param value The value to check.
 * @param name Option name, for the error message.
 * @param who Name of the public method, for the error message.
 * @throws {TypeError} If `value` is not a number.
 * @throws {RangeError} If `value` is negative or `NaN`.
 */
export function checkNonNegative(value: unknown, name: string, who: string): void {
  if (typeof value !== 'number') {
    throw new TypeError(`Matrix.${who}: ${name} must be a number, got ${typeName(value)}.`);
  }
  if (!(value >= 0)) throw new RangeError(`Matrix.${who}: ${name} must be >= 0, got ${value}.`);
}

/**
 * Validates an iteration-budget option: a positive integer.
 * @param value The value to check.
 * @param name Option name, for the error message.
 * @param who Name of the public method, for the error message.
 * @throws {TypeError} If `value` is not a number.
 * @throws {RangeError} If `value` is not a positive integer.
 */
export function checkPositiveInteger(value: unknown, name: string, who: string): void {
  if (typeof value !== 'number') {
    throw new TypeError(`Matrix.${who}: ${name} must be a number, got ${typeName(value)}.`);
  }
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`Matrix.${who}: ${name} must be a positive integer, got ${value}.`);
  }
}

/**
 * Checks that `value` is an object whose listed properties are `Matrix`
 * instances, `Vector` instances and arrays, as a factorization record
 * (`{ L, U, perm }`, `{ QR, tau, perm }`, `{ Q, R }`) must be.
 * @param value The argument to check.
 * @param spec Property names grouped by required type.
 * @param who Name of the public method, for the error message.
 * @param role The argument's name, for the error message.
 * @param label Description of the expected record, with its article, e.g. `an LUDecomposition { L, U, perm, sign }`.
 * @throws {TypeError} If `value` is not such a record.
 */
export function checkRecord(
  value: unknown,
  spec: { matrices: string[]; vectors?: string[]; arrays?: string[] },
  who: string,
  role: string,
  label: string
): void {
  const v = value as Record<string, unknown> | null | undefined;
  const ok =
    typeof v === 'object' &&
    v !== null &&
    spec.matrices.every((k) => v[k] instanceof Matrix) &&
    (spec.vectors ?? []).every((k) => v[k] instanceof Vector) &&
    (spec.arrays ?? []).every((k) => Array.isArray(v[k]));
  if (!ok) {
    throw new TypeError(`Matrix.${who}: ${role} must be ${label}, got ${typeName(value)}.`);
  }
}

/**
 * Throws unless `A` is square.
 * @param A The matrix to check.
 * @param who Name of the calling method, without the `Matrix.` prefix.
 * @throws {RangeError} If `A.rows !== A.cols`: "Matrix.who: expected a square matrix, got RxC."
 */
export function assertSquare(A: Matrix, who: string): void {
  if (A.rows !== A.cols)
    throw new RangeError(`Matrix.${who}: expected a square matrix, got ${A.rows}x${A.cols}.`);
}
