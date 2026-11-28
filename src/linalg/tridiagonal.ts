import { Vector, type VectorLike } from './Vector.js';
import { checkVectorOut, typeName } from './checks.js';
import { ldexp, safeScale } from './Matrix.linalg.js';

/**
 * LU factorization of a general tridiagonal matrix `A` with partial pivoting (the scheme of
 * LAPACK `dgttrf`), as returned by `tridiagonalLU()` and consumed by `solveTridiagonalLU()`.
 * Together, `P * A = L * U`, where `L` is unit lower bidiagonal with the multipliers `dl`, `U` is
 * upper triangular with the three diagonals `d`, `du` and `du2`, and `P` is the sequence of row
 * interchanges recorded in `ipiv`.
 *
 * For a matrix whose entries are close to the limits of the double range, the factors are those
 * of `A / scale` (see `scale`); for every ordinary matrix `scale` is `1`.
 */
export interface TridiagonalLU {
  /** Order of the system. */
  n: number;
  /** Multipliers of `L` (length `n - 1`). */
  dl: Float64Array;
  /** Diagonal of `U` (length `n`). */
  d: Float64Array;
  /** First superdiagonal of `U` (length `n - 1`). */
  du: Float64Array;
  /** Second superdiagonal of `U`, created by row interchanges (length `n - 2`, or `0`). */
  du2: Float64Array;
  /** Row interchanges: row `i` was swapped with row `ipiv[i]` (`i` or `i + 1`), zero-based. */
  ipiv: Int32Array;
  /**
   * A power of two `σ`: the factors above are those of `A / σ`, so that a matrix with entries close
   * to the limits of the double range can be eliminated without overflow or underflow. It is `1`
   * unless the largest entry of `A` is outside `[2^-900, 2^900]`.
   */
  scale: number;
}

/**
 * Factors the tridiagonal matrix with subdiagonal `dl`, diagonal `d` and superdiagonal `du`
 * (lengths `n - 1`, `n`, `n - 1`) in place, with partial pivoting as `dgttrf` does: the three
 * arrays are overwritten and become part of the returned factorization. No validation, no scaling
 * and no scan for non-finite entries: this is the allocation-free kernel behind
 * `tridiagonalLU()`, and what the cubic splines use directly.
 *
 * Plain Thomas elimination is not enough for cubic-spline systems: the not-a-knot boundary rows
 * are not diagonally dominant, so pivoting is needed for backward stability.
 * @param dl Subdiagonal (length `n - 1`), overwritten with the multipliers of `L`.
 * @param d Diagonal (length `n`), overwritten with the diagonal of `U`.
 * @param du Superdiagonal (length `n - 1`), overwritten with the first superdiagonal of `U`.
 * @param who Name of the public function, for the error message.
 * @returns The factorization, which shares the three input arrays. Its `scale` is `1`.
 * @throws {Error} If the matrix is singular (a zero pivot is found).
 * @internal
 */
export function factorTridiagonalInPlace(
  dl: Float64Array,
  d: Float64Array,
  du: Float64Array,
  who = 'tridiagonalLU'
): TridiagonalLU {
  const n = d.length;
  const du2 = new Float64Array(Math.max(n - 2, 0));
  const ipiv = new Int32Array(n);

  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(d[i]) >= Math.abs(dl[i])) {
      // No row interchange
      ipiv[i] = i;
      if (d[i] !== 0) {
        const fact = dl[i] / d[i];
        dl[i] = fact;
        d[i + 1] -= fact * du[i];
      }
    } else {
      // Interchange rows i and i + 1
      ipiv[i] = i + 1;
      const fact = d[i] / dl[i];
      d[i] = dl[i];
      dl[i] = fact;
      const temp = du[i];
      du[i] = d[i + 1];
      d[i + 1] = temp - fact * d[i + 1];
      if (i < n - 2) {
        du2[i] = du[i + 1];
        du[i + 1] = -fact * du2[i];
      }
    }
  }
  ipiv[n - 1] = n - 1;

  for (let i = 0; i < n; i++) {
    if (d[i] === 0) {
      throw new Error(`${who}: matrix is singular.`);
    }
  }

  return { n, dl, d, du, du2, ipiv, scale: 1 };
}

/**
 * Solves `A x = b` in place using the factorization of `A` (the `scale` is ignored): on entry `b`
 * holds the right-hand side, on exit the solution. No validation: the allocation-free kernel
 * behind `solveTridiagonalLU()`, and what the cubic splines use directly.
 * @param lu The factorization from `factorTridiagonalInPlace()`.
 * @param b Right-hand side of length `lu.n`, overwritten with the solution.
 * @internal
 */
export function solveTridiagonalInPlace(lu: TridiagonalLU, b: Float64Array): void {
  const { n, dl, d, du, du2, ipiv } = lu;

  // Forward substitution: apply the row interchanges and L^-1
  for (let i = 0; i < n - 1; i++) {
    if (ipiv[i] === i) {
      b[i + 1] -= dl[i] * b[i];
    } else {
      const temp = b[i];
      b[i] = b[i + 1];
      b[i + 1] = temp - dl[i] * b[i];
    }
  }

  // Back substitution with U
  b[n - 1] /= d[n - 1];
  if (n > 1) {
    b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2];
  }
  for (let i = n - 3; i >= 0; i--) {
    b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / d[i];
  }
}

/** Implementation of `tridiagonalLU()`; `who` names the public function in error messages. */
function factorChecked(dl: VectorLike, d: VectorLike, du: VectorLike, who: string): TridiagonalLU {
  const sub = Vector.asVector(dl).data;
  const diag = Vector.asVector(d).data;
  const sup = Vector.asVector(du).data;
  const n = diag.length;
  if (n < 1 || sub.length !== n - 1 || sup.length !== n - 1) {
    throw new RangeError(
      `${who}: expected dl, d and du of lengths n - 1, n and n - 1 with n >= 1, got ${sub.length}, ${diag.length} and ${sup.length}.`
    );
  }

  // One buffer holds the three diagonals (as views), so that a single pass can scan and scale it.
  const w = new Float64Array(3 * n - 2);
  w.set(sub, 0);
  w.set(diag, n - 1);
  w.set(sup, 2 * n - 1);

  let max = 0;
  let nan = false;
  for (let k = 0; k < w.length; k++) {
    const v = w[k];
    // `v - v` is 0 for every finite v and NaN for NaN and ±Infinity.
    if (v - v !== 0) {
      if (v === v) {
        const name =
          k < n - 1 ? `dl[${k}]` : k < 2 * n - 1 ? `d[${k - (n - 1)}]` : `du[${k - (2 * n - 1)}]`;
        throw new RangeError(`${who}: matrix contains an infinite entry in ${name}, got ${v}.`);
      }
      nan = true;
    } else {
      const a = v < 0 ? -v : v;
      if (a > max) max = a;
    }
  }

  if (nan) {
    // A NaN entry is not singularity: every number of the factorization is NaN.
    w.fill(NaN);
    const ipiv = new Int32Array(n);
    for (let i = 0; i < n; i++) ipiv[i] = i;
    return {
      n,
      dl: w.subarray(0, n - 1),
      d: w.subarray(n - 1, 2 * n - 1),
      du: w.subarray(2 * n - 1),
      du2: new Float64Array(Math.max(n - 2, 0)).fill(NaN),
      ipiv,
      scale: 1
    };
  }

  const sigma = safeScale(w, max);
  if (sigma !== 1) for (let k = 0; k < w.length; k++) w[k] /= sigma;
  const lu = factorTridiagonalInPlace(
    w.subarray(0, n - 1),
    w.subarray(n - 1, 2 * n - 1),
    w.subarray(2 * n - 1),
    who
  );
  lu.scale = sigma;
  return lu;
}

/** Implementation of `solveTridiagonalLU()`; `who` names the public function in error messages. */
function solveChecked(
  lu: TridiagonalLU,
  b: VectorLike,
  out: Vector | undefined,
  who: string
): Vector {
  const v = lu as Partial<TridiagonalLU> | null | undefined;
  if (
    typeof v !== 'object' ||
    v === null ||
    typeof v.n !== 'number' ||
    !(v.dl instanceof Float64Array) ||
    !(v.d instanceof Float64Array) ||
    !(v.du instanceof Float64Array) ||
    !(v.du2 instanceof Float64Array) ||
    !(v.ipiv instanceof Int32Array) ||
    typeof v.scale !== 'number'
  ) {
    throw new TypeError(
      `${who}: lu must be a TridiagonalLU { n, dl, d, du, du2, ipiv, scale }, got ${typeName(lu)}.`
    );
  }
  const { n, dl, d, du, du2, ipiv, scale: sigma } = lu;
  if (
    n < 1 ||
    d.length !== n ||
    dl.length !== n - 1 ||
    du.length !== n - 1 ||
    du2.length !== Math.max(n - 2, 0) ||
    ipiv.length !== n
  ) {
    throw new RangeError(`${who}: inconsistent TridiagonalLU shapes.`);
  }
  const rhs = Vector.asVector(b);
  if (rhs.size !== n) {
    throw new RangeError(`${who}: shape mismatch: ${n}x${n} vs vector of size ${rhs.size}.`);
  }
  if (out !== undefined) {
    checkVectorOut(out, who);
    if (out.size !== n) throw new RangeError(`${who}: out has size ${out.size}, expected ${n}.`);
    if (out === rhs) throw new RangeError(`${who}: out must not be b.`);
  }
  for (let i = 0; i < n; i++) {
    if (d[i] === 0) throw new Error(`${who}: zero diagonal entry, matrix is singular.`);
  }

  // The factors are those of `A / sigma`, solved against `b / tau`: the solution of the scaled
  // system, `y`, relates to the one wanted by `x = y * tau / sigma`. Applying that factor last
  // gives `±Infinity` for an `x` that is out of range, rather than `NaN`.
  const tau = safeScale(rhs.data);
  const x = out ?? new Vector(n);
  x.data.set(rhs.data);
  if (tau !== 1) for (let i = 0; i < n; i++) x.data[i] /= tau;
  solveTridiagonalInPlace(lu, x.data);
  if (sigma !== 1 || tau !== 1) {
    const shift = Math.round(Math.log2(tau)) - Math.round(Math.log2(sigma));
    for (let i = 0; i < n; i++) x.data[i] = ldexp(x.data[i], shift);
  }
  return x;
}

/**
 * LU factorization with partial pivoting of the tridiagonal matrix with subdiagonal `dl`, diagonal
 * `d` and superdiagonal `du`, in the style of LAPACK `dgttrf`. It costs `O(n)` time and memory and
 * the result can be reused for any number of right-hand sides with `solveTridiagonalLU()`. Unlike
 * plain Thomas elimination, pivoting keeps the factorization backward stable for matrices that are
 * not diagonally dominant, and it never divides by a zero that a nonsingular matrix could avoid.
 *
 * A pivot counts as missing only if it is exactly `0` (the LAPACK convention): a nearly singular
 * matrix returns finite, possibly inaccurate factors. A finite matrix is never reported singular
 * because of its magnitude: if its largest entry is outside `[2^-900, 2^900]`, the elimination runs
 * on the matrix divided by a power of two, recorded in `scale`.
 *
 * A `NaN` entry is not an error and not singularity: every number of the result is `NaN`, with
 * `ipiv` the identity and `scale` `1`. An infinite entry throws `RangeError` (it wins over `NaN`).
 * The inputs are not modified.
 * @param dl The subdiagonal, of length `n - 1` (`A[i + 1][i]`). A `Vector` or array-like of numbers.
 * @param d The diagonal, of length `n >= 1` (`A[i][i]`). A `Vector` or array-like of numbers.
 * @param du The superdiagonal, of length `n - 1` (`A[i][i + 1]`). A `Vector` or array-like of numbers.
 * @returns The `{ n, dl, d, du, du2, ipiv, scale }` factorization, as a new object.
 * @throws {RangeError} If `d` is empty or `dl` or `du` do not have length `n - 1`.
 * @throws {RangeError} If an entry is infinite.
 * @throws {Error} If the matrix is singular (a zero pivot is found).
 * @throws {TypeError} If an argument is neither a `Vector` nor an array of numbers.
 *
 * @example
 * ```ts
 * import { tridiagonalLU, solveTridiagonalLU } from 'numerics-js/linalg';
 *
 * // [[0, 2, 0], [1, 1, 3], [0, 4, 5]]: the zero on the diagonal forces a row interchange.
 * const lu = tridiagonalLU([1, 4], [0, 1, 5], [2, 3]);
 * console.log(Array.from(lu.ipiv));
 * for (const b of [[2, 5, 9], [0, 1, 0]]) {
 *   console.log(solveTridiagonalLU(lu, b).toString());
 * }
 * ```
 *
 * Output:
 * ```text
 * [ 1, 2, 2 ]
 * Vector(1, 1, 1)
 * Vector(1, 0, 0)
 * ```
 */
export function tridiagonalLU(dl: VectorLike, d: VectorLike, du: VectorLike): TridiagonalLU {
  return factorChecked(dl, d, du, 'tridiagonalLU');
}

/**
 * Solves `A * x = b` for the tridiagonal matrix `A`, given its factorization as returned by
 * `tridiagonalLU()`: forward substitution with the row interchanges and `L`, then back
 * substitution with `U`. Costs `O(n)`; if you need to solve against the same matrix with several
 * right-hand sides, call `tridiagonalLU()` once and reuse this instead of refactoring each time.
 *
 * Neither `lu` nor `b` is modified. A zero on the diagonal of `U` throws `Error`. Entries are not
 * scanned: `NaN` propagates through the arithmetic (so a factorization that `tridiagonalLU()`
 * filled with `NaN` gives an all-`NaN` solution) and infinite values follow IEEE rules. A solution
 * that is out of range is `±Infinity`, never `NaN`.
 * @param lu The factorization from `tridiagonalLU()`.
 * @param b The right-hand side (a `Vector` or array-like of numbers). Must have `b.size === lu.n`.
 * @param out Optional vector of size `lu.n` that receives the solution, avoiding an allocation. It
 * must not be `b`. If this function throws, the contents of `out` are undefined.
 * @returns The solution vector `x`: `out` if given, otherwise a new vector.
 * @throws {RangeError} If `b` or `out` does not match the size of the factorization, if `out` is `b`,
 * or if the arrays of `lu` have inconsistent lengths.
 * @throws {Error} If `U` has a zero diagonal entry.
 * @throws {TypeError} If `lu` is not a `TridiagonalLU`, or if `out` is given but is not a `Vector`.
 */
export function solveTridiagonalLU(lu: TridiagonalLU, b: VectorLike, out?: Vector): Vector {
  return solveChecked(lu, b, out, 'solveTridiagonalLU');
}

/**
 * Solves `A * x = b` for the tridiagonal matrix `A` with subdiagonal `dl`, diagonal `d` and
 * superdiagonal `du`, via `tridiagonalLU()` followed by `solveTridiagonalLU()`: `O(n)` time and
 * memory, with partial pivoting. If you need to solve against the same matrix with several
 * right-hand sides, call `tridiagonalLU()` once yourself and reuse `solveTridiagonalLU()`.
 *
 * The singularity, `NaN`, infinity and scaling rules are those of `tridiagonalLU()`.
 * @param dl The subdiagonal, of length `n - 1`. A `Vector` or array-like of numbers.
 * @param d The diagonal, of length `n >= 1`. A `Vector` or array-like of numbers.
 * @param du The superdiagonal, of length `n - 1`. A `Vector` or array-like of numbers.
 * @param b The right-hand side (a `Vector` or array-like of numbers). Must have length `n`.
 * @param out Optional vector of size `n` that receives the solution. It must not be `b`.
 * @returns The solution vector `x`: `out` if given, otherwise a new vector.
 * @throws {RangeError} If `d` is empty, if `dl` or `du` do not have length `n - 1`, if `b` or `out`
 * does not have length `n`, if `out` is `b`, or if an entry of the matrix is infinite.
 * @throws {Error} If the matrix is singular.
 * @throws {TypeError} If an argument is neither a `Vector` nor an array of numbers, or if `out` is
 * given but is not a `Vector`.
 *
 * @example
 * ```ts
 * import { solveTridiagonal } from 'numerics-js/linalg';
 *
 * // -x'' = 1 on a grid of 4 interior points, with h^2 = 1.
 * const x = solveTridiagonal([-1, -1, -1], [2, 2, 2, 2], [-1, -1, -1], [1, 1, 1, 1]);
 * console.log(Array.from(x.data, (v) => v.toFixed(6)).join(' '));
 * ```
 *
 * Output:
 * ```text
 * 2.000000 3.000000 3.000000 2.000000
 * ```
 */
export function solveTridiagonal(
  dl: VectorLike,
  d: VectorLike,
  du: VectorLike,
  b: VectorLike,
  out?: Vector
): Vector {
  return solveChecked(factorChecked(dl, d, du, 'solveTridiagonal'), b, out, 'solveTridiagonal');
}
