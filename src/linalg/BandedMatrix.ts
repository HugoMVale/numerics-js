import { Matrix } from './Matrix.js';
import { Vector, type VectorLike } from './Vector.js';
import { checkVectorOut, typeName } from './checks.js';
import { ldexp, safeScale } from './Matrix.linalg.js';

/**
 * Result of `BandedMatrix.lu()`: the LU factorization with partial pivoting of a banded matrix,
 * in the storage of LAPACK `dgbtrf`, so that `P * A = L * U`.
 *
 * `ab` has `ldab = 2 * kl + ku + 1` rows and `n` columns, stored column by column (column `j`
 * occupies `ab[j * ldab .. (j + 1) * ldab - 1]`). With `kv = kl + ku`, row interchanges widen the
 * band of `U` to `kv` superdiagonals: `U[i][j]` (for `j - kv <= i <= j`) is at
 * `ab[kv + i - j + j * ldab]`, and the multiplier of `L` that eliminates `A[j + k][j]`
 * (`1 <= k <= min(kl, n - 1 - j)`) is at `ab[kv + k + j * ldab]`. The first `kl` entries of each
 * column are workspace for the fill-in. `ipiv[j]` is the row (`j <= ipiv[j] <= j + kl`) that was
 * interchanged with row `j` at step `j`; unlike in the dense `LUDecomposition`, the multipliers of
 * `L` are not permuted.
 *
 * For a matrix whose entries are close to the limits of the double range, the factors are those of
 * `A / scale`; for every ordinary matrix `scale` is `1`.
 */
export interface BandedLUDecomposition {
  /** Order of the matrix. */
  n: number;
  /** Number of subdiagonals of the matrix (after clamping to `n - 1`). */
  kl: number;
  /** Number of superdiagonals of the matrix (after clamping to `n - 1`). */
  ku: number;
  /** The factors `L` and `U` in LAPACK band storage, length `(2 * kl + ku + 1) * n`. */
  ab: Float64Array;
  /** Row interchanges (zero-based), length `n`. */
  ipiv: Int32Array;
  /**
   * A power of two `σ`: the factors in `ab` are those of `A / σ`, so that a matrix with entries
   * close to the limits of the double range can be eliminated without overflow or underflow. It is
   * `1` unless the largest entry of `A` is outside `[2^-900, 2^900]`.
   */
  scale: number;
}

const LU_LABEL = 'a BandedLUDecomposition { n, kl, ku, ab, ipiv, scale }';

function checkIndex(n: number, i: number, j: number, who: string): void {
  if (!Number.isInteger(i) || !Number.isInteger(j) || i < 0 || j < 0 || i >= n || j >= n) {
    throw new RangeError(
      `BandedMatrix.${who}: index (${i}, ${j}) is outside the ${n}x${n} matrix.`
    );
  }
}

function checkBandwidth(value: unknown, name: string, who: string): number {
  if (typeof value !== 'number') {
    throw new TypeError(`BandedMatrix.${who}: ${name} must be a number, got ${typeName(value)}.`);
  }
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(
      `BandedMatrix.${who}: ${name} must be a non-negative integer, got ${value}.`
    );
  }
  return value;
}

function checkOrder(n: unknown, who: string): number {
  if (typeof n !== 'number') {
    throw new TypeError(`BandedMatrix.${who}: n must be a number, got ${typeName(n)}.`);
  }
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError(`BandedMatrix.${who}: n must be a positive integer, got ${n}.`);
  }
  return n;
}

/** Allocates `length` doubles, turning the engine's allocation `RangeError` into the library's. */
function allocate(length: number, who: string): Float64Array {
  try {
    return new Float64Array(length);
  } catch (e) {
    if (!(e instanceof RangeError)) throw e;
    throw new RangeError(`BandedMatrix.${who}: the band storage is too large to allocate.`);
  }
}

/**
 * A square `n x n` matrix with `kl` subdiagonals and `ku` superdiagonals, stored in the band
 * storage of LAPACK (and of SciPy's `solve_banded`): `ab[ku + i - j][j] = A[i][j]` for
 * `max(0, j - ku) <= i <= min(n - 1, j + kl)`, in a flat column-major `Float64Array` `data` with
 * `ldab = kl + ku + 1` rows, so that element `(i, j)` lives at `data[ku + i - j + j * ldab]`.
 * Memory and the cost of a matrix-vector product are `O(n * (kl + ku))`; the factorization of
 * `lu()` costs `O(n * kl * (kl + ku))` and each solve `O(n * (kl + ku))`, which makes systems from
 * finite-difference or finite-element grids tractable where a dense `Matrix` is not.
 *
 * Entries outside the band are structurally zero: `get` returns `0` for them, `set` rejects them,
 * and the arithmetic never reads them (so, unlike a dense product, `0 * Infinity` does not
 * produce `NaN` for them). The unused corners of `data` (positions that would be outside the
 * matrix) are always `0`.
 *
 * **Errors.** As for `Matrix`: `RangeError` for bad values (shapes, sizes, indices, an entry
 * outside the band, an infinite entry), `TypeError` for wrong types, plain `Error` for a singular
 * matrix. Messages have the form `BandedMatrix.method: description.`.
 *
 * @example
 * ```ts
 * import { BandedMatrix } from 'numerics-js/linalg';
 *
 * // The second-difference matrix [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].
 * const A = new BandedMatrix(3, 1, 1);
 * for (let i = 0; i < 3; i++) {
 *   A.set(i, i, 2);
 *   if (i > 0) A.set(i, i - 1, -1);
 *   if (i < 2) A.set(i, i + 1, -1);
 * }
 * const x = A.solve([1, 0, 1]);
 * console.log(Array.from(x.data, (v) => v.toFixed(6)).join(' '));
 * console.log(Array.from(A.mulVec(x).data, (v) => v.toFixed(6)).join(' '));
 * ```
 *
 * Output:
 * ```text
 * 1.000000 1.000000 1.000000
 * 1.000000 0.000000 1.000000
 * ```
 */
export class BandedMatrix {
  /** Order of the matrix (it has `n` rows and `n` columns). */
  readonly n: number;
  /** Number of subdiagonals, at most `n - 1`. */
  readonly kl: number;
  /** Number of superdiagonals, at most `n - 1`. */
  readonly ku: number;
  /**
   * The band, column-major with `ldab = kl + ku + 1` rows: element `(i, j)` of the matrix is
   * `data[ku + i - j + j * ldab]`. Writing to it changes the matrix; entries at positions that are
   * outside the matrix (the unused corners) must stay `0`.
   */
  readonly data: Float64Array;

  /**
   * Creates a zero `n x n` banded matrix with `kl` subdiagonals and `ku` superdiagonals. A
   * bandwidth of `n` or more is clamped to `n - 1`, which describes the same set of matrices.
   * @param n Order of the matrix (a positive integer).
   * @param kl Number of subdiagonals (a non-negative integer).
   * @param ku Number of superdiagonals (a non-negative integer).
   * @throws {RangeError} If `n` is not a positive integer, if `kl` or `ku` is not a non-negative
   * integer, or if the storage is too large to allocate.
   * @throws {TypeError} If an argument is not a number.
   */
  constructor(n: number, kl: number, ku: number) {
    checkOrder(n, 'constructor');
    checkBandwidth(kl, 'kl', 'constructor');
    checkBandwidth(ku, 'ku', 'constructor');
    this.n = n;
    this.kl = Math.min(kl, n - 1);
    this.ku = Math.min(ku, n - 1);
    this.data = allocate((this.kl + this.ku + 1) * n, 'constructor');
  }

  /** Number of rows of `data`: `kl + ku + 1`. */
  get ldab(): number {
    return this.kl + this.ku + 1;
  }

  /**
   * Creates the banded matrix equal to the dense square matrix `A`. With `kl` and `ku` given,
   * every entry of `A` outside that band must be exactly zero; without them, the smallest band
   * that contains all nonzero entries of `A` is used (a zero matrix gives `kl = ku = 0`). A `NaN`
   * is a nonzero entry.
   * @param A The dense matrix. Must be square. Not modified.
   * @param kl Number of subdiagonals. Give both `kl` and `ku`, or neither.
   * @param ku Number of superdiagonals. Give both `kl` and `ku`, or neither.
   * @returns A new banded matrix with the entries of `A`.
   * @throws {TypeError} If `A` is not a `Matrix`, or `kl` or `ku` is not a number.
   * @throws {RangeError} If `A` is not square, if only one of `kl` and `ku` is given, if either is
   * not a non-negative integer, or if `A` has a nonzero entry outside the given band.
   *
   * @example
   * ```ts
   * import { BandedMatrix, Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[4, 1, 0], [2, 5, 1], [0, 3, 6]]);
   * const B = BandedMatrix.fromMatrix(A);
   * console.log(B.kl, B.ku);
   * console.log(B.toMatrix().toString());
   * ```
   *
   * Output:
   * ```text
   * 1 1
   * Matrix[[4, 1, 0], [2, 5, 1], [0, 3, 6]]
   * ```
   */
  static fromMatrix(A: Matrix, kl?: number, ku?: number): BandedMatrix {
    if (!(A instanceof Matrix)) {
      throw new TypeError(`BandedMatrix.fromMatrix: expected a Matrix, got ${typeName(A)}.`);
    }
    if (A.rows !== A.cols) {
      throw new RangeError(
        `BandedMatrix.fromMatrix: expected a square matrix, got ${A.rows}x${A.cols}.`
      );
    }
    if ((kl === undefined) !== (ku === undefined)) {
      throw new RangeError('BandedMatrix.fromMatrix: give both kl and ku, or neither.');
    }
    const n = A.rows;
    const a = A.data;
    let lower: number;
    let upper: number;
    if (kl === undefined || ku === undefined) {
      lower = 0;
      upper = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (a[i * n + j] !== 0) {
            if (i - j > lower) lower = i - j;
            if (j - i > upper) upper = j - i;
          }
        }
      }
    } else {
      lower = checkBandwidth(kl, 'kl', 'fromMatrix');
      upper = checkBandwidth(ku, 'ku', 'fromMatrix');
    }
    const B = new BandedMatrix(n, lower, upper);
    const { kl: l, ku: u, ldab } = B;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = a[i * n + j];
        if (i - j > l || j - i > u) {
          if (v !== 0) {
            throw new RangeError(
              `BandedMatrix.fromMatrix: entry (${i}, ${j}) is ${v} but lies outside the band (kl = ${l}, ku = ${u}).`
            );
          }
        } else {
          B.data[u + i - j + j * ldab] = v;
        }
      }
    }
    return B;
  }

  /**
   * Creates a banded matrix from its band `ab` in the layout of LAPACK and SciPy's
   * `solve_banded`: a `(kl + ku + 1) x n` matrix with `ab[ku + i - j][j] = A[i][j]`, that is, row
   * `ku` holds the diagonal, rows above it the superdiagonals (each shifted right) and rows below
   * it the subdiagonals (each shifted left). The entries of `ab` at positions that fall outside the
   * matrix (the top-left and bottom-right corners) are ignored. A bandwidth of `n` or more is
   * clamped to `n - 1`; the entries of `ab` for the diagonals beyond that are all unused.
   * @param ab The band, a `Matrix` with `kl + ku + 1` rows and `n` columns. Not modified.
   * @param kl Number of subdiagonals.
   * @param ku Number of superdiagonals.
   * @returns A new banded matrix; it does not share storage with `ab`.
   * @throws {TypeError} If `ab` is not a `Matrix`, or `kl` or `ku` is not a number.
   * @throws {RangeError} If `kl` or `ku` is not a non-negative integer, or `ab.rows !== kl + ku + 1`.
   *
   * @example
   * ```ts
   * import { BandedMatrix, Matrix } from 'numerics-js/linalg';
   *
   * // Rows: superdiagonal, diagonal, subdiagonal (the corner entries marked 0 are unused).
   * const ab = Matrix.from([
   *   [0, 1, 1],
   *   [4, 5, 6],
   *   [2, 3, 0]
   * ]);
   * console.log(BandedMatrix.fromBands(ab, 1, 1).toMatrix().toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[4, 1, 0], [2, 5, 1], [0, 3, 6]]
   * ```
   */
  static fromBands(ab: Matrix, kl: number, ku: number): BandedMatrix {
    if (!(ab instanceof Matrix)) {
      throw new TypeError(`BandedMatrix.fromBands: expected a Matrix, got ${typeName(ab)}.`);
    }
    checkBandwidth(kl, 'kl', 'fromBands');
    checkBandwidth(ku, 'ku', 'fromBands');
    if (ab.rows !== kl + ku + 1) {
      throw new RangeError(
        `BandedMatrix.fromBands: expected kl + ku + 1 = ${kl + ku + 1} rows, got ${ab.rows}.`
      );
    }
    const n = ab.cols;
    const B = new BandedMatrix(n, kl, ku);
    const { kl: l, ku: u, ldab } = B;
    for (let j = 0; j < n; j++) {
      const i0 = Math.max(0, j - u);
      const i1 = Math.min(n - 1, j + l);
      for (let i = i0; i <= i1; i++) {
        // `ku` (not `u`) addresses the caller's layout, which was not clamped.
        B.data[u + i - j + j * ldab] = ab.data[(ku + i - j) * n + j];
      }
    }
    return B;
  }

  /**
   * Gets the element at row `i`, column `j`: `0` for a position outside the band.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @returns The value at `(i, j)`.
   * @throws {RangeError} If `i` or `j` is not an integer inside the matrix.
   */
  get(i: number, j: number): number {
    checkIndex(this.n, i, j, 'get');
    if (i - j > this.kl || j - i > this.ku) return 0;
    return this.data[this.ku + i - j + j * this.ldab];
  }

  /**
   * Sets the element at row `i`, column `j`, which must lie inside the band.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @param value The value to store.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `i` or `j` is not an integer inside the matrix, or `(i, j)` is outside
   * the band (`i - j > kl` or `j - i > ku`): such an entry cannot be stored.
   */
  set(i: number, j: number, value: number): this {
    checkIndex(this.n, i, j, 'set');
    if (i - j > this.kl || j - i > this.ku) {
      throw new RangeError(
        `BandedMatrix.set: index (${i}, ${j}) is outside the band (kl = ${this.kl}, ku = ${this.ku}).`
      );
    }
    this.data[this.ku + i - j + j * this.ldab] = value;
    return this;
  }

  /**
   * Computes the matrix-vector product `A * v` in `O(n * (kl + ku))`. Only the entries inside the
   * band are read.
   * @param v The vector (a `Vector` or array-like of numbers), of length `n`.
   * @param out Optional vector of length `n` that receives the product, avoiding an allocation. It
   * must not be `v`.
   * @returns The product: `out` if given, otherwise a new vector.
   * @throws {RangeError} If `v` or `out` does not have length `n`, or if `out` is `v`.
   * @throws {TypeError} If `v` is neither a `Vector` nor an array-like of numbers, or if `out` is
   * given but is not a `Vector`.
   */
  mulVec(v: VectorLike, out?: Vector): Vector {
    const x = Vector.asVector(v);
    const { n, kl, ku, ldab, data } = this;
    if (x.size !== n) {
      throw new RangeError(
        `BandedMatrix.mulVec: shape mismatch: ${n}x${n} * vector of size ${x.size}.`
      );
    }
    if (out !== undefined) {
      checkVectorOut(out, 'BandedMatrix.mulVec');
      if (out.size !== n) {
        throw new RangeError(`BandedMatrix.mulVec: out has size ${out.size}, expected ${n}.`);
      }
      if (out === x) throw new RangeError('BandedMatrix.mulVec: out must not be v.');
    }
    const y = out ?? new Vector(n);
    const yd = y.data;
    yd.fill(0);
    for (let j = 0; j < n; j++) {
      const xj = x.data[j];
      const i0 = j > ku ? j - ku : 0;
      const i1 = j + kl < n ? j + kl : n - 1;
      const base = ku - j + j * ldab;
      for (let i = i0; i <= i1; i++) yd[i] += data[base + i] * xj;
    }
    return y;
  }

  /**
   * Converts to a dense `n x n` matrix (zero outside the band).
   * @returns A new `Matrix`.
   * @throws {RangeError} If the dense matrix is too large to allocate.
   */
  toMatrix(): Matrix {
    const { n, kl, ku, ldab, data } = this;
    const A = new Matrix(n, n);
    for (let j = 0; j < n; j++) {
      const i0 = j > ku ? j - ku : 0;
      const i1 = j + kl < n ? j + kl : n - 1;
      for (let i = i0; i <= i1; i++) A.data[i * n + j] = data[ku + i - j + j * ldab];
    }
    return A;
  }

  /**
   * LU factorization with partial pivoting of this matrix, with the semantics of LAPACK
   * `dgbtrf`/`dgbtf2`: `P * A = L * U`, where the row interchanges widen the band of `U` to
   * `kl + ku` superdiagonals. It costs `O(n * kl * (kl + ku))` and the result can be reused for
   * any number of right-hand sides with `BandedMatrix.solveLU()`, each costing `O(n * (kl + ku))`.
   *
   * A pivot counts as missing only if it is exactly `0` (the LAPACK convention): a nearly
   * singular matrix returns finite, possibly inaccurate factors. A finite matrix is never reported
   * singular because of its magnitude: if its largest entry is outside `[2^-900, 2^900]` the
   * elimination runs on the matrix divided by a power of two, recorded in `scale`.
   *
   * A `NaN` entry is not an error and not singularity: every number of `ab` is `NaN`, with `ipiv`
   * the identity and `scale` `1`. An infinite entry throws `RangeError` (it wins over `NaN`).
   *
   * Repeated factorizations of same-shaped matrices can avoid all allocation by passing the
   * factorization of an earlier call as `out`.
   * @param out Optional storage to reuse: a factorization for a matrix with the same `n`, `kl` and
   * `ku`, as returned by an earlier call. It is overwritten and returned. If this method throws,
   * its contents are undefined. Omit to allocate a new factorization.
   * @returns The `{ n, kl, ku, ab, ipiv, scale }` factorization: `out` if given, otherwise a new
   * object. This matrix is not modified.
   * @throws {Error} If the matrix is singular (a column's largest available pivot is exactly `0`).
   * @throws {RangeError} If an entry is infinite, or `out` does not match this matrix.
   * @throws {TypeError} If `out` is given but is not a `BandedLUDecomposition`.
   *
   * @example
   * ```ts
   * import { BandedMatrix, Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = BandedMatrix.fromMatrix(Matrix.from([[0, 2, 0], [1, 1, 3], [0, 4, 5]]));
   * const dec = A.lu();
   * console.log(Array.from(dec.ipiv));
   * for (const b of [[2, 5, 9], [0, 1, 0]]) {
   *   console.log(BandedMatrix.solveLU(dec, Vector.from(b)).toString());
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
  lu(out?: BandedLUDecomposition): BandedLUDecomposition {
    const { n, kl, ku, ldab, data } = this;
    const kv = kl + ku;
    const ldlu = 2 * kl + ku + 1;
    const dec = prepareLU(this, out, 'lu');
    const { ab, ipiv } = dec;

    // One pass: scan for non-finite entries and find the largest magnitude.
    let max = 0;
    let nan = false;
    for (let k = 0; k < data.length; k++) {
      const v = data[k];
      // `v - v` is 0 for every finite v and NaN for NaN and ±Infinity.
      if (v - v !== 0) {
        if (v === v) {
          const j = Math.floor(k / ldab);
          const i = k - j * ldab - ku + j;
          throw new RangeError(
            `BandedMatrix.lu: matrix contains an infinite entry at (${i}, ${j}), got ${v}.`
          );
        }
        nan = true;
      } else {
        const a = v < 0 ? -v : v;
        if (a > max) max = a;
      }
    }
    if (nan) {
      ab.fill(NaN);
      for (let i = 0; i < n; i++) ipiv[i] = i;
      dec.scale = 1;
      return dec;
    }

    const sigma = safeScale(data, max);
    dec.scale = sigma;
    // Copy the band `kl` rows down, leaving the first `kl` rows of each column for fill-in.
    ab.fill(0);
    for (let j = 0; j < n; j++) {
      const src = j * ldab;
      const dst = j * ldlu + kl;
      if (sigma === 1) {
        for (let r = 0; r < ldab; r++) ab[dst + r] = data[src + r];
      } else {
        for (let r = 0; r < ldab; r++) ab[dst + r] = data[src + r] / sigma;
      }
    }

    // Unblocked elimination of `dgbtf2`. Column `j` of `ab` holds A(i, j) at `kv + i - j`.
    let ju = 0; // last column that the elimination has touched so far
    for (let j = 0; j < n; j++) {
      const col = j * ldlu;
      const km = kl < n - 1 - j ? kl : n - 1 - j;
      // Pivot search in A(j..j+km, j).
      let jp = 0;
      let pivot = Math.abs(ab[col + kv]);
      for (let k = 1; k <= km; k++) {
        const v = Math.abs(ab[col + kv + k]);
        if (v > pivot) {
          pivot = v;
          jp = k;
        }
      }
      ipiv[j] = j + jp;
      if (pivot === 0) throw new Error('BandedMatrix.lu: matrix is singular.');
      const reach = j + ku + jp < n - 1 ? j + ku + jp : n - 1;
      if (reach > ju) ju = reach;
      if (jp !== 0) {
        // Swap rows j and j + jp over columns j..ju: element (j, c) is at kv + j + c * (ldlu - 1).
        const stride = ldlu - 1;
        for (let c = j; c <= ju; c++) {
          const p = kv + j + c * stride;
          const tmp = ab[p];
          ab[p] = ab[p + jp];
          ab[p + jp] = tmp;
        }
      }
      if (km > 0) {
        const d = ab[col + kv];
        for (let k = 1; k <= km; k++) ab[col + kv + k] /= d;
        // Rank-1 update of the trailing block A(j+1..j+km, j+1..ju).
        for (let c = j + 1; c <= ju; c++) {
          const p = kv + j + c * (ldlu - 1); // U(j, c)
          const t = ab[p];
          if (t !== 0) {
            for (let k = 1; k <= km; k++) ab[p + k] -= ab[col + kv + k] * t;
          }
        }
      }
    }
    return dec;
  }

  /**
   * Solves `A * x = b` for `x`, given the LU factorization of `A` from `lu()`: applies the row
   * interchanges and the multipliers of `L`, then back substitution with `U` (the `dgbtrs`
   * semantics). Costs `O(n * (kl + ku))`; if you need to solve against the same matrix with
   * several right-hand sides, call `lu()` once and reuse this instead of refactoring each time.
   *
   * Neither `decomposition` nor `b` is modified. A zero on the diagonal of `U` throws `Error`.
   * Entries are not scanned: `NaN` propagates through the arithmetic (so a factorization that
   * `lu()` filled with `NaN` gives an all-`NaN` solution) and infinite values follow IEEE rules.
   * A solution that is out of range is `±Infinity`, never `NaN`. With `out` given and `b` a
   * `Vector`, the call allocates nothing.
   * @param decomposition The factorization returned by `lu()`.
   * @param b The right-hand side (a `Vector` or array-like of numbers), of length `n`.
   * @param out Optional vector of length `n` that receives the solution, avoiding an allocation.
   * It must not be `b`. If this method throws, the contents of `out` are undefined.
   * @returns The solution vector `x`: `out` if given, otherwise a new vector.
   * @throws {RangeError} If `b` or `out` does not match the size of the factorization, if `out` is
   * `b`, or if the factorization has inconsistent shapes.
   * @throws {Error} If `U` has a zero diagonal entry.
   * @throws {TypeError} If `decomposition` is not a `BandedLUDecomposition`, or if `out` is given
   * but is not a `Vector`.
   */
  static solveLU(decomposition: BandedLUDecomposition, b: VectorLike, out?: Vector): Vector {
    return solveBanded(decomposition, b, out, 'solveLU');
  }

  /**
   * Solves `A * x = b` for `x` via `lu()` followed by `BandedMatrix.solveLU()`. If you need to
   * solve against the same matrix with several right-hand sides, call `lu()` once yourself and
   * reuse `BandedMatrix.solveLU()`.
   * @param b The right-hand side (a `Vector` or array-like of numbers), of length `n`.
   * @returns The solution vector `x`.
   * @throws {RangeError} If `b` does not have length `n`, or if an entry of the matrix is infinite.
   * @throws {Error} If the matrix is singular.
   * @throws {TypeError} If `b` is neither a `Vector` nor an array-like of numbers.
   */
  solve(b: VectorLike): Vector {
    const rhs = Vector.asVector(b);
    if (rhs.size !== this.n) {
      throw new RangeError(
        `BandedMatrix.solve: shape mismatch: ${this.n}x${this.n} vs vector of size ${rhs.size}.`
      );
    }
    return solveBanded(this.lu(), rhs, undefined, 'solve');
  }
}

/** Validates `out` of `lu()` or allocates the storage of a factorization of `A`. */
function prepareLU(
  A: BandedMatrix,
  out: BandedLUDecomposition | undefined,
  who: string
): BandedLUDecomposition {
  const { n, kl, ku } = A;
  const length = (2 * kl + ku + 1) * n;
  if (out === undefined) {
    return {
      n,
      kl,
      ku,
      ab: allocate(length, who),
      ipiv: new Int32Array(n),
      scale: 1
    };
  }
  checkLU(out, `BandedMatrix.${who}`, 'out');
  if (out.n !== n || out.kl !== kl || out.ku !== ku) {
    throw new RangeError(
      `BandedMatrix.${who}: out is for n = ${out.n}, kl = ${out.kl}, ku = ${out.ku}, expected n = ${n}, kl = ${kl}, ku = ${ku}.`
    );
  }
  return out;
}

/** Checks the type and the internal consistency of a factorization record. */
function checkLU(dec: unknown, who: string, role: string): void {
  const v = dec as Partial<BandedLUDecomposition> | null | undefined;
  if (
    typeof v !== 'object' ||
    v === null ||
    typeof v.n !== 'number' ||
    typeof v.kl !== 'number' ||
    typeof v.ku !== 'number' ||
    !(v.ab instanceof Float64Array) ||
    !(v.ipiv instanceof Int32Array) ||
    typeof v.scale !== 'number'
  ) {
    throw new TypeError(`${who}: ${role} must be ${LU_LABEL}, got ${typeName(dec)}.`);
  }
  const { n, kl, ku, ab, ipiv } = v as BandedLUDecomposition;
  if (
    !Number.isInteger(n) ||
    !Number.isInteger(kl) ||
    !Number.isInteger(ku) ||
    n < 1 ||
    kl < 0 ||
    ku < 0 ||
    kl >= n ||
    ku >= n ||
    ab.length !== (2 * kl + ku + 1) * n ||
    ipiv.length !== n
  ) {
    throw new RangeError(`${who}: inconsistent BandedLUDecomposition shapes.`);
  }
}

/** Implementation of `BandedMatrix.solveLU()` and `solve()`. */
function solveBanded(
  dec: BandedLUDecomposition,
  b: VectorLike,
  out: Vector | undefined,
  who: string
): Vector {
  const name = `BandedMatrix.${who}`;
  checkLU(dec, name, 'decomposition');
  const { n, kl, ku, ab, ipiv, scale: sigma } = dec;
  const rhs = Vector.asVector(b);
  if (rhs.size !== n) {
    throw new RangeError(`${name}: shape mismatch: ${n}x${n} vs vector of size ${rhs.size}.`);
  }
  if (out !== undefined) {
    checkVectorOut(out, name);
    if (out.size !== n) throw new RangeError(`${name}: out has size ${out.size}, expected ${n}.`);
    if (out === rhs) throw new RangeError(`${name}: out must not be b.`);
  }
  const kv = kl + ku;
  const ld = 2 * kl + ku + 1;
  for (let j = 0; j < n; j++) {
    if (ab[kv + j * ld] === 0) {
      throw new Error(`${name}: zero diagonal entry, matrix is singular.`);
    }
  }

  // The factors are those of `A / sigma`, solved against `b / tau`; see `solveTridiagonalLU`.
  const tau = safeScale(rhs.data);
  const x = out ?? new Vector(n);
  const xd = x.data;
  xd.set(rhs.data);
  if (tau !== 1) for (let i = 0; i < n; i++) xd[i] /= tau;

  // Forward substitution: interchanges and multipliers of L, column by column.
  if (kl > 0) {
    for (let j = 0; j < n - 1; j++) {
      const lm = kl < n - 1 - j ? kl : n - 1 - j;
      const l = ipiv[j];
      if (l !== j) {
        const tmp = xd[l];
        xd[l] = xd[j];
        xd[j] = tmp;
      }
      const xj = xd[j];
      if (xj !== 0) {
        const base = kv + j * ld;
        for (let k = 1; k <= lm; k++) xd[j + k] -= ab[base + k] * xj;
      }
    }
  }
  // Back substitution with U, which has kv superdiagonals.
  for (let j = n - 1; j >= 0; j--) {
    const col = j * ld;
    const xj = xd[j] / ab[col + kv];
    xd[j] = xj;
    const i0 = j > kv ? j - kv : 0;
    for (let i = i0; i < j; i++) xd[i] -= ab[col + kv + i - j] * xj;
  }

  if (sigma !== 1 || tau !== 1) {
    const shift = Math.round(Math.log2(tau)) - Math.round(Math.log2(sigma));
    for (let i = 0; i < n; i++) xd[i] = ldexp(xd[i], shift);
  }
  return x;
}
