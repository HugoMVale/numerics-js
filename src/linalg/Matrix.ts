import { ArrayND } from './ArrayND.js';
import { Vector } from './Vector.js';
import * as linalg from './Matrix.linalg.js';

// The heavier linear-algebra algorithms (factorizations, linear-system
// solves, eigendecomposition) live in Matrix.linalg.ts, not here — see
// that file's header comment for why. Matrix's own methods below
// (`lu()`, `qr()`, `eig()`, etc.) are thin forwarders to it, and these
// re-exports keep `import { Matrix, type LUDecomposition } from
// './Matrix.js'`-style imports working unchanged.
export type {
  Eigenvalue,
  QRDecomposition,
  QRPivotDecomposition,
  Eigenpair,
  LUDecomposition
} from './Matrix.linalg.js';
import type {
  Eigenvalue,
  QRDecomposition,
  QRPivotDecomposition,
  Eigenpair,
  LUDecomposition
} from './Matrix.linalg.js';

/**
 * A rows x cols matrix backed by a flat, row-major Float64Array.
 *
 * All element access (`get`, `set`, `row`, `col`, `setRow`, `setCol`,
 * `swapRows`, `scaleRow`, `addScaledRow`) is **0-based**: valid row indices
 * are `0..rows-1` and valid column indices are `0..cols-1`, matching JS's
 * usual 0-based indexing convention.
 *
 * Elementwise arithmetic (`add`/`sub`/`mult`/`div` + `Self` variants,
 * `abs`/`pow`/`sqrt`/`clip` + `Self` variants), tolerance comparisons
 * (`isClose`/`allClose`), `normSq`/`norm`/`dot`/`dist` (Frobenius), and
 * `copy`/`fill` come from `ArrayND`; see that class for their docs.
 * `add`/`sub`/`mult`/`div` and their `Self` variants accept another
 * `Matrix` of the same shape (elementwise), a scalar, or — declared
 * here, on top of `ArrayND`'s overloads — a `Vector`, which is
 * broadcast across the rows following numpy's rules: a plain `Vector` is
 * a *row* vector (numpy shape `(cols,)`), so `A.add(v)` adds `v[j]` to
 * every element of column `j`. Broadcasting is deliberately limited to
 * this case: a `Vector` of size `1` (a scalar) or of size `cols`, and
 * only when the result keeps this matrix's shape. Matrix-with-matrix
 * operands must still match exactly (no `1 x n` / `m x 1` broadcasting).
 * `toArray` is not inherited (its natural shape differs per subclass)
 * and is defined here directly, as an array of row arrays.
 */
export class Matrix extends ArrayND {
  private _rows: number;
  private _cols: number;
  public data: Float64Array;

  /**
   * @param rows Number of rows (must be a positive integer).
   * @param cols Number of columns (must be a positive integer).
   * @param input Optional initial data in row-major order
   *   (i.e. row 0 followed by row 1, etc.), length `rows * cols`.
   *   If omitted, the matrix is initialized to all zeros.
   * @throws {RangeError} If `rows` or `cols` is not a positive integer, or
   *   if `input` is provided and its length is not exactly `rows * cols`.
   */
  constructor(rows: number, cols: number, input?: ArrayLike<number>) {
    super();
    if (!Number.isInteger(rows) || rows < 1) {
      throw new RangeError(`Matrix: rows must be a positive integer, got ${rows}`);
    }
    if (!Number.isInteger(cols) || cols < 1) {
      throw new RangeError(`Matrix: cols must be a positive integer, got ${cols}`);
    }
    if (input !== undefined && input.length !== rows * cols) {
      throw new RangeError(
        `Matrix: expected ${rows * cols} values for a ${rows}x${cols} matrix, got ${input.length}`
      );
    }
    this._rows = rows;
    this._cols = cols;
    this.data = new Float64Array(rows * cols);
    if (input !== undefined) this.data.set(input);
  }

  /**
   * The number of rows in this matrix. Read-only (like `Vector.size`) so
   * it can never desync from `data`.
   */
  get rows(): number {
    return this._rows;
  }

  /**
   * The number of columns in this matrix. Read-only (like `Vector.size`)
   * so it can never desync from `data`.
   */
  get cols(): number {
    return this._cols;
  }

  // -----------------------------------------------------------------
  // Structural predicates. All default to `tol = 0` (exact), matching
  // `lu()`/`determinant()`'s "exact zero only" convention — pass a
  // small positive `tol` when checking a matrix built by floating-point
  // computation (e.g. the `R` from `qr()`/`qrUpdate()`), whose
  // off-triangle entries are typically tiny-but-nonzero rather than
  // exactly `0`.
  // -----------------------------------------------------------------

  /**
   * Checks whether this matrix is square (`rows === cols`).
   * @returns `true` if `rows === cols`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * const B = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.isSquare());
   * console.log(B.isSquare());
   * ```
   *
   * Output:
   * ```text
   * false
   * true
   * ```
   */
  isSquare(): boolean {
    return this.rows === this.cols;
  }

  /**
   * Checks whether this matrix equals its own transpose, i.e.
   * `this[i][j] === this[j][i]` for every `i`, `j`. Always `false` for a
   * non-square matrix (transposing would change its shape, so it can
   * never equal itself).
   * @param tol Absolute tolerance: a pair `(i, j)`/`(j, i)` counts as
   * equal if `|this[i][j] - this[j][i]| <= tol`. Defaults to `0` (exact).
   * @returns `true` if this matrix is symmetric within `tol`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const S = Matrix.from([[1, 2], [2, 3]]);
   * const N = Matrix.from([[1, 2], [3, 4]]);
   * console.log(S.isSymmetric());
   * console.log(N.isSymmetric());
   * ```
   *
   * Output:
   * ```text
   * true
   * false
   * ```
   */
  isSymmetric(tol: number = 0): boolean {
    if (!this.isSquare()) return false;
    for (let i = 0; i < this.rows; i++) {
      for (let j = i + 1; j < this.cols; j++) {
        if (Math.abs(this.getUnchecked(i, j) - this.getUnchecked(j, i)) > tol) return false;
      }
    }
    return true;
  }

  /**
   * Checks whether this matrix is lower triangular: every entry strictly
   * above the main diagonal (`j > i`) is zero. Unlike `isSymmetric()`,
   * this doesn't require a square matrix — a wide or tall matrix can
   * still satisfy "zero above the diagonal" (a trapezoidal shape), the
   * same generalization `qr()` relies on for `R` when `rows !== cols`.
   * @param tol Absolute tolerance: an entry above the diagonal counts as
   * zero if `|this[i][j]| <= tol`. Defaults to `0` (exact).
   * @returns `true` if every entry above the main diagonal is zero within `tol`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const L = Matrix.from([[1, 0], [2, 3]]);
   * console.log(L.isLowerTriangular());
   * console.log(Matrix.from([[1, 2], [0, 3]]).isLowerTriangular());
   * ```
   *
   * Output:
   * ```text
   * true
   * false
   * ```
   */
  isLowerTriangular(tol: number = 0): boolean {
    for (let i = 0; i < this.rows; i++) {
      for (let j = i + 1; j < this.cols; j++) {
        if (Math.abs(this.getUnchecked(i, j)) > tol) return false;
      }
    }
    return true;
  }

  /**
   * Checks whether this matrix is upper triangular (or, for a
   * non-square matrix, upper trapezoidal): every entry strictly below
   * the main diagonal (`i > j`) is zero. This is the shape `qr()`
   * guarantees for its `R` factor, including when `rows !== cols`.
   * @param tol Absolute tolerance: an entry below the diagonal counts as
   * zero if `|this[i][j]| <= tol`. Defaults to `0` (exact).
   * @returns `true` if every entry below the main diagonal is zero within `tol`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const U = Matrix.from([[1, 2], [0, 3]]);
   * console.log(U.isUpperTriangular());
   * console.log(Matrix.from([[1, 0], [2, 3]]).isUpperTriangular());
   * ```
   *
   * Output:
   * ```text
   * true
   * false
   * ```
   */
  isUpperTriangular(tol: number = 0): boolean {
    for (let i = 1; i < this.rows; i++) {
      for (let j = 0; j < Math.min(i, this.cols); j++) {
        if (Math.abs(this.getUnchecked(i, j)) > tol) return false;
      }
    }
    return true;
  }

  /**
   * Throws if `other` is not shape-compatible with this instance: for
   * Matrix, "compatible" means the same `rows` *and* `cols` — matching
   * `data.length` is not enough, since e.g. a 2x3 and a 3x2 matrix have
   * equal length but incompatible shape. Used internally (via `ArrayND`'s
   * arithmetic/isClose/dot/dist methods) to guard against silent shape
   * mismatches.
   * @param other The other matrix.
   * @param caller Name of the public method invoking this check, used to
   * produce a precise error message (e.g. `"add"`).
   * @throws {RangeError} If `other.rows !== this.rows || other.cols !== this.cols`.
   */
  protected _checkSameShape(other: this, caller: string): void {
    if (other.rows !== this.rows || other.cols !== this.cols) {
      throw new RangeError(
        `Matrix.${caller}: shape mismatch: ${this.rows}x${this.cols} vs ${other.rows}x${other.cols}`
      );
    }
  }

  /**
   * Internal-only fast constructor: wraps `data` directly as a new
   * Matrix of shape `rows x cols`, with no copying and no validation
   * whatsoever — `data` must already be a fresh `Float64Array` of length
   * `rows * cols`. Used by `_create()` (so `ArrayND`'s arithmetic
   * doesn't pay for a second allocation+copy, plus redundant
   * re-validation of already-known-good `rows`/`cols`, on top of the
   * buffer it already built) and by `row()` (which already owns a
   * freshly sliced, independent buffer by the time it gets here).
   * `public` only because TypeScript has no package-private — the name
   * (rather than a leading underscore) is the actual "don't call this
   * unless you already hold a validated, freshly-built buffer" signal:
   * skip it and you can construct a Matrix whose declared shape doesn't
   * match its buffer's length. Like `_create`'s `as this` cast, this
   * assumes Matrix is never itself subclassed.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data The buffer to wrap directly. Not copied. Must have length `rows * cols`.
   * @returns A new Matrix wrapping `data`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const buf = new Float64Array([1, 2, 3, 4]);
   * const A = Matrix.wrapUnchecked(2, 2, buf);
   * buf[0] = 10;
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[10, 2], [3, 4]]
   * ```
   */
  static wrapUnchecked(rows: number, cols: number, data: Float64Array): Matrix {
    const m = Object.create(Matrix.prototype) as Matrix;
    m._rows = rows;
    m._cols = cols;
    m.data = data;
    return m;
  }

  /**
   * Constructs a new Matrix with this instance's shape, wrapping the
   * given buffer directly.
   * @param data The buffer for the new matrix to wrap. Must already have
   * length `rows * cols`.
   * @returns A new `rows x cols` Matrix.
   */
  protected _create(data: Float64Array): this {
    return Matrix.wrapUnchecked(this.rows, this.cols, data) as this;
  }

  /**
   * Throws if `(i, j)` is not a valid 0-based index into this matrix.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @throws {RangeError} If `i` or `j` is out of range.
   */
  private _checkBounds(i: number, j: number): void {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
      throw new RangeError(
        `Matrix index (${i}, ${j}) out of bounds for ${this.rows}x${this.cols} matrix`
      );
    }
  }

  /**
   * Converts a 0-based `(i, j)` index into a flat index into `data`.
   * `public` rather than `private` (like `wrapUnchecked`) only so
   * `Matrix.linalg.ts`'s free functions — which operate on arbitrary
   * `Matrix` instances, not just `this` — can use it too; not
   * meaningfully part of the public API and shouldn't be relied on
   * outside this package. Does no bounds checking itself, but is also
   * not unsafe to call: it's pure arithmetic and can't corrupt
   * anything on its own, unlike `getUnchecked`/`setUnchecked` below,
   * which actually touch the buffer.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @returns The flat, 0-based index.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.flatIndex(0, 2));
   * console.log(A.flatIndex(1, 0));
   * ```
   *
   * Output:
   * ```text
   * 2
   * 3
   * ```
   */
  flatIndex(i: number, j: number): number {
    return i * this.cols + j;
  }

  /**
   * Unchecked element read: skips the bounds check `get()` does.
   * Internal-only, for hot loops (`matmul`, `inverse`, `lu`, etc.,
   * including the decomposition algorithms in `Matrix.linalg.ts`)
   * where the loop bounds already guarantee `i`/`j` are valid, so
   * `get()`'s check would be pure overhead. `public` rather than
   * `private` for the same reason as `flatIndex` — see its doc
   * comment. Callers are responsible for correctness: an out-of-range
   * `i`/`j` won't throw, it will silently read `undefined` (or another
   * element entirely) from the underlying buffer.
   * @param i Row index (0-based). Must be valid; not checked.
   * @param j Column index (0-based). Must be valid; not checked.
   * @returns The value at `(i, j)`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.getUnchecked(1, 0));
   * ```
   *
   * Output:
   * ```text
   * 3
   * ```
   */
  getUnchecked(i: number, j: number): number {
    return this.data[this.flatIndex(i, j)];
  }

  /**
   * Unchecked element write: skips the bounds check `set()` does.
   * Internal-only, for hot loops where the loop bounds already
   * guarantee `i`/`j` are valid. `public` rather than `private` for
   * the same reason as `flatIndex` — see its doc comment. Callers are
   * responsible for correctness: an out-of-range `i`/`j` won't throw —
   * it will either silently overwrite a different element (if the
   * flat index still lands inside the buffer) or silently do nothing
   * (a `Float64Array` write past the end is a no-op, not an error).
   * @param i Row index (0-based). Must be valid; not checked.
   * @param j Column index (0-based). Must be valid; not checked.
   * @param value The value to store.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.zero(2, 2);
   * A.setUnchecked(0, 1, 7);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 7], [0, 0]]
   * ```
   */
  setUnchecked(i: number, j: number, value: number): void {
    this.data[this.flatIndex(i, j)] = value;
  }

  // -----------------------------------------------------------------
  // Element / row / column access
  // -----------------------------------------------------------------

  /**
   * Gets the element at row `i`, column `j`.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @returns The value at `(i, j)`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.get(0, 1));
   * console.log(A.get(1, 1));
   * ```
   *
   * Output:
   * ```text
   * 2
   * 4
   * ```
   */
  get(i: number, j: number): number {
    this._checkBounds(i, j);
    return this.data[this.flatIndex(i, j)];
  }

  /**
   * Sets the element at row `i`, column `j`, mutating this matrix in place.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @param value The value to store.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.zero(2, 2);
   * A.set(0, 1, 5).set(1, 0, 6);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 5], [6, 0]]
   * ```
   */
  set(i: number, j: number, value: number): this {
    this._checkBounds(i, j);
    this.data[this.flatIndex(i, j)] = value;
    return this;
  }

  /**
   * Extracts row `i` as a vector.
   * @param i Row index (0-based).
   * @returns A new vector with `this.cols` components.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.row(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(4, 5, 6)
   * ```
   */
  row(i: number): Vector {
    if (i < 0 || i >= this.rows)
      throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
    return Vector.wrapUnchecked(
      this.data.slice(this.flatIndex(i, 0), this.flatIndex(i, 0) + this.cols)
    );
  }

  /**
   * Extracts column `j` as a vector.
   * @param j Column index (0-based).
   * @returns A new vector with `this.rows` components.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.col(2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(3, 6)
   * ```
   */
  col(j: number): Vector {
    if (j < 0 || j >= this.cols)
      throw new RangeError(`Matrix column ${j} out of bounds for ${this.cols} columns`);
    const res = new Vector(this.rows);
    for (let i = 0; i < this.rows; i++) res.data[i] = this.getUnchecked(i, j);
    return res;
  }

  /**
   * Extracts the main diagonal as a vector. Unlike `trace()`, this does
   * not require a square matrix: following numpy's `diag()` convention,
   * a non-square matrix yields `min(rows, cols)` entries, `(0,0),
   * (1,1), ..., (k-1,k-1)`. The static counterpart, `Matrix.diag(v)`,
   * builds a diagonal matrix from a vector — the same "extraction has a
   * constructor counterpart" pairing `row`/`setRow` and `col`/`setCol`
   * already follow.
   * @returns A new vector of length `min(rows, cols)` holding the
   * diagonal entries.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.diag().toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 4)
   * ```
   */
  diag(): Vector {
    const n = Math.min(this.rows, this.cols);
    const res = new Vector(n);
    for (let i = 0; i < n; i++) res.data[i] = this.getUnchecked(i, i);
    return res;
  }

  /**
   * Overwrites row `i` in place with the given values.
   * @param i Row index (0-based).
   * @param values Values to copy in; must have length `cols`.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `i` is out of bounds, or `values.length !== cols`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.zero(2, 3);
   * A.setRow(0, [1, 2, 3]);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2, 3], [0, 0, 0]]
   * ```
   */
  setRow(i: number, values: Vector | ArrayLike<number>): this {
    if (i < 0 || i >= this.rows)
      throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
    const src = values instanceof Vector ? values.data : values;
    if (src.length !== this.cols) {
      throw new RangeError(
        `Matrix setRow: expected ${this.cols} values for row ${i}, got ${src.length}`
      );
    }
    this.data.set(src, this.flatIndex(i, 0));
    return this;
  }

  /**
   * Overwrites column `j` in place with the given values.
   * @param j Column index (0-based).
   * @param values Values to copy in; must have length `rows`.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `j` is out of bounds, or `values.length !== rows`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.zero(2, 3);
   * A.setCol(1, Vector.from([7, 8]));
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 7, 0], [0, 8, 0]]
   * ```
   */
  setCol(j: number, values: Vector | ArrayLike<number>): this {
    if (j < 0 || j >= this.cols)
      throw new RangeError(`Matrix column ${j} out of bounds for ${this.cols} columns`);
    const src = values instanceof Vector ? values.data : values;
    if (src.length !== this.rows) {
      throw new RangeError(
        `Matrix setCol: expected ${this.rows} values for column ${j}, got ${src.length}`
      );
    }
    for (let i = 0; i < this.rows; i++) this.setUnchecked(i, j, src[i]);
    return this;
  }

  // -----------------------------------------------------------------
  // Elementwise arithmetic: `ArrayND`'s matrix/scalar overloads,
  // re-declared here so a `Vector` overload (row broadcasting) can be
  // added — a subclass override replaces the inherited signature set,
  // it doesn't extend it. The matrix/scalar paths just forward to
  // `ArrayND`; only the `Vector` path is new (`_broadcastRow`).
  // -----------------------------------------------------------------

  /**
   * Adds a matrix of the same shape, elementwise.
   * @param x The matrix. Must have the same shape as this one.
   * @returns A new matrix equal to `this + x`.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const B = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.add(B).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[11, 22], [33, 44]]
   * ```
   */
  add(x: this): this;

  /**
   * Adds a scalar, applied to every element.
   * @param s The scalar.
   * @returns A new matrix equal to `this + s`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.add(10).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[11, 12], [13, 14]]
   * ```
   */
  add(s: number): this;

  /**
   * Adds a row vector, broadcast across the rows (numpy: `A + v`
   * with `v` of shape `(cols,)`): `R[i][j] = this[i][j] + v[j]`.
   * Operand order is preserved, so for `sub`/`div` the vector is always
   * the right-hand side.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns A new matrix with this matrix's shape.
   * @throws {RangeError} If `v.size` is neither `1` nor `this.cols`; or
   * if `this.cols === 1` and `v.size > 1` (numpy would return a
   * `rows x v.size` matrix; that shape change is not supported here).
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.add(Vector.from([10, 20])).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[11, 22], [13, 24]]
   * ```
   */
  add(v: Vector): this;

  add(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'add', false);
    return typeof x === 'number' ? super.add(x) : super.add(x);
  }

  /**
   * Adds a matrix of the same shape in place, elementwise: `this += x`.
   * @param x The matrix. Must have the same shape as this one.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const B = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.addSelf(B).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[11, 22], [33, 44]]
   * Matrix[[11, 22], [33, 44]]
   * ```
   */
  addSelf(x: this): this;

  /**
   * Adds a scalar in place, applied to every element: `this += s`.
   * @param s The scalar.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.addSelf(10).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[11, 12], [13, 14]]
   * Matrix[[11, 12], [13, 14]]
   * ```
   */
  addSelf(s: number): this;

  /**
   * Adds a row vector in place, broadcast across the rows (numpy:
   * `A += v` with `v` of shape `(cols,)`): `this[i][j] += v[j]`. The
   * shape never changes, so every supported case can run in place.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns `this`, for chaining. Left unmodified if an error is thrown.
   * @throws {RangeError} Under the same conditions as `add(v)`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.addSelf(Vector.from([10, 20])).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[11, 22], [13, 24]]
   * Matrix[[11, 22], [13, 24]]
   * ```
   */
  addSelf(v: Vector): this;

  addSelf(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'add', true);
    return typeof x === 'number' ? super.addSelf(x) : super.addSelf(x);
  }

  /**
   * Subtracts a matrix of the same shape, elementwise.
   * @param x The matrix. Must have the same shape as this one.
   * @returns A new matrix equal to `this - x`.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * const B = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.sub(B).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[9, 18], [27, 36]]
   * ```
   */
  sub(x: this): this;

  /**
   * Subtracts a scalar, applied to every element.
   * @param s The scalar.
   * @returns A new matrix equal to `this - s`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.sub(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 1], [2, 3]]
   * ```
   */
  sub(s: number): this;

  /**
   * Subtracts a row vector, broadcast across the rows (numpy: `A - v`
   * with `v` of shape `(cols,)`): `R[i][j] = this[i][j] - v[j]`.
   * Operand order is preserved, so for `sub`/`div` the vector is always
   * the right-hand side.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns A new matrix with this matrix's shape.
   * @throws {RangeError} If `v.size` is neither `1` nor `this.cols`; or
   * if `this.cols === 1` and `v.size > 1` (numpy would return a
   * `rows x v.size` matrix; that shape change is not supported here).
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.sub(Vector.from([1, 2])).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[9, 18], [29, 38]]
   * ```
   */
  sub(v: Vector): this;

  sub(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'sub', false);
    return typeof x === 'number' ? super.sub(x) : super.sub(x);
  }

  /**
   * Subtracts a matrix of the same shape in place, elementwise: `this -= x`.
   * @param x The matrix. Must have the same shape as this one.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * const B = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.subSelf(B).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[9, 18], [27, 36]]
   * Matrix[[9, 18], [27, 36]]
   * ```
   */
  subSelf(x: this): this;

  /**
   * Subtracts a scalar in place, applied to every element: `this -= s`.
   * @param s The scalar.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.subSelf(1).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 1], [2, 3]]
   * Matrix[[0, 1], [2, 3]]
   * ```
   */
  subSelf(s: number): this;

  /**
   * Subtracts a row vector in place, broadcast across the rows (numpy:
   * `A -= v` with `v` of shape `(cols,)`): `this[i][j] -= v[j]`. The
   * shape never changes, so every supported case can run in place.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns `this`, for chaining. Left unmodified if an error is thrown.
   * @throws {RangeError} Under the same conditions as `sub(v)`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.subSelf(Vector.from([1, 2])).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[9, 18], [29, 38]]
   * Matrix[[9, 18], [29, 38]]
   * ```
   */
  subSelf(v: Vector): this;

  subSelf(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'sub', true);
    return typeof x === 'number' ? super.subSelf(x) : super.subSelf(x);
  }

  /**
   * Multiplies this matrix with a matrix of the same shape, elementwise.
   * @param x The matrix. Must have the same shape as this one.
   * @returns A new matrix equal to `this * x`.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const B = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.mult(B).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[10, 40], [90, 160]]
   * ```
   */
  mult(x: this): this;

  /**
   * Multiplies this matrix with a scalar, applied to every element.
   * @param s The scalar.
   * @returns A new matrix equal to `this * s`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.mult(3).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[3, 6], [9, 12]]
   * ```
   */
  mult(s: number): this;

  /**
   * Multiplies this matrix with a row vector, broadcast across the rows (numpy: `A * v`
   * with `v` of shape `(cols,)`): `R[i][j] = this[i][j] * v[j]`.
   * Operand order is preserved, so for `sub`/`div` the vector is always
   * the right-hand side.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns A new matrix with this matrix's shape.
   * @throws {RangeError} If `v.size` is neither `1` nor `this.cols`; or
   * if `this.cols === 1` and `v.size > 1` (numpy would return a
   * `rows x v.size` matrix; that shape change is not supported here).
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.mult(Vector.from([10, 20])).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[10, 40], [30, 80]]
   * ```
   */
  mult(v: Vector): this;

  mult(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'mult', false);
    return typeof x === 'number' ? super.mult(x) : super.mult(x);
  }

  /**
   * Multiplies this matrix with a matrix of the same shape in place, elementwise: `this *= x`.
   * @param x The matrix. Must have the same shape as this one.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const B = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.multSelf(B).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[10, 40], [90, 160]]
   * Matrix[[10, 40], [90, 160]]
   * ```
   */
  multSelf(x: this): this;

  /**
   * Multiplies this matrix with a scalar in place, applied to every element: `this *= s`.
   * @param s The scalar.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.multSelf(3).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[3, 6], [9, 12]]
   * Matrix[[3, 6], [9, 12]]
   * ```
   */
  multSelf(s: number): this;

  /**
   * Multiplies this matrix with a row vector in place, broadcast across the rows (numpy:
   * `A *= v` with `v` of shape `(cols,)`): `this[i][j] *= v[j]`. The
   * shape never changes, so every supported case can run in place.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns `this`, for chaining. Left unmodified if an error is thrown.
   * @throws {RangeError} Under the same conditions as `mult(v)`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.multSelf(Vector.from([10, 20])).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[10, 40], [30, 80]]
   * Matrix[[10, 40], [30, 80]]
   * ```
   */
  multSelf(v: Vector): this;

  multSelf(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'mult', true);
    return typeof x === 'number' ? super.multSelf(x) : super.multSelf(x);
  }

  /**
   * Divides this matrix by a matrix of the same shape, elementwise.
   * @param x The matrix. Must have the same shape as this one.
   * @returns A new matrix equal to `this / x`.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * const B = Matrix.from([[2, 4], [5, 8]]);
   * console.log(A.div(B).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[5, 5], [6, 5]]
   * ```
   */
  div(x: this): this;

  /**
   * Divides this matrix by a scalar, applied to every element.
   * @param s The scalar.
   * @returns A new matrix equal to `this / s`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.div(2).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0.5, 1], [1.5, 2]]
   * ```
   */
  div(s: number): this;

  /**
   * Divides this matrix by a row vector, broadcast across the rows (numpy: `A / v`
   * with `v` of shape `(cols,)`): `R[i][j] = this[i][j] / v[j]`.
   * Operand order is preserved, so for `sub`/`div` the vector is always
   * the right-hand side.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns A new matrix with this matrix's shape.
   * @throws {RangeError} If `v.size` is neither `1` nor `this.cols`; or
   * if `this.cols === 1` and `v.size > 1` (numpy would return a
   * `rows x v.size` matrix; that shape change is not supported here).
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.div(Vector.from([2, 4])).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[5, 5], [15, 10]]
   * ```
   */
  div(v: Vector): this;

  div(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'div', false);
    return typeof x === 'number' ? super.div(x) : super.div(x);
  }

  /**
   * Divides this matrix by a matrix of the same shape in place, elementwise: `this /= x`.
   * @param x The matrix. Must have the same shape as this one.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `x`'s shape differs from this matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * const B = Matrix.from([[2, 4], [5, 8]]);
   * console.log(A.divSelf(B).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[5, 5], [6, 5]]
   * Matrix[[5, 5], [6, 5]]
   * ```
   */
  divSelf(x: this): this;

  /**
   * Divides this matrix by a scalar in place, applied to every element: `this /= s`.
   * @param s The scalar.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.divSelf(2).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0.5, 1], [1.5, 2]]
   * Matrix[[0.5, 1], [1.5, 2]]
   * ```
   */
  divSelf(s: number): this;

  /**
   * Divides this matrix by a row vector in place, broadcast across the rows (numpy:
   * `A /= v` with `v` of shape `(cols,)`): `this[i][j] /= v[j]`. The
   * shape never changes, so every supported case can run in place.
   * @param v The row vector. Must have `v.size === this.cols`, or
   * `v.size === 1`, in which case it acts as a scalar (as in numpy).
   * @returns `this`, for chaining. Left unmodified if an error is thrown.
   * @throws {RangeError} Under the same conditions as `div(v)`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[10, 20], [30, 40]]);
   * console.log(A.divSelf(Vector.from([2, 4])).toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[5, 5], [15, 10]]
   * Matrix[[5, 5], [15, 10]]
   * ```
   */
  divSelf(v: Vector): this;

  divSelf(x: this | number | Vector): this {
    if (x instanceof Vector) return this._broadcastRow(x, 'div', true);
    return typeof x === 'number' ? super.divSelf(x) : super.divSelf(x);
  }

  /**
   * Shared implementation of `add`/`sub`/`mult`/`div` and their `Self`
   * variants for a `Vector` operand, broadcast as a row (see the class
   * header for the supported subset). Validates everything before
   * writing anything, so an error never leaves `this` half-modified.
   * `v.size === 1` forwards to the inherited scalar path; otherwise the
   * loop is specialised per operation, with the `switch` hoisted out of
   * the loops (an indirect per-element callback measured ~3-5x slower).
   * In-place is safe because each output element is written at the
   * exact index it was just read from, and `v` has its own buffer.
   * @param v The row vector.
   * @param op Which operation to apply, with `this` as the left operand.
   * @param inPlace If `true`, write the result into `this` and return it;
   * otherwise into a fresh matrix.
   * @returns `this` (in place) or a new matrix of the same shape.
   * @throws {RangeError} As documented on the public overloads.
   */
  private _broadcastRow(v: Vector, op: 'add' | 'sub' | 'mult' | 'div', inPlace: boolean): this {
    const m = this.rows;
    const n = this.cols;
    const k = v.size;

    if (k === 1) {
      const s = v.data[0];
      switch (op) {
        case 'add':
          return inPlace ? super.addSelf(s) : super.add(s);
        case 'sub':
          return inPlace ? super.subSelf(s) : super.sub(s);
        case 'mult':
          return inPlace ? super.multSelf(s) : super.mult(s);
        case 'div':
          return inPlace ? super.divSelf(s) : super.div(s);
      }
    }

    if (k !== n) {
      const caller = inPlace ? `${op}Self` : op;
      if (n === 1) {
        throw new RangeError(
          `Matrix.${caller}: broadcasting a vector of size ${k} against a ${m}x1 matrix would ` +
            `change the result shape to ${m}x${k}, which is not supported`
        );
      }
      throw new RangeError(
        `Matrix.${caller}: operands could not be broadcast together with shapes (${m},${n}) (${k})`
      );
    }

    const a = this.data;
    const b = v.data;
    const out = inPlace ? a : new Float64Array(a.length);
    switch (op) {
      case 'add':
        for (let i = 0; i < m; i++) {
          const o = i * n;
          for (let j = 0; j < n; j++) out[o + j] = a[o + j] + b[j];
        }
        break;
      case 'sub':
        for (let i = 0; i < m; i++) {
          const o = i * n;
          for (let j = 0; j < n; j++) out[o + j] = a[o + j] - b[j];
        }
        break;
      case 'mult':
        for (let i = 0; i < m; i++) {
          const o = i * n;
          for (let j = 0; j < n; j++) out[o + j] = a[o + j] * b[j];
        }
        break;
      case 'div':
        for (let i = 0; i < m; i++) {
          const o = i * n;
          for (let j = 0; j < n; j++) out[o + j] = a[o + j] / b[j];
        }
        break;
    }
    return inPlace ? this : this._create(out);
  }

  // -----------------------------------------------------------------
  // Matrix-specific operations: no Vector equivalent, or deliberately
  // not inherited from ArrayND (arity/shape differ too much to share).
  // -----------------------------------------------------------------

  /**
   * Applies a function to each element, elementwise. The general
   * escape hatch for an arbitrary elementwise transform — not just a
   * wrapper around `abs`/`pow`/`sqrt`/`clip` (inherited from
   * `ArrayND`), which only cover fixed operations. This is `Matrix`'s
   * counterpart to `Vector.map`; it isn't inherited from `ArrayND`
   * because the callback's arity differs (two indices here vs one on
   * `Vector`), same reason `get`/`set` aren't shared.
   * @param fn Called with each element's value, row index, and column
   * index; its return value becomes the corresponding element of the result.
   * @returns A new matrix, the same shape as this one, holding the mapped values.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.map((x) => x * x).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 4], [9, 16]]
   * ```
   */
  map(fn: (value: number, i: number, j: number) => number): Matrix {
    const res = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      const offset = this.flatIndex(i, 0);
      for (let j = 0; j < this.cols; j++) res.data[offset + j] = fn(this.data[offset + j], i, j);
    }
    return res;
  }

  /**
   * Multiplies this matrix by another: `this * m` (matrix product). Not
   * to be confused with `mult()`, the inherited elementwise/scalar
   * product.
   * @param m The right-hand matrix. Must have `m.rows === this.cols`.
   * @returns A new `this.rows x m.cols` matrix.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const B = Matrix.from([[0, 1], [1, 0]]);
   * console.log(A.matmul(B).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[2, 1], [4, 3]]
   * ```
   */
  matmul(m: Matrix): Matrix {
    if (this.cols !== m.rows) {
      throw new RangeError(
        `Matrix matmul shape mismatch: ${this.rows}x${this.cols} * ${m.rows}x${m.cols}`
      );
    }
    const res = new Matrix(this.rows, m.cols);
    const rowBuf = new Float64Array(m.cols);
    for (let i = 0; i < this.rows; i++) {
      rowBuf.fill(0);
      for (let k = 0; k < this.cols; k++) {
        const a = this.getUnchecked(i, k);
        if (a === 0) continue;
        const mOffset = m.flatIndex(k, 0);
        for (let j = 0; j < m.cols; j++) rowBuf[j] += a * m.data[mOffset + j];
      }
      res.data.set(rowBuf, res.flatIndex(i, 0));
    }
    return res;
  }

  /**
   * Multiplies this matrix by a column vector: `this * v`.
   * @param v The vector. Must have `v.size === this.cols`.
   * @returns A new vector with `this.rows` components.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.mulVec(Vector.from([1, 1])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(3, 7)
   * ```
   */
  mulVec(v: Vector): Vector {
    if (v.size !== this.cols) {
      throw new RangeError(
        `Matrix mulVec shape mismatch: ${this.rows}x${this.cols} * vec(${v.size})`
      );
    }
    const res = new Vector(this.rows);
    for (let i = 0; i < this.rows; i++) {
      const offset = this.flatIndex(i, 0);
      let sum = 0;
      for (let j = 0; j < this.cols; j++) sum += this.data[offset + j] * v.data[j];
      res.data[i] = sum;
    }
    return res;
  }

  /**
   * Computes the transpose of this matrix.
   * @returns A new `this.cols x this.rows` matrix equal to `this^T`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.transpose().toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 4], [2, 5], [3, 6]]
   * ```
   */
  transpose(): Matrix {
    const res = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      const offset = this.flatIndex(i, 0);
      for (let j = 0; j < this.cols; j++) res.setUnchecked(j, i, this.data[offset + j]);
    }
    return res;
  }

  /**
   * Computes the trace (sum of diagonal elements) of this matrix.
   * @returns The trace.
   * @throws {RangeError} If this matrix is not square.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.trace());
   * ```
   *
   * Output:
   * ```text
   * 5
   * ```
   */
  trace(): number {
    if (this.rows !== this.cols)
      throw new RangeError(`Matrix trace requires a square matrix, got ${this.rows}x${this.cols}`);
    let sum = 0;
    for (let i = 0; i < this.rows; i++) sum += this.getUnchecked(i, i);
    return sum;
  }

  /**
   * Computes the matrix infinity norm: the largest absolute row sum,
   * `max_i sum_j |this[i][j]|`. Besides being used internally to
   * auto-scale `rank()`'s default tolerance to this matrix's magnitude,
   * this is a general-purpose norm in its own right (unlike `ArrayND`'s
   * `norm()`, which is the Frobenius norm over the whole flat buffer
   * regardless of shape). See `norm1()` for its column-sum counterpart;
   * `this.norm1() === this.transpose().normInf()` always holds.
   *
   * This overrides `ArrayND.normInf()`'s entrywise "largest absolute
   * value" meaning (the correct one for `Vector`) with the induced/
   * operator norm above, which is the conventional meaning of "infinity
   * norm" for a matrix.
   * @returns The largest absolute row sum, or `0` for a matrix of all zeros.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, -2], [-3, 4]]);
   * console.log(A.normInf());
   * ```
   *
   * Output:
   * ```text
   * 7
   * ```
   */
  normInf(): number {
    let maxRowSum = 0;
    for (let i = 0; i < this.rows; i++) {
      const offset = this.flatIndex(i, 0);
      let rowSum = 0;
      for (let j = 0; j < this.cols; j++) rowSum += Math.abs(this.data[offset + j]);
      if (rowSum > maxRowSum) maxRowSum = rowSum;
    }
    return maxRowSum;
  }

  /**
   * Computes the matrix 1-norm: the largest absolute column sum,
   * `max_j sum_i |this[i][j]|`. The column-sum counterpart to
   * `normInf()`'s row-sum; `this.norm1() === this.transpose().normInf()`
   * always holds. Accumulates all column sums in a single pass over the
   * buffer, rather than summing each column separately.
   * @returns The largest absolute column sum, or `0` for a matrix of all zeros.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, -2], [-3, 4]]);
   * console.log(A.norm1());
   * ```
   *
   * Output:
   * ```text
   * 6
   * ```
   */
  norm1(): number {
    const colSums = new Float64Array(this.cols);
    for (let i = 0; i < this.rows; i++) {
      const offset = this.flatIndex(i, 0);
      for (let j = 0; j < this.cols; j++) colSums[j] += Math.abs(this.data[offset + j]);
    }
    let maxColSum = 0;
    for (let j = 0; j < this.cols; j++) {
      if (colSums[j] > maxColSum) maxColSum = colSums[j];
    }
    return maxColSum;
  }

  /**
   * Computes the rank of this matrix (the number of linearly independent
   * rows/columns): see `linalg.rank()` for the algorithm and the full
   * meaning of `tol`.
   * @param tol Absolute tolerance below which a pivot is treated as
   * zero; auto-scaled to this matrix's magnitude if omitted.
   * @returns The rank, between `0` and `min(rows, cols)`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [2, 4]]);
   * const B = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.rank());
   * console.log(B.rank());
   * ```
   *
   * Output:
   * ```text
   * 1
   * 2
   * ```
   */
  rank(tol?: number): number {
    return linalg.rank(this, tol);
  }

  /**
   * Computes the determinant of this matrix, via Gaussian elimination
   * with partial pivoting: see `linalg.determinant()` for the algorithm.
   * @returns The determinant.
   * @throws {RangeError} If this matrix is not square.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.determinant());
   * ```
   *
   * Output:
   * ```text
   * -2
   * ```
   */
  determinant(): number {
    return linalg.determinant(this);
  }

  /**
   * Computes the inverse of this matrix, via Gauss-Jordan elimination
   * with partial pivoting: see `linalg.inverse()` for the algorithm.
   * @returns A new matrix `M` such that `this.matmul(M)` is (up to
   *   floating-point error) the identity matrix.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If this matrix is singular (not invertible).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[4, 7], [2, 6]]);
   * console.log(A.inverse().toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0.6000000000000001, -0.7000000000000001], [-0.2, 0.4]]
   * ```
   */
  inverse(): Matrix {
    return linalg.inverse(this);
  }

  /**
   * Computes a partial-pivoted LU factorization of this matrix, such
   * that `P * this = L * U` for the row permutation `P` implied by
   * `perm`: see `linalg.lu()` for the algorithm and why it has no
   * tolerance parameter.
   * @param out Optional storage to reuse, as returned by an earlier call
   * for a matrix of the same size: its `L`, `U`, `perm` and `sign` are
   * overwritten and it is returned, so repeated factorizations allocate
   * nothing. This matrix may be `out.U` (factored in place), but not
   * `out.L`. If this method throws, the contents of `out` are undefined.
   * @returns The `{ L, U, perm, sign }` factorization: `out` if given,
   * otherwise a new object.
   * @throws {RangeError} If this matrix is not square, if `out` does not
   * match its size, or if this matrix is `out.L`.
   * @throws {Error} If this matrix is singular (some column's largest
   * available pivot is exactly `0`).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const dec = A.lu();
   * console.log(dec.L.toString());
   * console.log(dec.U.toString());
   * console.log(dec.perm);
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 0], [0.3333333333333333, 1]]
   * Matrix[[3, 4], [0, 0.6666666666666667]]
   * [ 1, 0 ]
   * ```
   */
  lu(out?: LUDecomposition): LUDecomposition {
    return linalg.lu(this, out);
  }

  /**
   * Solves `A * x = b` for `x`, given the LU factorization of `A` as
   * returned by `lu()`: see `linalg.luSolve()`. If you need to solve
   * against the same matrix with several right-hand sides, call `lu()`
   * once and reuse this instead of refactoring each time.
   * @param decomposition The `{ L, U, perm }` factorization of the
   * coefficient matrix, as returned by `lu()`.
   * @param b The right-hand side vector. Must have `b.size === L.rows`.
   * @param out Optional vector of size `L.rows` that receives the
   * solution, avoiding an allocation. It must not be `b`.
   * @returns The solution vector `x`: `out` if given, otherwise a new vector.
   * @throws {RangeError} If `b` or `out` does not match the size of the
   * factorization, or if `out` is `b`.
   * @throws {Error} If `U` has a zero diagonal entry.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const dec = A.lu();
   * console.log(Matrix.luSolve(dec, Vector.from([5, 11])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2)
   * ```
   */
  static luSolve(decomposition: LUDecomposition, b: Vector, out?: Vector): Vector {
    return linalg.luSolve(decomposition, b, out);
  }

  /**
   * Solves `this * x = b` for `x`, treating this matrix as lower
   * triangular: only entries on and below the diagonal are read, so
   * this can be called directly on the `L` factor from `lu()`.
   * @param b The right-hand side vector. Must have `b.size === this.rows`.
   * @param unitDiagonal If `true`, the diagonal is assumed to be all 1s
   *   (as `lu()`'s `L` always is) and is never read, avoiding a division.
   * @param out Optional vector of size `this.rows` that receives the
   *   solution, avoiding an allocation. It may be `b` itself, which is
   *   then overwritten with the solution.
   * @returns The solution vector `x`: `out` if given, otherwise a new vector.
   * @throws {RangeError} If this matrix is not square, `b.size !== this.rows`,
   *   or `out.size !== this.rows`.
   * @throws {Error} If `unitDiagonal` is `false` and a zero diagonal entry is encountered.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const L = Matrix.from([[2, 0], [1, 3]]);
   * console.log(L.solveLower(Vector.from([2, 5])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 1.3333333333333333)
   * ```
   */
  solveLower(b: Vector, unitDiagonal = false, out?: Vector): Vector {
    return linalg.solveLower(this, b, unitDiagonal, out);
  }

  /**
   * Solves `this * x = b` for `x`, treating this matrix as upper
   * triangular: only entries on and above the diagonal are read, so this
   * can be called directly on the `U` factor from `lu()`.
   * @param b The right-hand side vector. Must have `b.size === this.rows`.
   * @param out Optional vector of size `this.rows` that receives the
   *   solution, avoiding an allocation. It may be `b` itself, which is
   *   then overwritten with the solution.
   * @returns The solution vector `x`: `out` if given, otherwise a new vector.
   * @throws {RangeError} If this matrix is not square, `b.size !== this.rows`,
   *   or `out.size !== this.rows`.
   * @throws {Error} If a zero diagonal entry is encountered.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const U = Matrix.from([[2, 1], [0, 3]]);
   * console.log(U.solveUpper(Vector.from([4, 6])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2)
   * ```
   */
  solveUpper(b: Vector, out?: Vector): Vector {
    return linalg.solveUpper(this, b, out);
  }

  /**
   * Solves the linear system `this * x = b` for `x`, via LU
   * decomposition with partial pivoting (`lu()`) followed by forward
   * and back substitution: see `linalg.solve()`. If you need to solve
   * against the same matrix with several right-hand sides, call `lu()`
   * once yourself and reuse `Matrix.luSolve` instead of calling this
   * repeatedly.
   * @param b The right-hand side vector. Must have `b.size === this.rows`.
   * @returns The solution vector `x` such that `this.mulVec(x)` is (up to
   *   floating-point error) equal to `b`.
   * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
   * @throws {Error} If this matrix is singular (no unique solution).
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[4, 1], [2, 3]]);
   * console.log(A.solve(Vector.from([9, 13])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1.4, 3.4)
   * ```
   */
  solve(b: Vector): Vector {
    return linalg.solve(this, b);
  }

  // -----------------------------------------------------------------
  // Condition number estimation. Implementation lives in
  // Matrix.linalg.ts; see that file for the algorithms.
  // -----------------------------------------------------------------

  /**
   * Computes the exact 1-norm condition number of this matrix, `‖this‖₁ ·
   * ‖this⁻¹‖₁`, via `inverse()`: see `linalg.cond1()`. General-purpose,
   * but O(n³) — dominated by the explicit inverse. If this matrix is
   * triangular, `cond1Upper()`/`cond1Lower()` compute a reliable O(n²)
   * estimate instead, without ever forming the inverse; prefer those
   * when applicable.
   * @returns The 1-norm condition number, or `Infinity` if this matrix
   *   is (numerically) singular.
   * @throws {RangeError} If this matrix is not square.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[4, 1], [2, 3]]);
   * console.log(A.cond1());
   * ```
   *
   * Output:
   * ```text
   * 3
   * ```
   */
  cond1(): number {
    return linalg.cond1(this);
  }

  /**
   * Estimates the 1-norm condition number of this matrix, treating it as
   * upper triangular (only entries on and above the diagonal are read,
   * as with `solveUpper`), in O(n²) without ever forming the inverse:
   * see `linalg.cond1Upper()` for the algorithm (Cline–Moler–Stewart–
   * Wilkinson, 1979, as used by LINPACK's `dtrco` and described in
   * Dennis & Schnabel's Algorithm A3.3.1). Reliable in practice, though
   * (like any O(n²) estimator) it is a heuristic, not an exact value.
   * @returns The 1-norm condition number estimate, or `Infinity` if this
   *   matrix has a zero 1-norm.
   * @throws {RangeError} If this matrix is not square.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const U = Matrix.from([[2, 1], [0, 3]]);
   * console.log(U.cond1Upper());
   * ```
   *
   * Output:
   * ```text
   * 2
   * ```
   */
  cond1Upper(): number {
    return linalg.cond1Upper(this);
  }

  /**
   * Estimates the 1-norm condition number of this matrix, treating it as
   * lower triangular (only entries on and below the diagonal are read,
   * as with `solveLower`), in O(n²) without ever forming the inverse:
   * see `linalg.cond1Lower()` for the algorithm (Cline–Moler–Stewart–
   * Wilkinson, 1979, as used by LINPACK's `dtrco` and described in
   * Dennis & Schnabel's Algorithm A3.3.1). Reliable in practice, though
   * (like any O(n²) estimator) it is a heuristic, not an exact value.
   * @param unitDiagonal If `true`, the diagonal is assumed to be all 1s
   *   (as `lu()`'s `L` always is) and is never read.
   * @returns The 1-norm condition number estimate, or `Infinity` if this
   *   matrix has a zero 1-norm.
   * @throws {RangeError} If this matrix is not square.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const L = Matrix.from([[2, 0], [1, 3]]);
   * console.log(L.cond1Lower());
   * ```
   *
   * Output:
   * ```text
   * 1.6666666666666667
   * ```
   */
  cond1Lower(unitDiagonal = false): number {
    return linalg.cond1Lower(this, unitDiagonal);
  }

  // -----------------------------------------------------------------
  // Cholesky decomposition. Implementation lives in Matrix.linalg.ts;
  // see that file for the algorithm.
  // -----------------------------------------------------------------

  /**
   * Computes the Cholesky factorization of this matrix: `this = L *
   * L^T`, with `L` lower triangular and positive on the diagonal: see
   * `linalg.cholesky()` for the algorithm, and why it only reads the
   * lower triangle and has no tolerance parameter.
   * @returns The lower-triangular factor `L` such that `this = L * L^T`.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If this matrix is not positive definite (some
   * leading principal minor is not strictly positive).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const S = Matrix.from([[4, 2], [2, 3]]);
   * console.log(S.cholesky().toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[2, 0], [1, 1.4142135623730951]]
   * ```
   */
  cholesky(): Matrix {
    return linalg.cholesky(this);
  }

  /**
   * Solves `this * x = b` for `x`, treating this matrix as the Cholesky
   * factor `L` of a symmetric positive-definite matrix (as returned by
   * `cholesky()`): see `linalg.choleskySolve()`. If you need to solve
   * against the same matrix with several right-hand sides, call
   * `cholesky()` once and reuse this instead of refactoring each time.
   * @param b The right-hand side vector. Must have `b.size === this.rows`.
   * @returns The solution vector `x` such that
   * `this.matmul(this.transpose()).mulVec(x)` is (up to floating-point
   * error) equal to `b`.
   * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
   * @throws {Error} If this matrix has a zero diagonal entry.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const S = Matrix.from([[4, 2], [2, 3]]);
   * console.log(S.choleskySolve(Vector.from([8, 8])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0.2777777777777778, 0.4444444444444444)
   * ```
   */
  choleskySolve(b: Vector): Vector {
    return linalg.choleskySolve(this, b);
  }

  // -----------------------------------------------------------------
  // QR factorization and eigenvalues. Implementations live in
  // Matrix.linalg.ts; see that file for the algorithms.
  // -----------------------------------------------------------------

  /**
   * Computes a Householder QR factorization of this matrix: `this = Q *
   * R`, with `Q` orthogonal and `R` upper triangular (trapezoidal if
   * `this.rows > this.cols`).
   * @returns The `{ Q, R }` factorization.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * const dec = A.qr();
   * console.log(dec.Q.toString());
   * console.log(dec.R.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[-0.16903085094570325, 0.8970852271450607, 0.4082482904638628], [-0.50709255283711, 0.2760262237369415, -0.816496580927726], [-0.8451542547285166, -0.3450327796711769, 0.4082482904638632]]
   * Matrix[[-5.916079783099616, -7.437357441610946], [0, 0.8280786712108261], [0, 0]]
   * ```
   */
  qr(): QRDecomposition {
    return linalg.qr(this);
  }

  /**
   * Computes a Householder QR factorization with column pivoting of this
   * matrix (any shape), in compact storage: `this[:, perm] = Q * R`. The
   * orthogonal factor is never formed, so a tall matrix costs `O(rows *
   * cols)` memory instead of the `O(rows^2)` of `qr()`; the diagonal of `R`
   * has nonincreasing magnitude, which reveals the numerical rank. See
   * `linalg.qrPivot()` for the algorithm. Use `qrPivotMulQt`, `qrPivotQ` and
   * `qrPivotR` to work with the result.
   * @param out Optional storage to reuse, as returned by an earlier call
   * for a matrix of the same shape: overwritten and returned, so repeated
   * factorizations allocate no result storage. This matrix may be `out.QR`
   * (factored in place). If this method throws, the contents of `out` are
   * undefined.
   * @returns The `{ QR, tau, perm }` factorization: `out` if given,
   * otherwise a new object.
   * @throws {RangeError} If `out` does not match the shape of this matrix.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const a = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * const dec = a.qrPivot();
   * const Q = Matrix.qrPivotQ(dec); // 3x2
   * const R = Matrix.qrPivotR(dec); // 2x2
   * console.log(dec.perm);
   * console.log(Q.matmul(R).toString()); // a[:, perm]
   * ```
   *
   * Output:
   * ```text
   * [ 1, 0 ]
   * Matrix[[1.9999999999999993, 1.0000000000000007], [4, 3.0000000000000004], [6, 5.000000000000001]]
   * ```
   */
  qrPivot(out?: QRPivotDecomposition): QRPivotDecomposition {
    return linalg.qrPivot(this, out);
  }

  /**
   * Computes `Q^T * b` for the orthogonal factor `Q` of a pivoted QR
   * factorization, without forming `Q`: see `linalg.qrPivotMulQt()`.
   * @param decomposition A factorization, as returned by `qrPivot()`.
   * @param b The vector to transform. Must have `b.size === decomposition.QR.rows`.
   * @param out Optional vector of size `decomposition.QR.rows` that receives
   * the result, avoiding an allocation. It may be `b` itself.
   * @returns `Q^T * b` (length `rows`): `out` if given, otherwise a new
   * vector. For a tall matrix, the norm of the entries beyond the first
   * `min(rows, cols)` is the least-squares residual norm.
   * @throws {RangeError} If `b` or `out` does not match the factorization.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * const dec = A.qrPivot();
   * console.log(Matrix.qrPivotMulQt(dec, Vector.from([1, 2, 3])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(-3.7416573867739418, 9.447285969364182e-16, 3.0588234610191476e-16)
   * ```
   */
  static qrPivotMulQt(decomposition: QRPivotDecomposition, b: Vector, out?: Vector): Vector {
    return linalg.qrPivotMulQt(decomposition, b, out);
  }

  /**
   * Forms the thin orthogonal factor `Q` (`rows x min(rows, cols)`) of a
   * pivoted QR factorization: see `linalg.qrPivotQ()`.
   * @param decomposition A factorization, as returned by `qrPivot()`.
   * @returns A new matrix with orthonormal columns.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * const dec = A.qrPivot();
   * console.log(Matrix.qrPivotQ(dec).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[-0.2672612419124243, 0.8728715609439696], [-0.5345224838248488, 0.21821789023599258], [-0.8017837257372732, -0.43643578047198484]]
   * ```
   */
  static qrPivotQ(decomposition: QRPivotDecomposition): Matrix {
    return linalg.qrPivotQ(decomposition);
  }

  /**
   * Extracts the upper-trapezoidal factor `R` (`min(rows, cols) x cols`)
   * of a pivoted QR factorization: see `linalg.qrPivotR()`.
   * @param decomposition A factorization, as returned by `qrPivot()`.
   * @returns A new matrix with exact zeros below the diagonal.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * const dec = A.qrPivot();
   * console.log(Matrix.qrPivotR(dec).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[-7.483314773547883, -5.879747322073337], [0, -0.6546536707079759]]
   * ```
   */
  static qrPivotR(decomposition: QRPivotDecomposition): Matrix {
    return linalg.qrPivotR(decomposition);
  }

  /**
   * Updates a QR decomposition **in place** for the rank-1 modification
   * `A' = A + u * v^T`, without refactoring `A'` from scratch: see
   * `linalg.qrUpdateSelf()` for the algorithm.
   * @param qr An existing `{ Q, R }` factorization, as returned by
   * `qr()` (or a previous `qrUpdateSelf`/`qrUpdate` call) — `Q` must be
   * square (`m x m`) and `R` must be `m x n`. **Both `Q` and `R` are
   * mutated in place**; pass `qr()`'s freshly-created result (or your
   * own copies) if you need to keep the pre-update factorization around.
   * @param u The rank-1 update's left vector. Must have `u.size === qr.Q.rows`.
   * @param v The rank-1 update's right vector. Must have `v.size === qr.R.cols`.
   * @returns `qr` itself, mutated in place, for chaining.
   * @throws {RangeError} If `Q` isn't square, or `Q`/`R`/`u`/`v`'s shapes
   * are inconsistent with each other.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 0], [0, 1], [0, 0]]);
   * const dec = A.qr();
   * Matrix.qrUpdateSelf(dec, Vector.from([0, 0, 1]), Vector.from([1, 0]));
   * console.log(dec.Q.matmul(dec.R).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0.9999999999999998, 0], [0, 1], [0.9999999999999998, 0]]
   * ```
   */
  static qrUpdateSelf(qr: QRDecomposition, u: Vector, v: Vector): QRDecomposition {
    return linalg.qrUpdateSelf(qr, u, v);
  }

  /**
   * Non-mutating counterpart to `qrUpdateSelf`: copies `Q` and `R`
   * first, so the factorization passed in is left untouched.
   * @param qr An existing `{ Q, R }` factorization, as returned by `qr()`.
   * @param u The rank-1 update's left vector. Must have `u.size === qr.Q.rows`.
   * @param v The rank-1 update's right vector. Must have `v.size === qr.R.cols`.
   * @returns A new `{ Q, R }` factorization such that `Q * R` equals
   * (up to floating-point error) `qr.Q.matmul(qr.R)`'s value plus the
   * outer product of `u` and `v`.
   * @throws {RangeError} Same conditions as `qrUpdateSelf`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 0], [0, 1], [0, 0]]);
   * const dec = A.qr();
   * const updated = Matrix.qrUpdate(dec, Vector.from([0, 0, 1]), Vector.from([1, 0]));
   * console.log(updated.Q.matmul(updated.R).toString());
   * console.log(dec.Q.matmul(dec.R).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0.9999999999999998, 0], [0, 1], [0.9999999999999998, 0]]
   * Matrix[[1, 0], [0, 1], [0, 0]]
   * ```
   */
  static qrUpdate(qr: QRDecomposition, u: Vector, v: Vector): QRDecomposition {
    return linalg.qrUpdate(qr, u, v);
  }

  /**
   * Computes the eigenvalues of this (square) matrix: see
   * `linalg.eigenvalues()` for the algorithm (Hessenberg reduction +
   * double-shift QR iteration with deflation). If you also need
   * eigenvectors, use `eig()` instead: it computes both together from
   * the same Schur form, which is cheaper than computing eigenvalues
   * and then eigenvectors separately.
   * @param options.tol Relative tolerance for the deflation test.
   * Defaults to `Number.EPSILON`.
   * @param options.maxIterations Total shifted-QR iteration budget
   * across the whole computation. Defaults to `30 * this.rows`.
   * @returns The `rows` eigenvalues, in top-to-bottom diagonal position
   * within the underlying Schur form — *not* sorted by magnitude, and
   * *not* deflation order.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If the iteration budget is exhausted before every
   * block deflates (e.g. for a pathologically slow-converging matrix).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[2, 1], [1, 2]]);
   * console.log(A.eigenvalues());
   * ```
   *
   * Output:
   * ```text
   * [ { re: 3, im: 0 }, { re: 1, im: 0 } ]
   * ```
   */
  eigenvalues(options?: { maxIterations?: number; tol?: number }): Eigenvalue[] {
    return linalg.eigenvalues(this, options);
  }

  /**
   * Computes both the eigenvalues and eigenvectors of this (square)
   * matrix: see `linalg.eig()` for the algorithm. A real matrix's
   * eigenvector for a genuinely complex eigenvalue is itself genuinely
   * complex, so each pair carries `vectorRe`/`vectorIm` rather than a
   * single real `Vector`; `vectorIm` is all-zero whenever `value.im` is
   * `0`. Normalization follows LAPACK's convention (`dgeev`): each
   * eigenvector is scaled to Euclidean norm 1, with its
   * largest-magnitude component rotated to be real and positive.
   * @param options See `eigenvalues()` — same meaning and defaults.
   * @returns The `rows` eigenpairs, in the same order `eigenvalues()`
   * would return the values alone.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If the iteration budget is exhausted before every
   * block deflates.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[0, -1], [1, 0]]);
   * for (const { value, vectorRe, vectorIm } of A.eig()) {
   *   console.log(value, vectorRe.toString(), vectorIm.toString());
   * }
   * ```
   *
   * Output:
   * ```text
   * { re: 0, im: 1 } Vector(0.7071067811865475, 0) Vector(0, -0.7071067811865475)
   * { re: 0, im: -1 } Vector(0.7071067811865475, 0) Vector(0, 0.7071067811865475)
   * ```
   */
  eig(options?: { maxIterations?: number; tol?: number }): Eigenpair[] {
    return linalg.eig(this, options);
  }

  // -----------------------------------------------------------------
  // Axis-aware reductions, following numpy's convention: `axis`
  // omitted (or `undefined`) reduces over every element to a scalar
  // (delegating to ArrayND's whole-buffer helpers); `axis: 0` reduces
  // down each column (one result per column, an `Vector` of length
  // `cols`); `axis: 1` reduces across each row (one result per row, an
  // `Vector` of length `rows`). The underlying per-slice math (sum,
  // mean, min/max with NaN propagation, variance/std, argmin/argmax)
  // is not reimplemented here — `_reduceAxis` below reuses `ArrayND`'s
  // `_sumArr`/`_meanArr`/`_minArr`/`_maxArr`/`_varianceArr`/`_stdArr`/
  // `_argMinArr`/`_argMaxArr`, the same pure Float64Array reducers that
  // back Vector's and this class's own whole-buffer reductions.
  // -----------------------------------------------------------------

  /**
   * Applies `reduceFn` along the given axis: for `axis: 0`, once per
   * column, over that column's `rows` values; for `axis: 1`, once per
   * row, over that row's `cols` values.
   * @param axis `0` to reduce down columns, `1` to reduce across rows.
   * @param reduceFn Reduces one row's or column's values to a single number.
   * @returns An `Vector` of length `cols` (for `axis: 0`) or `rows` (for `axis: 1`).
   */
  private _reduceAxis(axis: 0 | 1, reduceFn: (values: Float64Array) => number): Vector {
    if (axis === 0) {
      const res = new Vector(this.cols);
      const col = new Float64Array(this.rows);
      for (let j = 0; j < this.cols; j++) {
        for (let i = 0; i < this.rows; i++) col[i] = this.getUnchecked(i, j);
        res.data[j] = reduceFn(col);
      }
      return res;
    } else {
      const res = new Vector(this.rows);
      for (let i = 0; i < this.rows; i++) {
        const row = this.data.subarray(this.flatIndex(i, 0), this.flatIndex(i, 0) + this.cols);
        res.data[i] = reduceFn(row);
      }
      return res;
    }
  }

  /**
   * Sums this matrix's elements to a scalar.
   * @returns The sum, or `0` if the buffer is empty.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.sum());
   * ```
   *
   * Output:
   * ```text
   * 21
   * ```
   */
  sum(axis?: undefined): number;
  /**
   * Sums this matrix's elements along one axis. `0` sums down each
   * column, returning an `Vector` of length `cols`; `1` sums across
   * each row, returning an `Vector` of length `rows`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.sum(0).toString());
   * console.log(A.sum(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(5, 7, 9)
   * Vector(6, 15)
   * ```
   */
  sum(axis: 0 | 1): Vector;
  sum(axis?: 0 | 1): number | Vector {
    if (axis === undefined) return this._sumAll();
    return this._reduceAxis(axis, ArrayND._sumArr);
  }

  /**
   * Computes the arithmetic mean of this matrix's elements to a scalar.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.mean());
   * ```
   *
   * Output:
   * ```text
   * 3.5
   * ```
   */
  mean(axis?: undefined): number;
  /**
   * Computes the arithmetic mean along one axis. `0` averages down each
   * column; `1` averages across each row.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.mean(0).toString());
   * console.log(A.mean(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(2.5, 3.5, 4.5)
   * Vector(2, 5)
   * ```
   */
  mean(axis: 0 | 1): Vector;
  mean(axis?: 0 | 1): number | Vector {
    if (axis === undefined) return this._meanAll();
    return this._reduceAxis(axis, ArrayND._meanArr);
  }

  /**
   * Finds the smallest element of this matrix. NaN wins the comparison
   * so it propagates, matching `Math.min`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.min());
   * ```
   *
   * Output:
   * ```text
   * 1
   * ```
   */
  min(axis?: undefined): number;
  /**
   * Finds the smallest element along one axis. `0` finds the minimum of
   * each column; `1` finds the minimum of each row. NaN propagates, as
   * in the no-axis form.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.min(0).toString());
   * console.log(A.min(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3)
   * Vector(1, 4)
   * ```
   */
  min(axis: 0 | 1): Vector;
  min(axis?: 0 | 1): number | Vector {
    if (axis === undefined) return this._minAll();
    return this._reduceAxis(axis, ArrayND._minArr);
  }

  /**
   * Finds the largest element of this matrix. NaN wins the comparison
   * so it propagates, matching `Math.max`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.max());
   * ```
   *
   * Output:
   * ```text
   * 6
   * ```
   */
  max(axis?: undefined): number;
  /**
   * Finds the largest element along one axis. `0` finds the maximum of
   * each column; `1` finds the maximum of each row. NaN propagates, as
   * in the no-axis form.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.max(0).toString());
   * console.log(A.max(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(4, 5, 6)
   * Vector(3, 6)
   * ```
   */
  max(axis: 0 | 1): Vector;
  max(axis?: 0 | 1): number | Vector {
    if (axis === undefined) return this._maxAll();
    return this._reduceAxis(axis, ArrayND._maxArr);
  }

  /**
   * Computes the variance of this matrix's elements to a scalar: the
   * mean of the squared deviations from the mean.
   * @param ddof Delta degrees of freedom. The divisor used is
   * `n - ddof`. Defaults to `0` (population variance); pass `1` for the
   * unbiased sample variance.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.variance());
   * console.log(A.variance(undefined, 1));
   * ```
   *
   * Output:
   * ```text
   * 2.9166666666666665
   * 3.5
   * ```
   */
  variance(axis?: undefined, ddof?: number): number;
  /**
   * Computes the variance along one axis. `0` computes the variance of
   * each column; `1` computes the variance of each row.
   * @param ddof Delta degrees of freedom, as in the no-axis form, applied
   * per row/column.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.variance(0).toString());
   * console.log(A.variance(1, 1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(2.25, 2.25, 2.25)
   * Vector(1, 1)
   * ```
   */
  variance(axis: 0 | 1, ddof?: number): Vector;
  variance(axis?: 0 | 1, ddof: number = 0): number | Vector {
    if (axis === undefined) return this._varianceAll(ddof);
    return this._reduceAxis(axis, (a) => ArrayND._varianceArr(a, ddof));
  }

  /**
   * Computes the standard deviation of this matrix's elements to a scalar.
   * @param ddof Delta degrees of freedom, forwarded to `variance()`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.std());
   * console.log(A.std(undefined, 1));
   * ```
   *
   * Output:
   * ```text
   * 1.707825127659933
   * 1.8708286933869707
   * ```
   */
  std(axis?: undefined, ddof?: number): number;
  /**
   * Computes the standard deviation along one axis. `0` computes it for
   * each column; `1` for each row.
   * @param ddof Delta degrees of freedom, forwarded to `variance()`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.std(0).toString());
   * console.log(A.std(1, 1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1.5, 1.5, 1.5)
   * Vector(1, 1)
   * ```
   */
  std(axis: 0 | 1, ddof?: number): Vector;
  std(axis?: 0 | 1, ddof: number = 0): number | Vector {
    if (axis === undefined) return this._stdAll(ddof);
    return this._reduceAxis(axis, (a) => ArrayND._stdArr(a, ddof));
  }

  /**
   * Computes the cumulative sum of this matrix's elements, flattened in
   * row-major order first (row 0 followed by row 1, etc.): a single
   * running total over that sequence, as an `Vector` of length `rows * cols`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.cumsum().toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 3, 6, 10, 15, 21)
   * ```
   */
  cumsum(axis?: undefined): Vector;
  /**
   * Computes the cumulative sum along one axis, keeping this matrix's
   * shape. `0` accumulates down each column independently, restarting
   * the running total at the top of each column. `1` accumulates across
   * each row independently, restarting at the start of each row.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * console.log(A.cumsum(0).toString());
   * console.log(A.cumsum(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2, 3], [5, 7, 9]]
   * Matrix[[1, 3, 6], [4, 9, 15]]
   * ```
   */
  cumsum(axis: 0 | 1): Matrix;
  cumsum(axis?: 0 | 1): Vector | Matrix {
    if (axis === undefined) {
      const res = new Vector(this.size);
      let running = 0;
      for (let i = 0; i < this.data.length; i++) {
        running += this.data[i];
        res.data[i] = running;
      }
      return res;
    }
    const res = this._create(new Float64Array(this.data.length));
    if (axis === 0) {
      for (let j = 0; j < this.cols; j++) {
        let running = 0;
        for (let i = 0; i < this.rows; i++) {
          running += this.getUnchecked(i, j);
          res.setUnchecked(i, j, running);
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        const offset = this.flatIndex(i, 0);
        let running = 0;
        for (let j = 0; j < this.cols; j++) {
          running += this.data[offset + j];
          res.data[offset + j] = running;
        }
      }
    }
    return res;
  }

  /**
   * Finds the index of this matrix's smallest element, as a single flat
   * index into the matrix flattened in row-major order. If multiple
   * elements tie for the minimum, the first flat index is returned. NaN
   * elements take priority, matching `min()`'s NaN-propagation semantics.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[3, 1, 2], [4, 5, 0]]);
   * console.log(A.argmin());
   * ```
   *
   * Output:
   * ```text
   * 5
   * ```
   */
  argmin(axis?: undefined): number;
  /**
   * Finds the index of the smallest element along one axis. `0` returns,
   * for each column, the *row* index (`0` to `rows - 1`) of that
   * column's minimum, as an `Vector` of length `cols`. `1` returns, for
   * each row, the *column* index (`0` to `cols - 1`) of that row's
   * minimum, as an `Vector` of length `rows`. Ties and NaN priority
   * follow the no-axis form, applied per row/column.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[3, 1, 2], [4, 5, 0]]);
   * console.log(A.argmin(0).toString());
   * console.log(A.argmin(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0, 0, 1)
   * Vector(1, 2)
   * ```
   */
  argmin(axis: 0 | 1): Vector;
  argmin(axis?: 0 | 1): number | Vector {
    if (axis === undefined) return ArrayND._argMinArr(this.data);
    return this._reduceAxis(axis, ArrayND._argMinArr);
  }

  /**
   * Finds the index of this matrix's largest element, as a single flat
   * index into the matrix flattened in row-major order. If multiple
   * elements tie for the maximum, the first flat index is returned. NaN
   * elements take priority, matching `max()`'s NaN-propagation semantics.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[3, 1, 2], [4, 5, 0]]);
   * console.log(A.argmax());
   * ```
   *
   * Output:
   * ```text
   * 4
   * ```
   */
  argmax(axis?: undefined): number;
  /**
   * Finds the index of the largest element along one axis. `0` returns,
   * for each column, the *row* index (`0` to `rows - 1`) of that
   * column's maximum, as an `Vector` of length `cols`. `1` returns, for
   * each row, the *column* index (`0` to `cols - 1`) of that row's
   * maximum, as an `Vector` of length `rows`. Ties and NaN priority
   * follow the no-axis form, applied per row/column.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[3, 1, 2], [4, 5, 0]]);
   * console.log(A.argmax(0).toString());
   * console.log(A.argmax(1).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 1, 0)
   * Vector(0, 1)
   * ```
   */
  argmax(axis: 0 | 1): Vector;
  argmax(axis?: 0 | 1): number | Vector {
    if (axis === undefined) return ArrayND._argMaxArr(this.data);
    return this._reduceAxis(axis, ArrayND._argMaxArr);
  }

  // -----------------------------------------------------------------
  // Axis-aware reshaping: unlike the reductions above, these don't
  // collapse a dimension — `sort` permutes each row/column in place
  // (same shape in, same shape out) and `slice` extracts a sub-matrix.
  // Grouped here by their shared per-axis structure, not by behavior.
  // -----------------------------------------------------------------

  /**
   * Returns a sorted copy of this matrix, leaving the original unchanged.
   * Unlike the reductions above (`sum`, `argmin`, etc.), where omitting
   * `axis` flattens the whole matrix, `sort` defaults to `axis: 1`
   * (sorting each row independently) since a flattened *ordering* isn't
   * a matrix anymore, so there's no sensible no-axis default here.
   * @param axis `0` sorts each column independently, top to bottom;
   * `1` (the default) sorts each row independently, left to right.
   * @param compareFn Optional comparator, as in `Array.prototype.sort`.
   * Defaults to ascending numeric order (unlike `Array.prototype.sort`'s
   * default, which sorts lexicographically).
   * @returns A new matrix, the same shape as this one, with each row or
   * column sorted independently.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[9, 1, 5], [3, 7, 2]]);
   * console.log(A.sort().toString());
   * console.log(A.sort(0).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 5, 9], [2, 3, 7]]
   * Matrix[[3, 1, 2], [9, 7, 5]]
   * ```
   */
  sort(axis: 0 | 1 = 1, compareFn?: (a: number, b: number) => number): Matrix {
    const res = this.copy();
    if (axis === 1) {
      for (let i = 0; i < res.rows; i++) {
        const start = res.flatIndex(i, 0);
        res.data.subarray(start, start + res.cols).sort(compareFn);
      }
    } else {
      const col = new Float64Array(res.rows);
      for (let j = 0; j < res.cols; j++) {
        for (let i = 0; i < res.rows; i++) col[i] = res.getUnchecked(i, j);
        col.sort(compareFn);
        for (let i = 0; i < res.rows; i++) res.setUnchecked(i, j, col[i]);
      }
    }
    return res;
  }

  /**
   * Resolves a `(start, end)` pair against an axis of the given
   * `length`, with the same semantics as `Array.prototype.slice`:
   * omitted means "from the start" / "to the end", negative indices
   * count back from the end, and everything is clamped to
   * `[0, length]`. Shared by `slice()` for its row and column axes,
   * independently.
   * @param start Start index, inclusive. Defaults to `0`.
   * @param end End index, exclusive. Defaults to `length`.
   * @param length The length of the axis being sliced.
   * @returns The resolved, clamped `[start, end)` range (`end >= start`, always).
   */
  private static _resolveRange(
    start: number | undefined,
    end: number | undefined,
    length: number
  ): { start: number; end: number } {
    const resolve = (idx: number | undefined, def: number): number => {
      if (idx === undefined) return def;
      return idx < 0 ? Math.max(length + idx, 0) : Math.min(idx, length);
    };
    const s = resolve(start, 0);
    const e = Math.max(resolve(end, length), s);
    return { start: s, end: e };
  }

  /**
   * Extracts a sub-matrix, with the same start/end/negative-index
   * semantics as `Array.prototype.slice`, applied independently to rows
   * and columns.
   * @param rowStart Row start index, inclusive. Defaults to `0`.
   * Negative values count back from the last row.
   * @param rowEnd Row end index, exclusive. Defaults to `rows`. Negative
   * values count back from the last row.
   * @param colStart Column start index, inclusive. Defaults to `0`.
   * Negative values count back from the last column.
   * @param colEnd Column end index, exclusive. Defaults to `cols`.
   * Negative values count back from the last column.
   * @returns A new, independent matrix holding the selected rows and columns.
   * @throws {RangeError} If the resolved row or column range is empty
   * (Matrix cannot represent a matrix with 0 rows or 0 columns).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
   * console.log(A.slice(0, 2, 1).toString());
   * console.log(A.slice(-1).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[2, 3], [5, 6]]
   * Matrix[[7, 8, 9]]
   * ```
   */
  slice(rowStart?: number, rowEnd?: number, colStart?: number, colEnd?: number): Matrix {
    const { start: rs, end: re } = Matrix._resolveRange(rowStart, rowEnd, this.rows);
    const { start: cs, end: ce } = Matrix._resolveRange(colStart, colEnd, this.cols);
    const newRows = re - rs;
    const newCols = ce - cs;
    if (newRows === 0 || newCols === 0) {
      throw new RangeError(
        `Matrix.slice: resolved range is ${newRows}x${newCols}, but Matrix cannot represent a matrix with 0 rows or 0 columns`
      );
    }
    const res = new Matrix(newRows, newCols);
    for (let i = 0; i < newRows; i++) {
      const srcOffset = this.flatIndex(rs + i, cs);
      res.data.set(this.data.subarray(srcOffset, srcOffset + newCols), res.flatIndex(i, 0));
    }
    return res;
  }

  /**
   * Repeats every element `repeats` times, over this matrix flattened in
   * row-major order, like `numpy.repeat(a, repeats)` with no axis:
   * `[[1, 2], [3, 4]].repeat(2)` is `Vector(1, 1, 2, 2, 3, 3, 4, 4)`.
   * Only a scalar count is supported.
   * @param repeats Must be a non-negative integer; `0` yields an empty vector.
   * @returns A new vector of length `rows * cols * repeats`.
   * @throws {RangeError} If `repeats` is not a non-negative integer.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.repeat(2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 1, 2, 2, 3, 3, 4, 4)
   * ```
   */
  repeat(repeats: number, axis?: undefined): Vector;
  /**
   * Repeats each row (`axis: 0`) or each column (`axis: 1`) `repeats`
   * times in place, like `numpy.repeat(a, repeats, axis)`: repeating
   * `[[1, 2], [3, 4]]` twice along axis 0 gives rows `[1,2],[1,2],[3,4],[3,4]`;
   * along axis 1 it gives `[1,1,2,2],[3,3,4,4]`. Contrast with `tile()`.
   * @param repeats Must be a non-negative integer.
   * @param axis `0` to repeat rows, `1` to repeat columns.
   * @returns A new, independent matrix, with `rows * repeats` rows (axis 0)
   * or `cols * repeats` columns (axis 1).
   * @throws {RangeError} If `repeats` is not a non-negative integer, `axis`
   * is not `0`/`1`, or `repeats` is `0` (Matrix cannot represent a matrix
   * with 0 rows or 0 columns).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.repeat(2, 0).toString());
   * console.log(A.repeat(2, 1).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [1, 2], [3, 4], [3, 4]]
   * Matrix[[1, 1, 2, 2], [3, 3, 4, 4]]
   * ```
   */
  repeat(repeats: number, axis: 0 | 1): Matrix;
  repeat(repeats: number, axis?: 0 | 1): Vector | Matrix {
    ArrayND._checkCount(repeats, 'Matrix.repeat');
    if (axis === undefined) {
      const res = new Vector(this.size * repeats);
      ArrayND._repeatElements(this.data, repeats, res.data, 0);
      return res;
    }
    if (axis !== 0 && axis !== 1) {
      throw new RangeError(`Matrix.repeat: axis must be 0, 1 or undefined, got ${axis}`);
    }
    if (repeats === 0) {
      throw new RangeError(
        `Matrix.repeat: repeats=0 along axis ${axis} would produce an empty matrix, ` +
          `but Matrix cannot represent a matrix with 0 rows or 0 columns`
      );
    }
    if (axis === 0) {
      const res = new Matrix(this.rows * repeats, this.cols);
      for (let i = 0; i < this.rows; i++) {
        const srcOffset = this.flatIndex(i, 0);
        const src = this.data.subarray(srcOffset, srcOffset + this.cols);
        for (let k = 0; k < repeats; k++) res.data.set(src, res.flatIndex(i * repeats + k, 0));
      }
      return res;
    }
    const res = new Matrix(this.rows, this.cols * repeats);
    for (let i = 0; i < this.rows; i++) {
      const srcOffset = this.flatIndex(i, 0);
      ArrayND._repeatElements(
        this.data.subarray(srcOffset, srcOffset + this.cols),
        repeats,
        res.data,
        res.flatIndex(i, 0)
      );
    }
    return res;
  }

  /**
   * Tiles this matrix, like `numpy.tile(a, reps)` for a 2-D `a`. A scalar
   * `reps` tiles along the columns only (numpy promotes it to `(1, reps)`);
   * a `[rowReps, colReps]` pair tiles along both axes. Contrast with
   * `repeat()`, which repeats each row/column in place.
   * @param reps Repetition counts; both must be `>= 1` here.
   * @returns A new, independent `rows * rowReps x cols * colReps` matrix.
   * @throws {RangeError} If a count is not a non-negative integer, or is
   * `0` (Matrix cannot represent a matrix with 0 rows or 0 columns).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.tile(2).toString());
   * console.log(A.tile([2, 1]).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2, 1, 2], [3, 4, 3, 4]]
   * Matrix[[1, 2], [3, 4], [1, 2], [3, 4]]
   * ```
   */
  tile(reps: number | [number, number]): Matrix {
    const [rowReps, colReps] = typeof reps === 'number' ? [1, reps] : reps;
    ArrayND._checkCount(rowReps, 'Matrix.tile', 'row reps');
    ArrayND._checkCount(colReps, 'Matrix.tile', 'column reps');
    if (rowReps === 0 || colReps === 0) {
      throw new RangeError(
        `Matrix.tile: reps (${rowReps}, ${colReps}) would produce an empty matrix, ` +
          `but Matrix cannot represent a matrix with 0 rows or 0 columns`
      );
    }
    const newCols = this.cols * colReps;
    const res = new Matrix(this.rows * rowReps, newCols);
    // Build the first block of `rows` rows (each source row tiled along
    // the columns), then replicate that block down the remaining rows.
    for (let i = 0; i < this.rows; i++) {
      const srcOffset = this.flatIndex(i, 0);
      ArrayND._tileArray(
        this.data.subarray(srcOffset, srcOffset + this.cols),
        colReps,
        res.data,
        i * newCols
      );
    }
    const blockLen = this.rows * newCols;
    for (let a = 1; a < rowReps; a++) res.data.copyWithin(a * blockLen, 0, blockLen);
    return res;
  }

  /**
   * Returns this matrix's elements as an array of row arrays.
   * @returns An array of `rows` arrays, each with `cols` numbers.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.toArray());
   * ```
   *
   * Output:
   * ```text
   * [ [ 1, 2 ], [ 3, 4 ] ]
   * ```
   */
  toArray(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i++) out.push(Array.from(this.row(i).data));
    return out;
  }

  /**
   * Returns a human-readable string representation of this matrix, one row per line.
   * @returns e.g. `"Matrix[[1, 2], [3, 4]]"`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [3, 4]]
   * ```
   */
  toString(): string {
    const rows = this.toArray().map((r) => `[${r.join(', ')}]`);
    return `Matrix[${rows.join(', ')}]`;
  }

  /**
   * Makes Matrix iterable over its rows, e.g. `for (const r of someMatrix)`.
   * Each yielded value is an Vector.
   */
  *[Symbol.iterator](): Generator<Vector, void, unknown> {
    for (let i = 0; i < this.rows; i++) yield this.row(i);
  }

  // -----------------------------------------------------------------
  // In-place (mutating) operations that stay matrix-specific.
  // -----------------------------------------------------------------

  /**
   * Transposes a square matrix in place.
   * @returns `this`, for chaining.
   * @throws {RangeError} If this matrix is not square.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * A.transposeSelf();
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 3], [2, 4]]
   * ```
   */
  transposeSelf(): this {
    if (this.rows !== this.cols)
      throw new RangeError(
        `Matrix transposeSelf requires a square matrix, got ${this.rows}x${this.cols}`
      );
    for (let i = 0; i < this.rows; i++) {
      for (let j = i + 1; j < this.cols; j++) {
        const tmp = this.getUnchecked(i, j);
        this.setUnchecked(i, j, this.getUnchecked(j, i));
        this.setUnchecked(j, i, tmp);
      }
    }
    return this;
  }

  /**
   * Swaps two rows in place, with no allocation. Useful when
   * implementing pivoting algorithms.
   * @param i First row index (0-based).
   * @param j Second row index (0-based).
   * @returns `this`, for chaining.
   * @throws {RangeError} If `i` or `j` is out of bounds.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * A.swapRows(0, 2);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[5, 6], [3, 4], [1, 2]]
   * ```
   */
  swapRows(i: number, j: number): this {
    if (i < 0 || i >= this.rows)
      throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
    if (j < 0 || j >= this.rows)
      throw new RangeError(`Matrix row ${j} out of bounds for ${this.rows} rows`);
    if (i === j) return this;
    const oi = this.flatIndex(i, 0);
    const oj = this.flatIndex(j, 0);
    for (let k = 0; k < this.cols; k++) {
      const tmp = this.data[oi + k];
      this.data[oi + k] = this.data[oj + k];
      this.data[oj + k] = tmp;
    }
    return this;
  }

  /**
   * Scales row `i` in place by a scalar: `row[i] *= s`.
   * @param i Row index (0-based).
   * @param s The scale factor.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `i` is out of bounds.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * A.scaleRow(1, 10);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [30, 40]]
   * ```
   */
  scaleRow(i: number, s: number): this {
    if (i < 0 || i >= this.rows)
      throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
    const offset = this.flatIndex(i, 0);
    for (let k = 0; k < this.cols; k++) this.data[offset + k] *= s;
    return this;
  }

  /**
   * Scales column `j` in place by a scalar: `col[j] *= s`.
   * @param j Column index (0-based).
   * @param s The scale factor.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `j` is out of bounds.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * A.scaleCol(0, -1);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[-1, 2], [-3, 4]]
   * ```
   */
  scaleCol(j: number, s: number): this {
    if (j < 0 || j >= this.cols)
      throw new RangeError(`Matrix column ${j} out of bounds for ${this.cols} columns`);
    for (let i = 0; i < this.rows; i++) this.data[this.flatIndex(i, j)] *= s;
    return this;
  }

  /**
   * Adds a scaled row to another row in place, in a single pass:
   * `row[i] += row[j] * s`. Useful when implementing Gaussian elimination.
   * @param i Row index to modify (0-based).
   * @param j Row index to read from and scale (0-based).
   * @param s The scale factor applied to row `j`.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `i` or `j` is out of bounds.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * A.addScaledRow(1, 0, -3);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [0, -2]]
   * ```
   */
  addScaledRow(i: number, j: number, s: number): this {
    if (i < 0 || i >= this.rows)
      throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
    if (j < 0 || j >= this.rows)
      throw new RangeError(`Matrix row ${j} out of bounds for ${this.rows} rows`);
    const offsetI = this.flatIndex(i, 0);
    const offsetJ = this.flatIndex(j, 0);
    for (let k = 0; k < this.cols; k++) this.data[offsetI + k] += this.data[offsetJ + k] * s;
    return this;
  }

  /**
   * Creates a `rows x cols` zero matrix.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @returns A new zero matrix.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * console.log(Matrix.zero(2, 3).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 0, 0], [0, 0, 0]]
   * ```
   */
  static zero(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  /**
   * Creates an `n x n` identity matrix.
   * @param n The matrix dimension.
   * @returns A new identity matrix.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * console.log(Matrix.identity(3).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
   * ```
   */
  static identity(n: number): Matrix {
    const res = new Matrix(n, n);
    for (let i = 0; i < n; i++) res.set(i, i, 1);
    return res;
  }

  /**
   * Creates a square diagonal matrix from a vector, with the vector's
   * components on the main diagonal and zeros elsewhere. The general
   * version of what `identity(n)` hand-rolls with a loop —
   * `Matrix.diag(Vector.ones(n))` is equivalent to `Matrix.identity(n)`.
   * Inverse of the instance method `diag()`.
   * @param v The diagonal entries.
   * @returns A new `v.size x v.size` matrix.
   * @throws {RangeError} If `v` is empty (`size === 0`).
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * console.log(Matrix.diag(Vector.from([1, 2, 3])).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 0, 0], [0, 2, 0], [0, 0, 3]]
   * ```
   */
  static diag(v: Vector): Matrix {
    if (v.size === 0) {
      throw new RangeError(
        'Matrix.diag: cannot construct a matrix from an empty vector (need at least one element)'
      );
    }
    const res = new Matrix(v.size, v.size);
    for (let i = 0; i < v.size; i++) res.setUnchecked(i, i, v.data[i]);
    return res;
  }

  /**
   * Creates an Matrix from an array of row arrays.
   * @param rows Source data; each inner array must have the same length.
   * @returns A new matrix with shape `rows.length x rows[0].length`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * console.log(Matrix.from([[1, 2, 3], [4, 5, 6]]).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2, 3], [4, 5, 6]]
   * ```
   */
  static from(rows: number[][]): Matrix {
    if (rows.length === 0) {
      throw new RangeError(
        'Matrix.from: cannot construct a matrix from an empty array (need at least one row)'
      );
    }
    const nRows = rows.length;
    const nCols = rows[0].length;
    if (nCols === 0) {
      throw new RangeError(
        'Matrix.from: cannot construct a matrix with empty rows (need at least one column)'
      );
    }
    const res = new Matrix(nRows, nCols);
    for (let i = 0; i < nRows; i++) res.setRow(i, rows[i]);
    return res;
  }

  /**
   * Horizontally concatenates matrices: glues them side by side, in
   * order, growing the column count while keeping the row count fixed.
   * The exact inverse of slicing out a column range with `slice()` (the
   * same "reassemble/grow" relationship `dot` has with `outer`).
   * @param matrices The matrices to concatenate, left to right. Must be
   * non-empty, and every matrix must have the same `rows`.
   * @returns A new matrix with `matrices[0].rows` rows and the sum of
   * every input's `cols`.
   * @throws {RangeError} If `matrices` is empty, or if any matrix's
   * `rows` doesn't match the first matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1], [2]]);
   * const B = Matrix.from([[3, 4], [5, 6]]);
   * console.log(Matrix.hstack([A, B]).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 3, 4], [2, 5, 6]]
   * ```
   */
  static hstack(matrices: Matrix[]): Matrix {
    if (matrices.length === 0) {
      throw new RangeError('Matrix.hstack: need at least one matrix to concatenate');
    }
    const rows = matrices[0].rows;
    let totalCols = 0;
    for (const m of matrices) {
      if (m.rows !== rows) {
        throw new RangeError(`Matrix.hstack: row count mismatch: ${rows} vs ${m.rows}`);
      }
      totalCols += m.cols;
    }
    const res = new Matrix(rows, totalCols);
    let colOffset = 0;
    for (const m of matrices) {
      for (let i = 0; i < rows; i++) {
        const srcOffset = m.flatIndex(i, 0);
        res.data.set(m.data.subarray(srcOffset, srcOffset + m.cols), res.flatIndex(i, colOffset));
      }
      colOffset += m.cols;
    }
    return res;
  }

  /**
   * Vertically concatenates matrices: stacks them top to bottom, in
   * order, growing the row count while keeping the column count fixed.
   * The exact inverse of slicing out a row range with `slice()` (the
   * same "reassemble/grow" relationship `dot` has with `outer`).
   * @param matrices The matrices to concatenate, top to bottom. Must be
   * non-empty, and every matrix must have the same `cols`.
   * @returns A new matrix with the sum of every input's `rows` and
   * `matrices[0].cols` columns.
   * @throws {RangeError} If `matrices` is empty, or if any matrix's
   * `cols` doesn't match the first matrix's.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2]]);
   * const B = Matrix.from([[3, 4], [5, 6]]);
   * console.log(Matrix.vstack([A, B]).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [3, 4], [5, 6]]
   * ```
   */
  static vstack(matrices: Matrix[]): Matrix {
    if (matrices.length === 0) {
      throw new RangeError('Matrix.vstack: need at least one matrix to concatenate');
    }
    const cols = matrices[0].cols;
    let totalRows = 0;
    for (const m of matrices) {
      if (m.cols !== cols) {
        throw new RangeError(`Matrix.vstack: column count mismatch: ${cols} vs ${m.cols}`);
      }
      totalRows += m.rows;
    }
    const res = new Matrix(totalRows, cols);
    let rowOffset = 0;
    for (const m of matrices) {
      res.data.set(m.data, res.flatIndex(rowOffset, 0));
      rowOffset += m.rows;
    }
    return res;
  }
}
