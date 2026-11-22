import { ArrayND } from './ArrayND.js';
import { Vector, type VectorLike } from './Vector.js';
import * as linalg from './Matrix.linalg.js';
import * as svdAlgs from './Matrix.svd.js';
import { asArrayLike, typeName } from './checks.js';

// The heavier linear-algebra algorithms (factorizations, linear-system
// solves, eigendecomposition) live in Matrix.linalg.ts, not here — see
// that file's header comment for why. Matrix's own methods below
// (`lu()`, `qr()`, `eig()`, etc.) are thin forwarders to it, and these
// re-exports keep `import { Matrix, type LUDecomposition } from
// './Matrix.js'`-style imports working unchanged. The singular value
// decomposition and the algorithms built on it (`svd()`, `lstsq()`,
// `pinv()`, ...) follow the same pattern with Matrix.svd.ts.
export type {
  Eigenvalue,
  QRDecomposition,
  QRPivotDecomposition,
  Eigenpair,
  LUDecomposition,
  RankOptions
} from './Matrix.linalg.js';
import type {
  Eigenvalue,
  QRDecomposition,
  QRPivotDecomposition,
  Eigenpair,
  LUDecomposition,
  RankOptions
} from './Matrix.linalg.js';
export type { SVDDecomposition, SVDOptions, SVDCutoffOptions, LstsqResult } from './Matrix.svd.js';
import type { SVDDecomposition, SVDOptions, SVDCutoffOptions, LstsqResult } from './Matrix.svd.js';

/**
 * A rows x cols matrix of doubles, stored in a single flat, row-major
 * `Float64Array` exposed as `data`: element `(i, j)` lives at
 * `data[i * cols + j]`, so row `i` is the contiguous slice
 * `data.subarray(i * cols, (i + 1) * cols)`. `rows` and `cols` are read-only.
 * Writing to the elements of `data` is allowed (it is what the algorithms do
 * in hot loops); replacing `data` by a buffer of another length is not.
 *
 * **Mutation convention.** Methods whose name ends in `Self` (`addSelf`,
 * `transposeSelf`, `swapRowsSelf`, ...) mutate the receiver and return
 * `this`; the setters `set`, `setRow`, `setCol` and `fill` also mutate (and
 * return `this`). Every other method returns a new object that does not share
 * storage with the receiver (the one exception is the internal
 * `wrapUnchecked`, which wraps the buffer it is given).
 *
 * **Errors.** Invalid arguments throw `RangeError` (a bad value: an index
 * outside the matrix or not an integer, a shape mismatch, an invalid `axis`
 * or tolerance) or `TypeError` (a bad type: a non-`Matrix` operand, a
 * non-array row). Messages read `Matrix.method: ... .`. Singular or
 * non-convergent numerical problems throw `Error`.
 *
 * All element access (`get`, `set`, `row`, `col`, `setRow`, `setCol`,
 * `swapRowsSelf`, `scaleRowSelf`, `scaleColSelf`, `addScaledRowSelf`) is **0-based**: valid row indices
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
 * Operands of an operation between two objects (these, `isClose`/`allClose`,
 * `dot`/`dist`, `matmul`) and all `out` arguments must be actual `Matrix` /
 * `Vector` instances; a plain array there is a `TypeError`, because converting
 * it on every call would allocate in hot loops and a bare array would be
 * ambiguous as a broadcast operand. Arguments that supply *data* to be read as
 * a vector (right-hand sides, `setRow`, `setCol`, `diag`) accept any
 * {@link VectorLike}.
 * `toArray` is not inherited (its natural shape differs per subclass)
 * and is defined here directly, as an array of row arrays.
 */
export class Matrix extends ArrayND {
  private _rows: number;
  private _cols: number;
  /**
   * The elements, as a flat row-major buffer of length `rows * cols`:
   * element `(i, j)` is `data[i * cols + j]`. Changes to it change the
   * matrix.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2, 3], [4, 5, 6]]);
   * A.data[4] = 50;
   * console.log(A.toString());
   * console.log(A.data.length);
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2, 3], [4, 50, 6]]
   * 6
   * ```
   */
  public data: Float64Array;

  /**
   * Creates a `rows x cols` matrix, zero-filled or initialised from
   * row-major `input` (copied).
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const Z = new Matrix(2, 3);
   * const A = new Matrix(2, 2, [1, 2, 3, 4]);
   * console.log(Z.toString());
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[0, 0, 0], [0, 0, 0]]
   * Matrix[[1, 2], [3, 4]]
   * ```
   * @param rows Number of rows (must be a positive integer).
   * @param cols Number of columns (must be a positive integer).
   * @param input Optional initial data (a `Vector` or any array-like of
   *   numbers) in row-major order (i.e. row 0 followed by row 1, etc.), of length `rows * cols`.
   *   If omitted, the matrix is initialized to all zeros.
   * @throws {RangeError} If `rows` or `cols` is not a positive integer, if
   *   `input` is provided and its length is not exactly `rows * cols`, or if
   *   the matrix is too large to allocate.
   * @throws {TypeError} If `input` is neither a `Vector` nor an array-like of numbers.
   */
  constructor(rows: number, cols: number, input?: Vector | ArrayLike<number>);
  /** @internal */
  constructor(rows: number, cols: number, input: null, adopt: Float64Array);
  constructor(
    rows: number,
    cols: number,
    input?: Vector | ArrayLike<number> | null,
    adopt?: Float64Array
  ) {
    super();
    if (adopt !== undefined) {
      // Internal fast path (see `wrapUnchecked`): take the buffer as is.
      this._rows = rows;
      this._cols = cols;
      this.data = adopt;
      return;
    }
    if (!Number.isInteger(rows) || rows < 1) {
      throw new RangeError(`Matrix.constructor: rows must be a positive integer, got ${rows}.`);
    }
    if (!Number.isInteger(cols) || cols < 1) {
      throw new RangeError(`Matrix.constructor: cols must be a positive integer, got ${cols}.`);
    }
    const src = input === undefined ? undefined : asArrayLike(input, 'Matrix.constructor', 'input');
    if (src !== undefined) {
      if (src.length !== rows * cols) {
        throw new RangeError(
          `Matrix.constructor: expected ${rows * cols} values for a ${rows}x${cols} matrix, got ${src.length}.`
        );
      }
    }
    this._rows = rows;
    this._cols = cols;
    try {
      this.data = new Float64Array(rows * cols);
    } catch {
      // The engine's own RangeError ("Invalid typed array length", "Array buffer allocation failed").
      throw new RangeError(
        `Matrix.constructor: a ${rows}x${cols} matrix is too large to allocate.`
      );
    }
    if (src !== undefined) this.data.set(src);
  }

  /**
   * The number of rows in this matrix. Read-only (like `Vector.size`) so
   * it can never desync from `data`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = new Matrix(2, 3);
   * console.log(A.rows);
   * ```
   *
   * Output:
   * ```text
   * 2
   * ```
   */
  get rows(): number {
    return this._rows;
  }

  /**
   * The number of columns in this matrix. Read-only (like `Vector.size`)
   * so it can never desync from `data`.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = new Matrix(2, 3);
   * console.log(A.cols);
   * ```
   *
   * Output:
   * ```text
   * 3
   * ```
   */
  get cols(): number {
    return this._cols;
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
    if (!(other instanceof Matrix)) {
      throw new TypeError(`Matrix.${caller}: expected a Matrix, got ${typeName(other)}.`);
    }
    if (other.rows !== this.rows || other.cols !== this.cols) {
      throw new RangeError(
        `Matrix.${caller}: shape mismatch: ${this.rows}x${this.cols} vs ${other.rows}x${other.cols}.`
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
   * match its buffer's length. It goes through the (internal) constructor
   * overload rather than `Object.create`, so that every matrix has the same
   * hidden class and the hot paths that mix them stay monomorphic. Like
   * `_create`'s `as this` cast, this assumes Matrix is never itself subclassed.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data The buffer to wrap directly. Not copied. Must have length `rows * cols`.
   * @returns A new Matrix wrapping `data`.
   *
   * @internal
   */
  static wrapUnchecked(rows: number, cols: number, data: Float64Array): Matrix {
    return new Matrix(rows, cols, null, data);
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

  protected _className(): string {
    return 'Matrix';
  }

  /**
   * Throws if `(i, j)` is not a valid 0-based index into this matrix.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @throws {RangeError} If `i` or `j` is out of range.
   */
  private _checkBounds(i: number, j: number, caller: string): void {
    if (
      !Number.isInteger(i) ||
      !Number.isInteger(j) ||
      i < 0 ||
      i >= this.rows ||
      j < 0 ||
      j >= this.cols
    ) {
      throw new RangeError(
        `Matrix.${caller}: index (${i}, ${j}) out of bounds for a ${this.rows}x${this.cols} matrix.`
      );
    }
  }

  /**
   * Throws if `i` is not a valid 0-based row index. Rejects NaN and
   * non-integers, which would otherwise address a wrong element of the
   * flat buffer (or none) without any error.
   * @param i Row index (0-based).
   * @param caller Name of the public method invoking this check.
   * @throws {RangeError} If `i` is not an integer in `0..rows-1`.
   */
  private _checkRow(i: number, caller: string): void {
    if (!Number.isInteger(i) || i < 0 || i >= this.rows) {
      throw new RangeError(`Matrix.${caller}: row index ${i} out of bounds for ${this.rows} rows.`);
    }
  }

  /**
   * Throws if `j` is not a valid 0-based column index.
   * @param j Column index (0-based).
   * @param caller Name of the public method invoking this check.
   * @throws {RangeError} If `j` is not an integer in `0..cols-1`.
   */
  private _checkCol(j: number, caller: string): void {
    if (!Number.isInteger(j) || j < 0 || j >= this.cols) {
      throw new RangeError(
        `Matrix.${caller}: column index ${j} out of bounds for ${this.cols} columns.`
      );
    }
  }

  /**
   * Throws if `tol` is not a usable absolute tolerance (negative or NaN).
   * @param tol The tolerance.
   * @param caller Name of the public method invoking this check.
   * @throws {RangeError} If `tol` is not `>= 0`.
   */
  private static _checkTol(tol: number, caller: string): void {
    if (typeof tol !== 'number') {
      throw new TypeError(`Matrix.${caller}: tol must be a number, got ${typeName(tol)}.`);
    }
    if (!(tol >= 0)) throw new RangeError(`Matrix.${caller}: tol must be >= 0, got ${tol}.`);
  }

  /**
   * Throws if `axis` is not `0` or `1`. Called only after the "no axis" case
   * has been handled, so that e.g. `sum(2)` or `sum(null)` can never silently
   * fall through to the row-wise code path.
   * @param axis The axis argument as received.
   * @param caller Name of the public method invoking this check.
   * @param optional Whether `undefined` is also a valid value (it selects the
   * whole-matrix variant), which only changes the wording of the message.
   * @throws {RangeError} If `axis` is neither `0` nor `1`.
   */
  private static _checkAxis(axis: unknown, caller: string, optional: boolean): void {
    if (axis !== 0 && axis !== 1) {
      const got = typeof axis === 'string' ? JSON.stringify(axis) : String(axis);
      throw new RangeError(
        `Matrix.${caller}: axis must be ${optional ? '0, 1 or undefined' : '0 or 1'}, got ${got}.`
      );
    }
  }

  /**
   * Validates the argument of `hstack`/`vstack`: a non-empty array whose
   * elements are all matrices.
   * @param matrices The argument as received.
   * @param caller Name of the public method invoking this check.
   * @throws {TypeError} If `matrices` is not an array, or an element is not a `Matrix`.
   * @throws {RangeError} If `matrices` is empty.
   */
  private static _checkMatrixList(matrices: unknown, caller: string): asserts matrices is Matrix[] {
    if (!Array.isArray(matrices)) {
      throw new TypeError(
        `Matrix.${caller}: expected an array of matrices, got ${typeName(matrices)}.`
      );
    }
    if (matrices.length === 0) {
      throw new RangeError(`Matrix.${caller}: need at least one matrix to concatenate.`);
    }
    for (let k = 0; k < matrices.length; k++) {
      if (!(matrices[k] instanceof Matrix)) {
        throw new TypeError(
          `Matrix.${caller}: element ${k} is not a Matrix, got ${typeName(matrices[k])}.`
        );
      }
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
   * @internal
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
   * @internal
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
   * @internal
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
   * @throws {RangeError} If `i` or `j` is not an integer inside the matrix.
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
    this._checkBounds(i, j, 'get');
    return this.data[this.flatIndex(i, j)];
  }

  /**
   * Sets the element at row `i`, column `j`, mutating this matrix in place.
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @param value The value to store.
   * @returns `this`, for chaining.
   * @throws {RangeError} If `i` or `j` is not an integer inside the matrix.
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
    this._checkBounds(i, j, 'set');
    this.data[this.flatIndex(i, j)] = value;
    return this;
  }

  /**
   * Extracts row `i` as a vector.
   * @param i Row index (0-based).
   * @returns A new vector with `this.cols` components.
   * @throws {RangeError} If `i` is not an integer in `0..rows-1`.
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
    this._checkRow(i, 'row');
    const start = this.flatIndex(i, 0);
    return Vector.wrapUnchecked(this.data.slice(start, start + this.cols));
  }

  /**
   * Extracts column `j` as a vector.
   * @param j Column index (0-based).
   * @returns A new vector with `this.rows` components.
   * @throws {RangeError} If `j` is not an integer in `0..cols-1`.
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
    this._checkCol(j, 'col');
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
   * @throws {TypeError} If `values` is neither a `Vector` nor an array-like of numbers.
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
    this._checkRow(i, 'setRow');
    const src = asArrayLike(values, 'Matrix.setRow', 'values');
    if (src.length !== this.cols) {
      throw new RangeError(
        `Matrix.setRow: expected ${this.cols} values for row ${i}, got ${src.length}.`
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
   * @throws {TypeError} If `values` is neither a `Vector` nor an array-like of numbers.
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
    this._checkCol(j, 'setCol');
    const given = asArrayLike(values, 'Matrix.setCol', 'values');
    if (given.length !== this.rows) {
      throw new RangeError(
        `Matrix.setCol: expected ${this.rows} values for column ${j}, got ${given.length}.`
      );
    }
    // The column is written one element at a time, so a source that is a view of
    // this very buffer (e.g. `A.data.subarray(...)`) must be snapshotted first.
    const src =
      given instanceof Float64Array && given.buffer === this.data.buffer ? given.slice() : given;
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
   * @throws {TypeError} If the operand is not a `Matrix`, a `Vector` or a number.
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
            `change the result shape to ${m}x${k}, which is not supported.`
        );
      }
      throw new RangeError(
        `Matrix.${caller}: operands could not be broadcast together with shapes (${m},${n}) (${k}).`
      );
    }

    const a = this.data;
    // In place, an operand that is a view of this very buffer would be read after
    // being overwritten (row 0 of `A` as `v`, say), so it is snapshotted first.
    const b = inPlace && v.data.buffer === a.buffer ? v.data.slice() : v.data;
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
   * product. Every product is formed, zeros included, so `NaN` and
   * `Infinity` propagate as in IEEE arithmetic (`0 * NaN` is `NaN`).
   * @param m The right-hand matrix. Must have `m.rows === this.cols`.
   * @returns A new `this.rows x m.cols` matrix.
   * @throws {RangeError} If `m.rows !== this.cols`.
   *
   * @throws {TypeError} If `m` is not a `Matrix`.
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
    if (!(m instanceof Matrix)) {
      throw new TypeError(`Matrix.matmul: expected a Matrix, got ${typeName(m)}.`);
    }
    if (this.cols !== m.rows) {
      throw new RangeError(
        `Matrix.matmul: shape mismatch: ${this.rows}x${this.cols} * ${m.rows}x${m.cols}.`
      );
    }
    const res = new Matrix(this.rows, m.cols);
    for (let i = 0; i < this.rows; i++) {
      const resOffset = res.flatIndex(i, 0);
      for (let k = 0; k < this.cols; k++) {
        const a = this.getUnchecked(i, k);
        const mOffset = m.flatIndex(k, 0);
        for (let j = 0; j < m.cols; j++) res.data[resOffset + j] += a * m.data[mOffset + j];
      }
    }
    return res;
  }

  /**
   * Multiplies this matrix by a column vector: `this * v`.
   * @param v The vector (a `Vector` or array-like of numbers). Must have `v.size === this.cols`.
   * @returns A new vector with `this.rows` components.
   * @throws {RangeError} If `v.size !== this.cols`.
   *
   * @throws {TypeError} If `v` is neither a `Vector` nor an array-like of numbers.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * // A plain array works too.
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * console.log(A.mulVec([1, 1]).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(3, 7)
   * ```
   */
  mulVec(v: VectorLike): Vector {
    v = Vector.asVector(v);
    if (v.size !== this.cols) {
      throw new RangeError(
        `Matrix.mulVec: shape mismatch: ${this.rows}x${this.cols} * vector of size ${v.size}.`
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
   * never equal itself). A pair containing `NaN` is never equal; equal
   * infinities are.
   * @param tol Absolute tolerance: a pair `(i, j)`/`(j, i)` counts as
   * equal if `|this[i][j] - this[j][i]| <= tol`. Defaults to `0` (exact).
   * @returns `true` if this matrix is symmetric within `tol`.
   * @throws {RangeError} If `tol` is negative or `NaN`.
   *
   * @throws {TypeError} If `tol` is not a number.
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
    Matrix._checkTol(tol, 'isSymmetric');
    if (!this.isSquare()) return false;
    for (let i = 0; i < this.rows; i++) {
      for (let j = i + 1; j < this.cols; j++) {
        const a = this.getUnchecked(i, j);
        const b = this.getUnchecked(j, i);
        // `a !== b` first so that equal infinities count as equal; NaN never does.
        if (a !== b && !(Math.abs(a - b) <= tol)) return false;
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
   * @throws {RangeError} If `tol` is negative or `NaN`.
   *
   * @throws {TypeError} If `tol` is not a number.
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
    Matrix._checkTol(tol, 'isLowerTriangular');
    for (let i = 0; i < this.rows; i++) {
      for (let j = i + 1; j < this.cols; j++) {
        if (!(Math.abs(this.getUnchecked(i, j)) <= tol)) return false;
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
   * @throws {RangeError} If `tol` is negative or `NaN`.
   *
   * @throws {TypeError} If `tol` is not a number.
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
    Matrix._checkTol(tol, 'isUpperTriangular');
    for (let i = 1; i < this.rows; i++) {
      for (let j = 0; j < Math.min(i, this.cols); j++) {
        if (!(Math.abs(this.getUnchecked(i, j)) <= tol)) return false;
      }
    }
    return true;
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
      throw new RangeError(
        `Matrix.trace: expected a square matrix, got ${this.rows}x${this.cols}.`
      );
    let sum = 0;
    for (let i = 0; i < this.rows; i++) sum += this.getUnchecked(i, i);
    return sum;
  }

  /**
   * Computes the matrix infinity norm: the largest absolute row sum,
   * `max_i sum_j |this[i][j]|`. This is the norm induced by the vector
   * infinity norm (`Vector.normInf()`), not the entrywise maximum. Besides
   * being used internally to auto-scale `rank()`'s default tolerance to this
   * matrix's magnitude, it is a general-purpose norm in its own right
   * (unlike `norm()`, which is the Frobenius norm over the whole flat
   * buffer). See `norm1()` for its column-sum counterpart;
   * `this.norm1() === this.transpose().normInf()` always holds.
   * @returns The largest absolute row sum, or `0` for a matrix of all zeros.
   * `NaN` if any element is NaN.
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
      else if (Number.isNaN(rowSum)) return NaN;
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
   * `NaN` if any element is NaN.
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
      else if (Number.isNaN(colSums[j])) return NaN;
    }
    return maxColSum;
  }

  /**
   * Computes the rank of this matrix (the number of linearly independent
   * rows/columns). By default it counts the singular values above
   * `max(atol, rtol * s[0])`, like `numpy.linalg.matrix_rank`, which is
   * reliable for ill-conditioned and badly scaled matrices.
   * `{ method: 'elimination' }` selects the cheaper but less robust
   * estimate based on Gaussian elimination, which was this method's only
   * algorithm before the SVD was available.
   *
   * Does not modify this matrix. The SVD method costs `O(rows * cols *
   * min(rows, cols))` with a larger constant than elimination, which has
   * the same order. A rank-deficient matrix is the normal use case, not an
   * error, and any shape is accepted; a 1x1 matrix has rank `1`, or `0` if
   * its entry is `0`. A `NaN` entry makes the result `NaN` (for both
   * methods); an infinite entry throws `RangeError`.
   * @param options Either an absolute tolerance on the singular values
   * (shorthand for `{ atol: tol, rtol: 0 }`), or `{ method, atol, rtol }`:
   * the method (`'svd'` by default), an absolute threshold (default `0`)
   * and a threshold relative to the largest singular value, or to the
   * infinity norm for elimination (defaults `max(rows, cols) * eps`, resp.
   * `10 * max(rows, cols) * eps`).
   * @returns The rank, between `0` and `min(rows, cols)`.
   * @throws {RangeError} If `atol` or `rtol` is negative or `NaN`, if
   * `method` is unknown, or if this matrix contains an infinite entry.
   * @throws {Error} If the SVD iteration fails to converge.
   * @throws {TypeError} If `options` is neither a number nor an object, or
   *   `atol` or `rtol` is not a number.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * // Unit diagonal, -1 above it: nonsingular, but its smallest singular
   * // value (about 3e-15) is below the rank tolerance. Gaussian elimination
   * // does not notice; the SVD does.
   * const T = new Matrix(50, 50).map((_, i, j) => (i === j ? 1 : j > i ? -1 : 0));
   * console.log(T.rank());
   * console.log(T.rank({ method: 'elimination' }));
   * ```
   *
   * Output:
   * ```text
   * 49
   * 50
   * ```
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * console.log(Matrix.from([[NaN, 1], [1, 1]]).rank());
   * console.log(Matrix.from([[1, 2, 3], [2, 4, 6]]).rank());
   * ```
   *
   * Output:
   * ```text
   * NaN
   * 1
   * ```
   */
  rank(options?: number | RankOptions): number {
    return linalg.rank(this, options);
  }

  /**
   * Computes the determinant of this matrix, via Gaussian elimination
   * with partial pivoting.
   *
   * Costs `O(n^3)` for `n = rows`; this matrix is not modified. A singular
   * matrix has determinant `0` without error (only an exactly zero pivot
   * counts, as in `lu()`, so a nearly singular matrix gives a tiny nonzero
   * value). The product of the pivots is not rescaled, so it overflows to
   * `±Infinity` or underflows to `0` for extreme magnitudes. A 1x1 matrix
   * returns its entry. A `NaN` entry makes the result `NaN`; an infinite
   * entry throws `RangeError`.
   * @returns The determinant.
   * @throws {RangeError} If this matrix is not square.
   * @throws {RangeError} If this matrix contains an infinite entry.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const singular = Matrix.from([[1, 2], [2, 4]]);
   * console.log(singular.determinant());
   * console.log(Matrix.from([[NaN, 1], [1, 1]]).determinant());
   * try {
   *   Matrix.from([[Infinity, 1], [1, 1]]).determinant();
   * } catch (e) {
   *   console.log((e as Error).message);
   * }
   * ```
   *
   * Output:
   * ```text
   * 0
   * NaN
   * Matrix.determinant: matrix contains an infinite entry at (0, 0), got Infinity.
   * ```
   */
  determinant(): number {
    return linalg.determinant(this);
  }

  /**
   * Computes the inverse of this matrix, via Gauss-Jordan elimination
   * with partial pivoting.
   *
   * Costs `O(n^3)` time and `O(n^2)` extra storage (an `n x 2n` augmented
   * copy); this matrix is not modified. Only an exactly zero pivot makes
   * the matrix singular (`Error`): a nearly singular matrix returns a
   * finite, possibly inaccurate inverse, so check `cond1()` or `rank()` if
   * that matters. To solve a system, `solve()` is faster and more accurate
   * than multiplying by the inverse. A 1x1 matrix returns the reciprocal of
   * its entry; if a pivot is subnormal and its reciprocal overflows, the
   * affected entries become `Infinity` instead of `NaN`. A `NaN` entry
   * makes every entry of the result `NaN`; an infinite entry throws
   * `RangeError`.
   * @returns A new matrix `M` such that `this.matmul(M)` is (up to
   *   floating-point error) the identity matrix.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If this matrix is singular (not invertible).
   * @throws {RangeError} If this matrix contains an infinite entry.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * try {
   *   Matrix.from([[1, 2], [2, 4]]).inverse();
   * } catch (e) {
   *   console.log((e as Error).message);
   * }
   * console.log(Matrix.from([[NaN, 1], [1, 1]]).inverse().toString());
   * try {
   *   Matrix.from([[Infinity, 1], [1, 1]]).inverse();
   * } catch (e) {
   *   console.log((e as Error).message);
   * }
   * ```
   *
   * Output:
   * ```text
   * Matrix.inverse: matrix is singular.
   * Matrix[[NaN, NaN], [NaN, NaN]]
   * Matrix.inverse: matrix contains an infinite entry at (0, 0), got Infinity.
   * ```
   */
  inverse(): Matrix {
    return linalg.inverse(this);
  }

  /**
   * Computes a partial-pivoted LU factorization of this matrix, such
   * that `P * this = L * U` for the row permutation `P` implied by
   * `perm`. There is no tolerance parameter: a pivot counts as missing only
   * if it is exactly `0`.
   *
   * Costs about `(2/3) n^3` flops. This matrix is read, not modified,
   * unless it is `out.U`. A singular matrix throws `Error`; there is no
   * pivot tolerance, so a nearly singular matrix is factored normally (use
   * `rank()` or `cond1()` to detect it). A 1x1 matrix gives `L = [1]` and
   * `U = this`. A `NaN` entry makes `L` and `U` all `NaN`, with `perm` the
   * identity and `sign` `1`, instead of throwing; an infinite entry throws
   * `RangeError`.
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
   * @throws {RangeError} If this matrix contains an infinite entry.
   * @throws {TypeError} If `out` is given but is not an `LUDecomposition`.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const dec = Matrix.from([[NaN, 1], [1, 1]]).lu();
   * console.log(dec.U.toString(), dec.perm, dec.sign);
   * ```
   *
   * Output:
   * ```text
   * Matrix[[NaN, NaN], [NaN, NaN]] [ 0, 1 ] 1
   * ```
   */
  lu(out?: LUDecomposition): LUDecomposition {
    return linalg.lu(this, out);
  }

  /**
   * Solves `A * x = b` for `x`, given the LU factorization of `A` as
   * returned by `lu()`. If you need to solve
   * against the same matrix with several right-hand sides, call `lu()`
   * once and reuse this instead of refactoring each time.
   *
   * Costs `O(n^2)` (forward, then back substitution); neither
   * `decomposition` nor `b` is modified. A zero on the diagonal of `U`
   * throws `Error`. A 1x1 factorization solves `U * x = b` directly.
   * Entries are not scanned: `NaN` propagates through the arithmetic (so a
   * factorization that `lu()` filled with `NaN` gives an all-`NaN`
   * solution) and infinite values follow IEEE rules.
   * @param decomposition The `{ L, U, perm }` factorization of the
   * coefficient matrix, as returned by `lu()`.
   * @param b The right-hand side (a `Vector` or array-like of numbers). Must have `b.size === L.rows`.
   * @param out Optional vector of size `L.rows` that receives the
   * solution, avoiding an allocation. It must not be `b`.
   * @returns The solution vector `x`: `out` if given, otherwise a new vector.
   * @throws {RangeError} If `b` or `out` does not match the size of the
   * factorization, or if `out` is `b`.
   * @throws {Error} If `U` has a zero diagonal entry.
   * @throws {TypeError} If `decomposition` is not an `LUDecomposition`.
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * const dec = A.lu();
   * console.log(Matrix.solveLU(dec, Vector.from([5, 11])).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2)
   * ```
   */
  static solveLU(decomposition: LUDecomposition, b: VectorLike, out?: Vector): Vector {
    return linalg.solveLU(decomposition, Vector.asVector(b), out);
  }

  /**
   * Solves `this * x = b` for `x`, treating this matrix as lower
   * triangular: only entries on and below the diagonal are read, so
   * this can be called directly on the `L` factor from `lu()`.
   *
   * Costs `O(n^2)`. This matrix and `b` are not modified (unless `out` is
   * `b`). A zero diagonal entry, which is only read if `unitDiagonal` is
   * `false`, throws `Error`: the matrix is singular. A 1x1 matrix divides
   * `b` by its entry. Entries are not scanned: `NaN` and infinite values
   * follow IEEE arithmetic.
   * @param b The right-hand side (a `Vector` or array-like of numbers). Must have `b.size === this.rows`.
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
   * @throws {TypeError} If `b` is neither a `Vector` nor an array-like of numbers.
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
  solveLower(b: VectorLike, unitDiagonal = false, out?: Vector): Vector {
    return linalg.solveLower(this, Vector.asVector(b), unitDiagonal, out);
  }

  /**
   * Solves `this * x = b` for `x`, treating this matrix as upper
   * triangular: only entries on and above the diagonal are read, so this
   * can be called directly on the `U` factor from `lu()`.
   *
   * Costs `O(n^2)`. This matrix and `b` are not modified (unless `out` is
   * `b`). A zero diagonal entry throws `Error`: the matrix is singular. A
   * 1x1 matrix divides `b` by its entry. Entries are not scanned: `NaN` and
   * infinite values follow IEEE arithmetic.
   * @param b The right-hand side (a `Vector` or array-like of numbers). Must have `b.size === this.rows`.
   * @param out Optional vector of size `this.rows` that receives the
   *   solution, avoiding an allocation. It may be `b` itself, which is
   *   then overwritten with the solution.
   * @returns The solution vector `x`: `out` if given, otherwise a new vector.
   * @throws {RangeError} If this matrix is not square, `b.size !== this.rows`,
   *   or `out.size !== this.rows`.
   * @throws {Error} If a zero diagonal entry is encountered.
   *
   * @throws {TypeError} If `b` is neither a `Vector` nor an array-like of numbers.
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
  solveUpper(b: VectorLike, out?: Vector): Vector {
    return linalg.solveUpper(this, Vector.asVector(b), out);
  }

  /**
   * Solves the linear system `this * x = b` for `x`, via LU
   * decomposition with partial pivoting (`lu()`) followed by forward
   * and back substitution. If you need to solve
   * against the same matrix with several right-hand sides, call `lu()`
   * once yourself and reuse `Matrix.solveLU` instead of calling this
   * repeatedly.
   *
   * Costs about `(2/3) n^3` flops, dominated by `lu()`; neither this matrix
   * nor `b` is modified. A singular matrix throws `Error` (an exactly zero
   * pivot; a nearly singular matrix returns finite, possibly inaccurate
   * numbers, see `cond1()`). For a rank-deficient or non-square system, use
   * `lstsq()`. A 1x1 system returns `b / this`. A `NaN` entry in this
   * matrix makes every entry of the result `NaN`; an infinite entry throws
   * `RangeError`. `b` is not scanned: a `NaN` in `b` gives `NaN` results.
   * @param b The right-hand side (a `Vector` or array-like of numbers). Must have `b.size === this.rows`.
   * @returns The solution vector `x` such that `this.mulVec(x)` is (up to
   *   floating-point error) equal to `b`.
   * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
   * @throws {Error} If this matrix is singular (no unique solution).
   *
   * @throws {TypeError} If `b` is neither a `Vector` nor an array-like of numbers.
   * @throws {RangeError} If this matrix contains an infinite entry.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * try {
   *   Matrix.from([[1, 2], [2, 4]]).solve([1, 1]);
   * } catch (e) {
   *   console.log((e as Error).message);
   * }
   * console.log(Matrix.from([[NaN, 1], [1, 1]]).solve([1, 1]).toString());
   * // A rank-deficient system has a minimum-norm least-squares solution instead.
   * console.log(Matrix.from([[1, 2], [2, 4]]).lstsq([1, 2]).x.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix.solve: matrix is singular.
   * Vector(NaN, NaN)
   * Vector(0.19999999999999998, 0.39999999999999997)
   * ```
   */
  solve(b: VectorLike): Vector {
    return linalg.solve(this, Vector.asVector(b));
  }

  // -----------------------------------------------------------------
  // Condition number estimation. Implementation lives in
  // Matrix.linalg.ts; see that file for the algorithms.
  // -----------------------------------------------------------------

  /**
   * Computes the exact 1-norm condition number of this matrix, `‖this‖₁ ·
   * ‖this⁻¹‖₁`, via `inverse()`. General-purpose,
   * but O(n³) — dominated by the explicit inverse. If this matrix is
   * triangular, `cond1Upper()`/`cond1Lower()` compute a reliable O(n²)
   * estimate instead, without ever forming the inverse; prefer those
   * when applicable.
   *
   * This matrix is not modified. An exactly singular matrix gives
   * `Infinity` rather than an error (a nearly singular one gives a very
   * large value). A 1x1 matrix has condition number `1` up to rounding, or
   * `Infinity` if it is `0`. A `NaN` entry makes the result `NaN`; an
   * infinite entry throws `RangeError`.
   * @returns The 1-norm condition number, or `Infinity` if this matrix
   *   is (numerically) singular.
   * @throws {RangeError} If this matrix is not square.
   * @throws {RangeError} If this matrix contains an infinite entry.
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
   * as with `solveUpper`), in O(n²) without ever forming the inverse,
   * with the algorithm of Cline–Moler–Stewart–
   * Wilkinson (1979, as used by LINPACK's `dtrco` and described in
   * Dennis & Schnabel's Algorithm A3.3.1). Reliable in practice, though
   * (like any O(n²) estimator) it is a heuristic, not an exact value.
   *
   * This matrix is not modified. A zero diagonal entry gives `Infinity`. A
   * 1x1 matrix gives `1`, or `Infinity` if it is `0`. Entries are not
   * scanned: with a `NaN` or infinite entry the result is `NaN` or
   * meaningless.
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
   * as with `solveLower`), in O(n²) without ever forming the inverse,
   * with the algorithm of Cline–Moler–Stewart–
   * Wilkinson (1979, as used by LINPACK's `dtrco` and described in
   * Dennis & Schnabel's Algorithm A3.3.1). Reliable in practice, though
   * (like any O(n²) estimator) it is a heuristic, not an exact value.
   *
   * This matrix is not modified. A zero diagonal entry gives `Infinity`. A
   * 1x1 matrix gives `1`, or `Infinity` if it is `0`. Entries are not
   * scanned: with a `NaN` or infinite entry the result is `NaN` or
   * meaningless.
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
   * L^T`, with `L` lower triangular and positive on the diagonal. There is no tolerance parameter.
   *
   * Costs about `(1/3) n^3` flops; this matrix is not modified. Only its
   * lower triangle, diagonal included, is read, so the upper triangle can
   * hold anything, including `NaN`. A matrix that is not positive definite,
   * which includes a singular positive semidefinite one, throws `Error`. A
   * 1x1 matrix gives `sqrt` of its entry, or throws `Error` if it is not
   * positive. A `NaN` entry in the lower triangle makes every entry of the
   * result `NaN`; an infinite one throws `RangeError`.
   * @returns The lower-triangular factor `L` such that `this = L * L^T`.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If this matrix is not positive definite (some
   * leading principal minor is not strictly positive).
   * @throws {RangeError} If the lower triangle of this matrix, diagonal
   *   included, contains an infinite entry.
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
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * try {
   *   Matrix.from([[1, 2], [2, 1]]).cholesky();
   * } catch (e) {
   *   console.log((e as Error).message);
   * }
   * // Only the lower triangle is read.
   * console.log(Matrix.from([[4, NaN], [2, 3]]).cholesky().toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix.cholesky: matrix is not positive definite (leading minor of order 2 is not positive).
   * Matrix[[2, 0], [1, 1.4142135623730951]]
   * ```
   */
  cholesky(): Matrix {
    return linalg.cholesky(this);
  }

  /**
   * Solves `this * x = b` for `x`, treating this matrix as the Cholesky
   * factor `L` of a symmetric positive-definite matrix (as returned by
   * `cholesky()`). If you need to solve
   * against the same matrix with several right-hand sides, call
   * `cholesky()` once and reuse this instead of refactoring each time.
   *
   * Costs `O(n^2)`; neither this matrix nor `b` is modified. A zero
   * diagonal entry throws `Error`. A 1x1 matrix `[l]` solves `l * l * x =
   * b`. Entries are not scanned: `NaN` and infinite values follow IEEE
   * arithmetic.
   * @param b The right-hand side (a `Vector` or array-like of numbers). Must have `b.size === this.rows`.
   * @returns The solution vector `x` such that
   * `this.matmul(this.transpose()).mulVec(x)` is (up to floating-point
   * error) equal to `b`.
   * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
   * @throws {Error} If this matrix has a zero diagonal entry.
   *
   * @throws {TypeError} If `b` is neither a `Vector` nor an array-like of numbers.
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
  choleskySolve(b: VectorLike): Vector {
    return linalg.choleskySolve(this, Vector.asVector(b));
  }

  // -----------------------------------------------------------------
  // QR factorization and eigenvalues. Implementations live in
  // Matrix.linalg.ts; see that file for the algorithms.
  // -----------------------------------------------------------------

  /**
   * Computes a Householder QR factorization of this matrix: `this = Q *
   * R`, with `Q` orthogonal and `R` upper triangular (trapezoidal if
   * `this.rows > this.cols`).
   *
   * Costs `O(rows * cols * min(rows, cols))` time to factor, plus
   * `O(rows^2 * cols)` to form the `rows x rows` factor `Q` explicitly
   * (a tall matrix is dominated by this), and `O(rows^2)` storage
   * (`qrPivot()` forms no `Q`); this matrix is
   * not modified. Any shape is accepted and rank deficiency is not an
   * error: `R` then has zero or tiny diagonal entries. A 1x1 matrix gives
   * `Q = [1]` and `R = this`. A `NaN` entry makes every entry of `Q` and
   * `R` `NaN`; an infinite entry throws `RangeError`.
   * @returns The `{ Q, R }` factorization.
   * @throws {RangeError} If this matrix contains an infinite entry.
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
   * has nonincreasing magnitude, which reveals the numerical rank. Use `qrPivotMulQt`, `qrPivotQ` and
   * `qrPivotR` to work with the result.
   *
   * Costs `O(rows * cols * min(rows, cols))` time. This matrix is not
   * modified, unless it is `out.QR`. Any shape is accepted and rank
   * deficiency is not an error: the diagonal of `R` shows it by decreasing
   * to zero or tiny values. A `NaN` entry makes `QR` and `tau` all `NaN`,
   * with `perm` the identity, instead of throwing; an infinite entry throws
   * `RangeError`.
   * @param out Optional storage to reuse, as returned by an earlier call
   * for a matrix of the same shape: overwritten and returned, so repeated
   * factorizations allocate no result storage. This matrix may be `out.QR`
   * (factored in place). If this method throws, the contents of `out` are
   * undefined.
   * @returns The `{ QR, tau, perm }` factorization: `out` if given,
   * otherwise a new object.
   * @throws {RangeError} If `out` does not match the shape of this matrix.
   * @throws {RangeError} If this matrix contains an infinite entry.
   * @throws {TypeError} If `out` is given but is not a
   *   `QRPivotDecomposition`.
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
   * factorization, without forming `Q`.
   *
   * Costs `O(rows * min(rows, cols))`; `decomposition` and `b` are not
   * modified (unless `out` is `b`, which is overwritten). Entries are not
   * scanned: `NaN` propagates under IEEE arithmetic.
   * @param decomposition A factorization, as returned by `qrPivot()`.
   * @param b The vector to transform (a `Vector` or array-like of numbers). Must have `b.size === decomposition.QR.rows`.
   * @param out Optional vector of size `decomposition.QR.rows` that receives
   * the result, avoiding an allocation. It may be `b` itself.
   * @returns `Q^T * b` (length `rows`): `out` if given, otherwise a new
   * vector. For a tall matrix, the norm of the entries beyond the first
   * `min(rows, cols)` is the least-squares residual norm.
   * @throws {RangeError} If `b` or `out` does not match the factorization.
   * @throws {TypeError} If `decomposition` is not a `QRPivotDecomposition`.
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
  static qrPivotMulQt(decomposition: QRPivotDecomposition, b: VectorLike, out?: Vector): Vector {
    return linalg.qrPivotMulQt(decomposition, Vector.asVector(b), out);
  }

  /**
   * Forms the thin orthogonal factor `Q` (`rows x min(rows, cols)`) of a
   * pivoted QR factorization.
   *
   * Costs `O(rows^2 * min(rows, cols))`; `decomposition` is not modified.
   * Entries are not scanned: `NaN` propagates under IEEE arithmetic.
   * @param decomposition A factorization, as returned by `qrPivot()`.
   * @returns A new matrix with orthonormal columns.
   * @throws {TypeError} If `decomposition` is not a `QRPivotDecomposition`.
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
   * of a pivoted QR factorization.
   *
   * Costs `O(min(rows, cols) * cols)`; `decomposition` is not modified.
   * @param decomposition A factorization, as returned by `qrPivot()`.
   * @returns A new matrix with exact zeros below the diagonal.
   * @throws {TypeError} If `decomposition` is not a `QRPivotDecomposition`.
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
   * `A' = A + u * v^T`, without refactoring `A'` from scratch.
   *
   * Costs `O(m * (m + n))` for an `m x n` matrix, against `O(m * n * min(m,
   * n))` to refactor. Entries are not scanned: `NaN` and infinite values
   * follow IEEE arithmetic.
   * @param qr An existing `{ Q, R }` factorization, as returned by
   * `qr()` (or a previous `qrUpdateSelf`/`qrUpdate` call) — `Q` must be
   * square (`m x m`) and `R` must be `m x n`. **Both `Q` and `R` are
   * mutated in place**; pass `qr()`'s freshly-created result (or your
   * own copies) if you need to keep the pre-update factorization around.
   * @param u The rank-1 update's left vector (a `Vector` or array-like of numbers). Must have `u.size === qr.Q.rows`.
   * @param v The rank-1 update's right vector (a `Vector` or array-like of numbers). Must have `v.size === qr.R.cols`.
   * @returns `qr` itself, mutated in place, for chaining.
   * @throws {RangeError} If `Q` isn't square, or `Q`/`R`/`u`/`v`'s shapes
   * are inconsistent with each other.
   * @throws {TypeError} If `qr` is not a `QRDecomposition`.
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
  static qrUpdateSelf(qr: QRDecomposition, u: VectorLike, v: VectorLike): QRDecomposition {
    return linalg.qrUpdateSelf(qr, Vector.asVector(u), Vector.asVector(v));
  }

  /**
   * Non-mutating counterpart to `qrUpdateSelf`: copies `Q` and `R`
   * first, so the factorization passed in is left untouched.
   *
   * Costs `O(m * (m + n))` for an `m x n` matrix, plus the copy of `Q` and
   * `R`. Entries are not scanned: `NaN` and infinite values follow IEEE
   * arithmetic.
   * @param qr An existing `{ Q, R }` factorization, as returned by `qr()`.
   * @param u The rank-1 update's left vector (a `Vector` or array-like of numbers). Must have `u.size === qr.Q.rows`.
   * @param v The rank-1 update's right vector (a `Vector` or array-like of numbers). Must have `v.size === qr.R.cols`.
   * @returns A new `{ Q, R }` factorization such that `Q * R` equals
   * (up to floating-point error) `qr.Q.matmul(qr.R)`'s value plus the
   * outer product of `u` and `v`.
   * @throws {RangeError} Same conditions as `qrUpdateSelf`.
   * @throws {TypeError} If `qr` is not a `QRDecomposition`.
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
  static qrUpdate(qr: QRDecomposition, u: VectorLike, v: VectorLike): QRDecomposition {
    return linalg.qrUpdate(qr, Vector.asVector(u), Vector.asVector(v));
  }

  /**
   * Computes the eigenvalues of this (square) matrix, via Hessenberg reduction +
   * double-shift QR iteration with deflation. If you also need
   * eigenvectors, use `eig()` instead: it computes both together from
   * the same Schur form, which is cheaper than computing eigenvalues
   * and then eigenvectors separately.
   *
   * Costs `O(n^3)`; this matrix is not modified. Singular and defective
   * matrices are fine (a singular matrix has a zero eigenvalue); only a
   * square matrix is accepted. A 1x1 matrix returns its entry. A `NaN`
   * entry makes every eigenvalue `NaN + NaN i`; an infinite entry throws
   * `RangeError`.
   * @param options Optional settings.
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
   * @throws {RangeError} If `tol` is negative or `NaN`, if `maxIterations`
   *   is not a positive integer, or if this matrix contains an infinite
   *   entry.
   * @throws {TypeError} If `options` is not an object, or `tol` or
   *   `maxIterations` is not a number.
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
   * matrix. A real matrix's
   * eigenvector for a genuinely complex eigenvalue is itself genuinely
   * complex, so each pair carries `vectorRe`/`vectorIm` rather than a
   * single real `Vector`; `vectorIm` is all-zero whenever `value.im` is
   * `0`. Normalization follows LAPACK's convention (`dgeev`): each
   * eigenvector is scaled to Euclidean norm 1, with its
   * largest-magnitude component rotated to be real and positive.
   *
   * Costs `O(n^3)`; this matrix is not modified. A defective matrix has no
   * complete set of independent eigenvectors: the returned vectors are then
   * (nearly) parallel. A 1x1 matrix returns its entry with eigenvector
   * `[1]`. A `NaN` entry makes every eigenvalue and eigenvector entry
   * `NaN`; an infinite entry throws `RangeError`.
   * @param options See `eigenvalues()` — same meaning and defaults.
   * @returns The `rows` eigenpairs, in the same order `eigenvalues()`
   * would return the values alone.
   * @throws {RangeError} If this matrix is not square.
   * @throws {Error} If the iteration budget is exhausted before every
   * block deflates.
   * @throws {RangeError} If `tol` is negative or `NaN`, if `maxIterations`
   *   is not a positive integer, or if this matrix contains an infinite
   *   entry.
   * @throws {TypeError} If `options` is not an object, or `tol` or
   *   `maxIterations` is not a number.
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
  // Singular value decomposition and what is built on it: least
  // squares, pseudoinverse, 2-norm condition number and numerical rank.
  // Implementations live in Matrix.svd.ts; see that file for the
  // algorithms.
  // -----------------------------------------------------------------

  /**
   * Computes the singular value decomposition `A = U * diag(s) * V^T` of
   * this matrix (any shape): see `svd()` in `Matrix.svd.ts` for the
   * algorithm (Householder bidiagonalization + implicit-shift QR iteration,
   * as in LAPACK `dgesvd`) and its accuracy. The singular values come out in
   * nonincreasing order; the singular vectors are orthonormal, with
   * arbitrary but deterministic signs. If you only need the singular
   * values, `singularValues()` is roughly 1.5 to 3 times faster.
   *
   * Costs `O(rows * cols * min(rows, cols))`, or `O(max(rows, cols)^2 *
   * min(rows, cols))` with `full: true` on a non-square matrix, where the
   * square singular-vector factor is formed; this matrix is not modified.
   * Any shape is accepted and rank deficiency is not an error: the trailing
   * singular values are zero or tiny and the matching singular vectors are
   * an arbitrary orthonormal completion. A 1x1 matrix `[a]` gives `s =
   * [|a|]`, with the sign in `U` or `V`. A `NaN` entry makes `U`, `s` and
   * `V` all `NaN` (with their usual shapes); an infinite entry throws
   * `RangeError`.
   * @param options `{ full: true }` returns square `U` and `V` instead of
   * the thin `rows x k` and `cols x k` factors, `k = min(rows, cols)`.
   * @returns The `{ U, s, V }` decomposition, `V` not transposed.
   * @throws {RangeError} If this matrix contains an infinite entry.
   * @throws {TypeError} If `options` is not an object.
   * @throws {Error} If the SVD iteration fails to converge.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[3, 0], [4, 5]]);
   * const { U, s, V } = A.svd();
   * console.log(s.toString());
   * console.log(U.matmul(Matrix.diag(s)).matmul(V.transpose()).allClose(A));
   * ```
   *
   * Output:
   * ```text
   * Vector(6.70820393249937, 2.2360679774997894)
   * true
   * ```
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const { s } = Matrix.from([[1, 2], [2, 4]]).svd();
   * console.log(s.get(0).toFixed(6), s.get(1) < 1e-15);
   * console.log(Matrix.from([[1, NaN], [3, 4]]).svd().s.toString());
   * try {
   *   Matrix.from([[1, Infinity], [3, 4]]).svd();
   * } catch (e) {
   *   console.log((e as Error).message);
   * }
   * ```
   *
   * Output:
   * ```text
   * 5.000000 true
   * Vector(NaN, NaN)
   * Matrix.svd: matrix contains an infinite entry at (0, 1), got Infinity.
   * ```
   */
  svd(options?: SVDOptions): SVDDecomposition {
    return svdAlgs.svd(this, options);
  }

  /**
   * Computes the singular values of this matrix (any shape), in
   * nonincreasing order, without forming the singular vectors: see
   * `svd()` for the algorithm. They are exactly the values `svd()` returns,
   * at roughly a third to two thirds of its cost.
   *
   * Costs `O(rows * cols * min(rows, cols))`; this matrix is not modified.
   * Any shape is accepted; a rank-deficient matrix has zero or tiny
   * trailing values. A `NaN` entry makes every value `NaN`; an infinite
   * entry throws `RangeError`.
   * @returns The `min(rows, cols)` singular values.
   * @throws {RangeError} If this matrix contains an infinite entry.
   * @throws {Error} If the SVD iteration fails to converge.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[3, 0], [4, 5]]);
   * console.log(A.singularValues().toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(6.70820393249937, 2.2360679774997894)
   * ```
   */
  singularValues(): Vector {
    return svdAlgs.singularValues(this);
  }

  /**
   * Computes the minimum-norm least-squares solution of `this * x = b`, for
   * any shape and rank: among the `x` minimizing `||this*x - b||_2`, the one
   * of smallest norm, i.e. `pinv() * b`, but computed directly from the SVD
   * without forming the left singular vectors: see `lstsq()` in `Matrix.svd.ts`. Singular
   * values at or below `max(atol, rtol * s[0])` count as zero, which makes
   * the result stable for (numerically) rank-deficient matrices; for a full-rank
   * square matrix this is just `solve()`, only slower and more robust.
   *
   * Costs `O(rows * cols * min(rows, cols))`; neither this matrix nor `b`
   * is modified. A rank-deficient system is the normal use case (the
   * minimum-norm solution is returned and `rank` reports the deficiency);
   * an inconsistent system gets the least-squares solution, with
   * `residualNorm` above zero; a singular 1x1 matrix `[0]` gives `x = [0]`
   * and `rank` `0`. A `NaN` entry in this matrix makes `x`, `s`, `rank` and
   * `residualNorm` all `NaN`, and a `NaN` in `b` makes `x` `NaN`; an
   * infinite entry in this matrix throws `RangeError`. Invalid `atol` or
   * `rtol` is reported even then.
   * @param b The right-hand side, of size `rows`.
   * @param options The singular-value cutoff, `atol` and `rtol`.
   * @returns `{ x, rank, residualNorm, s }`: the solution, the numerical
   * rank, the residual norm `||this*x - b||_2`, and the singular values.
   * @throws {RangeError} If `b.size !== rows`, if this matrix contains an
   * infinite entry, or if `atol` or `rtol` is negative.
   * @throws {Error} If the SVD iteration fails to converge.
   * @throws {TypeError} If `options`, `atol` or `rtol` has the wrong type.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * // Straight-line fit y = c0 + c1 * t through (1, 1), (2, 2), (3, 2).
   * const A = Matrix.from([[1, 1], [1, 2], [1, 3]]);
   * const { x, rank, residualNorm } = A.lstsq([1, 2, 2]);
   * console.log(x.toString());
   * console.log(rank);
   * console.log(residualNorm);
   * ```
   *
   * Output:
   * ```text
   * Vector(0.6666666666666662, 0.5000000000000002)
   * 2
   * 0.4082482904638632
   * ```
   */
  lstsq(b: VectorLike, options?: SVDCutoffOptions): LstsqResult {
    return svdAlgs.lstsq(this, b, options);
  }

  /**
   * Computes the Moore-Penrose pseudoinverse of this matrix (any shape; the
   * result is `cols x rows`) from the SVD: see `pinv()` in `Matrix.svd.ts`.
   * Singular values at or below `max(atol, rtol * s[0])` count as zero. To
   * solve one least-squares problem, `lstsq()` is cheaper than
   * `pinv()` followed by a product.
   *
   * Costs `O(rows * cols * min(rows, cols))`; this matrix is not modified.
   * A singular or rank-deficient matrix is the normal use case: singular
   * values at or below the cutoff are treated as zero, so the pseudoinverse
   * of `[0]` is `[0]`. A `NaN` entry makes every entry of the result `NaN`;
   * an infinite entry throws `RangeError`. Invalid `atol` or `rtol` is
   * reported even then.
   * @param options The singular-value cutoff, `atol` and `rtol`.
   * @returns A new `cols x rows` matrix.
   * @throws {RangeError} If this matrix contains an infinite entry, or if
   * `atol` or `rtol` is negative.
   * @throws {Error} If the SVD iteration fails to converge.
   * @throws {TypeError} If `options`, `atol` or `rtol` has the wrong type.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4], [5, 6]]);
   * console.log(A.pinv().toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[-1.3333333333333337, -0.33333333333333287, 0.6666666666666665], [1.0833333333333337, 0.333333333333333, -0.4166666666666666]]
   * ```
   */
  pinv(options?: SVDCutoffOptions): Matrix {
    return svdAlgs.pinv(this, options);
  }

  /**
   * Computes the 2-norm condition number `s_max / s_min` of this matrix (any
   * shape) from its singular values: see `cond2()` in `Matrix.svd.ts`.
   * `Infinity` if the matrix is singular. Unlike `cond1()` it needs no
   * inverse and also applies to rectangular matrices.
   *
   * Costs `O(rows * cols * min(rows, cols))`; this matrix is not modified.
   * A singular matrix, including `[0]`, gives `Infinity` rather than an
   * error; a nonzero 1x1 matrix gives `1`. A `NaN` entry makes the result
   * `NaN`; an infinite entry throws `RangeError`.
   * @returns The condition number, at least `1`, or `Infinity`.
   * @throws {RangeError} If this matrix contains an infinite entry.
   * @throws {Error} If the SVD iteration fails to converge.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[4, 1], [2, 3]]);
   * console.log(A.cond2());
   * ```
   *
   * Output:
   * ```text
   * 2.6180339887498945
   * ```
   */
  cond2(): number {
    return svdAlgs.cond2(this);
  }

  // -----------------------------------------------------------------
  // Axis-aware reductions, following numpy's convention: `axis`
  // omitted (or `undefined`) reduces over every element to a scalar
  // (delegating to ArrayND's whole-buffer helpers); `axis: 0` reduces
  // down each column (one result per column, a `Vector` of length
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
   * @returns A `Vector` of length `cols` (for `axis: 0`) or `rows` (for `axis: 1`).
   */
  private _reduceAxis(
    axis: 0 | 1,
    reduceFn: (values: Float64Array) => number,
    caller: string
  ): Vector {
    Matrix._checkAxis(axis, caller, true);
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
   * @param axis Omit to reduce over all elements.
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
   * column, returning a `Vector` of length `cols`; `1` sums across
   * each row, returning a `Vector` of length `rows`.
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with the sums.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, ArrayND._sumArr, 'sum');
  }

  /**
   * Computes the arithmetic mean of this matrix's elements to a scalar.
   * @param axis Omit to reduce over all elements.
   * @returns The arithmetic mean of all elements.
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
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with the means.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, ArrayND._meanArr, 'mean');
  }

  /**
   * Finds the smallest element of this matrix. NaN wins the comparison
   * so it propagates, matching `Math.min`.
   * @param axis Omit to reduce over all elements.
   * @returns The smallest element.
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
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with the minima.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, ArrayND._minArr, 'min');
  }

  /**
   * Finds the largest element of this matrix. NaN wins the comparison
   * so it propagates, matching `Math.max`.
   * @param axis Omit to reduce over all elements.
   * @returns The largest element.
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
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with the maxima.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, ArrayND._maxArr, 'max');
  }

  /**
   * Computes the variance of this matrix's elements to a scalar: the
   * mean of the squared deviations from the mean.
   * @param ddof Delta degrees of freedom. The divisor used is
   * `n - ddof`. Defaults to `0` (population variance); pass `1` for the
   * unbiased sample variance.
   * @param axis Omit to reduce over all elements.
   * @returns The variance over all elements.
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
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with the variances.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, (a) => ArrayND._varianceArr(a, ddof), 'variance');
  }

  /**
   * Computes the standard deviation of this matrix's elements to a scalar.
   * @param ddof Delta degrees of freedom, forwarded to `variance()`.
   * @param axis Omit to reduce over all elements.
   * @returns The standard deviation over all elements.
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
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with the standard deviations.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, (a) => ArrayND._stdArr(a, ddof), 'std');
  }

  /**
   * Computes the cumulative sum of this matrix's elements, flattened in
   * row-major order first (row 0 followed by row 1, etc.): a single
   * running total over that sequence, as a `Vector` of length `rows * cols`.
   * @param axis Omit to reduce over all elements.
   * @returns A vector of length `rows * cols` with the running sums over the row-major flattened elements.
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
   * @param axis `0` to accumulate down each column, `1` across each row.
   * @returns A new matrix of the same shape with running sums down each column (axis `0`) or across each row (axis `1`).
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    Matrix._checkAxis(axis, 'cumsum', true);
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
   * @param axis Omit to reduce over all elements.
   * @returns The flat (row-major) index of the smallest element; the first one if several are equal.
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
   * column's minimum, as a `Vector` of length `cols`. `1` returns, for
   * each row, the *column* index (`0` to `cols - 1`) of that row's
   * minimum, as a `Vector` of length `rows`. Ties and NaN priority
   * follow the no-axis form, applied per row/column.
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with, for each column or row, the index within it of its smallest element.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, ArrayND._argMinArr, 'argmin');
  }

  /**
   * Finds the index of this matrix's largest element, as a single flat
   * index into the matrix flattened in row-major order. If multiple
   * elements tie for the maximum, the first flat index is returned. NaN
   * elements take priority, matching `max()`'s NaN-propagation semantics.
   * @param axis Omit to reduce over all elements.
   * @returns The flat (row-major) index of the largest element; the first one if several are equal.
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
   * column's maximum, as a `Vector` of length `cols`. `1` returns, for
   * each row, the *column* index (`0` to `cols - 1`) of that row's
   * maximum, as a `Vector` of length `rows`. Ties and NaN priority
   * follow the no-axis form, applied per row/column.
   * @param axis `0` to reduce down each column, `1` to reduce across each row.
   * @returns A `Vector` of length `cols` (axis `0`) or `rows` (axis `1`) with, for each column or row, the index within it of its largest element.
   * @throws {RangeError} If `axis` is not `0`, `1` or `undefined`.
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
    return this._reduceAxis(axis, ArrayND._argMaxArr, 'argmax');
  }

  // -----------------------------------------------------------------
  // Axis-aware reshaping: unlike the reductions above, these don't
  // collapse a dimension — `sort` returns a copy with each row/column
  // sorted (same shape in, same shape out), `slice` extracts a
  // sub-matrix and `repeat`/`tile` enlarge the matrix. All return new
  // matrices; none mutates the receiver. Grouped here by their shared
  // per-axis structure, not by behavior.
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
   * @throws {RangeError} If `axis` is not `0` or `1`.
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
    Matrix._checkAxis(axis, 'sort', false);
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
      const k = Math.trunc(idx) || 0; // truncate toward zero; NaN (and -0) become 0
      return k < 0 ? Math.max(length + k, 0) : Math.min(k, length);
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
   * Negative values count back from the last row. Like all four bounds,
   * it is truncated toward zero (`1.5` is `1`, `-1.5` is `-1`) and `NaN` counts
   * as `0`.
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
        `Matrix.slice: resolved range is ${newRows}x${newCols}, but Matrix cannot represent a matrix with 0 rows or 0 columns.`
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
   * @param axis Omit to repeat each element of the row-major flattening.
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
      throw new RangeError(`Matrix.repeat: axis must be 0, 1 or undefined, got ${axis}.`);
    }
    if (repeats === 0) {
      throw new RangeError(
        `Matrix.repeat: repeats=0 along axis ${axis} would produce an empty matrix, ` +
          `but Matrix cannot represent a matrix with 0 rows or 0 columns.`
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
   * @param reps A column count, or a `[rowReps, colReps]` pair; each count must be an integer `>= 1`.
   * @returns A new, independent `rows * rowReps x cols * colReps` matrix.
   * @throws {RangeError} If a count is not an integer `>= 1` (Matrix cannot
   * represent a matrix with 0 rows or 0 columns).
   * @throws {TypeError} If `reps` is neither a number nor a `[rowReps, colReps]` pair.
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
    if (typeof reps !== 'number' && !(Array.isArray(reps) && reps.length === 2)) {
      const got = Array.isArray(reps) ? `an array of length ${reps.length}` : typeName(reps);
      throw new TypeError(
        `Matrix.tile: reps must be a number or a [rowReps, colReps] pair, got ${got}.`
      );
    }
    const [rowReps, colReps] = typeof reps === 'number' ? [1, reps] : reps;
    ArrayND._checkCount(rowReps, 'Matrix.tile', 'row reps');
    ArrayND._checkCount(colReps, 'Matrix.tile', 'column reps');
    if (rowReps === 0 || colReps === 0) {
      throw new RangeError(
        `Matrix.tile: reps (${rowReps}, ${colReps}) would produce an empty matrix, ` +
          `but Matrix cannot represent a matrix with 0 rows or 0 columns.`
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
   * Each yielded value is a new, independent `Vector` copy of the row.
   * @returns A generator over the rows, from the first to the last.
   *
   * @example
   * ```ts
   * import { Matrix } from 'numerics-js/linalg';
   *
   * const A = Matrix.from([[1, 2], [3, 4]]);
   * for (const row of A) console.log(row.toString());
   * console.log([...A].length);
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2)
   * Vector(3, 4)
   * 2
   * ```
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
        `Matrix.transposeSelf: expected a square matrix, got ${this.rows}x${this.cols}.`
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
   * A.swapRowsSelf(0, 2);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[5, 6], [3, 4], [1, 2]]
   * ```
   */
  swapRowsSelf(i: number, j: number): this {
    this._checkRow(i, 'swapRowsSelf');
    this._checkRow(j, 'swapRowsSelf');
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
   * A.scaleRowSelf(1, 10);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [30, 40]]
   * ```
   */
  scaleRowSelf(i: number, s: number): this {
    this._checkRow(i, 'scaleRowSelf');
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
   * A.scaleColSelf(0, -1);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[-1, 2], [-3, 4]]
   * ```
   */
  scaleColSelf(j: number, s: number): this {
    this._checkCol(j, 'scaleColSelf');
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
   * A.addScaledRowSelf(1, 0, -3);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [0, -2]]
   * ```
   */
  addScaledRowSelf(i: number, j: number, s: number): this {
    this._checkRow(i, 'addScaledRowSelf');
    this._checkRow(j, 'addScaledRowSelf');
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
   * @param v The diagonal entries (a `Vector` or array-like of numbers).
   * @returns A new `v.size x v.size` matrix.
   * @throws {RangeError} If `v` is empty (`size === 0`).
   *
   * @throws {TypeError} If `v` is neither a `Vector` nor an array-like of numbers.
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
  static diag(v: VectorLike): Matrix {
    v = Vector.asVector(v);
    if (v.size === 0) {
      throw new RangeError(
        'Matrix.diag: cannot construct a matrix from an empty vector (need at least one element).'
      );
    }
    const res = new Matrix(v.size, v.size);
    for (let i = 0; i < v.size; i++) res.setUnchecked(i, i, v.data[i]);
    return res;
  }

  /**
   * Creates a Matrix from an array of rows.
   * @param rows Source data: one entry per row, each an array of numbers, a typed array or a `Vector`; all rows must have the same, non-zero length. The data is copied.
   * @returns A new matrix with shape `rows.length x rows[0].length`.
   * @throws {TypeError} If `rows` is not an array, or a row is neither an array-like of numbers nor a `Vector`.
   * @throws {RangeError} If `rows` is empty, a row is empty, or the rows have different lengths.
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
   *
   * @example
   * ```ts
   * import { Matrix, Vector } from 'numerics-js/linalg';
   *
   * // Rows may be arrays, typed arrays or vectors.
   * const A = Matrix.from([[1, 2], new Float64Array([3, 4]), Vector.from([5, 6])]);
   * console.log(A.toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[1, 2], [3, 4], [5, 6]]
   * ```
   */
  static from(rows: ReadonlyArray<Vector | ArrayLike<number>>): Matrix {
    if (!Array.isArray(rows)) {
      throw new TypeError(`Matrix.from: expected an array of rows, got ${typeName(rows)}.`);
    }
    if (rows.length === 0) {
      throw new RangeError(
        'Matrix.from: cannot construct a matrix from an empty array (need at least one row).'
      );
    }
    const src: ArrayLike<number>[] = [];
    for (let i = 0; i < rows.length; i++) {
      const r: unknown = rows[i] instanceof Vector ? (rows[i] as Vector).data : rows[i];
      if (
        r === null ||
        typeof r !== 'object' ||
        typeof (r as ArrayLike<number>).length !== 'number'
      ) {
        throw new TypeError(
          `Matrix.from: row ${i} is not an array of numbers or a Vector, got ${typeName(r)}.`
        );
      }
      src.push(r as ArrayLike<number>);
    }
    const nRows = rows.length;
    const nCols = src[0].length;
    if (nCols === 0) {
      throw new RangeError(
        'Matrix.from: cannot construct a matrix with empty rows (need at least one column).'
      );
    }
    for (let i = 1; i < nRows; i++) {
      if (src[i].length !== nCols) {
        throw new RangeError(
          `Matrix.from: row ${i} has ${src[i].length} elements, expected ${nCols} (as in row 0).`
        );
      }
    }
    const res = new Matrix(nRows, nCols);
    for (let i = 0; i < nRows; i++) res.data.set(src[i], i * nCols);
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
   * @throws {TypeError} If `matrices` is not an array or contains a non-`Matrix` element.
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
    Matrix._checkMatrixList(matrices, 'hstack');
    const rows = matrices[0].rows;
    let totalCols = 0;
    for (const m of matrices) {
      if (m.rows !== rows) {
        throw new RangeError(`Matrix.hstack: row count mismatch: ${rows} vs ${m.rows}.`);
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
   * @throws {TypeError} If `matrices` is not an array or contains a non-`Matrix` element.
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
    Matrix._checkMatrixList(matrices, 'vstack');
    const cols = matrices[0].cols;
    let totalRows = 0;
    for (const m of matrices) {
      if (m.cols !== cols) {
        throw new RangeError(`Matrix.vstack: column count mismatch: ${cols} vs ${m.cols}.`);
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
