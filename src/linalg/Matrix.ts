import { ArrayND } from './arraynd.js';
import { Vector } from './Vector.js';
import * as linalg from './Matrix.linalg.js';

// The heavier linear-algebra algorithms (factorizations, linear-system
// solves, eigendecomposition) live in Matrix.linalg.ts, not here — see
// that file's header comment for why. Matrix's own methods below
// (`lu()`, `qr()`, `eig()`, etc.) are thin forwarders to it, and these
// re-exports keep `import { Matrix, type LUDecomposition } from
// './Matrix.js'`-style imports working unchanged.
export type { Eigenvalue, QRDecomposition, Eigenpair, LUDecomposition } from './Matrix.linalg.js';
import type { Eigenvalue, QRDecomposition, Eigenpair, LUDecomposition } from './Matrix.linalg.js';

/**
 * A rows x cols matrix backed by a flat, row-major Float64Array.
 *
 * All element access (`get`, `set`, `row`, `col`, `setRow`, `setCol`,
 * `swapRows`, `scaleRow`, `addScaledRow`) is **0-based**: valid row indices
 * are `0..rows-1` and valid column indices are `0..cols-1`, matching JS's
 * usual 0-based indexing convention.
 *
 * Elementwise arithmetic (`add`/`sub`/`mult`/`div` + `Self` variants,
 * `abs`/`pow`/`sqrt`/`clip` + `Self` variants — `mult`/`div` accept either
 * another `Matrix` of the same shape, elementwise, or a scalar), tolerance
 * comparisons (`isClose`/`allClose`), `normSq`/`norm`/`dot`/`dist`
 * (Frobenius), and `copy`/`fill` are inherited from `ArrayND` unchanged;
 * see that class for their docs. There is currently no broadcasting
 * against an `Vector` (row/column vector) — both operands must be the
 * same shape, or a plain scalar. `toArray` is not inherited (its natural
 * shape differs per subclass) and is defined here directly, as an array of row arrays.
 */
export class Matrix extends ArrayND {
    private _rows: number;
    private _cols: number;
    public data: Float64Array;

    /**
     * @param rows Number of rows (must be a positive integer).
     * @param cols Number of columns (must be a positive integer).
     * @param input Optional initial data in
     *   row-major order (i.e. row 0 followed by row 1, etc.), length `rows * cols`.
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
            throw new RangeError(`Matrix.${caller}: shape mismatch: ${this.rows}x${this.cols} vs ${other.rows}x${other.cols}`);
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
            throw new RangeError(`Matrix index (${i}, ${j}) out of bounds for ${this.rows}x${this.cols} matrix`);
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
     */
    row(i: number): Vector {
        if (i < 0 || i >= this.rows) throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
        return Vector.wrapUnchecked(this.data.slice(this.flatIndex(i, 0), this.flatIndex(i, 0) + this.cols));
    }

    /**
     * Extracts column `j` as a vector.
     * @param j Column index (0-based).
     * @returns A new vector with `this.rows` components.
     */
    col(j: number): Vector {
        if (j < 0 || j >= this.cols) throw new RangeError(`Matrix column ${j} out of bounds for ${this.cols} columns`);
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
     */
    setRow(i: number, values: Vector | ArrayLike<number>): this {
        if (i < 0 || i >= this.rows) throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
        const src = values instanceof Vector ? values.data : values;
        if (src.length !== this.cols) {
            throw new RangeError(`Matrix setRow: expected ${this.cols} values for row ${i}, got ${src.length}`);
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
     */
    setCol(j: number, values: Vector | ArrayLike<number>): this {
        if (j < 0 || j >= this.cols) throw new RangeError(`Matrix column ${j} out of bounds for ${this.cols} columns`);
        const src = values instanceof Vector ? values.data : values;
        if (src.length !== this.rows) {
            throw new RangeError(`Matrix setCol: expected ${this.rows} values for column ${j}, got ${src.length}`);
        }
        for (let i = 0; i < this.rows; i++) this.setUnchecked(i, j, src[i]);
        return this;
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
     */
    matmul(m: Matrix): Matrix {
        if (this.cols !== m.rows) {
            throw new RangeError(`Matrix matmul shape mismatch: ${this.rows}x${this.cols} * ${m.rows}x${m.cols}`);
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
     */
    mulVec(v: Vector): Vector {
        if (v.size !== this.cols) {
            throw new RangeError(`Matrix mulVec shape mismatch: ${this.rows}x${this.cols} * vec(${v.size})`);
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
     */
    trace(): number {
        if (this.rows !== this.cols) throw new RangeError(`Matrix trace requires a square matrix, got ${this.rows}x${this.cols}`);
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
     * @returns The largest absolute row sum, or `0` for a matrix of all zeros.
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
     */
    rank(tol?: number): number {
        return linalg.rank(this, tol);
    }

    /**
     * Computes the determinant of this matrix, via Gaussian elimination
     * with partial pivoting: see `linalg.determinant()` for the algorithm.
     * @returns The determinant.
     * @throws {RangeError} If this matrix is not square.
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
     */
    inverse(): Matrix {
        return linalg.inverse(this);
    }

    /**
     * Computes a partial-pivoted LU factorization of this matrix, such
     * that `P * this = L * U` for the row permutation `P` implied by
     * `perm`: see `linalg.lu()` for the algorithm and why it has no
     * tolerance parameter.
     * @returns The `{ L, U, perm, sign }` factorization.
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If this matrix is singular (some column's largest
     * available pivot is exactly `0`).
     */
    lu(): LUDecomposition {
        return linalg.lu(this);
    }

    /**
     * Solves `this * x = b` for `x`, treating this matrix as lower
     * triangular: only entries on and below the diagonal are read, so
     * this can be called directly on the `L` factor from `lu()`.
     * @param b The right-hand side vector. Must have `b.size === this.rows`.
     * @param unitDiagonal If `true`, the diagonal is assumed to be all 1s
     *   (as `lu()`'s `L` always is) and is never read, avoiding a division.
     * @returns The solution vector `x`.
     * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
     * @throws {Error} If `unitDiagonal` is `false` and a zero diagonal entry is encountered.
     */
    solveLower(b: Vector, unitDiagonal = false): Vector {
        return linalg.solveLower(this, b, unitDiagonal);
    }

    /**
     * Solves `this * x = b` for `x`, treating this matrix as upper
     * triangular: only entries on and above the diagonal are read, so this
     * can be called directly on the `U` factor from `lu()`.
     * @param b The right-hand side vector. Must have `b.size === this.rows`.
     * @returns The solution vector `x`.
     * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
     * @throws {Error} If a zero diagonal entry is encountered.
     */
    solveUpper(b: Vector): Vector {
        return linalg.solveUpper(this, b);
    }

    /**
     * Solves the linear system `this * x = b` for `x`, via LU
     * decomposition with partial pivoting (`lu()`) followed by forward
     * and back substitution: see `linalg.solve()`. If you need to solve
     * against the same matrix with several right-hand sides, call `lu()`
     * once yourself and reuse `solveLower`/`solveUpper` directly instead
     * of calling this repeatedly.
     * @param b The right-hand side vector. Must have `b.size === this.rows`.
     * @returns The solution vector `x` such that `this.mulVec(x)` is (up to
     *   floating-point error) equal to `b`.
     * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
     * @throws {Error} If this matrix is singular (no unique solution).
     */
    solve(b: Vector): Vector {
        return linalg.solve(this, b);
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
     */
    qr(): QRDecomposition {
        return linalg.qr(this);
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
     */
    sum(axis?: undefined): number;
    /**
     * Sums this matrix's elements along one axis. `0` sums down each
     * column, returning an `Vector` of length `cols`; `1` sums across
     * each row, returning an `Vector` of length `rows`.
     */
    sum(axis: 0 | 1): Vector;
    sum(axis?: 0 | 1): number | Vector {
        if (axis === undefined) return this._sumAll();
        return this._reduceAxis(axis, ArrayND._sumArr);
    }

    /**
     * Computes the arithmetic mean of this matrix's elements to a scalar.
     */
    mean(axis?: undefined): number;
    /**
     * Computes the arithmetic mean along one axis. `0` averages down each
     * column; `1` averages across each row.
     */
    mean(axis: 0 | 1): Vector;
    mean(axis?: 0 | 1): number | Vector {
        if (axis === undefined) return this._meanAll();
        return this._reduceAxis(axis, ArrayND._meanArr);
    }

    /**
     * Finds the smallest element of this matrix. NaN wins the comparison
     * so it propagates, matching `Math.min`.
     */
    min(axis?: undefined): number;
    /**
     * Finds the smallest element along one axis. `0` finds the minimum of
     * each column; `1` finds the minimum of each row. NaN propagates, as
     * in the no-axis form.
     */
    min(axis: 0 | 1): Vector;
    min(axis?: 0 | 1): number | Vector {
        if (axis === undefined) return this._minAll();
        return this._reduceAxis(axis, ArrayND._minArr);
    }

    /**
     * Finds the largest element of this matrix. NaN wins the comparison
     * so it propagates, matching `Math.max`.
     */
    max(axis?: undefined): number;
    /**
     * Finds the largest element along one axis. `0` finds the maximum of
     * each column; `1` finds the maximum of each row. NaN propagates, as
     * in the no-axis form.
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
     */
    variance(axis?: undefined, ddof?: number): number;
    /**
     * Computes the variance along one axis. `0` computes the variance of
     * each column; `1` computes the variance of each row.
     * @param ddof Delta degrees of freedom, as in the no-axis form, applied
     * per row/column.
     */
    variance(axis: 0 | 1, ddof?: number): Vector;
    variance(axis?: 0 | 1, ddof: number = 0): number | Vector {
        if (axis === undefined) return this._varianceAll(ddof);
        return this._reduceAxis(axis, (a) => ArrayND._varianceArr(a, ddof));
    }

    /**
     * Computes the standard deviation of this matrix's elements to a scalar.
     * @param ddof Delta degrees of freedom, forwarded to `variance()`.
     */
    std(axis?: undefined, ddof?: number): number;
    /**
     * Computes the standard deviation along one axis. `0` computes it for
     * each column; `1` for each row.
     * @param ddof Delta degrees of freedom, forwarded to `variance()`.
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
     */
    cumsum(axis?: undefined): Vector;
    /**
     * Computes the cumulative sum along one axis, keeping this matrix's
     * shape. `0` accumulates down each column independently, restarting
     * the running total at the top of each column. `1` accumulates across
     * each row independently, restarting at the start of each row.
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
     */
    argmin(axis?: undefined): number;
    /**
     * Finds the index of the smallest element along one axis. `0` returns,
     * for each column, the *row* index (`0` to `rows - 1`) of that
     * column's minimum, as an `Vector` of length `cols`. `1` returns, for
     * each row, the *column* index (`0` to `cols - 1`) of that row's
     * minimum, as an `Vector` of length `rows`. Ties and NaN priority
     * follow the no-axis form, applied per row/column.
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
     */
    argmax(axis?: undefined): number;
    /**
     * Finds the index of the largest element along one axis. `0` returns,
     * for each column, the *row* index (`0` to `rows - 1`) of that
     * column's maximum, as an `Vector` of length `cols`. `1` returns, for
     * each row, the *column* index (`0` to `cols - 1`) of that row's
     * maximum, as an `Vector` of length `rows`. Ties and NaN priority
     * follow the no-axis form, applied per row/column.
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
    private static _resolveRange(start: number | undefined, end: number | undefined, length: number): { start: number; end: number } {
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
     */
    slice(rowStart?: number, rowEnd?: number, colStart?: number, colEnd?: number): Matrix {
        const { start: rs, end: re } = Matrix._resolveRange(rowStart, rowEnd, this.rows);
        const { start: cs, end: ce } = Matrix._resolveRange(colStart, colEnd, this.cols);
        const newRows = re - rs;
        const newCols = ce - cs;
        if (newRows === 0 || newCols === 0) {
            throw new RangeError(`Matrix.slice: resolved range is ${newRows}x${newCols}, but Matrix cannot represent a matrix with 0 rows or 0 columns`);
        }
        const res = new Matrix(newRows, newCols);
        for (let i = 0; i < newRows; i++) {
            const srcOffset = this.flatIndex(rs + i, cs);
            res.data.set(this.data.subarray(srcOffset, srcOffset + newCols), res.flatIndex(i, 0));
        }
        return res;
    }

    /**
     * Returns this matrix's elements as an array of row arrays.
     * @returns An array of `rows` arrays, each with `cols` numbers.
     */
    toArray(): number[][] {
        const out: number[][] = [];
        for (let i = 0; i < this.rows; i++) out.push(Array.from(this.row(i).data));
        return out;
    }

    /**
     * Returns a human-readable string representation of this matrix, one row per line.
     * @returns e.g. `"Matrix[[1, 2], [3, 4]]"`.
     */
    toString(): string {
        const rows = this.toArray().map(r => `[${r.join(', ')}]`);
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
     */
    transposeSelf(): this {
        if (this.rows !== this.cols) throw new RangeError(`Matrix transposeSelf requires a square matrix, got ${this.rows}x${this.cols}`);
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
     */
    swapRows(i: number, j: number): this {
        if (i < 0 || i >= this.rows) throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
        if (j < 0 || j >= this.rows) throw new RangeError(`Matrix row ${j} out of bounds for ${this.rows} rows`);
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
     */
    scaleRow(i: number, s: number): this {
        if (i < 0 || i >= this.rows) throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
        const offset = this.flatIndex(i, 0);
        for (let k = 0; k < this.cols; k++) this.data[offset + k] *= s;
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
     */
    addScaledRow(i: number, j: number, s: number): this {
        if (i < 0 || i >= this.rows) throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
        if (j < 0 || j >= this.rows) throw new RangeError(`Matrix row ${j} out of bounds for ${this.rows} rows`);
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
     */
    static zero(rows: number, cols: number): Matrix {
        return new Matrix(rows, cols);
    }

    /**
     * Creates an `n x n` identity matrix.
     * @param n The matrix dimension.
     * @returns A new identity matrix.
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
     */
    static diag(v: Vector): Matrix {
        if (v.size === 0) {
            throw new RangeError('Matrix.diag: cannot construct a matrix from an empty vector (need at least one element)');
        }
        const res = new Matrix(v.size, v.size);
        for (let i = 0; i < v.size; i++) res.setUnchecked(i, i, v.data[i]);
        return res;
    }

    /**
     * Creates an Matrix from an array of row arrays.
     * @param rows Source data; each inner array must have the same length.
     * @returns A new matrix with shape `rows.length x rows[0].length`.
     */
    static from(rows: number[][]): Matrix {
        if (rows.length === 0) {
            throw new RangeError('Matrix.from: cannot construct a matrix from an empty array (need at least one row)');
        }
        const nRows = rows.length;
        const nCols = rows[0].length;
        if (nCols === 0) {
            throw new RangeError('Matrix.from: cannot construct a matrix with empty rows (need at least one column)');
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