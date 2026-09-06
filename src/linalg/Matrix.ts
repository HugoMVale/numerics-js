import { ArrayND } from './arraynd.js';
import { Vector } from './Vector.js';

/**
 * Result of `Matrix.lu()`: a partial-pivoted LU factorization such that
 * `P * A = L * U`, where `P` is the row permutation implied by `perm`.
 */
/**
 * A single eigenvalue, real or complex. Complex eigenvalues of a real
 * matrix always occur in conjugate pairs; each conjugate is reported as
 * its own entry (same `re`, opposite-signed `im`).
 */
export interface Eigenvalue {
    /** Real part. */
    re: number;
    /** Imaginary part; `0` for a real eigenvalue. */
    im: number;
}

/**
 * Result of `Matrix.qr()`: a Householder QR factorization such that
 * `this = Q * R`.
 */
export interface QRDecomposition {
    /** Orthogonal factor, `rows x rows`. */
    Q: Matrix;
    /** Upper-triangular factor, `rows x cols` (trapezoidal if `rows > cols`). */
    R: Matrix;
}

/**
 * A single eigenvalue paired with its eigenvector, as returned by
 * `Matrix.eig()`. A real matrix's eigenvector for a genuinely complex
 * eigenvalue is itself genuinely complex, so it's split into real and
 * imaginary parts rather than using a single real `Vector`; `vectorIm` is
 * all-zero whenever `value.im` is `0`.
 */
export interface Eigenpair {
    value: Eigenvalue;
    /** Real part of the eigenvector. */
    vectorRe: Vector;
    /** Imaginary part of the eigenvector; all-zero for a real eigenvalue. */
    vectorIm: Vector;
}

/**
 * Result of `Matrix.lu()`: a partial-pivoted LU factorization such that
 * `P * A = L * U`, where `P` is the row permutation implied by `perm`.
 */
export interface LUDecomposition {
    /** Unit lower-triangular factor (1s on the diagonal). */
    L: Matrix;
    /** Upper-triangular factor. */
    U: Matrix;
    /**
     * Row permutation applied during pivoting: `perm[i]` is the index, in
     * the original matrix, of the row now in position `i`. Apply it to a
     * vector `b` with `perm.map(p => b.data[p])` before the triangular
     * solves.
     */
    perm: number[];
    /**
     * Parity of the row permutation: `+1` for an even number of row swaps,
     * `-1` for an odd number. `sign * product(diag(U))` equals `det(A)`.
     */
    sign: number;
}

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
     * freshly sliced, independent buffer by the time it gets here). Not
     * part of the public API — despite being a public static method
     * (TypeScript has no package-private), treat the leading underscore as
     * a hard "don't call this from outside the array module." Like
     * `_create`'s `as this` cast, this assumes Matrix is never itself
     * subclassed.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param data The buffer to wrap directly. Not copied. Must have length `rows * cols`.
     * @returns A new Matrix wrapping `data`.
     */
    static _wrapUnchecked(rows: number, cols: number, data: Float64Array): Matrix {
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
        return Matrix._wrapUnchecked(this.rows, this.cols, data) as this;
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
     * @param i Row index (0-based).
     * @param j Column index (0-based).
     * @returns The flat, 0-based index.
     */
    private _idx(i: number, j: number): number {
        return i * this.cols + j;
    }

    /**
     * Unchecked element read: skips `_checkBounds`. Internal-only, for
     * hot loops (`matmul`, `inverse`, `lu`, etc.) where the loop bounds
     * already guarantee `i`/`j` are valid, so the public `get`'s bounds
     * check would be pure overhead. Callers are responsible for correctness.
     * @param i Row index (0-based). Must be valid; not checked.
     * @param j Column index (0-based). Must be valid; not checked.
     * @returns The value at `(i, j)`.
     */
    private _get(i: number, j: number): number {
        return this.data[this._idx(i, j)];
    }

    /**
     * Unchecked element write: skips `_checkBounds`. Internal-only, for
     * hot loops where the loop bounds already guarantee `i`/`j` are valid.
     * Callers are responsible for correctness.
     * @param i Row index (0-based). Must be valid; not checked.
     * @param j Column index (0-based). Must be valid; not checked.
     * @param value The value to store.
     */
    private _set(i: number, j: number, value: number): void {
        this.data[this._idx(i, j)] = value;
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
        return this.data[this._idx(i, j)];
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
        this.data[this._idx(i, j)] = value;
        return this;
    }

    /**
     * Extracts row `i` as a vector.
     * @param i Row index (0-based).
     * @returns A new vector with `this.cols` components.
     */
    row(i: number): Vector {
        if (i < 0 || i >= this.rows) throw new RangeError(`Matrix row ${i} out of bounds for ${this.rows} rows`);
        return Vector._wrapUnchecked(this.data.slice(this._idx(i, 0), this._idx(i, 0) + this.cols));
    }

    /**
     * Extracts column `j` as a vector.
     * @param j Column index (0-based).
     * @returns A new vector with `this.rows` components.
     */
    col(j: number): Vector {
        if (j < 0 || j >= this.cols) throw new RangeError(`Matrix column ${j} out of bounds for ${this.cols} columns`);
        const res = new Vector(this.rows);
        for (let i = 0; i < this.rows; i++) res.data[i] = this._get(i, j);
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
        this.data.set(src, this._idx(i, 0));
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
        for (let i = 0; i < this.rows; i++) this._set(i, j, src[i]);
        return this;
    }

    // -----------------------------------------------------------------
    // Matrix-specific operations: no Vector equivalent, or deliberately
    // not inherited from ArrayND (arity/shape differ too much to share).
    // -----------------------------------------------------------------

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
                const a = this._get(i, k);
                if (a === 0) continue;
                const mOffset = m._idx(k, 0);
                for (let j = 0; j < m.cols; j++) rowBuf[j] += a * m.data[mOffset + j];
            }
            res.data.set(rowBuf, res._idx(i, 0));
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
            const offset = this._idx(i, 0);
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
            const offset = this._idx(i, 0);
            for (let j = 0; j < this.cols; j++) res._set(j, i, this.data[offset + j]);
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
        for (let i = 0; i < this.rows; i++) sum += this._get(i, i);
        return sum;
    }

    /**
     * Finds the best pivot row for column `col`, searching rows `startRow..m.rows-1`.
     * Used internally by `_forwardEliminate`, `inverse`, and `solve` to share
     * the same partial-pivoting (largest-magnitude-entry) selection logic.
     * @param m The matrix to search (may be a working copy or an augmented matrix).
     * @param col Column to search (0-based).
     * @param startRow First row to consider (0-based).
     * @param tol Entries with absolute value at or below this are never chosen as a pivot.
     * @returns The 0-based row index of the best pivot, or `-1` if none exceeds `tol`.
     */
    private static _findPivotRow(m: Matrix, col: number, startRow: number, tol: number): number {
        let pivotRow = -1;
        let pivotVal = tol;
        for (let i = startRow; i < m.rows; i++) {
            const v = Math.abs(m._get(i, col));
            if (v > pivotVal) { pivotVal = v; pivotRow = i; }
        }
        return pivotRow;
    }

    /**
     * Reduces a copy of this matrix to row echelon form via Gaussian
     * elimination with partial pivoting (forward elimination only, no
     * back-substitution or row scaling). Shared building block for `rank()`
     * and `determinant()`, which both need the same elimination but reduce
     * the result differently — and, deliberately, use different `tol`
     * values: `determinant()` always passes `0` (matching LAPACK/numpy's
     * exact-zero-only convention, same as `lu()`), while `rank()` defaults
     * to an auto-scaled nonzero tolerance (see its own docstring), since
     * detecting near-singularity (not just exact singularity) is rank's
     * whole job.
     * @param tol Absolute tolerance below which a candidate pivot is treated as zero.
     * @returns `rank` is the number of pivots found; `sign` is `+1`/`-1` tracking the row-swap
     *   parity; `pivots` are the pivot values in the order they were chosen.
     */
    private _forwardEliminate(tol = 0): { rank: number; sign: number; pivots: number[] } {
        const m = this.copy();
        let rank = 0;
        let sign = 1;
        const pivots: number[] = [];
        for (let col = 0; col < m.cols && rank < m.rows; col++) {
            const pivotRow = Matrix._findPivotRow(m, col, rank, tol);
            if (pivotRow === -1) continue;
            if (pivotRow !== rank) { m.swapRows(rank, pivotRow); sign = -sign; }
            const pivot = m._get(rank, col);
            pivots.push(pivot);
            for (let i = rank + 1; i < m.rows; i++) {
                const factor = m._get(i, col) / pivot;
                if (factor !== 0) m.addScaledRow(i, rank, -factor);
            }
            rank++;
        }
        return { rank, sign, pivots };
    }

    /**
     * The scale factor `c` in `rank()`'s default tolerance,
     * `c * max(rows, cols) * Number.EPSILON * normInf(this)`.
     */
    private static readonly _RANK_TOL_SCALE = 10;

    /**
     * Computes the matrix infinity norm: the largest absolute row sum.
     * Used only to auto-scale `rank()`'s default tolerance to this
     * matrix's magnitude — this is a private implementation detail, not a
     * general-purpose norm method (unlike `ArrayND`'s `norm()`, which is
     * the Frobenius norm and applies to the whole flat buffer regardless
     * of shape).
     */
    private _normInf(): number {
        let maxRowSum = 0;
        for (let i = 0; i < this.rows; i++) {
            const offset = this._idx(i, 0);
            let rowSum = 0;
            for (let j = 0; j < this.cols; j++) rowSum += Math.abs(this.data[offset + j]);
            if (rowSum > maxRowSum) maxRowSum = rowSum;
        }
        return maxRowSum;
    }

    /**
     * Computes the rank of this matrix (the number of linearly independent
     * rows/columns), via Gaussian elimination with partial pivoting.
     *
     * This is inherently an approximation: elimination-based rank is a
     * fundamentally less numerically robust way to estimate rank than the
     * SVD-based approach production libraries like numpy actually use
     * (`numpy.linalg.matrix_rank`), which this library doesn't implement.
     * Treat this as a reasonable estimate for well-scaled matrices, not as
     * a numpy-equivalent result for pathological or ill-conditioned ones.
     * @param tol Absolute tolerance below which a pivot is treated as
     * zero. If omitted, defaults to `c * max(rows, cols) * Number.EPSILON
     * * normInf(this)` with `c = 10` — the same shape of scale-relative
     * formula numpy's SVD-based `matrix_rank` uses (`S.max() * max(M, N) *
     * eps`), substituting this matrix's infinity norm for numpy's largest
     * singular value, since this library doesn't have singular values
     * available without SVD. Pass `0` explicitly for exact-zero-only
     * pivot rejection, matching `lu()`/`determinant()`'s convention.
     * @returns The rank, between `0` and `min(rows, cols)`.
     */
    rank(tol?: number): number {
        const effectiveTol = tol ?? Matrix._RANK_TOL_SCALE * Math.max(this.rows, this.cols) * Number.EPSILON * this._normInf();
        return this._forwardEliminate(effectiveTol).rank;
    }

    /**
     * Computes the determinant of this matrix, via Gaussian elimination with
     * partial pivoting. Like `lu()`, there is no tolerance parameter: a
     * pivot only counts as missing if it's *exactly* `0`, matching the
     * convention LAPACK and numpy use internally (they compute the
     * determinant from the same kind of LU factorization). A near-singular
     * matrix will generally return a small-but-nonzero float here, not
     * `0` — if you need near-singularity detection, use `rank()` instead,
     * which is built for exactly that.
     * @returns The determinant.
     * @throws {RangeError} If this matrix is not square.
     */
    determinant(): number {
        if (this.rows !== this.cols) throw new RangeError(`Matrix determinant requires a square matrix, got ${this.rows}x${this.cols}`);
        const { rank, sign, pivots } = this._forwardEliminate(0);
        if (rank < this.rows) return 0;
        let det = sign;
        for (const p of pivots) det *= p;
        return det;
    }

    /**
     * Computes the inverse of this matrix, via Gauss-Jordan elimination with
     * partial pivoting.
     * @returns A new matrix `M` such that `this.matmul(M)` is (up to
     *   floating-point error) the identity matrix.
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If this matrix is singular (not invertible).
     */
    inverse(): Matrix {
        if (this.rows !== this.cols) throw new RangeError(`Matrix inverse requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        // Augment [this | I] and row-reduce the left half to I; the right
        // half then becomes this^-1.
        const aug = new Matrix(n, 2 * n);
        for (let i = 0; i < n; i++) {
            aug.data.set(this.data.subarray(this._idx(i, 0), this._idx(i, 0) + n), aug._idx(i, 0));
            aug._set(i, n + i, 1);
        }
        for (let col = 0; col < n; col++) {
            const pivotRow = Matrix._findPivotRow(aug, col, col, 0);
            if (pivotRow === -1) throw new Error('Matrix inverse: matrix is singular');
            if (pivotRow !== col) aug.swapRows(col, pivotRow);
            aug.scaleRow(col, 1 / aug._get(col, col));
            for (let i = 0; i < n; i++) {
                if (i === col) continue;
                const factor = aug._get(i, col);
                if (factor !== 0) aug.addScaledRow(i, col, -factor);
            }
        }
        const res = new Matrix(n, n);
        for (let i = 0; i < n; i++) {
            res.data.set(aug.data.subarray(aug._idx(i, n), aug._idx(i, n) + n), res._idx(i, 0));
        }
        return res;
    }

    /**
     * Computes a partial-pivoted LU factorization of this matrix, such that
     * `P * this = L * U` for the row permutation `P` implied by `perm`.
     * Factoring once and reusing the result via `solveLower`/`solveUpper`
     * is much cheaper than calling `solve()` repeatedly against the same
     * matrix with different right-hand sides (e.g. inside a Newton solver
     * that reuses a Jacobian across corrector steps).
     *
     * There is no tolerance parameter: matching the convention used by
     * LAPACK's `dgetrf` (which this is modeled on), the pivot in each
     * column is always the largest-magnitude entry available, regardless
     * of how small it is — only a pivot that is *exactly* `0` counts as
     * missing. A fuzzy "close enough to zero" threshold isn't part of
     * plain LU pivoting in reference implementations; if you need that
     * kind of near-singularity detection, use `rank()`, which does expose
     * a tolerance for exactly this purpose.
     * @returns The `{ L, U, perm, sign }` factorization.
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If this matrix is singular (some column's largest
     * available pivot is exactly `0`).
     */
    lu(): LUDecomposition {
        if (this.rows !== this.cols) throw new RangeError(`Matrix lu requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        const U = this.copy();
        const L = Matrix.identity(n);
        const perm = Array.from({ length: n }, (_, i) => i);
        let sign = 1;

        for (let col = 0; col < n; col++) {
            const pivotRow = Matrix._findPivotRow(U, col, col, 0);
            if (pivotRow === -1) throw new Error('Matrix lu: matrix is singular');
            if (pivotRow !== col) {
                U.swapRows(col, pivotRow);
                // Rows of L to the left of `col` already hold computed
                // multipliers; swap those along with U's rows so that
                // P * this === L * U still holds after the pivot.
                for (let k = 0; k < col; k++) {
                    const tmp = L._get(col, k);
                    L._set(col, k, L._get(pivotRow, k));
                    L._set(pivotRow, k, tmp);
                }
                [perm[col], perm[pivotRow]] = [perm[pivotRow], perm[col]];
                sign = -sign;
            }
            const pivot = U._get(col, col);
            for (let i = col + 1; i < n; i++) {
                const factor = U._get(i, col) / pivot;
                if (factor !== 0) {
                    U.addScaledRow(i, col, -factor);
                    L._set(i, col, factor);
                }
            }
        }
        return { L, U, perm, sign };
    }

    /**
     * Solves `this * x = b` for `x`, treating this matrix as lower
     * triangular: only entries on and below the diagonal are read, so this
     * can be called directly on the `L` factor from `lu()`.
     * @param b The right-hand side vector. Must have `b.size === this.rows`.
     * @param unitDiagonal If `true`, the diagonal is assumed to be all 1s
     *   (as `lu()`'s `L` always is) and is never read, avoiding a division.
     * @returns The solution vector `x`.
     * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
     * @throws {Error} If `unitDiagonal` is `false` and a zero diagonal entry is encountered.
     */
    solveLower(b: Vector, unitDiagonal = false): Vector {
        if (this.rows !== this.cols) throw new RangeError(`Matrix solveLower requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.size !== this.rows) throw new RangeError(`Matrix solveLower shape mismatch: ${this.rows}x${this.cols} vs vec(${b.size})`);
        const n = this.rows;
        const x = new Vector(n);
        for (let i = 0; i < n; i++) {
            const offset = this._idx(i, 0);
            let s = b.data[i];
            for (let j = 0; j < i; j++) s -= this.data[offset + j] * x.data[j];
            if (unitDiagonal) {
                x.data[i] = s;
            } else {
                const d = this.data[offset + i];
                if (d === 0) throw new Error('Matrix solveLower: zero diagonal entry, matrix is singular');
                x.data[i] = s / d;
            }
        }
        return x;
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
        if (this.rows !== this.cols) throw new RangeError(`Matrix solveUpper requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.size !== this.rows) throw new RangeError(`Matrix solveUpper shape mismatch: ${this.rows}x${this.cols} vs vec(${b.size})`);
        const n = this.rows;
        const x = new Vector(n);
        for (let i = n - 1; i >= 0; i--) {
            const offset = this._idx(i, 0);
            let s = b.data[i];
            for (let j = i + 1; j < n; j++) s -= this.data[offset + j] * x.data[j];
            const d = this.data[offset + i];
            if (d === 0) throw new Error('Matrix solveUpper: zero diagonal entry, matrix is singular');
            x.data[i] = s / d;
        }
        return x;
    }

    /**
     * Solves the linear system `this * x = b` for `x`, via LU decomposition
     * with partial pivoting (`lu()`) followed by forward and back
     * substitution. If you need to solve against the same matrix with
     * several right-hand sides, call `lu()` once yourself and reuse
     * `solveLower`/`solveUpper` directly instead of calling this repeatedly.
     * @param b The right-hand side vector. Must have `b.size === this.rows`.
     * @returns The solution vector `x` such that `this.mulVec(x)` is (up to
     *   floating-point error) equal to `b`.
     * @throws {RangeError} If this matrix is not square, or `b.size !== this.rows`.
     * @throws {Error} If this matrix is singular (no unique solution).
     */
    solve(b: Vector): Vector {
        if (this.rows !== this.cols) throw new RangeError(`Matrix solve requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.size !== this.rows) throw new RangeError(`Matrix solve shape mismatch: ${this.rows}x${this.cols} vs vec(${b.size})`);
        const { L, U, perm } = this.lu();
        const pb = new Vector(perm.map(p => b.data[p]));
        const y = L.solveLower(pb, true);
        return U.solveUpper(y);
    }

    // -----------------------------------------------------------------
    // QR factorization and eigenvalues.
    // -----------------------------------------------------------------

    /**
     * Raw Householder QR factorization: `A = Q * R`. Shared implementation
     * used both by the public `qr()` (called on `this` directly) and
     * internally by `eigenvalues()` (called on small working submatrices
     * each shifted-QR iteration). Works for any shape, not just square.
     * @param A The matrix to factor. Not mutated.
     * @returns The `{ Q, R }` factorization, `Q` being `A.rows x A.rows`
     * and `R` being `A.rows x A.cols` with (up to floating-point cleanup)
     * zeros below the diagonal.
     */
    private static _householderQR(A: Matrix): QRDecomposition {
        const m = A.rows;
        const n = A.cols;
        const R = A.copy();
        const Q = Matrix.identity(m);
        const v = new Float64Array(m);
        const kMax = Math.min(m - 1, n);

        for (let k = 0; k < kMax; k++) {
            let normX = 0;
            for (let i = k; i < m; i++) { const x = R._get(i, k); normX += x * x; }
            normX = Math.sqrt(normX);
            if (normX === 0) continue;

            const x0 = R._get(k, k);
            const alpha = x0 >= 0 ? -normX : normX;
            let vnormSq = 0;
            for (let i = k; i < m; i++) {
                const val = i === k ? x0 - alpha : R._get(i, k);
                v[i] = val;
                vnormSq += val * val;
            }
            if (vnormSq === 0) continue;
            const vnorm = Math.sqrt(vnormSq);
            for (let i = k; i < m; i++) v[i] /= vnorm;

            // Apply the reflector I - 2vv^T to R on the left (columns k..n-1).
            for (let j = k; j < n; j++) {
                let dot = 0;
                for (let i = k; i < m; i++) dot += v[i] * R._get(i, j);
                dot *= 2;
                if (dot === 0) continue;
                for (let i = k; i < m; i++) R._set(i, j, R._get(i, j) - dot * v[i]);
            }
            // Accumulate it into Q on the right: Q := Q * (I - 2vv^T).
            for (let i = 0; i < m; i++) {
                let dot = 0;
                for (let c = k; c < m; c++) dot += Q._get(i, c) * v[c];
                dot *= 2;
                if (dot === 0) continue;
                for (let c = k; c < m; c++) Q._set(i, c, Q._get(i, c) - dot * v[c]);
            }
        }

        // Explicit cleanup: force exact zeros below the diagonal, matching
        // the guarantee numpy/LAPACK-based `qr()` wrappers provide (the
        // arithmetic above already drives these to ~0, up to rounding).
        for (let i = 1; i < m; i++) {
            for (let j = 0; j < Math.min(i, n); j++) R._set(i, j, 0);
        }
        return { Q, R };
    }

    /**
     * Computes a Householder QR factorization of this matrix: `this = Q *
     * R`, with `Q` orthogonal and `R` upper triangular (trapezoidal if
     * `this.rows > this.cols`).
     * @returns The `{ Q, R }` factorization.
     */
    qr(): QRDecomposition {
        return Matrix._householderQR(this);
    }

    /**
     * Computes the stable Givens rotation `{ c, s }` (`c*c + s*s === 1`)
     * that zeros the second component when applied to the pair `[a, b]`:
     * `[c*a + s*b, -s*a + c*b] === [r, 0]`. The classic `drotg`-style
     * formula (Golub & Van Loan, *Matrix Computations*, Algorithm 5.1.3):
     * it divides by whichever of `a`/`b` is larger in magnitude rather
     * than ever squaring-then-rooting the larger one directly, so it
     * doesn't overflow for large inputs the way a naive
     * `hypot`-then-divide computation could.
     * @param a First component.
     * @param b Second component — the one the rotation zeros.
     * @returns The rotation coefficients.
     */
    private static _givens(a: number, b: number): { c: number; s: number } {
        if (b === 0) return { c: 1, s: 0 };
        if (a === 0) return { c: 0, s: 1 };
        if (Math.abs(a) > Math.abs(b)) {
            const t = b / a;
            const u = Math.sign(a) * Math.sqrt(1 + t * t);
            const c = 1 / u;
            return { c, s: c * t };
        } else {
            const t = a / b;
            const u = Math.sign(b) * Math.sqrt(1 + t * t);
            const s = 1 / u;
            return { c: s * t, s };
        }
    }

    /**
     * Left-multiplies `M` in place by a Givens rotation acting on rows
     * `i`/`j`, across columns `[colStart, colEnd)`: the pair `(row_i,
     * row_j)` becomes `(c*row_i + s*row_j, -s*row_i + c*row_j)`. Shared by
     * both sweeps of `qrUpdateSelf`, which use it to chase a Hessenberg
     * bulge up and then back down through `R`.
     * @param M The matrix to rotate, mutated in place.
     * @param i First row index.
     * @param j Second row index — the one a `_givens(a, b)` pair computed
     * from `(M[i][col], M[j][col])` would zero.
     * @param c Rotation cosine.
     * @param s Rotation sine.
     * @param colStart First column to touch, inclusive.
     * @param colEnd Last column to touch, exclusive.
     */
    private static _applyGivensRows(M: Matrix, i: number, j: number, c: number, s: number, colStart: number, colEnd: number): void {
        for (let col = colStart; col < colEnd; col++) {
            const a = M._get(i, col);
            const b = M._get(j, col);
            M._set(i, col, c * a + s * b);
            M._set(j, col, -s * a + c * b);
        }
    }

    /**
     * Right-multiplies `M` in place by the same Givens rotation
     * `_applyGivensRows` would apply on the left, but acting on columns
     * `i`/`j` instead of rows: `(col_i, col_j)` becomes `(c*col_i +
     * s*col_j, -s*col_i + c*col_j)`. Used to accumulate each rotation
     * from `qrUpdateSelf`'s two sweeps into `Q`, the same way
     * `_householderQR` accumulates its reflectors into `Q` via
     * right-multiplication.
     * @param M The matrix to rotate, mutated in place.
     * @param i First column index.
     * @param j Second column index.
     * @param c Rotation cosine.
     * @param s Rotation sine.
     */
    private static _applyGivensCols(M: Matrix, i: number, j: number, c: number, s: number): void {
        for (let row = 0; row < M.rows; row++) {
            const a = M._get(row, i);
            const b = M._get(row, j);
            M._set(row, i, c * a + s * b);
            M._set(row, j, -s * a + c * b);
        }
    }

    /**
     * Updates a QR decomposition **in place** for the rank-1 modification
     * `A' = A + u * v^T`, without refactoring `A'` from scratch —
     * TypeScript/JS analogue of `scipy.linalg.qr_update`.
     *
     * Algorithm (Gill, Golub, Murray & Saunders 1974; written up as
     * Algorithm 12.5.1 in Golub & Van Loan's *Matrix Computations*; the
     * same approach underlies LINPACK's `dchud`/`dqrup` and, ultimately,
     * `scipy.linalg.qr_update` — the standard, most-used way to do this,
     * because unlike a fresh `qr()` it doesn't cost an extra factor of
     * `min(m, n)` in the flop count):
     *
     * 1. Since `A = Q * R`, `A' = Q * (R + w * v^T)` where `w = Q^T * u`.
     * 2. A sweep of Givens rotations, applied to adjacent rows from the
     *    bottom up, collapses `w` to a multiple of `e_1` — and, applied to
     *    the same row pairs of `R`, simultaneously turns `R` into upper
     *    Hessenberg form (one nonzero sub-diagonal).
     * 3. The now-scalar leftover from `w` is folded into `R`'s first row
     *    as `+= w[0] * v^T`, which doesn't disturb the Hessenberg shape.
     * 4. A second sweep of Givens rotations, applied top-down, chases the
     *    sub-diagonal bulge off the bottom, restoring upper-triangular form.
     *
     * Both sweeps are accumulated into `Q` via right-multiplication (the
     * same trick `_householderQR` uses for its reflectors), so `Q * R`
     * keeps equaling `A'` throughout. Total cost is `O(m*n)`, versus
     * `O(m*n*min(m,n))` for `A.add(u.outer(v)).qr()` from scratch — worth
     * it whenever updates are applied repeatedly, e.g. recursive least
     * squares, online/Kalman-filter-style estimation, or a quasi-Newton
     * solver refactoring its Jacobian after every step.
     *
     * For a rank-`k` update, call this `k` times in a loop, once per
     * column of `U`/`V` — the same strategy LINPACK's rank-`k` routines
     * use internally, since a rank-`k` update is just `k` rank-1 updates
     * applied in sequence.
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
        const { Q, R } = qr;
        const m = Q.rows;
        const n = R.cols;
        if (Q.rows !== Q.cols) throw new RangeError(`Matrix.qrUpdateSelf: Q must be square, got ${Q.rows}x${Q.cols}`);
        if (R.rows !== m) throw new RangeError(`Matrix.qrUpdateSelf: shape mismatch: Q is ${m}x${m} but R has ${R.rows} rows`);
        if (u.size !== m) throw new RangeError(`Matrix.qrUpdateSelf: u must have size ${m} (Q.rows), got ${u.size}`);
        if (v.size !== n) throw new RangeError(`Matrix.qrUpdateSelf: v must have size ${n} (R.cols), got ${v.size}`);

        // w = Q^T * u
        const w = new Float64Array(m);
        for (let i = 0; i < m; i++) {
            let sum = 0;
            for (let k = 0; k < m; k++) sum += Q._get(k, i) * u.data[k];
            w[i] = sum;
        }

        // Phase 1: sweep bottom-to-top, collapsing w to w[0]*e_1 while
        // simultaneously reducing R to upper Hessenberg form.
        for (let k = m - 1; k >= 1; k--) {
            if (w[k] === 0) continue;
            const { c, s } = Matrix._givens(w[k - 1], w[k]);
            const a = w[k - 1], b = w[k];
            w[k - 1] = c * a + s * b;
            w[k] = -s * a + c * b;
            Matrix._applyGivensRows(R, k - 1, k, c, s, 0, n);
            Matrix._applyGivensCols(Q, k - 1, k, c, s);
        }

        // Fold the (now scalar) update into R's first row.
        const tau = w[0];
        if (tau !== 0) {
            for (let j = 0; j < n; j++) R._set(0, j, R._get(0, j) + tau * v.data[j]);
        }

        // Phase 2: sweep top-to-bottom, eliminating the sub-diagonal
        // bulge introduced above and restoring upper-triangular form.
        const kMax = Math.min(m - 1, n);
        for (let k = 0; k < kMax; k++) {
            const sub = R._get(k + 1, k);
            if (sub === 0) continue;
            const { c, s } = Matrix._givens(R._get(k, k), sub);
            Matrix._applyGivensRows(R, k, k + 1, c, s, k, n);
            Matrix._applyGivensCols(Q, k, k + 1, c, s);
        }

        // Explicit cleanup: force exact zeros below the diagonal, the same
        // guarantee `_householderQR` provides (see its own cleanup pass).
        for (let i = 1; i < m; i++) {
            for (let j = 0; j < Math.min(i, n); j++) R._set(i, j, 0);
        }
        return qr;
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
        return Matrix.qrUpdateSelf({ Q: qr.Q.copy(), R: qr.R.copy() }, u, v);
    }

    /**
     * Applies an orthogonal similarity transform confined to rows/columns
     * `[lo, hi]`, but correctly to the *whole* matrix: `H := Qfull^T * H *
     * Qfull` where `Qfull` is the identity everywhere except the `[lo,
     * hi] x [lo, hi]` block, which is `Qlocal`. Concretely this means
     * left-multiplying rows `[lo, hi]` (across every column) by
     * `Qlocal^T`, then right-multiplying columns `[lo, hi]` (across every
     * row) by `Qlocal` — touching the coupling entries outside the block,
     * not just the block itself. If `Q` is supplied, the same
     * right-multiplication is applied to its columns `[lo, hi]`, so it
     * keeps accumulating the overall change of basis.
     *
     * Shared by every step of `_schurForm()`'s QR iteration and by its
     * final real-2x2-block cleanup pass — both are exactly this operation,
     * just with a different (locally-derived) `Qlocal`.
     */
    private static _applySimilarity(H: Matrix, Q: Matrix | null, lo: number, hi: number, Qlocal: Matrix): void {
        const n = H.rows;
        const k = hi - lo + 1;

        // H[lo:hi+1, :] := Qlocal^T * H[lo:hi+1, :]
        const rowBuf = new Float64Array(k * n);
        for (let i = 0; i < k; i++) for (let j = 0; j < n; j++) rowBuf[i * n + j] = H._get(lo + i, j);
        for (let i = 0; i < k; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let m = 0; m < k; m++) sum += Qlocal._get(m, i) * rowBuf[m * n + j];
                H._set(lo + i, j, sum);
            }
        }

        // H[:, lo:hi+1] := H[:, lo:hi+1] * Qlocal
        const colBuf = new Float64Array(n * k);
        for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) colBuf[i * k + j] = H._get(i, lo + j);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < k; j++) {
                let sum = 0;
                for (let m = 0; m < k; m++) sum += colBuf[i * k + m] * Qlocal._get(m, j);
                H._set(i, lo + j, sum);
            }
        }

        if (Q) {
            const qBuf = new Float64Array(n * k);
            for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) qBuf[i * k + j] = Q._get(i, lo + j);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < k; j++) {
                    let sum = 0;
                    for (let m = 0; m < k; m++) sum += qBuf[i * k + m] * Qlocal._get(m, j);
                    Q._set(i, lo + j, sum);
                }
            }
        }
    }

    /**
     * Reduces a copy of this (square) matrix to upper Hessenberg form via
     * orthogonal similarity transforms (Householder reflectors applied on
     * both sides), preserving eigenvalues. Used internally by
     * `_schurForm()` as a preprocessing step: it collapses each QR
     * iteration from O(n^3) to O(n^2) and gives shifted QR its usual fast
     * convergence behavior. Not exposed publicly since a Hessenberg-form
     * result isn't useful on its own without the rest of the eigenvalue
     * pipeline.
     * @param accumulateQ If true, also builds and returns the orthogonal
     * matrix `Q` such that `H = Q^T * this * Q` (needed by `eig()`, which
     * has to map eigenvectors back out of the Hessenberg/Schur basis; not
     * needed by `eigenvalues()`, so it's skipped there to avoid the extra
     * O(n^3) of accumulation work).
     * @returns `H`, upper Hessenberg (zero below the subdiagonal) and
     * similar to this matrix; and `Q` (or `null` if not requested).
     */
    private _hessenberg(accumulateQ = false): { H: Matrix; Q: Matrix | null } {
        const n = this.rows;
        const H = this.copy();
        const Q = accumulateQ ? Matrix.identity(n) : null;
        if (n < 3) return { H, Q };
        const v = new Float64Array(n);

        for (let k = 0; k < n - 2; k++) {
            let normX = 0;
            for (let i = k + 1; i < n; i++) { const x = H._get(i, k); normX += x * x; }
            normX = Math.sqrt(normX);
            if (normX === 0) continue;

            const x0 = H._get(k + 1, k);
            const alpha = x0 >= 0 ? -normX : normX;
            let vnormSq = 0;
            for (let i = k + 1; i < n; i++) {
                const val = i === k + 1 ? x0 - alpha : H._get(i, k);
                v[i] = val;
                vnormSq += val * val;
            }
            if (vnormSq === 0) continue;
            const vnorm = Math.sqrt(vnormSq);
            for (let i = k + 1; i < n; i++) v[i] /= vnorm;

            // Left multiply: H[k+1:, :] -= 2 v (v^T H[k+1:, :]).
            for (let j = 0; j < n; j++) {
                let dot = 0;
                for (let i = k + 1; i < n; i++) dot += v[i] * H._get(i, j);
                dot *= 2;
                if (dot === 0) continue;
                for (let i = k + 1; i < n; i++) H._set(i, j, H._get(i, j) - dot * v[i]);
            }
            // Right multiply (completes the similarity transform):
            // H[:, k+1:] -= 2 (H[:, k+1:] v) v^T.
            for (let i = 0; i < n; i++) {
                let dot = 0;
                for (let j = k + 1; j < n; j++) dot += H._get(i, j) * v[j];
                dot *= 2;
                if (dot === 0) continue;
                for (let j = k + 1; j < n; j++) H._set(i, j, H._get(i, j) - dot * v[j]);
            }
            // Accumulate: Q := Q * Hk (same reflector, right-applied).
            if (Q) {
                for (let i = 0; i < n; i++) {
                    let dot = 0;
                    for (let j = k + 1; j < n; j++) dot += Q._get(i, j) * v[j];
                    dot *= 2;
                    if (dot === 0) continue;
                    for (let j = k + 1; j < n; j++) Q._set(i, j, Q._get(i, j) - dot * v[j]);
                }
            }
        }

        // Explicit cleanup: force exact zeros below the subdiagonal.
        for (let i = 2; i < n; i++) {
            for (let j = 0; j < i - 1; j++) H._set(i, j, 0);
        }
        return { H, Q };
    }

    /**
     * Solves the characteristic equation of a 2x2 block `[[a, b], [c,
     * d]]` directly, in closed form. Used by `_schurForm()` to finish off
     * a trailing 2x2 block once the active submatrix has shrunk that far —
     * shifted QR alone never fully deflates a block whose eigenvalues are
     * a complex-conjugate pair, so those are extracted this way instead of
     * by further iteration.
     * @returns The two eigenvalues. If the discriminant is negative, a
     * complex-conjugate pair with the positive-imaginary-part root first;
     * otherwise two reals, larger root first.
     */
    private static _solve2x2Eigs(a: number, b: number, c: number, d: number): [Eigenvalue, Eigenvalue] {
        const tr = a + d;
        const det = a * d - b * c;
        const disc = tr * tr - 4 * det;
        if (disc >= 0) {
            const s = Math.sqrt(disc);
            return [
                { re: (tr + s) / 2, im: 0 },
                { re: (tr - s) / 2, im: 0 },
            ];
        }
        const s = Math.sqrt(-disc);
        return [
            { re: tr / 2, im: s / 2 },
            { re: tr / 2, im: -s / 2 },
        ];
    }

    /** Complex multiplication, as a `[re, im]` pair. */
    private static _cmul(aRe: number, aIm: number, bRe: number, bIm: number): [number, number] {
        return [aRe * bRe - aIm * bIm, aRe * bIm + aIm * bRe];
    }

    /**
     * Complex division `a / b`, as a `[re, im]` pair. If `|b|` is smaller
     * than `floor`, `b` is treated as `floor` (real) instead of dividing
     * by (near-)zero — guards the eigenvector back-substitution below
     * against (near-)repeated eigenvalues, the same role LAPACK's dtrevc
     * safe-minimum plays.
     */
    private static _cdiv(aRe: number, aIm: number, bRe: number, bIm: number, floor: number): [number, number] {
        let dRe = bRe, dIm = bIm;
        if (dRe * dRe + dIm * dIm < floor * floor) { dRe = floor; dIm = 0; }
        const dMagSq = dRe * dRe + dIm * dIm;
        return [(aRe * dRe + aIm * dIm) / dMagSq, (aIm * dRe - aRe * dIm) / dMagSq];
    }

    /**
     * Back-substitutes for the rows *above* an already-seeded eigenvector
     * block, mutating `yRe`/`yIm` in place. `T` is only *quasi*-upper
     * triangular (a 2x2 complex-conjugate block has a nonzero subdiagonal
     * entry within itself), so this can't just solve one row at a time
     * for every row above the target block: whenever it reaches an
     * earlier block that is itself a 2x2 (i.e. some *other* eigenvalue of
     * `T` also happens to be complex), that block's two rows are coupled
     * and have to be solved as one 2x2 complex linear system rather than
     * two independent rows. Every other row is an ordinary 1x1 solve.
     * @param blocks The full, `lo`-ascending block list from
     * `_schurForm()`.
     * @param blockIdx Index of the target (already-seeded) block within
     * `blocks`; only earlier entries (smaller index, smaller `lo`) are
     * processed.
     * @param hiTarget The target block's `hi` — the upper bound (inclusive)
     * of every summation below, since `y` is zero past it by construction.
     */
    private static _backSubstitute(
        T: Matrix, blocks: [number, number][], blockIdx: number, hiTarget: number,
        lambdaRe: number, lambdaIm: number, yRe: Float64Array, yIm: Float64Array, floor: number
    ): void {
        for (let bi = blockIdx - 1; bi >= 0; bi--) {
            const [p, q] = blocks[bi];

            if (p === q) {
                let sumRe = 0, sumIm = 0;
                for (let j = p + 1; j <= hiTarget; j++) { sumRe += T._get(p, j) * yRe[j]; sumIm += T._get(p, j) * yIm[j]; }
                const dRe = T._get(p, p) - lambdaRe, dIm = -lambdaIm;
                const [yr, yi] = Matrix._cdiv(-sumRe, -sumIm, dRe, dIm, floor);
                yRe[p] = yr; yIm[p] = yi;
                continue;
            }

            // Coupled 2x2 solve for rows p and q = p+1:
            //   [ (T[p,p]-lambda)   T[p,q]           ] [y_p]   [rhs_p]
            //   [ T[q,p]            (T[q,q]-lambda)  ] [y_q] = [rhs_q]
            let rpRe = 0, rpIm = 0, rqRe = 0, rqIm = 0;
            for (let j = q + 1; j <= hiTarget; j++) {
                rpRe -= T._get(p, j) * yRe[j]; rpIm -= T._get(p, j) * yIm[j];
                rqRe -= T._get(q, j) * yRe[j]; rqIm -= T._get(q, j) * yIm[j];
            }
            const m00Re = T._get(p, p) - lambdaRe, m00Im = -lambdaIm;
            const m01Re = T._get(p, q);
            const m10Re = T._get(q, p);
            const m11Re = T._get(q, q) - lambdaRe, m11Im = -lambdaIm;

            const [detARe, detAIm] = Matrix._cmul(m00Re, m00Im, m11Re, m11Im);
            const [detBRe, detBIm] = Matrix._cmul(m01Re, 0, m10Re, 0);
            const detRe = detARe - detBRe, detIm = detAIm - detBIm;

            const [numPARe, numPAIm] = Matrix._cmul(rpRe, rpIm, m11Re, m11Im);
            const [numPBRe, numPBIm] = Matrix._cmul(m01Re, 0, rqRe, rqIm);
            const [ypRe, ypIm] = Matrix._cdiv(numPARe - numPBRe, numPAIm - numPBIm, detRe, detIm, floor);

            const [numQARe, numQAIm] = Matrix._cmul(m00Re, m00Im, rqRe, rqIm);
            const [numQBRe, numQBIm] = Matrix._cmul(m10Re, 0, rpRe, rpIm);
            const [yqRe, yqIm] = Matrix._cdiv(numQARe - numQBRe, numQAIm - numQBIm, detRe, detIm, floor);

            yRe[p] = ypRe; yIm[p] = ypIm;
            yRe[q] = yqRe; yIm[q] = yqIm;
        }
    }

    /**
     * Core of both `eigenvalues()` and `eig()`: reduces this (square)
     * matrix to upper Hessenberg form, then runs double-shift QR iteration
     * with deflation until every diagonal block has shrunk to size 1 (a
     * real eigenvalue) or size 2 (necessarily a complex-conjugate pair —
     * see below). The result is a genuine real Schur form: `this = Q * T *
     * Q^T` with `Q` orthogonal and `T` quasi-upper-triangular.
     *
     * Each QR iteration shifts by the trace `s` and determinant `t` of the
     * active block's trailing 2x2 submatrix — both always real, even when
     * that 2x2's own eigenvalues are complex — via `M = S^2 - s*S + t*I`,
     * `M = QR`, `S := Q^T * S * Q`. This is the *explicit* form of the
     * real double-shift ("Francis") QR step LAPACK (`dhseqr`/`dlahqr`)
     * implements *implicitly* via bulge-chasing: the same shift strategy
     * and the same fast (locally cubic) convergence on both real and
     * complex-conjugate eigenvalues, but computed by forming `S^2`
     * explicitly each iteration rather than the implicit bulge-chase,
     * trading some performance for a much simpler, easier-to-verify
     * implementation. Every 10th iteration on a block that hasn't
     * deflated, an ad hoc "exceptional shift" (a real value repeated
     * twice, following EISPACK's/Numerical Recipes' `hqr`) is used in
     * place of the trace/determinant pair, to break the rare stagnation
     * cycles that can otherwise trap any fixed-shift strategy.
     *
     * When `accumulateQ` is true, an extra cleanup pass runs at the end:
     * any size-2 block whose eigenvalues turn out to be real (rather than
     * a complex-conjugate pair) gets one further 2x2 rotation applied,
     * splitting it into two independent size-1 blocks. This isn't needed
     * for eigenvalues alone (the 2x2 closed-form solve already handles
     * real roots correctly either way), but it means `eig()` never has to
     * special-case "a 2x2 block that happens to have real eigenvalues" —
     * every returned block is either a real 1x1 or a genuinely
     * irreducible complex 2x2.
     * @param accumulateQ Whether to build and return `Q` (needed for
     * eigenvectors; skipped for eigenvalues alone to save the O(n^3) of
     * accumulation work, similarly to `_hessenberg()`'s same-named option).
     * @returns `T` (the Schur form), `Q` (or `null`), and `blocks`: the
     * final list of `[lo, hi]` diagonal block ranges, sorted by `lo`
     * ascending — i.e. top-to-bottom position in `T`, *not* the order
     * blocks happened to finish deflating in.
     * @throws {Error} If the iteration budget is exhausted before every
     * block deflates (e.g. for a pathologically slow-converging matrix).
     */
    private _schurForm(accumulateQ: boolean, tol: number, maxIterations: number): { T: Matrix; Q: Matrix | null; blocks: [number, number][] } {
        const n = this.rows;
        const { H, Q } = this._hessenberg(accumulateQ);
        if (n === 1) return { T: H, Q, blocks: [[0, 0]] };

        // Hessenberg-form norm (sum of magnitudes on/above the diagonal,
        // plus the subdiagonal): used as a floor for the deflation test
        // when the local diagonal entries are themselves ~0, matching
        // EISPACK's `anorm` fallback.
        let anorm = 0;
        for (let i = 0; i < n; i++) {
            for (let j = Math.max(0, i - 1); j < n; j++) anorm += Math.abs(H._get(i, j));
        }

        // Stack of active [lo, hi] (inclusive) diagonal blocks still to
        // resolve. A block deflates either by splitting into two smaller
        // blocks (an interior subdiagonal entry vanishes) or, once it
        // shrinks to size 1 or 2, by direct read-off.
        const stack: [number, number][] = [[0, n - 1]];
        const blocks: [number, number][] = [];
        let budget = maxIterations;

        while (stack.length > 0) {
            const [lo, hi] = stack.pop()!;

            if (hi === lo) { blocks.push([lo, hi]); continue; }
            if (hi === lo + 1) { blocks.push([lo, hi]); continue; }

            // Look for an interior subdiagonal entry small enough to treat
            // as zero, splitting the block in two. If none is found, take
            // one shifted QR step on the whole active block and check
            // again. `its` counts iterations spent on *this* block since
            // it was last split off, purely to trigger the exceptional
            // shift below — it does not affect the shared `budget`.
            let its = 0;
            for (; ;) {
                let splitAt = -1;
                for (let k = hi; k > lo; k--) {
                    const s = Math.abs(H._get(k - 1, k - 1)) + Math.abs(H._get(k, k));
                    const threshold = tol * (s === 0 ? anorm : s);
                    if (Math.abs(H._get(k, k - 1)) <= threshold) { splitAt = k; break; }
                }
                if (splitAt !== -1) {
                    stack.push([lo, splitAt - 1]);
                    stack.push([splitAt, hi]);
                    break;
                }

                if (budget-- <= 0) {
                    throw new Error(`Matrix eigenvalues: failed to converge within the iteration budget (${maxIterations})`);
                }

                its++;
                const size = hi - lo + 1;
                let s: number, t: number;
                if (its % 10 === 0) {
                    // Exceptional shift: an ad hoc real value used twice
                    // (equivalent to `M = (S - sigma*I)^2`) in place of the
                    // usual trace/determinant pair, to break rare
                    // stagnation cycles a fixed shift strategy can otherwise
                    // get stuck in.
                    const sigma = 0.75 * (Math.abs(H._get(hi, hi - 1)) + Math.abs(H._get(hi - 1, hi - 2)));
                    s = 2 * sigma;
                    t = sigma * sigma;
                } else {
                    const a = H._get(hi - 1, hi - 1), b = H._get(hi - 1, hi);
                    const c = H._get(hi, hi - 1), d = H._get(hi, hi);
                    s = a + d;
                    t = a * d - b * c;
                }

                const S = new Matrix(size, size);
                for (let i = 0; i < size; i++) {
                    for (let j = 0; j < size; j++) S._set(i, j, H._get(lo + i, lo + j));
                }
                // M = S^2 - s*S + t*I: the real quadratic shift polynomial.
                const S2 = S.matmul(S);
                const M = new Matrix(size, size);
                for (let i = 0; i < size; i++) {
                    for (let j = 0; j < size; j++) {
                        let v = S2._get(i, j) - s * S._get(i, j);
                        if (i === j) v += t;
                        M._set(i, j, v);
                    }
                }
                const { Q: Qlocal } = Matrix._householderQR(M);
                Matrix._applySimilarity(H, Q, lo, hi, Qlocal);
            }
        }

        blocks.sort((x, y) => x[0] - y[0]);
        if (!accumulateQ) return { T: H, Q, blocks };

        // Cleanup pass: split any size-2 block with real eigenvalues into
        // two size-1 blocks (see the doc comment above).
        const finalBlocks: [number, number][] = [];
        for (const [lo, hi] of blocks) {
            if (hi === lo) { finalBlocks.push([lo, hi]); continue; }

            const a = H._get(lo, lo), b = H._get(lo, hi), c = H._get(hi, lo), d = H._get(hi, hi);
            const disc = (a - d) * (a - d) + 4 * b * c;
            if (disc < 0) { finalBlocks.push([lo, hi]); continue; } // genuine complex pair: leave as-is

            // Triangularize via the eigenvector for one root: solving
            // (block - lambda*I)x = 0 gives x = [b, lambda - a] (or the
            // symmetric [lambda - d, c] if b happens to be ~0); the
            // rotation with x/|x| as its first column then triangularizes
            // the block by construction.
            const lambda = (a + d + Math.sqrt(disc)) / 2;
            let x0 = b, x1 = lambda - a;
            if (Math.abs(x0) < 1e-300 && Math.abs(x1) < 1e-300) { x0 = lambda - d; x1 = c; }
            if (Math.abs(x0) < 1e-300 && Math.abs(x1) < 1e-300) { x0 = 1; x1 = 0; }
            const norm = Math.hypot(x0, x1);
            const q0 = x0 / norm, q1 = x1 / norm;
            const Qlocal = Matrix.from([[q0, -q1], [q1, q0]]);

            Matrix._applySimilarity(H, Q, lo, hi, Qlocal);
            H._set(hi, lo, 0); // clean up floating-point noise below the diagonal
            finalBlocks.push([lo, lo]);
            finalBlocks.push([hi, hi]);
        }
        return { T: H, Q, blocks: finalBlocks };
    }

    /**
     * Computes the eigenvalues of this (square) matrix, via `_schurForm()`
     * (Hessenberg reduction + double-shift QR iteration with deflation —
     * see that method's doc comment for the algorithm). If you also need
     * eigenvectors, use `eig()` instead: it computes both together from
     * the same Schur form, which is cheaper than computing eigenvalues
     * and then eigenvectors separately.
     * @param options.tol Relative tolerance for the deflation test: a
     * subdiagonal entry `H[k, k-1]` is treated as converged once
     * `abs(H[k, k-1]) <= tol * (abs(H[k-1,k-1]) + abs(H[k,k]))` (falling
     * back to `tol * norm` when that sum is `0`). Defaults to
     * `Number.EPSILON`, i.e. one unit in the last place — the same
     * machine-precision convention LAPACK (`dlamch('P')`) and EISPACK use
     * for this test.
     * @param options.maxIterations Total shifted-QR iteration budget
     * across the whole computation (shared across all deflations, not
     * per-eigenvalue). Defaults to `30 * this.rows`, matching EISPACK's
     * `hqr` (`itn = 30*n`) and LAPACK's similar per-eigenvalue allowance.
     * @returns The `rows` eigenvalues, in top-to-bottom diagonal position
     * within the underlying Schur form — *not* sorted by magnitude, and
     * *not* deflation order (an early-finishing block and a
     * later-finishing one can end up adjacent either way).
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If the iteration budget is exhausted before every
     * block deflates (e.g. for a pathologically slow-converging matrix).
     */
    eigenvalues(options?: { maxIterations?: number; tol?: number }): Eigenvalue[] {
        if (this.rows !== this.cols) throw new RangeError(`Matrix eigenvalues requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        const tol = options?.tol ?? Number.EPSILON;
        const maxIterations = options?.maxIterations ?? 30 * n;

        const { T, blocks } = this._schurForm(false, tol, maxIterations);
        const result: Eigenvalue[] = new Array(n);
        for (const [lo, hi] of blocks) {
            if (hi === lo) {
                result[lo] = { re: T._get(lo, lo), im: 0 };
            } else {
                const [e1, e2] = Matrix._solve2x2Eigs(T._get(lo, lo), T._get(lo, hi), T._get(hi, lo), T._get(hi, hi));
                result[lo] = e1;
                result[hi] = e2;
            }
        }
        return result;
    }

    /**
     * Computes both the eigenvalues and eigenvectors of this (square)
     * matrix: first the real Schur form via `_schurForm()`, then each
     * eigenvector by back-substitution on the (quasi-)upper-triangular `T`
     * followed by mapping back through `Q` (`A = Q * T * Q^T`, so a vector
     * `y` with `T*y = lambda*y` gives `Q*y` with `A*(Q*y) = lambda*(Q*y)`).
     *
     * A real matrix's eigenvector for a genuinely complex eigenvalue is
     * itself genuinely complex — that's unavoidable, not a limitation of
     * this implementation — so each pair carries `vectorRe`/`vectorIm`
     * rather than a single real `Vector`; `vectorIm` is all-zero whenever
     * `value.im` is `0`. For a complex-conjugate pair of eigenvalues, the
     * two eigenvectors are exact complex conjugates of one another, so
     * only one is computed directly and the other is derived by negating
     * `vectorIm` (and `value.im`).
     *
     * Normalization follows LAPACK's convention (`dgeev`): each
     * eigenvector is scaled to Euclidean norm 1, with its largest-magnitude
     * component rotated to be real and positive (a real eigenvector is
     * already real, so only the norm applies; its sign is otherwise
     * whatever the computation happens to produce).
     * @param options See `eigenvalues()` — same meaning and defaults.
     * @returns The `rows` eigenpairs, in the same order `eigenvalues()`
     * would return the values alone.
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If the iteration budget is exhausted before every
     * block deflates.
     */
    eig(options?: { maxIterations?: number; tol?: number }): Eigenpair[] {
        if (this.rows !== this.cols) throw new RangeError(`Matrix eig requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        const tol = options?.tol ?? Number.EPSILON;
        const maxIterations = options?.maxIterations ?? 30 * n;

        if (n === 1) {
            return [{ value: { re: this._get(0, 0), im: 0 }, vectorRe: new Vector([1]), vectorIm: new Vector([0]) }];
        }

        const { T, Q, blocks } = this._schurForm(true, tol, maxIterations);
        const Qt = Q!; // accumulateQ was true, so Q is always populated here.

        // Floor for the back-substitution denominator, guarding against
        // (near-)repeated eigenvalues landing on the diagonal — the same
        // role LAPACK's dtrevc safe-minimum plays.
        let tnorm = 0;
        for (let i = 0; i < n; i++) for (let j = i; j < n; j++) tnorm += Math.abs(T._get(i, j));
        const floor = Number.EPSILON * Math.max(tnorm, 1);

        const result: Eigenpair[] = new Array(n);

        for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
            const [lo, hi] = blocks[blockIdx];
            if (hi === lo) {
                // Real eigenvalue: back-substitute for a real eigenvector
                // of T, supported only on indices [0, lo] (T is upper
                // triangular, so the invariant subspace for the k-th
                // diagonal eigenvalue is spanned by the first k+1 Schur
                // basis vectors).
                const lambda = T._get(lo, lo);
                const yRe = new Float64Array(n), yIm = new Float64Array(n);
                yRe[lo] = 1;
                Matrix._backSubstitute(T, blocks, blockIdx, lo, lambda, 0, yRe, yIm, floor);

                const vRe = new Array<number>(n).fill(0);
                for (let i = 0; i < n; i++) {
                    let sum = 0;
                    for (let k = 0; k <= lo; k++) sum += Qt._get(i, k) * yRe[k];
                    vRe[i] = sum;
                }
                let norm = 0;
                for (let i = 0; i < n; i++) norm += vRe[i] * vRe[i];
                norm = Math.sqrt(norm);
                if (norm > 0) for (let i = 0; i < n; i++) vRe[i] /= norm;

                result[lo] = {
                    value: { re: lambda, im: 0 },
                    vectorRe: new Vector(vRe),
                    vectorIm: new Vector(new Array<number>(n).fill(0)),
                };
            } else {
                // Complex-conjugate pair: same idea, but the local 2-entry
                // "seed" of the eigenvector and every step of the
                // back-substitution are complex.
                const a = T._get(lo, lo), b = T._get(lo, hi), c = T._get(hi, lo), d = T._get(hi, hi);
                const [e1] = Matrix._solve2x2Eigs(a, b, c, d); // e1.im > 0 by construction
                const lamRe = e1.re, lamIm = e1.im;

                // Local eigenvector of the 2x2 block: [b, lambda - a]
                // solves (block - lambda*I)v = 0 directly (or the
                // symmetric form if b happens to be ~0).
                let v0re = b, v0im = 0;
                let v1re = lamRe - a, v1im = lamIm;
                if (Math.abs(b) < floor) { v0re = lamRe - d; v0im = lamIm; v1re = c; v1im = 0; }

                const yRe = new Float64Array(n), yIm = new Float64Array(n);
                yRe[lo] = v0re; yIm[lo] = v0im;
                yRe[hi] = v1re; yIm[hi] = v1im;
                Matrix._backSubstitute(T, blocks, blockIdx, hi, lamRe, lamIm, yRe, yIm, floor);

                const vRe = new Array<number>(n).fill(0), vIm = new Array<number>(n).fill(0);
                for (let i = 0; i < n; i++) {
                    let sRe = 0, sIm = 0;
                    for (let k = 0; k <= hi; k++) { const q = Qt._get(i, k); sRe += q * yRe[k]; sIm += q * yIm[k]; }
                    vRe[i] = sRe; vIm[i] = sIm;
                }

                // LAPACK normalization: rotate phase so the
                // largest-magnitude component is real and positive, then
                // scale to unit Euclidean norm.
                let maxMagSq = -1, maxIdx = 0;
                for (let i = 0; i < n; i++) {
                    const m = vRe[i] * vRe[i] + vIm[i] * vIm[i];
                    if (m > maxMagSq) { maxMagSq = m; maxIdx = i; }
                }
                if (maxMagSq > 0) {
                    const mag = Math.sqrt(maxMagSq);
                    const cosT = vRe[maxIdx] / mag, sinT = vIm[maxIdx] / mag;
                    for (let i = 0; i < n; i++) {
                        const re = vRe[i], im = vIm[i];
                        vRe[i] = re * cosT + im * sinT;
                        vIm[i] = im * cosT - re * sinT;
                    }
                }
                let norm = 0;
                for (let i = 0; i < n; i++) norm += vRe[i] * vRe[i] + vIm[i] * vIm[i];
                norm = Math.sqrt(norm);
                if (norm > 0) for (let i = 0; i < n; i++) { vRe[i] /= norm; vIm[i] /= norm; }

                const vImConj = vIm.map(x => -x);
                result[lo] = { value: { re: lamRe, im: lamIm }, vectorRe: new Vector(vRe), vectorIm: new Vector(vIm) };
                result[hi] = { value: { re: lamRe, im: -lamIm }, vectorRe: new Vector(vRe), vectorIm: new Vector(vImConj) };
            }
        }

        return result;
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
                for (let i = 0; i < this.rows; i++) col[i] = this._get(i, j);
                res.data[j] = reduceFn(col);
            }
            return res;
        } else {
            const res = new Vector(this.rows);
            for (let i = 0; i < this.rows; i++) {
                const row = this.data.subarray(this._idx(i, 0), this._idx(i, 0) + this.cols);
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
                    running += this._get(i, j);
                    res._set(i, j, running);
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                const offset = this._idx(i, 0);
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
                const start = res._idx(i, 0);
                res.data.subarray(start, start + res.cols).sort(compareFn);
            }
        } else {
            const col = new Float64Array(res.rows);
            for (let j = 0; j < res.cols; j++) {
                for (let i = 0; i < res.rows; i++) col[i] = res._get(i, j);
                col.sort(compareFn);
                for (let i = 0; i < res.rows; i++) res._set(i, j, col[i]);
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
            const srcOffset = this._idx(rs + i, cs);
            res.data.set(this.data.subarray(srcOffset, srcOffset + newCols), res._idx(i, 0));
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
                const tmp = this._get(i, j);
                this._set(i, j, this._get(j, i));
                this._set(j, i, tmp);
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
        const oi = this._idx(i, 0);
        const oj = this._idx(j, 0);
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
        const offset = this._idx(i, 0);
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
        const offsetI = this._idx(i, 0);
        const offsetJ = this._idx(j, 0);
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
}