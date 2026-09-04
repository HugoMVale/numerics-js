import { ArrayND } from './arraynd.js';
import { Vector } from './Vector.js';

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