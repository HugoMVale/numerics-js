import { Array1D } from './array1d';

/**
 * Result of `Array2D.lu()`: a partial-pivoted LU factorization such that
 * `P * A = L * U`, where `P` is the row permutation implied by `perm`.
 */
export interface LUDecomposition {
    /** Unit lower-triangular factor (1s on the diagonal). */
    L: Array2D;
    /** Upper-triangular factor. */
    U: Array2D;
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
 */
export class Array2D {
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
        if (!Number.isInteger(rows) || rows < 1) {
            throw new RangeError(`Array2D: rows must be a positive integer, got ${rows}`);
        }
        if (!Number.isInteger(cols) || cols < 1) {
            throw new RangeError(`Array2D: cols must be a positive integer, got ${cols}`);
        }
        if (input !== undefined && input.length !== rows * cols) {
            throw new RangeError(
                `Array2D: expected ${rows * cols} values for a ${rows}x${cols} matrix, got ${input.length}`
            );
        }
        this._rows = rows;
        this._cols = cols;
        this.data = new Float64Array(rows * cols);
        if (input !== undefined) this.data.set(input);
    }

    /**
     * The number of rows in this matrix. Read-only (like `Array1D.size`) so
     * it can never desync from `data`.
     */
    get rows(): number {
        return this._rows;
    }

    /**
     * The number of columns in this matrix. Read-only (like `Array1D.size`)
     * so it can never desync from `data`.
     */
    get cols(): number {
        return this._cols;
    }

    /**
     * The total number of elements in this matrix, i.e. `rows * cols`.
     * Derived directly from `data.length` so it can never desync.
     */
    get size(): number {
        return this.data.length;
    }

    /**
     * Throws if `m` is not an Array2D with the same shape as this one. Used
     * internally to guard binary operations against silent shape mismatches.
     * @param m The other matrix.
     * @throws {RangeError} If `m.rows !== this.rows || m.cols !== this.cols`.
     */
    private _checkShape(m: Array2D): void {
        if (m.rows !== this.rows || m.cols !== this.cols) {
            throw new RangeError(`Array2D shape mismatch: ${this.rows}x${this.cols} vs ${m.rows}x${m.cols}`);
        }
    }

    /**
     * Throws if `(i, j)` is not a valid 0-based index into this matrix.
     * @param i Row index (0-based).
     * @param j Column index (0-based).
     * @throws {RangeError} If `i` or `j` is out of range.
     */
    private _checkBounds(i: number, j: number): void {
        if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
            throw new RangeError(`Array2D index (${i}, ${j}) out of bounds for ${this.rows}x${this.cols} matrix`);
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
    row(i: number): Array1D {
        if (i < 0 || i >= this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
        return new Array1D(this.data.slice(this._idx(i, 0), this._idx(i, 0) + this.cols));
    }

    /**
     * Extracts column `j` as a vector.
     * @param j Column index (0-based).
     * @returns A new vector with `this.rows` components.
     */
    col(j: number): Array1D {
        if (j < 0 || j >= this.cols) throw new RangeError(`Array2D column ${j} out of bounds for ${this.cols} columns`);
        const res = new Array1D(this.rows);
        for (let i = 0; i < this.rows; i++) res.data[i] = this.get(i, j);
        return res;
    }

    /**
     * Overwrites row `i` in place with the given values.
     * @param i Row index (0-based).
     * @param values Values to copy in; must have length `cols`.
     * @returns `this`, for chaining.
     * @throws {RangeError} If `i` is out of bounds, or `values.length !== cols`.
     */
    setRow(i: number, values: Array1D | ArrayLike<number>): this {
        if (i < 0 || i >= this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
        const src = values instanceof Array1D ? values.data : values;
        if (src.length !== this.cols) {
            throw new RangeError(`Array2D setRow: expected ${this.cols} values for row ${i}, got ${src.length}`);
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
    setCol(j: number, values: Array1D | ArrayLike<number>): this {
        if (j < 0 || j >= this.cols) throw new RangeError(`Array2D column ${j} out of bounds for ${this.cols} columns`);
        const src = values instanceof Array1D ? values.data : values;
        if (src.length !== this.rows) {
            throw new RangeError(`Array2D setCol: expected ${this.rows} values for column ${j}, got ${src.length}`);
        }
        for (let i = 0; i < this.rows; i++) this.set(i, j, src[i]);
        return this;
    }

    // -----------------------------------------------------------------
    // Immutable operations (return new instances)
    // -----------------------------------------------------------------

    /**
     * Adds another matrix to this one, elementwise.
     * @param m The matrix to add. Must have the same shape as this one.
     * @returns A new matrix equal to `this + m`.
     */
    add(m: Array2D): Array2D {
        this._checkShape(m);
        const res = new Array2D(this.rows, this.cols);
        for (let k = 0; k < this.data.length; k++) res.data[k] = this.data[k] + m.data[k];
        return res;
    }

    /**
     * Subtracts another matrix from this one, elementwise.
     * @param m The matrix to subtract. Must have the same shape as this one.
     * @returns A new matrix equal to `this - m`.
     */
    sub(m: Array2D): Array2D {
        this._checkShape(m);
        const res = new Array2D(this.rows, this.cols);
        for (let k = 0; k < this.data.length; k++) res.data[k] = this.data[k] - m.data[k];
        return res;
    }

    /**
     * Scales every element of this matrix by a scalar.
     * @param s The scale factor.
     * @returns A new matrix equal to `this * s`.
     */
    mult(s: number): Array2D {
        const res = new Array2D(this.rows, this.cols);
        for (let k = 0; k < this.data.length; k++) res.data[k] = this.data[k] * s;
        return res;
    }

    /**
     * Multiplies this matrix by another: `this * m` (matrix product).
     * @param m The right-hand matrix. Must have `m.rows === this.cols`.
     * @returns A new `this.rows x m.cols` matrix.
     */
    matmul(m: Array2D): Array2D {
        if (this.cols !== m.rows) {
            throw new RangeError(`Array2D matmul shape mismatch: ${this.rows}x${this.cols} * ${m.rows}x${m.cols}`);
        }
        const res = new Array2D(this.rows, m.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let k = 0; k < this.cols; k++) {
                const a = this.get(i, k);
                if (a === 0) continue;
                for (let j = 0; j < m.cols; j++) {
                    res.set(i, j, res.get(i, j) + a * m.get(k, j));
                }
            }
        }
        return res;
    }

    /**
     * Multiplies this matrix by a column vector: `this * v`.
     * @param v The vector. Must have `v.size === this.cols`.
     * @returns A new vector with `this.rows` components.
     */
    mulVec(v: Array1D): Array1D {
        if (v.size !== this.cols) {
            throw new RangeError(`Array2D mulVec shape mismatch: ${this.rows}x${this.cols} * vec(${v.size})`);
        }
        const res = new Array1D(this.rows);
        for (let i = 0; i < this.rows; i++) {
            let sum = 0;
            for (let j = 0; j < this.cols; j++) sum += this.get(i, j) * v.data[j];
            res.data[i] = sum;
        }
        return res;
    }

    /**
     * Computes the transpose of this matrix.
     * @returns A new `this.cols x this.rows` matrix equal to `this^T`.
     */
    transpose(): Array2D {
        const res = new Array2D(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) res.set(j, i, this.get(i, j));
        }
        return res;
    }

    /**
     * Computes the trace (sum of diagonal elements) of this matrix.
     * @returns The trace.
     * @throws {RangeError} If this matrix is not square.
     */
    trace(): number {
        if (this.rows !== this.cols) throw new RangeError(`Array2D trace requires a square matrix, got ${this.rows}x${this.cols}`);
        let sum = 0;
        for (let i = 0; i < this.rows; i++) sum += this.get(i, i);
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
    private static _findPivotRow(m: Array2D, col: number, startRow: number, tol: number): number {
        let pivotRow = -1;
        let pivotVal = tol;
        for (let i = startRow; i < m.rows; i++) {
            const v = Math.abs(m.get(i, col));
            if (v > pivotVal) { pivotVal = v; pivotRow = i; }
        }
        return pivotRow;
    }

    /**
     * Reduces a copy of this matrix to row echelon form via Gaussian
     * elimination with partial pivoting (forward elimination only, no
     * back-substitution or row scaling). Shared building block for `rank()`
     * and `determinant()`, which both need the same elimination but reduce
     * the result differently.
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
            const pivotRow = Array2D._findPivotRow(m, col, rank, tol);
            if (pivotRow === -1) continue;
            if (pivotRow !== rank) { m.swapRows(rank, pivotRow); sign = -sign; }
            const pivot = m.get(rank, col);
            pivots.push(pivot);
            for (let i = rank + 1; i < m.rows; i++) {
                const factor = m.get(i, col) / pivot;
                if (factor !== 0) m.addScaledRow(i, rank, -factor);
            }
            rank++;
        }
        return { rank, sign, pivots };
    }

    /**
     * Computes the rank of this matrix (the number of linearly independent
     * rows/columns), via Gaussian elimination with partial pivoting.
     * @param tol Absolute tolerance below which a pivot is treated as zero.
     * @returns The rank, between `0` and `min(rows, cols)`.
     */
    rank(tol = 1e-10): number {
        return this._forwardEliminate(tol).rank;
    }

    /**
     * Computes the determinant of this matrix, via Gaussian elimination with
     * partial pivoting.
     * @returns The determinant.
     * @throws {RangeError} If this matrix is not square.
     */
    determinant(): number {
        if (this.rows !== this.cols) throw new RangeError(`Array2D determinant requires a square matrix, got ${this.rows}x${this.cols}`);
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
    inverse(): Array2D {
        if (this.rows !== this.cols) throw new RangeError(`Array2D inverse requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        // Augment [this | I] and row-reduce the left half to I; the right
        // half then becomes this^-1.
        const aug = new Array2D(n, 2 * n);
        for (let i = 0; i < n; i++) {
            aug.data.set(this.data.subarray(this._idx(i, 0), this._idx(i, 0) + n), aug._idx(i, 0));
            aug.set(i, n + i, 1);
        }
        for (let col = 0; col < n; col++) {
            const pivotRow = Array2D._findPivotRow(aug, col, col, 0);
            if (pivotRow === -1) throw new Error('Array2D inverse: matrix is singular');
            if (pivotRow !== col) aug.swapRows(col, pivotRow);
            aug.scaleRow(col, 1 / aug.get(col, col));
            for (let i = 0; i < n; i++) {
                if (i === col) continue;
                const factor = aug.get(i, col);
                if (factor !== 0) aug.addScaledRow(i, col, -factor);
            }
        }
        const res = new Array2D(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) res.set(i, j, aug.get(i, n + j));
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
     * @param tol Absolute tolerance below which a candidate pivot is treated as zero.
     * @returns The `{ L, U, perm, sign }` factorization.
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If this matrix is singular.
     */
    lu(tol = 0): LUDecomposition {
        if (this.rows !== this.cols) throw new RangeError(`Array2D lu requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        const U = this.copy();
        const L = Array2D.identity(n);
        const perm = Array.from({ length: n }, (_, i) => i);
        let sign = 1;

        for (let col = 0; col < n; col++) {
            const pivotRow = Array2D._findPivotRow(U, col, col, tol);
            if (pivotRow === -1) throw new Error('Array2D lu: matrix is singular');
            if (pivotRow !== col) {
                U.swapRows(col, pivotRow);
                // Rows of L to the left of `col` already hold computed
                // multipliers; swap those along with U's rows so that
                // P * this === L * U still holds after the pivot.
                for (let k = 0; k < col; k++) {
                    const tmp = L.get(col, k);
                    L.set(col, k, L.get(pivotRow, k));
                    L.set(pivotRow, k, tmp);
                }
                [perm[col], perm[pivotRow]] = [perm[pivotRow], perm[col]];
                sign = -sign;
            }
            const pivot = U.get(col, col);
            for (let i = col + 1; i < n; i++) {
                const factor = U.get(i, col) / pivot;
                if (factor !== 0) {
                    U.addScaledRow(i, col, -factor);
                    L.set(i, col, factor);
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
    solveLower(b: Array1D, unitDiagonal = false): Array1D {
        if (this.rows !== this.cols) throw new RangeError(`Array2D solveLower requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.size !== this.rows) throw new RangeError(`Array2D solveLower shape mismatch: ${this.rows}x${this.cols} vs vec(${b.size})`);
        const n = this.rows;
        const x = new Array1D(n);
        for (let i = 0; i < n; i++) {
            let s = b.data[i];
            for (let j = 0; j < i; j++) s -= this.get(i, j) * x.data[j];
            if (unitDiagonal) {
                x.data[i] = s;
            } else {
                const d = this.get(i, i);
                if (d === 0) throw new Error('Array2D solveLower: zero diagonal entry, matrix is singular');
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
    solveUpper(b: Array1D): Array1D {
        if (this.rows !== this.cols) throw new RangeError(`Array2D solveUpper requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.size !== this.rows) throw new RangeError(`Array2D solveUpper shape mismatch: ${this.rows}x${this.cols} vs vec(${b.size})`);
        const n = this.rows;
        const x = new Array1D(n);
        for (let i = n - 1; i >= 0; i--) {
            let s = b.data[i];
            for (let j = i + 1; j < n; j++) s -= this.get(i, j) * x.data[j];
            const d = this.get(i, i);
            if (d === 0) throw new Error('Array2D solveUpper: zero diagonal entry, matrix is singular');
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
    solve(b: Array1D): Array1D {
        if (this.rows !== this.cols) throw new RangeError(`Array2D solve requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.size !== this.rows) throw new RangeError(`Array2D solve shape mismatch: ${this.rows}x${this.cols} vs vec(${b.size})`);
        const { L, U, perm } = this.lu();
        const pb = new Array1D(perm.map(p => b.data[p]));
        const y = L.solveLower(pb, true);
        return U.solveUpper(y);
    }

    /**
     * Computes the sum of all elements of this matrix.
     * @returns The sum, or `0` if the matrix is empty.
     */
    sum(): number {
        let s = 0;
        for (let k = 0; k < this.data.length; k++) s += this.data[k];
        return s;
    }

    /**
     * Finds the smallest element of this matrix.
     * @returns The minimum value.
     * @throws {RangeError} If the matrix is empty.
     */
    min(): number {
        if (this.data.length === 0) throw new RangeError('Array2D min() called on an empty matrix');
        let m = this.data[0];
        for (let k = 1; k < this.data.length; k++) if (this.data[k] < m) m = this.data[k];
        return m;
    }

    /**
     * Finds the largest element of this matrix.
     * @returns The maximum value.
     * @throws {RangeError} If the matrix is empty.
     */
    max(): number {
        if (this.data.length === 0) throw new RangeError('Array2D max() called on an empty matrix');
        let m = this.data[0];
        for (let k = 1; k < this.data.length; k++) if (this.data[k] > m) m = this.data[k];
        return m;
    }

    /**
     * Creates an independent copy of this matrix.
     * @returns A new Array2D with the same shape and values.
     */
    copy(): Array2D {
        return new Array2D(this.rows, this.cols, this.data);
    }

    /**
     * Checks whether this matrix is elementwise close to another. Symmetric
     * in `this` and `m`: an element `a` is close to `b` if
     * `|a - b| <= atol + rtol * max(|a|, |b|)`, so `a.isClose(b) === b.isClose(a)`.
     * @param m The other matrix.
     * @param rtol Relative tolerance.
     * @param atol Absolute tolerance.
     * @returns `true` if `m` has the same shape and all elements of `this` are close to `m`'s.
     */
    isClose(m: Array2D, rtol = 1e-5, atol = 1e-8): boolean {
        if (m.rows !== this.rows || m.cols !== this.cols) return false;
        for (let k = 0; k < this.data.length; k++) {
            const a = this.data[k];
            const b = m.data[k];
            if (Math.abs(a - b) > atol + rtol * Math.max(Math.abs(a), Math.abs(b))) return false;
        }
        return true;
    }

    /**
     * Checks whether this matrix is exactly elementwise equal to another.
     * For tolerance-based comparison, use `isClose` instead.
     * @param m The other matrix.
     * @returns `true` if `m` has the same shape and all elements are exactly equal.
     */
    equals(m: Array2D): boolean {
        if (m.rows !== this.rows || m.cols !== this.cols) return false;
        for (let k = 0; k < this.data.length; k++) {
            if (this.data[k] !== m.data[k]) return false;
        }
        return true;
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
     * @returns e.g. `"Array2D[[1, 2], [3, 4]]"`.
     */
    toString(): string {
        const rows = this.toArray().map(r => `[${r.join(', ')}]`);
        return `Array2D[${rows.join(', ')}]`;
    }

    /**
     * Makes Array2D iterable over its rows, e.g. `for (const r of someMatrix)`.
     * Each yielded value is an Array1D.
     */
    *[Symbol.iterator](): Generator<Array1D, void, unknown> {
        for (let i = 0; i < this.rows; i++) yield this.row(i);
    }

    // -----------------------------------------------------------------
    // In-place (mutating) operations.
    // -----------------------------------------------------------------

    /**
     * Resets this matrix to all zeros in place.
     * @returns `this`, for chaining.
     */
    reset(): this {
        this.data.fill(0);
        return this;
    }

    /**
     * Adds another matrix to this one in place, elementwise: `this += m`.
     * @param m The matrix to add. Must have the same shape as this one.
     * @returns `this`, for chaining.
     */
    addSelf(m: Array2D): this {
        this._checkShape(m);
        for (let k = 0; k < this.data.length; k++) this.data[k] += m.data[k];
        return this;
    }

    /**
     * Subtracts another matrix from this one in place, elementwise: `this -= m`.
     * @param m The matrix to subtract. Must have the same shape as this one.
     * @returns `this`, for chaining.
     */
    subSelf(m: Array2D): this {
        this._checkShape(m);
        for (let k = 0; k < this.data.length; k++) this.data[k] -= m.data[k];
        return this;
    }

    /**
     * Scales every element of this matrix in place: `this *= s`.
     * @param s The scale factor.
     * @returns `this`, for chaining.
     */
    multSelf(s: number): this {
        for (let k = 0; k < this.data.length; k++) this.data[k] *= s;
        return this;
    }

    /**
     * Transposes a square matrix in place.
     * @returns `this`, for chaining.
     * @throws {RangeError} If this matrix is not square.
     */
    transposeSelf(): this {
        if (this.rows !== this.cols) throw new RangeError(`Array2D transposeSelf requires a square matrix, got ${this.rows}x${this.cols}`);
        for (let i = 0; i < this.rows; i++) {
            for (let j = i + 1; j < this.cols; j++) {
                const tmp = this.get(i, j);
                this.set(i, j, this.get(j, i));
                this.set(j, i, tmp);
            }
        }
        return this;
    }

    /**
     * Swaps two rows in place. Useful when implementing pivoting algorithms.
     * @param i First row index (0-based).
     * @param j Second row index (0-based).
     * @returns `this`, for chaining.
     * @throws {RangeError} If `i` or `j` is out of bounds.
     */
    swapRows(i: number, j: number): this {
        if (i < 0 || i >= this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
        if (j < 0 || j >= this.rows) throw new RangeError(`Array2D row ${j} out of bounds for ${this.rows} rows`);
        if (i === j) return this;
        const a = this.row(i).toArray();
        this.setRow(i, this.row(j).data);
        this.setRow(j, a);
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
        if (i < 0 || i >= this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
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
        if (i < 0 || i >= this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
        if (j < 0 || j >= this.rows) throw new RangeError(`Array2D row ${j} out of bounds for ${this.rows} rows`);
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
    static zero(rows: number, cols: number): Array2D {
        return new Array2D(rows, cols);
    }

    /**
     * Creates an `n x n` identity matrix.
     * @param n The matrix dimension.
     * @returns A new identity matrix.
     */
    static identity(n: number): Array2D {
        const res = new Array2D(n, n);
        for (let i = 0; i < n; i++) res.set(i, i, 1);
        return res;
    }

    /**
     * Creates an Array2D from an array of row arrays.
     * @param rows Source data; each inner array must have the same length.
     * @returns A new matrix with shape `rows.length x rows[0].length`.
     */
    static from(rows: number[][]): Array2D {
        if (rows.length === 0) {
            throw new RangeError('Array2D.from: cannot construct a matrix from an empty array (need at least one row)');
        }
        const nRows = rows.length;
        const nCols = rows[0].length;
        if (nCols === 0) {
            throw new RangeError('Array2D.from: cannot construct a matrix with empty rows (need at least one column)');
        }
        const res = new Array2D(nRows, nCols);
        for (let i = 0; i < nRows; i++) res.setRow(i, rows[i]);
        return res;
    }
}