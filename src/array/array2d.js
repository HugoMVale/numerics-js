import { Array1D } from '../index.js';

/**
 * A rows x cols matrix backed by a flat, row-major Float64Array.
 *
 * All element access (`get`, `set`, `row`, `col`, `setRow`, `setCol`,
 * `swapRows`, `scaleRow`, `addScaledRow`) is **1-based**: valid row indices
 * are `1..rows` and valid column indices are `1..cols`, matching common
 * mathematical / textbook convention rather than JS's usual 0-based indexing.
 */
export class Array2D {
    /**
     * @param {number} rows - Number of rows (must be >= 1).
     * @param {number} cols - Number of columns (must be >= 1).
     * @param {number[]|Float64Array} [input] - Optional initial data in
     *   row-major order (i.e. row 1 followed by row 2, etc.), length `rows * cols`.
     *   If omitted, the matrix is initialized to all zeros.
     */
    constructor(rows, cols, input) {
        this.rows = rows;
        this.cols = cols;
        this.data = new Float64Array(rows * cols);
        if (input !== undefined) this.data.set(input);
    }

    /**
     * Throws if `m` is not an Array2D with the same shape as this one. Used
     * internally to guard binary operations against silent shape mismatches.
     * @param {Array2D} m - The other matrix.
     * @throws {RangeError} If `m.rows !== this.rows || m.cols !== this.cols`.
     * @private
     */
    _checkShape(m) {
        if (m.rows !== this.rows || m.cols !== this.cols) {
            throw new RangeError(`Array2D shape mismatch: ${this.rows}x${this.cols} vs ${m.rows}x${m.cols}`);
        }
    }

    /**
     * Throws if `(i, j)` is not a valid 1-based index into this matrix.
     * @param {number} i - Row index (1-based).
     * @param {number} j - Column index (1-based).
     * @throws {RangeError} If `i` or `j` is out of range.
     * @private
     */
    _checkBounds(i, j) {
        if (i < 1 || i > this.rows || j < 1 || j > this.cols) {
            throw new RangeError(`Array2D index (${i}, ${j}) out of bounds for ${this.rows}x${this.cols} matrix`);
        }
    }

    /**
     * Converts a 1-based `(i, j)` index into a flat index into `data`.
     * @param {number} i - Row index (1-based).
     * @param {number} j - Column index (1-based).
     * @returns {number} The flat, 0-based index.
     * @private
     */
    _idx(i, j) { return (i - 1) * this.cols + (j - 1); }

    // -----------------------------------------------------------------
    // Element / row / column access
    // -----------------------------------------------------------------

    /**
     * Gets the element at row `i`, column `j`.
     * @param {number} i - Row index (1-based).
     * @param {number} j - Column index (1-based).
     * @returns {number} The value at `(i, j)`.
     */
    get(i, j) {
        this._checkBounds(i, j);
        return this.data[this._idx(i, j)];
    }

    /**
     * Sets the element at row `i`, column `j`, mutating this matrix in place.
     * @param {number} i - Row index (1-based).
     * @param {number} j - Column index (1-based).
     * @param {number} value - The value to store.
     * @returns {Array2D} `this`, for chaining.
     */
    set(i, j, value) {
        this._checkBounds(i, j);
        this.data[this._idx(i, j)] = value;
        return this;
    }

    /**
     * Extracts row `i` as a vector.
     * @param {number} i - Row index (1-based).
     * @returns {Array1D} A new vector with `this.cols` components.
     */
    row(i) {
        if (i < 1 || i > this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
        return new Array1D(this.data.subarray(this._idx(i, 1), this._idx(i, 1) + this.cols));
    }

    /**
     * Extracts column `j` as a vector.
     * @param {number} j - Column index (1-based).
     * @returns {Array1D} A new vector with `this.rows` components.
     */
    col(j) {
        if (j < 1 || j > this.cols) throw new RangeError(`Array2D column ${j} out of bounds for ${this.cols} columns`);
        const res = new Array1D(this.rows);
        for (let i = 1; i <= this.rows; i++) res.data[i - 1] = this.get(i, j);
        return res;
    }

    /**
     * Overwrites row `i` in place with the given values.
     * @param {number} i - Row index (1-based).
     * @param {Array1D|number[]|Float64Array} values - Values to copy in; must have length `cols`.
     * @returns {Array2D} `this`, for chaining.
     */
    setRow(i, values) {
        if (i < 1 || i > this.rows) throw new RangeError(`Array2D row ${i} out of bounds for ${this.rows} rows`);
        const src = values instanceof Array1D ? values.data : values;
        this.data.set(src, this._idx(i, 1));
        return this;
    }

    /**
     * Overwrites column `j` in place with the given values.
     * @param {number} j - Column index (1-based).
     * @param {Array1D|number[]|Float64Array} values - Values to copy in; must have length `rows`.
     * @returns {Array2D} `this`, for chaining.
     */
    setCol(j, values) {
        if (j < 1 || j > this.cols) throw new RangeError(`Array2D column ${j} out of bounds for ${this.cols} columns`);
        const src = values instanceof Array1D ? values.data : values;
        for (let i = 1; i <= this.rows; i++) this.set(i, j, src[i - 1]);
        return this;
    }

    // -----------------------------------------------------------------
    // Immutable operations (return new instances)
    // -----------------------------------------------------------------

    /**
     * Adds another matrix to this one, elementwise.
     * @param {Array2D} m - The matrix to add. Must have the same shape as this one.
     * @returns {Array2D} A new matrix equal to `this + m`.
     */
    add(m) {
        this._checkShape(m);
        const res = new Array2D(this.rows, this.cols);
        for (let k = 0; k < this.data.length; k++) res.data[k] = this.data[k] + m.data[k];
        return res;
    }

    /**
     * Subtracts another matrix from this one, elementwise.
     * @param {Array2D} m - The matrix to subtract. Must have the same shape as this one.
     * @returns {Array2D} A new matrix equal to `this - m`.
     */
    sub(m) {
        this._checkShape(m);
        const res = new Array2D(this.rows, this.cols);
        for (let k = 0; k < this.data.length; k++) res.data[k] = this.data[k] - m.data[k];
        return res;
    }

    /**
     * Scales every element of this matrix by a scalar.
     * @param {number} s - The scale factor.
     * @returns {Array2D} A new matrix equal to `this * s`.
     */
    mult(s) {
        const res = new Array2D(this.rows, this.cols);
        for (let k = 0; k < this.data.length; k++) res.data[k] = this.data[k] * s;
        return res;
    }

    /**
     * Multiplies this matrix by another: `this * m` (matrix product).
     * @param {Array2D} m - The right-hand matrix. Must have `m.rows === this.cols`.
     * @returns {Array2D} A new `this.rows x m.cols` matrix.
     */
    matmul(m) {
        if (this.cols !== m.rows) {
            throw new RangeError(`Array2D matmul shape mismatch: ${this.rows}x${this.cols} * ${m.rows}x${m.cols}`);
        }
        const res = new Array2D(this.rows, m.cols);
        for (let i = 1; i <= this.rows; i++) {
            for (let k = 1; k <= this.cols; k++) {
                const a = this.get(i, k);
                if (a === 0) continue;
                for (let j = 1; j <= m.cols; j++) {
                    res.set(i, j, res.get(i, j) + a * m.get(k, j));
                }
            }
        }
        return res;
    }

    /**
     * Multiplies this matrix by a column vector: `this * v`.
     * @param {Array1D} v - The vector. Must have `v.dim === this.cols`.
     * @returns {Array1D} A new vector with `this.rows` components.
     */
    mulVec(v) {
        if (v.dim !== this.cols) {
            throw new RangeError(`Array2D mulVec shape mismatch: ${this.rows}x${this.cols} * vec(${v.dim})`);
        }
        const res = new Array1D(this.rows);
        for (let i = 1; i <= this.rows; i++) {
            let sum = 0;
            for (let j = 1; j <= this.cols; j++) sum += this.get(i, j) * v.data[j - 1];
            res.data[i - 1] = sum;
        }
        return res;
    }

    /**
     * Computes the transpose of this matrix.
     * @returns {Array2D} A new `this.cols x this.rows` matrix equal to `this^T`.
     */
    transpose() {
        const res = new Array2D(this.cols, this.rows);
        for (let i = 1; i <= this.rows; i++) {
            for (let j = 1; j <= this.cols; j++) res.set(j, i, this.get(i, j));
        }
        return res;
    }

    /**
     * Computes the trace (sum of diagonal elements) of this matrix.
     * @returns {number} The trace.
     * @throws {RangeError} If this matrix is not square.
     */
    trace() {
        if (this.rows !== this.cols) throw new RangeError(`Array2D trace requires a square matrix, got ${this.rows}x${this.cols}`);
        let sum = 0;
        for (let i = 1; i <= this.rows; i++) sum += this.get(i, i);
        return sum;
    }

    /**
     * Finds the best pivot row for column `col`, searching rows `startRow..m.rows`.
     * Used internally by `_forwardEliminate`, `inverse`, and `solve` to share
     * the same partial-pivoting (largest-magnitude-entry) selection logic.
     * @param {Array2D} m - The matrix to search (may be a working copy or an augmented matrix).
     * @param {number} col - Column to search (1-based).
     * @param {number} startRow - First row to consider (1-based).
     * @param {number} tol - Entries with absolute value at or below this are never chosen as a pivot.
     * @returns {number} The 1-based row index of the best pivot, or `-1` if none exceeds `tol`.
     * @private
     */
    static _findPivotRow(m, col, startRow, tol) {
        let pivotRow = -1;
        let pivotVal = tol;
        for (let i = startRow; i <= m.rows; i++) {
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
     * @param {number} [tol=0] - Absolute tolerance below which a candidate pivot is treated as zero.
     * @returns {{rank: number, sign: number, pivots: number[]}} `rank` is the
     *   number of pivots found; `sign` is `+1`/`-1` tracking the row-swap
     *   parity (meaningful only for square matrices, for determinant sign);
     *   `pivots` are the pivot values in the order they were chosen (their
     *   product, times `sign`, is the determinant for a square matrix).
     * @private
     */
    _forwardEliminate(tol = 0) {
        const m = this.copy();
        let rank = 0;
        let sign = 1;
        const pivots = [];
        for (let col = 1; col <= m.cols && rank < m.rows; col++) {
            const pivotRow = Array2D._findPivotRow(m, col, rank + 1, tol);
            if (pivotRow === -1) continue;
            rank++;
            if (pivotRow !== rank) { m.swapRows(rank, pivotRow); sign = -sign; }
            const pivot = m.get(rank, col);
            pivots.push(pivot);
            for (let i = rank + 1; i <= m.rows; i++) {
                const factor = m.get(i, col) / pivot;
                if (factor !== 0) m.addScaledRow(i, rank, -factor);
            }
        }
        return { rank, sign, pivots };
    }

    /**
     * Computes the rank of this matrix (the number of linearly independent
     * rows/columns), via Gaussian elimination with partial pivoting.
     * @param {number} [tol=1e-10] - Absolute tolerance below which a pivot is treated as zero.
     * @returns {number} The rank, between `0` and `min(rows, cols)`.
     */
    rank(tol = 1e-10) {
        return this._forwardEliminate(tol).rank;
    }

    /**
     * Computes the determinant of this matrix, via Gaussian elimination with
     * partial pivoting.
     * @returns {number} The determinant.
     * @throws {RangeError} If this matrix is not square.
     */
    determinant() {
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
     * @returns {Array2D} A new matrix `M` such that `this.matmul(M)` is (up to
     *   floating-point error) the identity matrix.
     * @throws {RangeError} If this matrix is not square.
     * @throws {Error} If this matrix is singular (not invertible).
     */
    inverse() {
        if (this.rows !== this.cols) throw new RangeError(`Array2D inverse requires a square matrix, got ${this.rows}x${this.cols}`);
        const n = this.rows;
        // Augment [this | I] and row-reduce the left half to I; the right
        // half then becomes this^-1.
        const aug = new Array2D(n, 2 * n);
        for (let i = 1; i <= n; i++) {
            aug.setRow(i, this.row(i).toArray());
            aug.set(i, n + i, 1);
        }
        for (let col = 1; col <= n; col++) {
            const pivotRow = Array2D._findPivotRow(aug, col, col, 0);
            if (pivotRow === -1) throw new Error('Array2D inverse: matrix is singular');
            if (pivotRow !== col) aug.swapRows(col, pivotRow);
            aug.scaleRow(col, 1 / aug.get(col, col));
            for (let i = 1; i <= n; i++) {
                if (i === col) continue;
                const factor = aug.get(i, col);
                if (factor !== 0) aug.addScaledRow(i, col, -factor);
            }
        }
        const res = new Array2D(n, n);
        for (let i = 1; i <= n; i++) {
            for (let j = 1; j <= n; j++) res.set(i, j, aug.get(i, n + j));
        }
        return res;
    }

    /**
     * Solves the linear system `this * x = b` for `x`, via Gaussian
     * elimination with partial pivoting followed by back-substitution.
     *
     * This is the preferred way to solve a single system: it's roughly 3x
     * fewer floating-point operations than `this.inverse().mulVec(b)|matmul(B)`,
     * since it never forms the full inverse, and it shares the same
     * partial-pivoting numerical stability.
     * @param {Array1D} b - The right-hand side vector. Must have `b.dim === this.rows`.
     * @returns {Array1D} The solution vector `x` such that `this.mulVec(x)` is (up to
     *   floating-point error) equal to `b`.
     * @throws {RangeError} If this matrix is not square, or `b.dim !== this.rows`.
     * @throws {Error} If this matrix is singular (no unique solution).
     */
    solve(b) {
        if (this.rows !== this.cols) throw new RangeError(`Array2D solve requires a square matrix, got ${this.rows}x${this.cols}`);
        if (b.dim !== this.rows) throw new RangeError(`Array2D solve shape mismatch: ${this.rows}x${this.cols} vs vec(${b.dim})`);
        const n = this.rows;
        // Augment [this | b] and forward-eliminate to row echelon form,
        // then back-substitute for x - this avoids ever computing this^-1.
        const aug = new Array2D(n, n + 1);
        for (let i = 1; i <= n; i++) {
            aug.setRow(i, this.row(i).toArray());
            aug.set(i, n + 1, b.data[i - 1]);
        }
        for (let col = 1; col <= n; col++) {
            const pivotRow = Array2D._findPivotRow(aug, col, col, 0);
            if (pivotRow === -1) throw new Error('Array2D solve: matrix is singular, no unique solution');
            if (pivotRow !== col) aug.swapRows(col, pivotRow);
            const pivot = aug.get(col, col);
            for (let i = col + 1; i <= n; i++) {
                const factor = aug.get(i, col) / pivot;
                if (factor !== 0) aug.addScaledRow(i, col, -factor);
            }
        }
        const x = new Array1D(n);
        for (let i = n; i >= 1; i--) {
            let s = aug.get(i, n + 1);
            for (let j = i + 1; j <= n; j++) s -= aug.get(i, j) * x.data[j - 1];
            x.data[i - 1] = s / aug.get(i, i);
        }
        return x;
    }

    /**
     * Computes the sum of all elements of this matrix.
     * @returns {number} The sum, or `0` if the matrix is empty.
     */
    sum() {
        let s = 0;
        for (let k = 0; k < this.data.length; k++) s += this.data[k];
        return s;
    }

    /**
     * Finds the smallest element of this matrix.
     * @returns {number} The minimum value.
     * @throws {RangeError} If the matrix is empty.
     */
    min() {
        if (this.data.length === 0) throw new RangeError('Array2D min() called on an empty matrix');
        let m = this.data[0];
        for (let k = 1; k < this.data.length; k++) if (this.data[k] < m) m = this.data[k];
        return m;
    }

    /**
     * Finds the largest element of this matrix.
     * @returns {number} The maximum value.
     * @throws {RangeError} If the matrix is empty.
     */
    max() {
        if (this.data.length === 0) throw new RangeError('Array2D max() called on an empty matrix');
        let m = this.data[0];
        for (let k = 1; k < this.data.length; k++) if (this.data[k] > m) m = this.data[k];
        return m;
    }

    /**
     * Creates an independent copy of this matrix.
     * @returns {Array2D} A new Array2D with the same shape and values.
     */
    copy() { return new Array2D(this.rows, this.cols, this.data); }

    /**
     * Checks whether this matrix is elementwise close to another, modeled on
     * `numpy.isclose`: an element `a` is close to `b` if
     * `|a - b| <= atol + rtol * |b|`.
     * @param {Array2D} m - The other matrix.
     * @param {number} [rtol=1e-5] - Relative tolerance.
     * @param {number} [atol=1e-8] - Absolute tolerance.
     * @returns {boolean} `true` if `m` has the same shape and all elements of `this` are close to `m`'s.
     */
    isClose(m, rtol = 1e-5, atol = 1e-8) {
        if (m.rows !== this.rows || m.cols !== this.cols) return false;
        for (let k = 0; k < this.data.length; k++) {
            if (Math.abs(this.data[k] - m.data[k]) > atol + rtol * Math.abs(m.data[k])) return false;
        }
        return true;
    }

    /**
     * Returns this matrix's elements as an array of row arrays.
     * @returns {number[][]} An array of `rows` arrays, each with `cols` numbers.
     */
    toArray() {
        const out = [];
        for (let i = 1; i <= this.rows; i++) out.push(Array.from(this.row(i).data));
        return out;
    }

    /**
     * Returns a human-readable string representation of this matrix, one row per line.
     * @returns {string} e.g. `"Array2D[[1, 2], [3, 4]]"`.
     */
    toString() {
        const rows = this.toArray().map(r => `[${r.join(', ')}]`);
        return `Array2D[${rows.join(', ')}]`;
    }

    /**
     * Makes Array2D iterable over its rows, e.g. `for (const r of someMatrix)`.
     * Each yielded value is an Array1D.
     * @returns {Iterator<Array1D>}
     */
    *[Symbol.iterator]() {
        for (let i = 1; i <= this.rows; i++) yield this.row(i);
    }

    // -----------------------------------------------------------------
    // In-place (mutating) operations.
    // -----------------------------------------------------------------

    /**
     * Resets this matrix to all zeros in place.
     * @returns {Array2D} `this`, for chaining.
     */
    reset() {
        this.data.fill(0);
        return this;
    }

    /**
     * Adds another matrix to this one in place, elementwise: `this += m`.
     * @param {Array2D} m - The matrix to add. Must have the same shape as this one.
     * @returns {Array2D} `this`, for chaining.
     */
    addSelf(m) {
        this._checkShape(m);
        for (let k = 0; k < this.data.length; k++) this.data[k] += m.data[k];
        return this;
    }

    /**
     * Subtracts another matrix from this one in place, elementwise: `this -= m`.
     * @param {Array2D} m - The matrix to subtract. Must have the same shape as this one.
     * @returns {Array2D} `this`, for chaining.
     */
    subSelf(m) {
        this._checkShape(m);
        for (let k = 0; k < this.data.length; k++) this.data[k] -= m.data[k];
        return this;
    }

    /**
     * Scales every element of this matrix in place: `this *= s`.
     * @param {number} s - The scale factor.
     * @returns {Array2D} `this`, for chaining.
     */
    multSelf(s) {
        for (let k = 0; k < this.data.length; k++) this.data[k] *= s;
        return this;
    }

    /**
     * Transposes a square matrix in place.
     * @returns {Array2D} `this`, for chaining.
     * @throws {RangeError} If this matrix is not square (transposing a
     *   non-square matrix would change its shape; use `transpose()` instead).
     */
    transposeSelf() {
        if (this.rows !== this.cols) throw new RangeError(`Array2D transposeSelf requires a square matrix, got ${this.rows}x${this.cols}`);
        for (let i = 1; i <= this.rows; i++) {
            for (let j = i + 1; j <= this.cols; j++) {
                const tmp = this.get(i, j);
                this.set(i, j, this.get(j, i));
                this.set(j, i, tmp);
            }
        }
        return this;
    }

    /**
     * Swaps two rows in place. Useful when implementing pivoting algorithms.
     * @param {number} i - First row index (1-based).
     * @param {number} j - Second row index (1-based).
     * @returns {Array2D} `this`, for chaining.
     */
    swapRows(i, j) {
        if (i === j) return this;
        const a = this.row(i).toArray();
        this.setRow(i, this.row(j).data);
        this.setRow(j, a);
        return this;
    }

    /**
     * Scales row `i` in place by a scalar: `row[i] *= s`.
     * @param {number} i - Row index (1-based).
     * @param {number} s - The scale factor.
     * @returns {Array2D} `this`, for chaining.
     */
    scaleRow(i, s) {
        for (let j = 1; j <= this.cols; j++) this.set(i, j, this.get(i, j) * s);
        return this;
    }

    /**
     * Adds a scaled row to another row in place, in a single pass:
     * `row[i] += row[j] * s`. Useful when implementing Gaussian elimination.
     * @param {number} i - Row index to modify (1-based).
     * @param {number} j - Row index to read from and scale (1-based).
     * @param {number} s - The scale factor applied to row `j`.
     * @returns {Array2D} `this`, for chaining.
     */
    addScaledRow(i, j, s) {
        for (let k = 1; k <= this.cols; k++) this.set(i, k, this.get(i, k) + this.get(j, k) * s);
        return this;
    }

    /**
     * Creates a `rows x cols` zero matrix.
     * @param {number} rows - Number of rows.
     * @param {number} cols - Number of columns.
     * @returns {Array2D} A new zero matrix.
     */
    static zero(rows, cols) { return new Array2D(rows, cols); }

    /**
     * Creates an `n x n` identity matrix.
     * @param {number} n - The matrix dimension.
     * @returns {Array2D} A new identity matrix.
     */
    static identity(n) {
        const res = new Array2D(n, n);
        for (let i = 1; i <= n; i++) res.set(i, i, 1);
        return res;
    }

    /**
     * Creates an Array2D from an array of row arrays.
     * @param {number[][]} rows - Source data; each inner array must have the same length.
     * @returns {Array2D} A new matrix with shape `rows.length x rows[0].length`.
     */
    static from(rows) {
        const nRows = rows.length;
        const nCols = nRows > 0 ? rows[0].length : 0;
        const res = new Array2D(nRows, nCols);
        for (let i = 1; i <= nRows; i++) res.setRow(i, rows[i - 1]);
        return res;
    }
}