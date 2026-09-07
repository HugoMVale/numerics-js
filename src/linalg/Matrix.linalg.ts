import { Matrix } from './Matrix.js';
import { Vector } from './Vector.js';

// =======================================================================
// Matrix.linalg.ts
//
// Linear-algebra algorithms for Matrix, including factorizations, linear
// system solves, rank and determinant calculations, QR updates, and
// eigendecomposition. Matrix instance and static methods delegate to the
// exported functions in this module; the remaining helpers are private
// implementation details.
//
// The algorithms operate on arbitrary Matrix instances and use the
// unchecked getUnchecked, setUnchecked, and flatIndex accessors in their
// inner loops. These accessors are public for this internal cooperation,
// but callers should use Matrix's checked API instead.
// =======================================================================

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
 * `A = Q * R`.
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

// -----------------------------------------------------------------
// Gaussian elimination: shared building block for rank/determinant/
// inverse/lu.
// -----------------------------------------------------------------

/**
 * Finds the best pivot row for column `col`, searching rows `startRow..m.rows-1`.
 * Used internally by `forwardEliminate`, `inverse`, and `lu` to share the
 * same partial-pivoting (largest-magnitude-entry) selection logic.
 * @param m The matrix to search (may be a working copy or an augmented matrix).
 * @param col Column to search (0-based).
 * @param startRow First row to consider (0-based).
 * @param tol Entries with absolute value at or below this are never chosen as a pivot.
 * @returns The 0-based row index of the best pivot, or `-1` if none exceeds `tol`.
 */
function findPivotRow(m: Matrix, col: number, startRow: number, tol: number): number {
    let pivotRow = -1;
    let pivotVal = tol;
    for (let i = startRow; i < m.rows; i++) {
        const v = Math.abs(m.getUnchecked(i, col));
        if (v > pivotVal) { pivotVal = v; pivotRow = i; }
    }
    return pivotRow;
}

/**
 * Reduces a copy of `A` to row echelon form via Gaussian elimination with
 * partial pivoting (forward elimination only, no back-substitution or row
 * scaling). Shared building block for `rank()` and `determinant()`, which
 * both need the same elimination but reduce the result differently — and,
 * deliberately, use different `tol` values: `determinant()` always passes
 * `0` (matching LAPACK/numpy's exact-zero-only convention, same as
 * `lu()`), while `rank()` defaults to an auto-scaled nonzero tolerance
 * (see its own docstring), since detecting near-singularity (not just
 * exact singularity) is rank's whole job.
 * @param A The matrix to eliminate. Not mutated (a copy is worked on).
 * @param tol Absolute tolerance below which a candidate pivot is treated as zero.
 * @returns `rank` is the number of pivots found; `sign` is `+1`/`-1` tracking the row-swap
 *   parity; `pivots` are the pivot values in the order they were chosen.
 */
function forwardEliminate(A: Matrix, tol = 0): { rank: number; sign: number; pivots: number[] } {
    const m = A.copy();
    let rank = 0;
    let sign = 1;
    const pivots: number[] = [];
    for (let col = 0; col < m.cols && rank < m.rows; col++) {
        const pivotRow = findPivotRow(m, col, rank, tol);
        if (pivotRow === -1) continue;
        if (pivotRow !== rank) { m.swapRows(rank, pivotRow); sign = -sign; }
        const pivot = m.getUnchecked(rank, col);
        pivots.push(pivot);
        for (let i = rank + 1; i < m.rows; i++) {
            const factor = m.getUnchecked(i, col) / pivot;
            if (factor !== 0) m.addScaledRow(i, rank, -factor);
        }
        rank++;
    }
    return { rank, sign, pivots };
}

/**
 * The scale factor `c` in `rank()`'s default tolerance,
 * `c * max(rows, cols) * Number.EPSILON * A.normInf()`.
 */
const RANK_TOL_SCALE = 10;

/**
 * Computes the rank of `A` (the number of linearly independent
 * rows/columns), via Gaussian elimination with partial pivoting.
 *
 * This is inherently an approximation: elimination-based rank is a
 * fundamentally less numerically robust way to estimate rank than the
 * SVD-based approach production libraries like numpy actually use
 * (`numpy.linalg.matrix_rank`), which this library doesn't implement.
 * Treat this as a reasonable estimate for well-scaled matrices, not as a
 * numpy-equivalent result for pathological or ill-conditioned ones.
 * @param A The matrix to inspect.
 * @param tol Absolute tolerance below which a pivot is treated as zero. If
 * omitted, defaults to `c * max(rows, cols) * Number.EPSILON *
 * A.normInf()` with `c = 10` — the same shape of scale-relative formula
 * numpy's SVD-based `matrix_rank` uses (`S.max() * max(M, N) * eps`),
 * substituting this matrix's infinity norm for numpy's largest singular
 * value, since this library doesn't have singular values available
 * without SVD. Pass `0` explicitly for exact-zero-only pivot rejection,
 * matching `lu()`/`determinant()`'s convention.
 * @returns The rank, between `0` and `min(rows, cols)`.
 */
export function rank(A: Matrix, tol?: number): number {
    const effectiveTol = tol ?? RANK_TOL_SCALE * Math.max(A.rows, A.cols) * Number.EPSILON * A.normInf();
    return forwardEliminate(A, effectiveTol).rank;
}

/**
 * Computes the determinant of `A`, via Gaussian elimination with partial
 * pivoting. Like `lu()`, there is no tolerance parameter: a pivot only
 * counts as missing if it's *exactly* `0`, matching the convention LAPACK
 * and numpy use internally (they compute the determinant from the same
 * kind of LU factorization). A near-singular matrix will generally return
 * a small-but-nonzero float here, not `0` — if you need near-singularity
 * detection, use `rank()` instead, which is built for exactly that.
 * @param A The matrix to compute the determinant of. Must be square.
 * @returns The determinant.
 * @throws {RangeError} If `A` is not square.
 */
export function determinant(A: Matrix): number {
    if (A.rows !== A.cols) throw new RangeError(`Matrix determinant requires a square matrix, got ${A.rows}x${A.cols}`);
    const { rank: r, sign, pivots } = forwardEliminate(A, 0);
    if (r < A.rows) return 0;
    let det = sign;
    for (const p of pivots) det *= p;
    return det;
}

/**
 * Computes the inverse of `A`, via Gauss-Jordan elimination with partial
 * pivoting.
 * @param A The matrix to invert. Must be square.
 * @returns A new matrix `M` such that `A.matmul(M)` is (up to
 *   floating-point error) the identity matrix.
 * @throws {RangeError} If `A` is not square.
 * @throws {Error} If `A` is singular (not invertible).
 */
export function inverse(A: Matrix): Matrix {
    if (A.rows !== A.cols) throw new RangeError(`Matrix inverse requires a square matrix, got ${A.rows}x${A.cols}`);
    const n = A.rows;
    // Augment [A | I] and row-reduce the left half to I; the right half
    // then becomes A^-1.
    const aug = new Matrix(n, 2 * n);
    for (let i = 0; i < n; i++) {
        aug.data.set(A.data.subarray(A.flatIndex(i, 0), A.flatIndex(i, 0) + n), aug.flatIndex(i, 0));
        aug.setUnchecked(i, n + i, 1);
    }
    for (let col = 0; col < n; col++) {
        const pivotRow = findPivotRow(aug, col, col, 0);
        if (pivotRow === -1) throw new Error('Matrix inverse: matrix is singular');
        if (pivotRow !== col) aug.swapRows(col, pivotRow);
        aug.scaleRow(col, 1 / aug.getUnchecked(col, col));
        for (let i = 0; i < n; i++) {
            if (i === col) continue;
            const factor = aug.getUnchecked(i, col);
            if (factor !== 0) aug.addScaledRow(i, col, -factor);
        }
    }
    const res = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
        res.data.set(aug.data.subarray(aug.flatIndex(i, n), aug.flatIndex(i, n) + n), res.flatIndex(i, 0));
    }
    return res;
}

/**
 * Computes a partial-pivoted LU factorization of `A`, such that `P * A =
 * L * U` for the row permutation `P` implied by `perm`. Factoring once
 * and reusing the result via `solveLower`/`solveUpper` is much cheaper
 * than calling `solve()` repeatedly against the same matrix with
 * different right-hand sides (e.g. inside a Newton solver that reuses a
 * Jacobian across corrector steps).
 *
 * There is no tolerance parameter: matching the convention used by
 * LAPACK's `dgetrf` (which this is modeled on), the pivot in each column
 * is always the largest-magnitude entry available, regardless of how
 * small it is — only a pivot that is *exactly* `0` counts as missing. A
 * fuzzy "close enough to zero" threshold isn't part of plain LU pivoting
 * in reference implementations; if you need that kind of near-singularity
 * detection, use `rank()`, which does expose a tolerance for exactly this
 * purpose.
 * @param A The matrix to factor. Must be square.
 * @returns The `{ L, U, perm, sign }` factorization.
 * @throws {RangeError} If `A` is not square.
 * @throws {Error} If `A` is singular (some column's largest available
 * pivot is exactly `0`).
 */
export function lu(A: Matrix): LUDecomposition {
    if (A.rows !== A.cols) throw new RangeError(`Matrix lu requires a square matrix, got ${A.rows}x${A.cols}`);
    const n = A.rows;
    const U = A.copy();
    const L = Matrix.identity(n);
    const perm = Array.from({ length: n }, (_, i) => i);
    let sign = 1;

    for (let col = 0; col < n; col++) {
        const pivotRow = findPivotRow(U, col, col, 0);
        if (pivotRow === -1) throw new Error('Matrix lu: matrix is singular');
        if (pivotRow !== col) {
            U.swapRows(col, pivotRow);
            // Rows of L to the left of `col` already hold computed
            // multipliers; swap those along with U's rows so that
            // P * A === L * U still holds after the pivot.
            for (let k = 0; k < col; k++) {
                const tmp = L.getUnchecked(col, k);
                L.setUnchecked(col, k, L.getUnchecked(pivotRow, k));
                L.setUnchecked(pivotRow, k, tmp);
            }
            [perm[col], perm[pivotRow]] = [perm[pivotRow], perm[col]];
            sign = -sign;
        }
        const pivot = U.getUnchecked(col, col);
        for (let i = col + 1; i < n; i++) {
            const factor = U.getUnchecked(i, col) / pivot;
            if (factor !== 0) {
                U.addScaledRow(i, col, -factor);
                L.setUnchecked(i, col, factor);
            }
        }
    }
    return { L, U, perm, sign };
}

/**
 * Solves `A * x = b` for `x`, treating `A` as lower triangular: only
 * entries on and below the diagonal are read, so this can be called
 * directly on the `L` factor from `lu()`.
 * @param A The (conceptually lower-triangular) coefficient matrix. Must be square.
 * @param b The right-hand side vector. Must have `b.size === A.rows`.
 * @param unitDiagonal If `true`, the diagonal is assumed to be all 1s (as
 *   `lu()`'s `L` always is) and is never read, avoiding a division.
 * @returns The solution vector `x`.
 * @throws {RangeError} If `A` is not square, or `b.size !== A.rows`.
 * @throws {Error} If `unitDiagonal` is `false` and a zero diagonal entry is encountered.
 */
export function solveLower(A: Matrix, b: Vector, unitDiagonal = false): Vector {
    if (A.rows !== A.cols) throw new RangeError(`Matrix solveLower requires a square matrix, got ${A.rows}x${A.cols}`);
    if (b.size !== A.rows) throw new RangeError(`Matrix solveLower shape mismatch: ${A.rows}x${A.cols} vs vec(${b.size})`);
    const n = A.rows;
    const x = new Vector(n);
    for (let i = 0; i < n; i++) {
        const offset = A.flatIndex(i, 0);
        let s = b.data[i];
        for (let j = 0; j < i; j++) s -= A.data[offset + j] * x.data[j];
        if (unitDiagonal) {
            x.data[i] = s;
        } else {
            const d = A.data[offset + i];
            if (d === 0) throw new Error('Matrix solveLower: zero diagonal entry, matrix is singular');
            x.data[i] = s / d;
        }
    }
    return x;
}

/**
 * Solves `A * x = b` for `x`, treating `A` as upper triangular: only
 * entries on and above the diagonal are read, so this can be called
 * directly on the `U` factor from `lu()`.
 * @param A The (conceptually upper-triangular) coefficient matrix. Must be square.
 * @param b The right-hand side vector. Must have `b.size === A.rows`.
 * @returns The solution vector `x`.
 * @throws {RangeError} If `A` is not square, or `b.size !== A.rows`.
 * @throws {Error} If a zero diagonal entry is encountered.
 */
export function solveUpper(A: Matrix, b: Vector): Vector {
    if (A.rows !== A.cols) throw new RangeError(`Matrix solveUpper requires a square matrix, got ${A.rows}x${A.cols}`);
    if (b.size !== A.rows) throw new RangeError(`Matrix solveUpper shape mismatch: ${A.rows}x${A.cols} vs vec(${b.size})`);
    const n = A.rows;
    const x = new Vector(n);
    for (let i = n - 1; i >= 0; i--) {
        const offset = A.flatIndex(i, 0);
        let s = b.data[i];
        for (let j = i + 1; j < n; j++) s -= A.data[offset + j] * x.data[j];
        const d = A.data[offset + i];
        if (d === 0) throw new Error('Matrix solveUpper: zero diagonal entry, matrix is singular');
        x.data[i] = s / d;
    }
    return x;
}

/**
 * Solves the linear system `A * x = b` for `x`, via LU decomposition with
 * partial pivoting (`lu()`) followed by forward and back substitution. If
 * you need to solve against the same matrix with several right-hand
 * sides, call `lu()` once yourself and reuse `solveLower`/`solveUpper`
 * directly instead of calling this repeatedly.
 * @param A The coefficient matrix. Must be square.
 * @param b The right-hand side vector. Must have `b.size === A.rows`.
 * @returns The solution vector `x` such that `A.mulVec(x)` is (up to
 *   floating-point error) equal to `b`.
 * @throws {RangeError} If `A` is not square, or `b.size !== A.rows`.
 * @throws {Error} If `A` is singular (no unique solution).
 */
export function solve(A: Matrix, b: Vector): Vector {
    if (A.rows !== A.cols) throw new RangeError(`Matrix solve requires a square matrix, got ${A.rows}x${A.cols}`);
    if (b.size !== A.rows) throw new RangeError(`Matrix solve shape mismatch: ${A.rows}x${A.cols} vs vec(${b.size})`);
    const { L, U, perm } = lu(A);
    const pb = new Vector(perm.map(p => b.data[p]));
    const y = solveLower(L, pb, true);
    return solveUpper(U, y);
}

// -----------------------------------------------------------------
// Cholesky decomposition: for symmetric positive-definite matrices.
// -----------------------------------------------------------------

/**
 * Computes the Cholesky factorization of `A`, via the Cholesky–Banachiewicz
 * algorithm: `A = L * L^T`, with `L` lower triangular and positive on the
 * diagonal. `A` must be symmetric positive definite.
 *
 * Only the lower triangle of `A` is ever read; the upper triangle is
 * ignored rather than checked against it — the same "just one triangle"
 * convention `solveLower`/`solveUpper` use for their input, and the same
 * convention LAPACK's `dpotrf` uses via its `UPLO` argument. If `A` isn't
 * actually symmetric, you silently get the factorization implied by its
 * lower triangle, not an error.
 *
 * There is no tolerance parameter, matching `lu()`'s LAPACK-aligned
 * convention: a diagonal entry only counts as failing if the running sum
 * is not strictly positive, exactly as `dpotrf` checks it. This means a
 * positive-*semidefinite* matrix (e.g. `A^T * A` for rank-deficient `A`)
 * that is only barely non-positive-definite due to rounding will still be
 * rejected here; there's no fuzzy fallback, by design.
 * @param A The matrix to factor. Must be square.
 * @returns The lower-triangular factor `L` such that `A = L * L^T`.
 * @throws {RangeError} If `A` is not square.
 * @throws {Error} If `A` is not positive definite (some leading principal
 * minor is not strictly positive).
 */
export function cholesky(A: Matrix): Matrix {
    if (A.rows !== A.cols) throw new RangeError(`Matrix cholesky requires a square matrix, got ${A.rows}x${A.cols}`);
    const n = A.rows;
    const L = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
        const rowOffsetA = A.flatIndex(i, 0);
        const rowOffsetL = L.flatIndex(i, 0);
        for (let j = 0; j <= i; j++) {
            const colOffsetL = L.flatIndex(j, 0);
            let sum = A.data[rowOffsetA + j];
            for (let k = 0; k < j; k++) sum -= L.data[rowOffsetL + k] * L.data[colOffsetL + k];
            if (i === j) {
                if (sum <= 0) throw new Error(`Matrix cholesky: matrix is not positive definite (leading minor of order ${i + 1} is not positive)`);
                L.data[rowOffsetL + j] = Math.sqrt(sum);
            } else {
                L.data[rowOffsetL + j] = sum / L.data[colOffsetL + j];
            }
        }
    }
    return L;
}

/**
 * Solves `A * x = b` for `x`, given the Cholesky factor `L` of `A` (i.e.
 * `A = L * L^T`, as returned by `cholesky()`), via forward substitution
 * against `L` followed by back substitution against `L^T` — reusing
 * `solveLower`/`solveUpper` rather than duplicating their loops. As with
 * `solve()` vs. `lu()`: if you need to solve against the same matrix with
 * several right-hand sides, call `cholesky()` once and reuse this instead
 * of refactoring each time.
 * @param L The lower-triangular Cholesky factor of the coefficient
 * matrix, as returned by `cholesky()`.
 * @param b The right-hand side vector. Must have `b.size === L.rows`.
 * @returns The solution vector `x` such that `L.matmul(L.transpose()).mulVec(x)`
 * is (up to floating-point error) equal to `b`.
 * @throws {RangeError} If `L` is not square, or `b.size !== L.rows`.
 * @throws {Error} If `L` has a zero diagonal entry.
 */
export function choleskySolve(L: Matrix, b: Vector): Vector {
    if (L.rows !== L.cols) throw new RangeError(`Matrix choleskySolve requires a square matrix, got ${L.rows}x${L.cols}`);
    if (b.size !== L.rows) throw new RangeError(`Matrix choleskySolve shape mismatch: ${L.rows}x${L.cols} vs vec(${b.size})`);
    const y = solveLower(L, b, false);
    return solveUpper(L.transpose(), y);
}

// -----------------------------------------------------------------
// QR factorization and rank-1 updates.
// -----------------------------------------------------------------

/**
 * Raw Householder QR factorization: `A = Q * R`. Shared implementation
 * used both by the public `qr()` (called on the matrix directly) and
 * internally by `eigenvalues()` (called on small working submatrices
 * each shifted-QR iteration). Works for any shape, not just square.
 * @param A The matrix to factor. Not mutated.
 * @returns The `{ Q, R }` factorization, `Q` being `A.rows x A.rows`
 * and `R` being `A.rows x A.cols` with (up to floating-point cleanup)
 * zeros below the diagonal.
 */
function householderQR(A: Matrix): QRDecomposition {
    const m = A.rows;
    const n = A.cols;
    const R = A.copy();
    const Q = Matrix.identity(m);
    const v = new Float64Array(m);
    const kMax = Math.min(m - 1, n);

    for (let k = 0; k < kMax; k++) {
        let normX = 0;
        for (let i = k; i < m; i++) { const x = R.getUnchecked(i, k); normX += x * x; }
        normX = Math.sqrt(normX);
        if (normX === 0) continue;

        const x0 = R.getUnchecked(k, k);
        const alpha = x0 >= 0 ? -normX : normX;
        let vnormSq = 0;
        for (let i = k; i < m; i++) {
            const val = i === k ? x0 - alpha : R.getUnchecked(i, k);
            v[i] = val;
            vnormSq += val * val;
        }
        if (vnormSq === 0) continue;
        const vnorm = Math.sqrt(vnormSq);
        for (let i = k; i < m; i++) v[i] /= vnorm;

        // Apply the reflector I - 2vv^T to R on the left (columns k..n-1).
        for (let j = k; j < n; j++) {
            let dot = 0;
            for (let i = k; i < m; i++) dot += v[i] * R.getUnchecked(i, j);
            dot *= 2;
            if (dot === 0) continue;
            for (let i = k; i < m; i++) R.setUnchecked(i, j, R.getUnchecked(i, j) - dot * v[i]);
        }
        // Accumulate it into Q on the right: Q := Q * (I - 2vv^T).
        for (let i = 0; i < m; i++) {
            let dot = 0;
            for (let c = k; c < m; c++) dot += Q.getUnchecked(i, c) * v[c];
            dot *= 2;
            if (dot === 0) continue;
            for (let c = k; c < m; c++) Q.setUnchecked(i, c, Q.getUnchecked(i, c) - dot * v[c]);
        }
    }

    // Explicit cleanup: force exact zeros below the diagonal, matching
    // the guarantee numpy/LAPACK-based `qr()` wrappers provide (the
    // arithmetic above already drives these to ~0, up to rounding).
    for (let i = 1; i < m; i++) {
        for (let j = 0; j < Math.min(i, n); j++) R.setUnchecked(i, j, 0);
    }
    return { Q, R };
}

/**
 * Computes a Householder QR factorization of `A`: `A = Q * R`, with `Q`
 * orthogonal and `R` upper triangular (trapezoidal if `A.rows > A.cols`).
 * @param A The matrix to factor.
 * @returns The `{ Q, R }` factorization.
 */
export function qr(A: Matrix): QRDecomposition {
    return householderQR(A);
}

/**
 * Computes the stable Givens rotation `{ c, s }` (`c*c + s*s === 1`) that
 * zeros the second component when applied to the pair `[a, b]`: `[c*a +
 * s*b, -s*a + c*b] === [r, 0]`. The classic `drotg`-style formula (Golub
 * & Van Loan, *Matrix Computations*, Algorithm 5.1.3): it divides by
 * whichever of `a`/`b` is larger in magnitude rather than ever
 * squaring-then-rooting the larger one directly, so it doesn't overflow
 * for large inputs the way a naive `hypot`-then-divide computation could.
 * @param a First component.
 * @param b Second component — the one the rotation zeros.
 * @returns The rotation coefficients.
 */
function givens(a: number, b: number): { c: number; s: number } {
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
 * Left-multiplies `M` in place by a Givens rotation acting on rows `i`/`j`,
 * across columns `[colStart, colEnd)`: the pair `(row_i, row_j)` becomes
 * `(c*row_i + s*row_j, -s*row_i + c*row_j)`. Shared by both sweeps of
 * `qrUpdateSelf`, which use it to chase a Hessenberg bulge up and then
 * back down through `R`.
 * @param M The matrix to rotate, mutated in place.
 * @param i First row index.
 * @param j Second row index — the one a `givens(a, b)` pair computed from
 * `(M[i][col], M[j][col])` would zero.
 * @param c Rotation cosine.
 * @param s Rotation sine.
 * @param colStart First column to touch, inclusive.
 * @param colEnd Last column to touch, exclusive.
 */
function applyGivensRows(M: Matrix, i: number, j: number, c: number, s: number, colStart: number, colEnd: number): void {
    for (let col = colStart; col < colEnd; col++) {
        const a = M.getUnchecked(i, col);
        const b = M.getUnchecked(j, col);
        M.setUnchecked(i, col, c * a + s * b);
        M.setUnchecked(j, col, -s * a + c * b);
    }
}

/**
 * Right-multiplies `M` in place by the same Givens rotation
 * `applyGivensRows` would apply on the left, but acting on columns
 * `i`/`j` instead of rows: `(col_i, col_j)` becomes `(c*col_i + s*col_j,
 * -s*col_i + c*col_j)`. Used to accumulate each rotation from
 * `qrUpdateSelf`'s two sweeps into `Q`, the same way `householderQR`
 * accumulates its reflectors into `Q` via right-multiplication.
 * @param M The matrix to rotate, mutated in place.
 * @param i First column index.
 * @param j Second column index.
 * @param c Rotation cosine.
 * @param s Rotation sine.
 */
function applyGivensCols(M: Matrix, i: number, j: number, c: number, s: number): void {
    for (let row = 0; row < M.rows; row++) {
        const a = M.getUnchecked(row, i);
        const b = M.getUnchecked(row, j);
        M.setUnchecked(row, i, c * a + s * b);
        M.setUnchecked(row, j, -s * a + c * b);
    }
}

/**
 * Updates a QR decomposition **in place** for the rank-1 modification `A'
 * = A + u * v^T`, without refactoring `A'` from scratch —
 * TypeScript/JS analogue of `scipy.linalg.qr_update`.
 *
 * Algorithm (Gill, Golub, Murray & Saunders 1974; written up as Algorithm
 * 12.5.1 in Golub & Van Loan's *Matrix Computations*; the same approach
 * underlies LINPACK's `dchud`/`dqrup` and, ultimately,
 * `scipy.linalg.qr_update` — the standard, most-used way to do this,
 * because unlike a fresh `qr()` it doesn't cost an extra factor of
 * `min(m, n)` in the flop count):
 *
 * 1. Since `A = Q * R`, `A' = Q * (R + w * v^T)` where `w = Q^T * u`.
 * 2. A sweep of Givens rotations, applied to adjacent rows from the
 *    bottom up, collapses `w` to a multiple of `e_1` — and, applied to
 *    the same row pairs of `R`, simultaneously turns `R` into upper
 *    Hessenberg form (one nonzero sub-diagonal).
 * 3. The now-scalar leftover from `w` is folded into `R`'s first row as
 *    `+= w[0] * v^T`, which doesn't disturb the Hessenberg shape.
 * 4. A second sweep of Givens rotations, applied top-down, chases the
 *    sub-diagonal bulge off the bottom, restoring upper-triangular form.
 *
 * Both sweeps are accumulated into `Q` via right-multiplication (the same
 * trick `householderQR` uses for its reflectors), so `Q * R` keeps
 * equaling `A'` throughout. Total cost is `O(m*n)`, versus
 * `O(m*n*min(m,n))` for `A.add(u.outer(v)).qr()` from scratch — worth it
 * whenever updates are applied repeatedly, e.g. recursive least squares,
 * online/Kalman-filter-style estimation, or a quasi-Newton solver
 * refactoring its Jacobian after every step.
 *
 * For a rank-`k` update, call this `k` times in a loop, once per column
 * of `U`/`V` — the same strategy LINPACK's rank-`k` routines use
 * internally, since a rank-`k` update is just `k` rank-1 updates applied
 * in sequence.
 * @param qrDecomp An existing `{ Q, R }` factorization, as returned by
 * `qr()` (or a previous `qrUpdateSelf`/`qrUpdate` call) — `Q` must be
 * square (`m x m`) and `R` must be `m x n`. **Both `Q` and `R` are
 * mutated in place**; pass `qr()`'s freshly-created result (or your own
 * copies) if you need to keep the pre-update factorization around.
 * @param u The rank-1 update's left vector. Must have `u.size === qrDecomp.Q.rows`.
 * @param v The rank-1 update's right vector. Must have `v.size === qrDecomp.R.cols`.
 * @returns `qrDecomp` itself, mutated in place, for chaining.
 * @throws {RangeError} If `Q` isn't square, or `Q`/`R`/`u`/`v`'s shapes
 * are inconsistent with each other.
 */
export function qrUpdateSelf(qrDecomp: QRDecomposition, u: Vector, v: Vector): QRDecomposition {
    const { Q, R } = qrDecomp;
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
        for (let k = 0; k < m; k++) sum += Q.getUnchecked(k, i) * u.data[k];
        w[i] = sum;
    }

    // Phase 1: sweep bottom-to-top, collapsing w to w[0]*e_1 while
    // simultaneously reducing R to upper Hessenberg form.
    for (let k = m - 1; k >= 1; k--) {
        if (w[k] === 0) continue;
        const { c, s } = givens(w[k - 1], w[k]);
        const a = w[k - 1], b = w[k];
        w[k - 1] = c * a + s * b;
        w[k] = -s * a + c * b;
        applyGivensRows(R, k - 1, k, c, s, 0, n);
        applyGivensCols(Q, k - 1, k, c, s);
    }

    // Fold the (now scalar) update into R's first row.
    const tau = w[0];
    if (tau !== 0) {
        for (let j = 0; j < n; j++) R.setUnchecked(0, j, R.getUnchecked(0, j) + tau * v.data[j]);
    }

    // Phase 2: sweep top-to-bottom, eliminating the sub-diagonal bulge
    // introduced above and restoring upper-triangular form.
    const kMax = Math.min(m - 1, n);
    for (let k = 0; k < kMax; k++) {
        const sub = R.getUnchecked(k + 1, k);
        if (sub === 0) continue;
        const { c, s } = givens(R.getUnchecked(k, k), sub);
        applyGivensRows(R, k, k + 1, c, s, k, n);
        applyGivensCols(Q, k, k + 1, c, s);
    }

    // Explicit cleanup: force exact zeros below the diagonal, the same
    // guarantee `householderQR` provides (see its own cleanup pass).
    for (let i = 1; i < m; i++) {
        for (let j = 0; j < Math.min(i, n); j++) R.setUnchecked(i, j, 0);
    }
    return qrDecomp;
}

/**
 * Non-mutating counterpart to `qrUpdateSelf`: copies `Q` and `R` first,
 * so the factorization passed in is left untouched.
 * @param qrDecomp An existing `{ Q, R }` factorization, as returned by `qr()`.
 * @param u The rank-1 update's left vector. Must have `u.size === qrDecomp.Q.rows`.
 * @param v The rank-1 update's right vector. Must have `v.size === qrDecomp.R.cols`.
 * @returns A new `{ Q, R }` factorization such that `Q * R` equals (up to
 * floating-point error) `qrDecomp.Q.matmul(qrDecomp.R)`'s value plus the
 * outer product of `u` and `v`.
 * @throws {RangeError} Same conditions as `qrUpdateSelf`.
 */
export function qrUpdate(qrDecomp: QRDecomposition, u: Vector, v: Vector): QRDecomposition {
    return qrUpdateSelf({ Q: qrDecomp.Q.copy(), R: qrDecomp.R.copy() }, u, v);
}

// -----------------------------------------------------------------
// Eigenvalues and eigenvectors, via Hessenberg reduction + shifted-QR
// (Francis double-shift) iteration to real Schur form.
// -----------------------------------------------------------------

/**
 * Applies an orthogonal similarity transform confined to rows/columns
 * `[lo, hi]`, but correctly to the *whole* matrix: `H := Qfull^T * H *
 * Qfull` where `Qfull` is the identity everywhere except the `[lo, hi] x
 * [lo, hi]` block, which is `Qlocal`. Concretely this means
 * left-multiplying rows `[lo, hi]` (across every column) by `Qlocal^T`,
 * then right-multiplying columns `[lo, hi]` (across every row) by
 * `Qlocal` — touching the coupling entries outside the block, not just
 * the block itself. If `Q` is supplied, the same right-multiplication is
 * applied to its columns `[lo, hi]`, so it keeps accumulating the overall
 * change of basis.
 *
 * Shared by every step of `schurForm()`'s QR iteration and by its final
 * real-2x2-block cleanup pass — both are exactly this operation, just
 * with a different (locally-derived) `Qlocal`.
 */
function applySimilarity(H: Matrix, Q: Matrix | null, lo: number, hi: number, Qlocal: Matrix): void {
    const n = H.rows;
    const k = hi - lo + 1;

    // H[lo:hi+1, :] := Qlocal^T * H[lo:hi+1, :]
    const rowBuf = new Float64Array(k * n);
    for (let i = 0; i < k; i++) for (let j = 0; j < n; j++) rowBuf[i * n + j] = H.getUnchecked(lo + i, j);
    for (let i = 0; i < k; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let m = 0; m < k; m++) sum += Qlocal.getUnchecked(m, i) * rowBuf[m * n + j];
            H.setUnchecked(lo + i, j, sum);
        }
    }

    // H[:, lo:hi+1] := H[:, lo:hi+1] * Qlocal
    const colBuf = new Float64Array(n * k);
    for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) colBuf[i * k + j] = H.getUnchecked(i, lo + j);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
            let sum = 0;
            for (let m = 0; m < k; m++) sum += colBuf[i * k + m] * Qlocal.getUnchecked(m, j);
            H.setUnchecked(i, lo + j, sum);
        }
    }

    if (Q) {
        const qBuf = new Float64Array(n * k);
        for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) qBuf[i * k + j] = Q.getUnchecked(i, lo + j);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < k; j++) {
                let sum = 0;
                for (let m = 0; m < k; m++) sum += qBuf[i * k + m] * Qlocal.getUnchecked(m, j);
                Q.setUnchecked(i, lo + j, sum);
            }
        }
    }
}

/**
 * Reduces a copy of `A` (square) to upper Hessenberg form via orthogonal
 * similarity transforms (Householder reflectors applied on both sides),
 * preserving eigenvalues. Used internally by `schurForm()` as a
 * preprocessing step: it collapses each QR iteration from O(n^3) to
 * O(n^2) and gives shifted QR its usual fast convergence behavior. Not
 * exposed publicly since a Hessenberg-form result isn't useful on its own
 * without the rest of the eigenvalue pipeline.
 * @param A The matrix to reduce. Not mutated.
 * @param accumulateQ If true, also builds and returns the orthogonal
 * matrix `Q` such that `H = Q^T * A * Q` (needed by `eig()`, which has to
 * map eigenvectors back out of the Hessenberg/Schur basis; not needed by
 * `eigenvalues()`, so it's skipped there to avoid the extra O(n^3) of
 * accumulation work).
 * @returns `H`, upper Hessenberg (zero below the subdiagonal) and similar
 * to `A`; and `Q` (or `null` if not requested).
 */
function hessenberg(A: Matrix, accumulateQ = false): { H: Matrix; Q: Matrix | null } {
    const n = A.rows;
    const H = A.copy();
    const Q = accumulateQ ? Matrix.identity(n) : null;
    if (n < 3) return { H, Q };
    const v = new Float64Array(n);

    for (let k = 0; k < n - 2; k++) {
        let normX = 0;
        for (let i = k + 1; i < n; i++) { const x = H.getUnchecked(i, k); normX += x * x; }
        normX = Math.sqrt(normX);
        if (normX === 0) continue;

        const x0 = H.getUnchecked(k + 1, k);
        const alpha = x0 >= 0 ? -normX : normX;
        let vnormSq = 0;
        for (let i = k + 1; i < n; i++) {
            const val = i === k + 1 ? x0 - alpha : H.getUnchecked(i, k);
            v[i] = val;
            vnormSq += val * val;
        }
        if (vnormSq === 0) continue;
        const vnorm = Math.sqrt(vnormSq);
        for (let i = k + 1; i < n; i++) v[i] /= vnorm;

        // Left multiply: H[k+1:, :] -= 2 v (v^T H[k+1:, :]).
        for (let j = 0; j < n; j++) {
            let dot = 0;
            for (let i = k + 1; i < n; i++) dot += v[i] * H.getUnchecked(i, j);
            dot *= 2;
            if (dot === 0) continue;
            for (let i = k + 1; i < n; i++) H.setUnchecked(i, j, H.getUnchecked(i, j) - dot * v[i]);
        }
        // Right multiply (completes the similarity transform):
        // H[:, k+1:] -= 2 (H[:, k+1:] v) v^T.
        for (let i = 0; i < n; i++) {
            let dot = 0;
            for (let j = k + 1; j < n; j++) dot += H.getUnchecked(i, j) * v[j];
            dot *= 2;
            if (dot === 0) continue;
            for (let j = k + 1; j < n; j++) H.setUnchecked(i, j, H.getUnchecked(i, j) - dot * v[j]);
        }
        // Accumulate: Q := Q * Hk (same reflector, right-applied).
        if (Q) {
            for (let i = 0; i < n; i++) {
                let dot = 0;
                for (let j = k + 1; j < n; j++) dot += Q.getUnchecked(i, j) * v[j];
                dot *= 2;
                if (dot === 0) continue;
                for (let j = k + 1; j < n; j++) Q.setUnchecked(i, j, Q.getUnchecked(i, j) - dot * v[j]);
            }
        }
    }

    // Explicit cleanup: force exact zeros below the subdiagonal.
    for (let i = 2; i < n; i++) {
        for (let j = 0; j < i - 1; j++) H.setUnchecked(i, j, 0);
    }
    return { H, Q };
}

/**
 * Solves the characteristic equation of a 2x2 block `[[a, b], [c, d]]`
 * directly, in closed form. Used by `schurForm()` to finish off a
 * trailing 2x2 block once the active submatrix has shrunk that far —
 * shifted QR alone never fully deflates a block whose eigenvalues are a
 * complex-conjugate pair, so those are extracted this way instead of by
 * further iteration.
 * @returns The two eigenvalues. If the discriminant is negative, a
 * complex-conjugate pair with the positive-imaginary-part root first;
 * otherwise two reals, larger root first.
 */
function solve2x2Eigs(a: number, b: number, c: number, d: number): [Eigenvalue, Eigenvalue] {
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
function cmul(aRe: number, aIm: number, bRe: number, bIm: number): [number, number] {
    return [aRe * bRe - aIm * bIm, aRe * bIm + aIm * bRe];
}

/**
 * Complex division `a / b`, as a `[re, im]` pair. If `|b|` is smaller
 * than `floor`, `b` is treated as `floor` (real) instead of dividing by
 * (near-)zero — guards the eigenvector back-substitution below against
 * (near-)repeated eigenvalues, the same role LAPACK's dtrevc
 * safe-minimum plays.
 */
function cdiv(aRe: number, aIm: number, bRe: number, bIm: number, floor: number): [number, number] {
    let dRe = bRe, dIm = bIm;
    if (dRe * dRe + dIm * dIm < floor * floor) { dRe = floor; dIm = 0; }
    const dMagSq = dRe * dRe + dIm * dIm;
    return [(aRe * dRe + aIm * dIm) / dMagSq, (aIm * dRe - aRe * dIm) / dMagSq];
}

/**
 * Back-substitutes for the rows *above* an already-seeded eigenvector
 * block, mutating `yRe`/`yIm` in place. `T` is only *quasi*-upper
 * triangular (a 2x2 complex-conjugate block has a nonzero subdiagonal
 * entry within itself), so this can't just solve one row at a time for
 * every row above the target block: whenever it reaches an earlier block
 * that is itself a 2x2 (i.e. some *other* eigenvalue of `T` also happens
 * to be complex), that block's two rows are coupled and have to be
 * solved as one 2x2 complex linear system rather than two independent
 * rows. Every other row is an ordinary 1x1 solve.
 * @param blocks The full, `lo`-ascending block list from `schurForm()`.
 * @param blockIdx Index of the target (already-seeded) block within
 * `blocks`; only earlier entries (smaller index, smaller `lo`) are
 * processed.
 * @param hiTarget The target block's `hi` — the upper bound (inclusive)
 * of every summation below, since `y` is zero past it by construction.
 */
function backSubstitute(
    T: Matrix, blocks: [number, number][], blockIdx: number, hiTarget: number,
    lambdaRe: number, lambdaIm: number, yRe: Float64Array, yIm: Float64Array, floor: number
): void {
    for (let bi = blockIdx - 1; bi >= 0; bi--) {
        const [p, q] = blocks[bi];

        if (p === q) {
            let sumRe = 0, sumIm = 0;
            for (let j = p + 1; j <= hiTarget; j++) { sumRe += T.getUnchecked(p, j) * yRe[j]; sumIm += T.getUnchecked(p, j) * yIm[j]; }
            const dRe = T.getUnchecked(p, p) - lambdaRe, dIm = -lambdaIm;
            const [yr, yi] = cdiv(-sumRe, -sumIm, dRe, dIm, floor);
            yRe[p] = yr; yIm[p] = yi;
            continue;
        }

        // Coupled 2x2 solve for rows p and q = p+1:
        //   [ (T[p,p]-lambda)   T[p,q]           ] [y_p]   [rhs_p]
        //   [ T[q,p]            (T[q,q]-lambda)  ] [y_q] = [rhs_q]
        let rpRe = 0, rpIm = 0, rqRe = 0, rqIm = 0;
        for (let j = q + 1; j <= hiTarget; j++) {
            rpRe -= T.getUnchecked(p, j) * yRe[j]; rpIm -= T.getUnchecked(p, j) * yIm[j];
            rqRe -= T.getUnchecked(q, j) * yRe[j]; rqIm -= T.getUnchecked(q, j) * yIm[j];
        }
        const m00Re = T.getUnchecked(p, p) - lambdaRe, m00Im = -lambdaIm;
        const m01Re = T.getUnchecked(p, q);
        const m10Re = T.getUnchecked(q, p);
        const m11Re = T.getUnchecked(q, q) - lambdaRe, m11Im = -lambdaIm;

        const [detARe, detAIm] = cmul(m00Re, m00Im, m11Re, m11Im);
        const [detBRe, detBIm] = cmul(m01Re, 0, m10Re, 0);
        const detRe = detARe - detBRe, detIm = detAIm - detBIm;

        const [numPARe, numPAIm] = cmul(rpRe, rpIm, m11Re, m11Im);
        const [numPBRe, numPBIm] = cmul(m01Re, 0, rqRe, rqIm);
        const [ypRe, ypIm] = cdiv(numPARe - numPBRe, numPAIm - numPBIm, detRe, detIm, floor);

        const [numQARe, numQAIm] = cmul(m00Re, m00Im, rqRe, rqIm);
        const [numQBRe, numQBIm] = cmul(m10Re, 0, rpRe, rpIm);
        const [yqRe, yqIm] = cdiv(numQARe - numQBRe, numQAIm - numQBIm, detRe, detIm, floor);

        yRe[p] = ypRe; yIm[p] = ypIm;
        yRe[q] = yqRe; yIm[q] = yqIm;
    }
}

/**
 * Core of both `eigenvalues()` and `eig()`: reduces `A` (square) to upper
 * Hessenberg form, then runs double-shift QR iteration with deflation
 * until every diagonal block has shrunk to size 1 (a real eigenvalue) or
 * size 2 (necessarily a complex-conjugate pair — see below). The result
 * is a genuine real Schur form: `A = Q * T * Q^T` with `Q` orthogonal and
 * `T` quasi-upper-triangular.
 *
 * Each QR iteration shifts by the trace `s` and determinant `t` of the
 * active block's trailing 2x2 submatrix — both always real, even when
 * that 2x2's own eigenvalues are complex — via `M = S^2 - s*S + t*I`, `M
 * = QR`, `S := Q^T * S * Q`. This is the *explicit* form of the real
 * double-shift ("Francis") QR step LAPACK (`dhseqr`/`dlahqr`) implements
 * *implicitly* via bulge-chasing: the same shift strategy and the same
 * fast (locally cubic) convergence on both real and complex-conjugate
 * eigenvalues, but computed by forming `S^2` explicitly each iteration
 * rather than the implicit bulge-chase, trading some performance for a
 * much simpler, easier-to-verify implementation. Every 10th iteration on
 * a block that hasn't deflated, an ad hoc "exceptional shift" (a real
 * value repeated twice, following EISPACK's/Numerical Recipes' `hqr`) is
 * used in place of the trace/determinant pair, to break the rare
 * stagnation cycles that can otherwise trap any fixed-shift strategy.
 *
 * When `accumulateQ` is true, an extra cleanup pass runs at the end: any
 * size-2 block whose eigenvalues turn out to be real (rather than a
 * complex-conjugate pair) gets one further 2x2 rotation applied,
 * splitting it into two independent size-1 blocks. This isn't needed for
 * eigenvalues alone (the 2x2 closed-form solve already handles real
 * roots correctly either way), but it means `eig()` never has to
 * special-case "a 2x2 block that happens to have real eigenvalues" —
 * every returned block is either a real 1x1 or a genuinely irreducible
 * complex 2x2.
 * @param A The matrix to decompose. Not mutated.
 * @param accumulateQ Whether to build and return `Q` (needed for
 * eigenvectors; skipped for eigenvalues alone to save the O(n^3) of
 * accumulation work, similarly to `hessenberg()`'s same-named option).
 * @returns `T` (the Schur form), `Q` (or `null`), and `blocks`: the final
 * list of `[lo, hi]` diagonal block ranges, sorted by `lo` ascending —
 * i.e. top-to-bottom position in `T`, *not* the order blocks happened to
 * finish deflating in.
 * @throws {Error} If the iteration budget is exhausted before every
 * block deflates (e.g. for a pathologically slow-converging matrix).
 */
function schurForm(A: Matrix, accumulateQ: boolean, tol: number, maxIterations: number): { T: Matrix; Q: Matrix | null; blocks: [number, number][] } {
    const n = A.rows;
    const { H, Q } = hessenberg(A, accumulateQ);
    if (n === 1) return { T: H, Q, blocks: [[0, 0]] };

    // Hessenberg-form norm (sum of magnitudes on/above the diagonal, plus
    // the subdiagonal): used as a floor for the deflation test when the
    // local diagonal entries are themselves ~0, matching EISPACK's
    // `anorm` fallback.
    let anorm = 0;
    for (let i = 0; i < n; i++) {
        for (let j = Math.max(0, i - 1); j < n; j++) anorm += Math.abs(H.getUnchecked(i, j));
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

        // Look for an interior subdiagonal entry small enough to treat as
        // zero, splitting the block in two. If none is found, take one
        // shifted QR step on the whole active block and check again.
        // `its` counts iterations spent on *this* block since it was
        // last split off, purely to trigger the exceptional shift below —
        // it does not affect the shared `budget`.
        let its = 0;
        for (; ;) {
            let splitAt = -1;
            for (let k = hi; k > lo; k--) {
                const s = Math.abs(H.getUnchecked(k - 1, k - 1)) + Math.abs(H.getUnchecked(k, k));
                const threshold = tol * (s === 0 ? anorm : s);
                if (Math.abs(H.getUnchecked(k, k - 1)) <= threshold) { splitAt = k; break; }
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
                // usual trace/determinant pair, to break rare stagnation
                // cycles a fixed shift strategy can otherwise get stuck
                // in.
                const sigma = 0.75 * (Math.abs(H.getUnchecked(hi, hi - 1)) + Math.abs(H.getUnchecked(hi - 1, hi - 2)));
                s = 2 * sigma;
                t = sigma * sigma;
            } else {
                const a = H.getUnchecked(hi - 1, hi - 1), b = H.getUnchecked(hi - 1, hi);
                const c = H.getUnchecked(hi, hi - 1), d = H.getUnchecked(hi, hi);
                s = a + d;
                t = a * d - b * c;
            }

            const S = new Matrix(size, size);
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) S.setUnchecked(i, j, H.getUnchecked(lo + i, lo + j));
            }
            // M = S^2 - s*S + t*I: the real quadratic shift polynomial.
            const S2 = S.matmul(S);
            const M = new Matrix(size, size);
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    let v = S2.getUnchecked(i, j) - s * S.getUnchecked(i, j);
                    if (i === j) v += t;
                    M.setUnchecked(i, j, v);
                }
            }
            const { Q: Qlocal } = householderQR(M);
            applySimilarity(H, Q, lo, hi, Qlocal);
        }
    }

    blocks.sort((x, y) => x[0] - y[0]);
    if (!accumulateQ) return { T: H, Q, blocks };

    // Cleanup pass: split any size-2 block with real eigenvalues into two
    // size-1 blocks (see the doc comment above).
    const finalBlocks: [number, number][] = [];
    for (const [lo, hi] of blocks) {
        if (hi === lo) { finalBlocks.push([lo, hi]); continue; }

        const a = H.getUnchecked(lo, lo), b = H.getUnchecked(lo, hi), c = H.getUnchecked(hi, lo), d = H.getUnchecked(hi, hi);
        const disc = (a - d) * (a - d) + 4 * b * c;
        if (disc < 0) { finalBlocks.push([lo, hi]); continue; } // genuine complex pair: leave as-is

        // Triangularize via the eigenvector for one root: solving (block -
        // lambda*I)x = 0 gives x = [b, lambda - a] (or the symmetric
        // [lambda - d, c] if b happens to be ~0); the rotation with x/|x|
        // as its first column then triangularizes the block by
        // construction.
        const lambda = (a + d + Math.sqrt(disc)) / 2;
        let x0 = b, x1 = lambda - a;
        if (Math.abs(x0) < 1e-300 && Math.abs(x1) < 1e-300) { x0 = lambda - d; x1 = c; }
        if (Math.abs(x0) < 1e-300 && Math.abs(x1) < 1e-300) { x0 = 1; x1 = 0; }
        const norm = Math.hypot(x0, x1);
        const q0 = x0 / norm, q1 = x1 / norm;
        const Qlocal = Matrix.from([[q0, -q1], [q1, q0]]);

        applySimilarity(H, Q, lo, hi, Qlocal);
        H.setUnchecked(hi, lo, 0); // clean up floating-point noise below the diagonal
        finalBlocks.push([lo, lo]);
        finalBlocks.push([hi, hi]);
    }
    return { T: H, Q, blocks: finalBlocks };
}

/**
 * Computes the eigenvalues of `A` (square), via `schurForm()` (Hessenberg
 * reduction + double-shift QR iteration with deflation — see that
 * function's doc comment for the algorithm). If you also need
 * eigenvectors, use `eig()` instead: it computes both together from the
 * same Schur form, which is cheaper than computing eigenvalues and then
 * eigenvectors separately.
 * @param A The matrix to decompose. Must be square.
 * @param options.tol Relative tolerance for the deflation test: a
 * subdiagonal entry `H[k, k-1]` is treated as converged once
 * `abs(H[k, k-1]) <= tol * (abs(H[k-1,k-1]) + abs(H[k,k]))` (falling back
 * to `tol * norm` when that sum is `0`). Defaults to `Number.EPSILON`,
 * i.e. one unit in the last place — the same machine-precision
 * convention LAPACK (`dlamch('P')`) and EISPACK use for this test.
 * @param options.maxIterations Total shifted-QR iteration budget across
 * the whole computation (shared across all deflations, not
 * per-eigenvalue). Defaults to `30 * A.rows`, matching EISPACK's `hqr`
 * (`itn = 30*n`) and LAPACK's similar per-eigenvalue allowance.
 * @returns The `A.rows` eigenvalues, in top-to-bottom diagonal position
 * within the underlying Schur form — *not* sorted by magnitude, and
 * *not* deflation order (an early-finishing block and a later-finishing
 * one can end up adjacent either way).
 * @throws {RangeError} If `A` is not square.
 * @throws {Error} If the iteration budget is exhausted before every
 * block deflates (e.g. for a pathologically slow-converging matrix).
 */
export function eigenvalues(A: Matrix, options?: { maxIterations?: number; tol?: number }): Eigenvalue[] {
    if (A.rows !== A.cols) throw new RangeError(`Matrix eigenvalues requires a square matrix, got ${A.rows}x${A.cols}`);
    const n = A.rows;
    const tol = options?.tol ?? Number.EPSILON;
    const maxIterations = options?.maxIterations ?? 30 * n;

    const { T, blocks } = schurForm(A, false, tol, maxIterations);
    const result: Eigenvalue[] = new Array(n);
    for (const [lo, hi] of blocks) {
        if (hi === lo) {
            result[lo] = { re: T.getUnchecked(lo, lo), im: 0 };
        } else {
            const [e1, e2] = solve2x2Eigs(T.getUnchecked(lo, lo), T.getUnchecked(lo, hi), T.getUnchecked(hi, lo), T.getUnchecked(hi, hi));
            result[lo] = e1;
            result[hi] = e2;
        }
    }
    return result;
}

/**
 * Computes both the eigenvalues and eigenvectors of `A` (square): first
 * the real Schur form via `schurForm()`, then each eigenvector by
 * back-substitution on the (quasi-)upper-triangular `T` followed by
 * mapping back through `Q` (`A = Q * T * Q^T`, so a vector `y` with `T*y
 * = lambda*y` gives `Q*y` with `A*(Q*y) = lambda*(Q*y)`).
 *
 * A real matrix's eigenvector for a genuinely complex eigenvalue is
 * itself genuinely complex — that's unavoidable, not a limitation of
 * this implementation — so each pair carries `vectorRe`/`vectorIm`
 * rather than a single real `Vector`; `vectorIm` is all-zero whenever
 * `value.im` is `0`. For a complex-conjugate pair of eigenvalues, the two
 * eigenvectors are exact complex conjugates of one another, so only one
 * is computed directly and the other is derived by negating `vectorIm`
 * (and `value.im`).
 *
 * Normalization follows LAPACK's convention (`dgeev`): each eigenvector
 * is scaled to Euclidean norm 1, with its largest-magnitude component
 * rotated to be real and positive (a real eigenvector is already real,
 * so only the norm applies; its sign is otherwise whatever the
 * computation happens to produce).
 * @param A The matrix to decompose. Must be square.
 * @param options See `eigenvalues()` — same meaning and defaults.
 * @returns The `A.rows` eigenpairs, in the same order `eigenvalues()`
 * would return the values alone.
 * @throws {RangeError} If `A` is not square.
 * @throws {Error} If the iteration budget is exhausted before every
 * block deflates.
 */
export function eig(A: Matrix, options?: { maxIterations?: number; tol?: number }): Eigenpair[] {
    if (A.rows !== A.cols) throw new RangeError(`Matrix eig requires a square matrix, got ${A.rows}x${A.cols}`);
    const n = A.rows;
    const tol = options?.tol ?? Number.EPSILON;
    const maxIterations = options?.maxIterations ?? 30 * n;

    if (n === 1) {
        return [{ value: { re: A.getUnchecked(0, 0), im: 0 }, vectorRe: new Vector([1]), vectorIm: new Vector([0]) }];
    }

    const { T, Q, blocks } = schurForm(A, true, tol, maxIterations);
    const Qt = Q!; // accumulateQ was true, so Q is always populated here.

    // Floor for the back-substitution denominator, guarding against
    // (near-)repeated eigenvalues landing on the diagonal — the same role
    // LAPACK's dtrevc safe-minimum plays.
    let tnorm = 0;
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) tnorm += Math.abs(T.getUnchecked(i, j));
    const floor = Number.EPSILON * Math.max(tnorm, 1);

    const result: Eigenpair[] = new Array(n);

    for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
        const [lo, hi] = blocks[blockIdx];
        if (hi === lo) {
            // Real eigenvalue: back-substitute for a real eigenvector of
            // T, supported only on indices [0, lo] (T is upper
            // triangular, so the invariant subspace for the k-th
            // diagonal eigenvalue is spanned by the first k+1 Schur
            // basis vectors).
            const lambda = T.getUnchecked(lo, lo);
            const yRe = new Float64Array(n), yIm = new Float64Array(n);
            yRe[lo] = 1;
            backSubstitute(T, blocks, blockIdx, lo, lambda, 0, yRe, yIm, floor);

            const vRe = new Array<number>(n).fill(0);
            for (let i = 0; i < n; i++) {
                let sum = 0;
                for (let k = 0; k <= lo; k++) sum += Qt.getUnchecked(i, k) * yRe[k];
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
            const a = T.getUnchecked(lo, lo), b = T.getUnchecked(lo, hi), c = T.getUnchecked(hi, lo), d = T.getUnchecked(hi, hi);
            const [e1] = solve2x2Eigs(a, b, c, d); // e1.im > 0 by construction
            const lamRe = e1.re, lamIm = e1.im;

            // Local eigenvector of the 2x2 block: [b, lambda - a] solves
            // (block - lambda*I)v = 0 directly (or the symmetric form if
            // b happens to be ~0).
            let v0re = b, v0im = 0;
            let v1re = lamRe - a, v1im = lamIm;
            if (Math.abs(b) < floor) { v0re = lamRe - d; v0im = lamIm; v1re = c; v1im = 0; }

            const yRe = new Float64Array(n), yIm = new Float64Array(n);
            yRe[lo] = v0re; yIm[lo] = v0im;
            yRe[hi] = v1re; yIm[hi] = v1im;
            backSubstitute(T, blocks, blockIdx, hi, lamRe, lamIm, yRe, yIm, floor);

            const vRe = new Array<number>(n).fill(0), vIm = new Array<number>(n).fill(0);
            for (let i = 0; i < n; i++) {
                let sRe = 0, sIm = 0;
                for (let k = 0; k <= hi; k++) { const q = Qt.getUnchecked(i, k); sRe += q * yRe[k]; sIm += q * yIm[k]; }
                vRe[i] = sRe; vIm[i] = sIm;
            }

            // LAPACK normalization: rotate phase so the largest-magnitude
            // component is real and positive, then scale to unit
            // Euclidean norm.
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