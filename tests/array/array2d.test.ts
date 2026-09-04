import { describe, it, expect } from 'vitest';
import { Array1D } from '../../src/array/array1d';
import { Array2D } from '../../src/array/array2d';

describe('Array2D', () => {
    it('enforces 0-based indexing for get and set', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(m.get(0, 0)).toBe(1);
        expect(m.get(0, 1)).toBe(2);
        expect(m.get(1, 0)).toBe(3);
        expect(m.get(1, 1)).toBe(4);

        expect(() => m.get(-1, 0)).toThrowError(RangeError);
        expect(() => m.get(2, 0)).toThrowError(RangeError);
    });

    it('extracts rows and columns accurately', () => {
        const m: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        expect(m.row(1).toArray()).toEqual([4, 5, 6]);
        expect(m.col(1).toArray()).toEqual([2, 5]);
    });

    it('performs matrix multiplication (matmul)', () => {
        const m1: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        const m2: Array2D = new Array2D(3, 2, [7, 8, 9, 10, 11, 12]);
        const result: Array2D = m1.matmul(m2);

        expect(result.rows).toBe(2);
        expect(result.cols).toBe(2);
        expect(result.toArray()).toEqual([
            [58, 64],
            [139, 154]
        ]);
    });

    it('computes determinants and inverses', () => {
        const m: Array2D = new Array2D(2, 2, [4, 7, 2, 6]);
        expect(m.determinant()).toBeCloseTo(10);

        const inv: Array2D = m.inverse();
        const identity: Array2D = m.matmul(inv);
        expect(identity.allClose(Array2D.identity(2))).toBe(true);
    });

    it('solves linear systems efficiently', () => {
        // System: 2x + y = 5, -x + y = 2  => x = 1, y = 3
        const m: Array2D = new Array2D(2, 2, [2, 1, -1, 1]);
        const b: Array1D = new Array1D([5, 2]);
        const x: Array1D = m.solve(b);

        expect(x.allClose(new Array1D([1, 3]))).toBe(true);
    });

    describe('lu()', () => {
        it('factors a matrix such that P*A = L*U', () => {
            const a: Array2D = Array2D.from([
                [4, 3, 0],
                [3, 4, -1],
                [0, -1, 4],
            ]);
            const { L, U, perm } = a.lu();

            const pa = new Array2D(3, 3);
            for (let i = 0; i < 3; i++) pa.setRow(i, a.row(perm[i]));

            expect(pa.allClose(L.matmul(U))).toBe(true);
        });

        it('returns a unit-lower-triangular L and an upper-triangular U', () => {
            const a: Array2D = Array2D.from([
                [2, -1, 3],
                [4, 2, 1],
                [-2, 5, 6],
            ]);
            const { L, U } = a.lu();

            for (let i = 0; i < 3; i++) {
                expect(L.get(i, i)).toBe(1);
                for (let j = i + 1; j < 3; j++) {
                    expect(L.get(i, j)).toBeCloseTo(0);
                    expect(U.get(j, i)).toBeCloseTo(0);
                }
            }
        });

        it('needs no pivoting on an already-triangular matrix: identity perm, sign +1', () => {
            const a: Array2D = Array2D.from([
                [5, 0, 0],
                [1, 3, 0],
                [2, 1, 4],
            ]);
            const { perm, sign } = a.lu();
            expect(perm).toEqual([0, 1, 2]);
            expect(sign).toBe(1);
        });

        it('tracks the row-swap parity in sign, consistent with determinant()', () => {
            const a: Array2D = Array2D.from([
                [0, 1],
                [1, 0],
            ]); // needs exactly one pivot swap
            const { U, sign } = a.lu();
            const detFromLU = sign * U.get(0, 0) * U.get(1, 1);
            expect(detFromLU).toBeCloseTo(a.determinant());
        });

        it('throws on a singular matrix', () => {
            const singular: Array2D = new Array2D(2, 2, [1, 2, 2, 4]); // row1 = 2*row0
            expect(() => singular.lu()).toThrowError();
        });

        it('rejects non-square input', () => {
            expect(() => new Array2D(2, 3).lu()).toThrowError(RangeError);
        });

        it('handles the 1x1 case', () => {
            const a: Array2D = Array2D.from([[7]]);
            const { L, U, perm, sign } = a.lu();
            expect(L.get(0, 0)).toBe(1);
            expect(U.get(0, 0)).toBe(7);
            expect(perm).toEqual([0]);
            expect(sign).toBe(1);
        });
    });

    describe('solveLower() and solveUpper()', () => {
        it('solveLower solves a unit-lower-triangular system when unitDiagonal is true', () => {
            const l: Array2D = Array2D.from([
                [1, 0, 0],
                [2, 1, 0],
                [-1, 3, 1],
            ]);
            const b: Array1D = new Array1D([1, 4, 2]);
            const x = l.solveLower(b, true);
            expect(l.mulVec(x).allClose(b)).toBe(true);
        });

        it('solveLower divides by the diagonal when unitDiagonal is false (the default)', () => {
            const l: Array2D = Array2D.from([
                [2, 0],
                [3, 4],
            ]);
            const b: Array1D = new Array1D([4, 11]);
            const x = l.solveLower(b);
            expect(x.allClose(new Array1D([2, 1.25]))).toBe(true);
        });

        it('solveUpper solves an upper-triangular system via back-substitution', () => {
            const u: Array2D = Array2D.from([
                [2, 1, -1],
                [0, 3, 2],
                [0, 0, 4],
            ]);
            const b: Array1D = new Array1D([3, 11, 8]);
            const x = u.solveUpper(b);
            expect(u.mulVec(x).allClose(b)).toBe(true);
        });

        it('ignores entries on the wrong side of the diagonal rather than validating shape', () => {
            // solveLower only reads j <= i; entries above the diagonal are
            // simply never consulted, even if the matrix isn't actually
            // lower triangular.
            const notLower: Array2D = Array2D.from([
                [2, 99],
                [1, 3],
            ]);
            const b: Array1D = new Array1D([4, 5]);
            const x = notLower.solveLower(b);
            expect(x.get(0)).toBeCloseTo(2); // 4 / 2, the "99" above the diagonal is ignored
        });

        it('throws on a zero diagonal entry', () => {
            const u: Array2D = Array2D.from([
                [2, 1],
                [0, 0],
            ]);
            expect(() => u.solveUpper(new Array1D([1, 1]))).toThrowError();

            const l: Array2D = Array2D.from([
                [0, 0],
                [1, 2],
            ]);
            expect(() => l.solveLower(new Array1D([1, 1]))).toThrowError();
        });

        it('rejects non-square input and shape-mismatched vectors', () => {
            expect(() => new Array2D(2, 3).solveLower(new Array1D(2))).toThrowError(RangeError);
            expect(() => new Array2D(2, 3).solveUpper(new Array1D(2))).toThrowError(RangeError);

            const square: Array2D = new Array2D(2, 2, [1, 0, 1, 1]);
            expect(() => square.solveLower(new Array1D(3))).toThrowError(RangeError);
            expect(() => square.solveUpper(new Array1D(3))).toThrowError(RangeError);
        });
    });

    it('solve() composes lu(), solveLower(), and solveUpper() consistently', () => {
        // Factoring once and reusing L/U/perm directly (the pattern a Newton
        // solver would use across several right-hand sides) must agree with
        // calling solve() fresh each time.
        const a: Array2D = Array2D.from([
            [4, 3, 0],
            [3, 4, -1],
            [0, -1, 4],
        ]);
        const { L, U, perm } = a.lu();

        for (const raw of [[1, 2, 3], [0, 1, 0], [-1, -2, -3]]) {
            const b = new Array1D(raw);
            const pb = new Array1D(perm.map(p => b.get(p)));
            const reused = U.solveUpper(L.solveLower(pb, true));
            expect(reused.allClose(a.solve(b))).toBe(true);
        }
    });

    it('rejects invalid shapes and mismatched input length in the constructor', () => {
        expect(() => new Array2D(0, 2)).toThrowError(RangeError);
        expect(() => new Array2D(2, 0)).toThrowError(RangeError);
        expect(() => new Array2D(1.5, 2)).toThrowError(RangeError);
        expect(() => new Array2D(2, 2, [1, 2, 3])).toThrowError(RangeError);
    });

    it('row() returns an independent copy, not a view into the matrix', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        const r = m.row(0);
        r.set(0, 999);
        expect(m.get(0, 0)).toBe(1); // original matrix must be untouched
        expect(r.get(0)).toBe(999);
    });

    it('throws when accessing an out-of-bounds row or column', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(() => m.row(5)).toThrowError(RangeError);
        expect(() => m.col(5)).toThrowError(RangeError);
    });

    it('validates length when overwriting a row via setRow()', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(() => m.setRow(0, [1, 2, 3])).toThrowError(RangeError); // too long
        expect(() => m.setRow(0, [1])).toThrowError(RangeError); // too short
        expect(() => m.setRow(5, [1, 2])).toThrowError(RangeError); // out of bounds
        m.setRow(0, [9, 9]);
        expect(m.row(0).toArray()).toEqual([9, 9]);
    });

    it('validates length when overwriting a column via setCol()', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(() => m.setCol(0, [1, 2, 3])).toThrowError(RangeError); // too long
        expect(() => m.setCol(0, [1])).toThrowError(RangeError); // too short
        expect(() => m.setCol(5, [1, 2])).toThrowError(RangeError); // out of bounds
        m.setCol(1, [7, 8]);
        expect(m.col(1).toArray()).toEqual([7, 8]);
    });

    it('swaps rows in place and validates both indices', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        m.swapRows(0, 1);
        expect(m.toArray()).toEqual([[3, 4], [1, 2]]);

        expect(() => m.swapRows(5, 0)).toThrowError(RangeError);
        expect(() => m.swapRows(0, 5)).toThrowError(RangeError);
        expect(() => m.swapRows(5, 5)).toThrowError(RangeError); // equal, out-of-bounds indices must still throw

        const unchanged: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        unchanged.swapRows(1, 1); // same in-bounds index is a no-op
        expect(unchanged.toArray()).toEqual([[1, 2], [3, 4]]);
    });

    it('scales a row and adds a scaled row in place', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        m.scaleRow(0, 10);
        expect(m.row(0).toArray()).toEqual([10, 20]);

        m.addScaledRow(1, 0, 2); // row1 += row0 * 2
        expect(m.row(1).toArray()).toEqual([23, 44]);

        expect(() => m.scaleRow(5, 1)).toThrowError(RangeError);
        expect(() => m.addScaledRow(5, 0, 1)).toThrowError(RangeError);
        expect(() => m.addScaledRow(0, 5, 1)).toThrowError(RangeError);
    });

    it('rejects shape mismatches in elementwise and product operations', () => {
        const a: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        const b: Array2D = new Array2D(3, 3);
        expect(() => a.add(b)).toThrowError(RangeError);
        expect(() => a.sub(b)).toThrowError(RangeError);
        expect(() => a.matmul(b)).toThrowError(RangeError);
        expect(() => a.mulVec(new Array1D(3))).toThrowError(RangeError);
    });

    it('multiplies a matrix by a column vector with mulVec', () => {
        const m: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        const v: Array1D = new Array1D([1, 0, 1]);
        expect(m.mulVec(v).toArray()).toEqual([4, 10]);
    });

    it('transposes matrices, including in place for square matrices', () => {
        const m: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        expect(m.transpose().toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
        expect(() => m.transposeSelf()).toThrowError(RangeError); // not square

        const sq: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        sq.transposeSelf();
        expect(sq.toArray()).toEqual([[1, 3], [2, 4]]);
    });

    it('computes the trace of a square matrix and rejects non-square input', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(m.trace()).toBe(5);
        expect(() => new Array2D(2, 3).trace()).toThrowError(RangeError);
    });

    it('treats a singular matrix as rank-deficient with zero determinant', () => {
        const singular: Array2D = new Array2D(2, 2, [1, 2, 2, 4]); // row1 = 2*row0
        expect(singular.determinant()).toBeCloseTo(0);
        expect(singular.rank()).toBe(1);
        expect(() => singular.inverse()).toThrowError();
        expect(() => singular.solve(new Array1D([1, 1]))).toThrowError();
    });

    it("rank()'s default tolerance treats tiny floating-point noise as rank-deficient, but an explicit tol=0 does not", () => {
        // row1 is row0*2 plus a perturbation far below float64 precision at
        // this matrix's scale: mathematically singular, numerically not
        // quite. determinant() (no tolerance, matching LAPACK/numpy) still
        // reports the tiny nonzero value rather than exactly 0.
        const nearSingular: Array2D = new Array2D(2, 2, [1, 2, 2, 4 + 1e-14]);
        expect(nearSingular.rank()).toBe(1);
        expect(nearSingular.rank(0)).toBe(2);
        expect(nearSingular.determinant()).not.toBe(0);
        expect(nearSingular.determinant()).toBeCloseTo(0);
    });

    it("rank()'s default tolerance scales with the matrix's own magnitude, unlike a fixed threshold", () => {
        // Same relative near-singularity as above, but every entry scaled
        // up by 1e8. A fixed absolute tolerance (e.g. the old default of
        // 1e-10) would be swamped by the larger elimination noise at this
        // scale and misreport this as full rank; the auto-scaled default
        // (proportional to the matrix's own infinity norm) still catches it.
        const scale = 1e8;
        const scaledNearSingular: Array2D = new Array2D(2, 2, [
            scale, 2 * scale,
            2 * scale, 4 * scale + scale * 1e-14,
        ]);
        expect(scaledNearSingular.rank()).toBe(1);
        expect(scaledNearSingular.rank(1e-10)).toBe(2);
    });

    it('rejects non-square input and shape-mismatched vectors for determinant/inverse/solve', () => {
        const nonSquare: Array2D = new Array2D(2, 3);
        expect(() => nonSquare.determinant()).toThrowError(RangeError);
        expect(() => nonSquare.inverse()).toThrowError(RangeError);
        expect(() => nonSquare.solve(new Array1D(2))).toThrowError(RangeError);

        const square: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(() => square.solve(new Array1D(3))).toThrowError(RangeError);
    });

    it('copy() produces an independent matrix', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        const c = m.copy();
        c.set(0, 0, 999);
        expect(m.get(0, 0)).toBe(1);
        expect(c.get(0, 0)).toBe(999);
    });

    it('returns a per-element boolean array from isClose, in row-major order', () => {
        const a: Array2D = new Array2D(2, 2, [1, 2, 100, 4]);
        const b: Array2D = new Array2D(2, 2, [1.00001, 2.5, 100, 4]);
        expect(a.isClose(b)).toEqual([true, false, true, true]);
    });

    it('isClose: rtol scales the argument, not `this` (asymmetric)', () => {
        const smaller: Array2D = new Array2D(1, 1, [0.5]);
        const larger: Array2D = new Array2D(1, 1, [1]);
        expect(smaller.isClose(larger, 0.5, 0)).toEqual([true]);
        expect(larger.isClose(smaller, 0.5, 0)).toEqual([false]);
    });

    it('throws on shape mismatch in isClose/allClose, rather than returning false', () => {
        const a: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(() => a.isClose(new Array2D(3, 3))).toThrowError(RangeError);
        expect(() => a.allClose(new Array2D(3, 3))).toThrowError(RangeError);
    });

    it('never considers NaN close to anything, including another NaN', () => {
        const a: Array2D = new Array2D(1, 1, [NaN]);
        const b: Array2D = new Array2D(1, 1, [NaN]);
        expect(a.isClose(b)).toEqual([false]);
        expect(a.allClose(b)).toBe(false);
    });

    it('allClose is true only when every element is close', () => {
        const a: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        const b: Array2D = new Array2D(2, 2, [1, 2, 3, 4.0000001]);
        const c: Array2D = new Array2D(2, 2, [1, 2, 3, 5]);
        expect(a.allClose(a.copy())).toBe(true);
        expect(a.allClose(b)).toBe(true);
        expect(a.allClose(c)).toBe(false);
    });

    it('computes sum/mean/min/max/variance/std over all elements when axis is omitted', () => {
        const m: Array2D = new Array2D(2, 2, [1, -2, 3, 4]);
        expect(m.sum()).toBe(6);
        expect(m.mean()).toBe(1.5);
        expect(m.min()).toBe(-2);
        expect(m.max()).toBe(4);
        expect(m.variance()).toBeCloseTo(5.25);
        expect(m.std()).toBeCloseTo(Math.sqrt(5.25));
    });

    it('propagates NaN in min()/max(), matching Math.min/Math.max semantics', () => {
        const m: Array2D = new Array2D(2, 2, [1, NaN, 3, 4]);
        expect(Number.isNaN(m.min())).toBe(true);
        expect(Number.isNaN(m.max())).toBe(true);
    });

    it('reduces along axis 0 (down each column) and axis 1 (across each row)', () => {
        const m: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        expect(m.sum(0).toArray()).toEqual([5, 7, 9]);
        expect(m.sum(1).toArray()).toEqual([6, 15]);

        expect(m.mean(0).toArray()).toEqual([2.5, 3.5, 4.5]);
        expect(m.mean(1).toArray()).toEqual([2, 5]);

        expect(m.min(0).toArray()).toEqual([1, 2, 3]);
        expect(m.min(1).toArray()).toEqual([1, 4]);

        expect(m.max(0).toArray()).toEqual([4, 5, 6]);
        expect(m.max(1).toArray()).toEqual([3, 6]);
    });

    it('computes variance/std per axis, with ddof forwarded correctly', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 5]);
        // Column 0: [1, 3], column 1: [2, 5]
        expect(m.variance(0).toArray()).toEqual([1, 2.25]);
        expect(m.variance(0, 1).toArray()).toEqual([2, 4.5]);
        expect(m.std(0).toArray()).toEqual([1, 1.5]);
    });

    it('cumsum flattens in row-major order when axis is omitted', () => {
        const m: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        expect(m.cumsum().toArray()).toEqual([1, 3, 6, 10, 15, 21]);
    });

    it('cumsum accumulates independently down each column (axis 0) or across each row (axis 1)', () => {
        const m: Array2D = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        expect(m.cumsum(0).toArray()).toEqual([
            [1, 2, 3],
            [5, 7, 9],
        ]);
        expect(m.cumsum(1).toArray()).toEqual([
            [1, 3, 6],
            [4, 9, 15],
        ]);
    });

    it('argmin/argmax return a flat index when axis is omitted', () => {
        const m: Array2D = new Array2D(2, 3, [5, 1, 4, 2, 8, 0]);
        expect(m.argmin()).toBe(5); // value 0 at flat index 5
        expect(m.argmax()).toBe(4); // value 8 at flat index 4
    });

    it('argmin/argmax return the row index per column (axis 0) or column index per row (axis 1)', () => {
        const m: Array2D = new Array2D(2, 3, [5, 1, 4, 2, 8, 0]);
        // Columns: [5,2], [1,8], [4,0]
        expect(m.argmin(0).toArray()).toEqual([1, 0, 1]);
        expect(m.argmax(0).toArray()).toEqual([0, 1, 0]);
        // Rows: [5,1,4], [2,8,0]
        expect(m.argmin(1).toArray()).toEqual([1, 2]);
        expect(m.argmax(1).toArray()).toEqual([0, 1]);
    });

    it('argmin/argmax return the first index on ties, and prioritize NaN, in every axis mode', () => {
        const ties: Array2D = new Array2D(2, 2, [3, 3, 3, 3]);
        expect(ties.argmin()).toBe(0);
        expect(ties.argmin(0).toArray()).toEqual([0, 0]);
        expect(ties.argmin(1).toArray()).toEqual([0, 0]);

        const withNaN: Array2D = new Array2D(2, 2, [1, NaN, 3, 4]);
        expect(withNaN.argmin()).toBe(1);
        expect(withNaN.argmin(0).toArray()).toEqual([0, 0]); // col 0 = [1, 3] -> index 0; col 1 = [NaN, 4] -> index 0
        expect(withNaN.argmin(1).toArray()).toEqual([1, 0]); // row 0 = [1, NaN] -> index 1; row 1 = [3, 4] -> index 0
    });

    it('sort() defaults to axis 1, sorting each row independently', () => {
        const m: Array2D = new Array2D(2, 3, [5, 6, 4, 1, 2, 3]);
        expect(m.sort().toArray()).toEqual([
            [4, 5, 6],
            [1, 2, 3],
        ]);
    });

    it('sort(0) sorts each column independently', () => {
        const m: Array2D = new Array2D(2, 3, [5, 6, 4, 1, 2, 3]);
        expect(m.sort(0).toArray()).toEqual([
            [1, 2, 3],
            [5, 6, 4],
        ]);
    });

    it('sort() does not mutate the original matrix', () => {
        const m: Array2D = new Array2D(2, 2, [4, 3, 2, 1]);
        const sorted = m.sort();
        expect(m.toArray()).toEqual([[4, 3], [2, 1]]);
        expect(sorted.toArray()).toEqual([[3, 4], [1, 2]]);
    });

    it('sort() accepts a custom comparator', () => {
        const m: Array2D = new Array2D(2, 3, [5, 6, 4, 1, 2, 3]);
        expect(m.sort(1, (a, b) => b - a).toArray()).toEqual([
            [6, 5, 4],
            [3, 2, 1],
        ]);
    });

    it('slices a sub-matrix with Array.prototype.slice semantics, per axis', () => {
        const m: Array2D = new Array2D(3, 4, [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
        ]);
        expect(m.slice(1, 3, 1, 3).toArray()).toEqual([
            [5, 6],
            [9, 10],
        ]);
        expect(m.slice(1).toArray()).toEqual([
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]);
        expect(m.slice(undefined, undefined, 2).toArray()).toEqual([
            [2, 3],
            [6, 7],
            [10, 11],
        ]);
        expect(m.slice().toArray()).toEqual(m.toArray());
    });

    it('slice() supports negative indices, counting back from the end, per axis', () => {
        const m: Array2D = new Array2D(3, 3, [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
        ]);
        expect(m.slice(-2).toArray()).toEqual([
            [3, 4, 5],
            [6, 7, 8],
        ]);
        expect(m.slice(undefined, undefined, -2).toArray()).toEqual([
            [1, 2],
            [4, 5],
            [7, 8],
        ]);
    });

    it('slice() returns an independent copy', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        const s = m.slice(0, 1, 0, 1);
        s.set(0, 0, 999);
        expect(m.get(0, 0)).toBe(1);
    });

    it('slice() throws if the resolved range is empty on either axis', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(() => m.slice(1, 1)).toThrowError(RangeError);
        expect(() => m.slice(undefined, undefined, 1, 1)).toThrowError(RangeError);
        expect(() => m.slice(5, 6)).toThrowError(RangeError);
    });

    it('multiplies and divides elementwise via mult/div with a matrix argument (Hadamard)', () => {
        const a: Array2D = new Array2D(2, 2, [2, 3, 4, 6]);
        const b: Array2D = new Array2D(2, 2, [1, 2, 4, 3]);
        expect(a.mult(b).toArray()).toEqual([[2, 6], [16, 18]]);
        expect(a.div(b).toArray()).toEqual([[2, 1.5], [1, 2]]);
        expect(() => a.mult(new Array2D(3, 3))).toThrowError(RangeError);
        expect(() => a.div(new Array2D(3, 3))).toThrowError(RangeError);
    });

    it('accepts a scalar in add/sub, matching Array1D', () => {
        const a: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(a.add(10).toArray()).toEqual([[11, 12], [13, 14]]);
        expect(a.sub(1).toArray()).toEqual([[0, 1], [2, 3]]);
    });

    it('divides by zero following standard float semantics', () => {
        const a: Array2D = new Array2D(1, 3, [1, -1, 0]);
        const zero: Array2D = new Array2D(1, 3);
        const result = a.div(zero).toArray()[0];
        expect(result[0]).toBe(Infinity);
        expect(result[1]).toBe(-Infinity);
        expect(Number.isNaN(result[2])).toBe(true);
    });

    it('performs addSelf/subSelf/multSelf/divSelf/fill in place', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        m.addSelf(new Array2D(2, 2, [1, 1, 1, 1]));
        expect(m.toArray()).toEqual([[2, 3], [4, 5]]);

        m.subSelf(new Array2D(2, 2, [1, 1, 1, 1]));
        m.multSelf(10);
        expect(m.toArray()).toEqual([[10, 20], [30, 40]]);

        m.divSelf(new Array2D(2, 2, [2, 2, 2, 2]));
        expect(m.toArray()).toEqual([[5, 10], [15, 20]]);

        m.fill(0);
        expect(m.toArray()).toEqual([[0, 0], [0, 0]]);
    });

    it('computes the Frobenius norm, inner product, and distance (inherited from ArrayND)', () => {
        const a: Array2D = new Array2D(2, 2, [3, 0, 0, 4]);
        expect(a.norm()).toBe(5);
        expect(a.normSq()).toBe(25);

        const b: Array2D = new Array2D(2, 2, [1, 0, 0, 1]);
        expect(a.dot(b)).toBe(7); // 3*1 + 0*0 + 0*0 + 4*1
        expect(() => a.dot(new Array2D(3, 3))).toThrowError(RangeError);

        const zero: Array2D = new Array2D(2, 2);
        expect(zero.dist(a)).toBe(5);
    });

    it('builds matrices via static zero(), identity(), and from()', () => {
        expect(Array2D.zero(2, 2).toArray()).toEqual([[0, 0], [0, 0]]);
        expect(Array2D.identity(2).toArray()).toEqual([[1, 0], [0, 1]]);
        expect(Array2D.from([[1, 2], [3, 4]]).toArray()).toEqual([[1, 2], [3, 4]]);
    });

    it('rejects empty input and ragged rows in Array2D.from()', () => {
        expect(() => Array2D.from([])).toThrowError(RangeError);
        expect(() => Array2D.from([[]])).toThrowError(RangeError);
        expect(() => Array2D.from([[1, 2], [3]])).toThrowError(RangeError); // ragged row
    });

    it('is iterable over its rows', () => {
        const m: Array2D = new Array2D(2, 2, [1, 2, 3, 4]);
        const rows = [...m].map(r => r.toArray());
        expect(rows).toEqual([[1, 2], [3, 4]]);
    });
});