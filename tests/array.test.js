import { describe, it, expect } from 'vitest';
import { Vec3, Array1D, Array2D } from '../src/index.js';

describe('Vec3', () => {
    it('initializes with default values or provided components', () => {
        const v1 = new Vec3();
        expect(v1.toArray()).toEqual([0, 0, 0]);

        const v2 = new Vec3(1, 2, 3);
        expect(v2.x).toBe(1);
        expect(v2.y).toBe(2);
        expect(v2.z).toBe(3);
    });

    it('performs immutable arithmetic operations', () => {
        const v1 = new Vec3(1, 2, 3);
        const v2 = new Vec3(4, 5, 6);

        const sum = v1.add(v2);
        expect(sum.toArray()).toEqual([5, 7, 9]);
        expect(v1.toArray()).toEqual([1, 2, 3]);

        const dot = v1.dot(v2);
        expect(dot).toBe(1 * 4 + 2 * 5 + 3 * 6);
    });

    it('performs mutating operations correctly', () => {
        const v = new Vec3(1, 1, 1);
        v.addSelf(new Vec3(2, 2, 2));
        expect(v.toArray()).toEqual([3, 3, 3]);
    });

    it('calculates norm and normalizes', () => {
        const v = new Vec3(0, 3, 4);
        expect(v.normSq()).toBe(25);
        expect(v.norm()).toBe(5);

        const unit = v.normalize();

        expect(unit.x).toBeCloseTo(0);
        expect(unit.y).toBeCloseTo(0.6);
        expect(unit.z).toBeCloseTo(0.8);
    });
});

describe('Array1D', () => {
    it('initializes and handles Float64Array data', () => {
        const v = new Array1D([1.5, 2.5, 3.5]);
        expect(v.dim).toBe(3);
        expect(v.sum()).toBe(7.5);
    });

    it('throws on dimension mismatch', () => {
        const v1 = new Array1D(3);
        const v2 = new Array1D(4);
        expect(() => v1.add(v2)).toThrowError(RangeError);
    });

    it('computes min and max', () => {
        const v = new Array1D([5, -1, 10, 3]);
        expect(v.min()).toBe(-1);
        expect(v.max()).toBe(10);
    });
});

describe('Array2D', () => {
    it('enforces 1-based indexing for get and set', () => {
        const m = new Array2D(2, 2, [1, 2, 3, 4]);
        expect(m.get(1, 1)).toBe(1);
        expect(m.get(1, 2)).toBe(2);
        expect(m.get(2, 1)).toBe(3);
        expect(m.get(2, 2)).toBe(4);

        expect(() => m.get(0, 1)).toThrowError(RangeError);
        expect(() => m.get(3, 1)).toThrowError(RangeError);
    });

    it('extracts rows and columns accurately', () => {
        const m = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        expect(m.row(2).toArray()).toEqual([4, 5, 6]);
        expect(m.col(2).toArray()).toEqual([2, 5]);
    });

    it('performs matrix multiplication (matmul)', () => {
        const m1 = new Array2D(2, 3, [1, 2, 3, 4, 5, 6]);
        const m2 = new Array2D(3, 2, [7, 8, 9, 10, 11, 12]);
        const result = m1.matmul(m2);

        expect(result.rows).toBe(2);
        expect(result.cols).toBe(2);
        expect(result.toArray()).toEqual([
            [58, 64],
            [139, 154]
        ]);
    });

    it('computes determinants and inverses', () => {
        const m = new Array2D(2, 2, [4, 7, 2, 6]);
        expect(m.determinant()).toBeCloseTo(10);

        const inv = m.inverse();
        const identity = m.matmul(inv);
        expect(identity.isClose(Array2D.identity(2))).toBe(true);
    });

    it('solves linear systems efficiently', () => {
        // System: 2x + y = 5, -x + y = 2  => x = 1, y = 3
        const m = new Array2D(2, 2, [2, 1, -1, 1]);
        const b = new Array1D([5, 2]);
        const x = m.solve(b);

        expect(x.isClose(new Array1D([1, 3]))).toBe(true);
    });
});