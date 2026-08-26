import { describe, it, expect } from 'vitest';
import { Array1D } from '../../src/array/array1d';

describe('Array1D', () => {
    it('initializes and handles Float64Array data', () => {
        const v: Array1D = new Array1D([1.5, 2.5, 3.5]);
        expect(v.dim).toBe(3);
        expect(v.sum()).toBe(7.5);
    });

    it('enforces 0-based indexing for component access', () => {
        const v: Array1D = new Array1D([1, 2, 3]);
        expect(v.get(0)).toBe(1);
        expect(v.get(2)).toBe(3);

        v.set(1, 20);
        expect(v.toArray()).toEqual([1, 20, 3]);
        expect(() => v.get(-1)).toThrowError(RangeError);
        expect(() => v.set(3, 0)).toThrowError(RangeError);
    });

    it('retains bulk replacement through set', () => {
        const v: Array1D = new Array1D(2);
        v.set([4, 5]);
        expect(v.toArray()).toEqual([4, 5]);
    });

    it('throws on dimension mismatch', () => {
        const v1: Array1D = new Array1D(3);
        const v2: Array1D = new Array1D(4);
        expect(() => v1.add(v2)).toThrowError(RangeError);
    });

    it('computes min and max', () => {
        const v: Array1D = new Array1D([5, -1, 10, 3]);
        expect(v.min()).toBe(-1);
        expect(v.max()).toBe(10);
    });

    it('uses a symmetric tolerance for isClose', () => {
        const smaller: Array1D = new Array1D([0.5]);
        const larger: Array1D = new Array1D([1]);

        expect(smaller.isClose(larger, 0.5, 0)).toBe(true);
        expect(larger.isClose(smaller, 0.5, 0)).toBe(true);
    });

    it('rejects invalid dimensions in the constructor', () => {
        expect(() => new Array1D(-1)).toThrowError(RangeError);
        expect(() => new Array1D(1.5)).toThrowError(RangeError);
        expect(new Array1D(0).dim).toBe(0);
    });

    it('validates length when bulk-replacing via set()', () => {
        const v: Array1D = new Array1D(3);
        expect(() => v.set([1, 2])).toThrowError(RangeError);
        expect(() => v.set([1, 2, 3, 4])).toThrowError(RangeError);
    });

    it('copy() produces an independent vector', () => {
        const v: Array1D = new Array1D([1, 2, 3]);
        const c: Array1D = v.copy();
        c.set(0, 999);
        expect(v.get(0)).toBe(1);
        expect(c.get(0)).toBe(999);
    });

    it('distinguishes equals (exact) from isClose (tolerant)', () => {
        const a: Array1D = new Array1D([1, 2, 3]);
        const b: Array1D = new Array1D([1, 2, 3.0000001]);
        expect(a.equals(a.copy())).toBe(true);
        expect(a.equals(b)).toBe(false);
        expect(a.isClose(b)).toBe(true);
        expect(a.equals(new Array1D([1, 2]))).toBe(false);
        expect(a.isClose(new Array1D([1, 2]))).toBe(false);
    });

    it('computes Euclidean distance between vectors', () => {
        const a: Array1D = new Array1D([0, 0]);
        const b: Array1D = new Array1D([3, 4]);
        expect(a.dist(b)).toBe(5);
        expect(() => a.dist(new Array1D(3))).toThrowError(RangeError);
    });

    it('limits magnitude in limit(), leaving short vectors unchanged', () => {
        const long: Array1D = new Array1D([3, 4]); // norm 5
        const limited: Array1D = long.limit(2);
        expect(limited.norm()).toBeCloseTo(2);
        expect(limited.get(0)).toBeCloseTo(1.2);
        expect(limited.get(1)).toBeCloseTo(1.6);

        const short: Array1D = new Array1D([1, 0]);
        const unchanged: Array1D = short.limit(10);
        expect(unchanged.toArray()).toEqual([1, 0]);
        expect(unchanged).not.toBe(short); // still a copy, not the same instance
        expect(() => short.limit(-1)).toThrowError(RangeError);
    });

    it('performs addSelf/subSelf/multSelf/addScaled/reset in place', () => {
        const v: Array1D = new Array1D([1, 2, 3]);
        v.subSelf(new Array1D([1, 1, 1]));
        expect(v.toArray()).toEqual([0, 1, 2]);

        v.multSelf(3);
        expect(v.toArray()).toEqual([0, 3, 6]);

        v.addScaled(new Array1D([1, 1, 1]), 2);
        expect(v.toArray()).toEqual([2, 5, 8]);

        v.reset();
        expect(v.toArray()).toEqual([0, 0, 0]);
    });

    it('sets this to a - b via subVectors without allocating a new vector', () => {
        const scratch: Array1D = new Array1D(2);
        const a: Array1D = new Array1D([5, 5]);
        const b: Array1D = new Array1D([2, 1]);
        const result = scratch.subVectors(a, b);
        expect(result).toBe(scratch); // returns this, for chaining
        expect(scratch.toArray()).toEqual([3, 4]);
    });

    it('throws when computing min/max of an empty vector', () => {
        const empty: Array1D = new Array1D(0);
        expect(() => empty.min()).toThrowError(RangeError);
        expect(() => empty.max()).toThrowError(RangeError);
    });

    it('propagates NaN in min/max instead of silently skipping it', () => {
        const v: Array1D = new Array1D([1, NaN, 3]);
        expect(Number.isNaN(v.min())).toBe(true);
        expect(Number.isNaN(v.max())).toBe(true);
    });

    it('supports iteration and destructuring', () => {
        const v: Array1D = new Array1D([7, 8, 9]);
        expect([...v]).toEqual([7, 8, 9]);
        const [a, b, c] = v;
        expect([a, b, c]).toEqual([7, 8, 9]);
    });

    it('builds vectors via static zero() and from()', () => {
        expect(Array1D.zero(3).toArray()).toEqual([0, 0, 0]);
        expect(Array1D.from([1, 2, 3]).toArray()).toEqual([1, 2, 3]);
    });
});
