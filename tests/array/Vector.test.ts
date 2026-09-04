import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/array/Vector.js';

describe('Vector', () => {
    it('initializes and handles Float64Array data', () => {
        const v: Vector = new Vector([1.5, 2.5, 3.5]);
        expect(v.size).toBe(3);
        expect(v.sum()).toBe(7.5);
    });

    it('enforces 0-based indexing for component access', () => {
        const v: Vector = new Vector([1, 2, 3]);
        expect(v.get(0)).toBe(1);
        expect(v.get(2)).toBe(3);

        v.set(1, 20);
        expect(v.toArray()).toEqual([1, 20, 3]);
        expect(() => v.get(-1)).toThrowError(RangeError);
        expect(() => v.set(3, 0)).toThrowError(RangeError);
    });

    it('retains bulk replacement through set', () => {
        const v: Vector = new Vector(2);
        v.set([4, 5]);
        expect(v.toArray()).toEqual([4, 5]);
    });

    it('throws on dimension mismatch', () => {
        const v1: Vector = new Vector(3);
        const v2: Vector = new Vector(4);
        expect(() => v1.add(v2)).toThrowError(RangeError);
    });

    it('computes min and max', () => {
        const v: Vector = new Vector([5, -1, 10, 3]);
        expect(v.min()).toBe(-1);
        expect(v.max()).toBe(10);
    });

    it('isClose: rtol scales the argument, not `this` (asymmetric)', () => {
        // |0.5 - 1| = 0.5. atol=0, rtol=0.5.
        // smaller.isClose(larger): threshold = 0.5 * |1|   = 0.5 -> close.
        // larger.isClose(smaller): threshold = 0.5 * |0.5| = 0.25 -> not close.
        const smaller: Vector = new Vector([0.5]);
        const larger: Vector = new Vector([1]);

        expect(smaller.isClose(larger, 0.5, 0)).toEqual([true]);
        expect(larger.isClose(smaller, 0.5, 0)).toEqual([false]);
    });

    it('returns a per-component boolean array from isClose', () => {
        const a: Vector = new Vector([1, 2, 100]);
        const b: Vector = new Vector([1.00001, 2.5, 100]);
        expect(a.isClose(b)).toEqual([true, false, true]);
    });

    it('throws on dimension mismatch in isClose', () => {
        const a: Vector = new Vector(3);
        const b: Vector = new Vector(4);
        expect(() => a.isClose(b)).toThrowError(RangeError);
    });

    it('never considers NaN close to anything, including another NaN', () => {
        const a: Vector = new Vector([NaN]);
        const b: Vector = new Vector([NaN]);
        expect(a.isClose(b)).toEqual([false]);
        expect(a.allClose(b)).toBe(false);
    });

    it('allClose is true only when every component is close', () => {
        const a: Vector = new Vector([1, 2, 3]);
        const b: Vector = new Vector([1, 2, 3.0000001]);
        const c: Vector = new Vector([1, 2, 4]);

        expect(a.allClose(b)).toBe(true);
        expect(a.allClose(c)).toBe(false);
        expect(() => a.allClose(new Vector(2))).toThrowError(RangeError);
    });

    it('rejects invalid dimensions in the constructor', () => {
        expect(() => new Vector(-1)).toThrowError(RangeError);
        expect(() => new Vector(1.5)).toThrowError(RangeError);
        expect(new Vector(0).size).toBe(0);
    });

    it('validates length when bulk-replacing via set()', () => {
        const v: Vector = new Vector(3);
        expect(() => v.set([1, 2])).toThrowError(RangeError);
        expect(() => v.set([1, 2, 3, 4])).toThrowError(RangeError);
    });

    it('copy() produces an independent vector', () => {
        const v: Vector = new Vector([1, 2, 3]);
        const c: Vector = v.copy();
        c.set(0, 999);
        expect(v.get(0)).toBe(1);
        expect(c.get(0)).toBe(999);
    });

    it('allClose treats near-equal vectors as close but not identical', () => {
        const a: Vector = new Vector([1, 2, 3]);
        const b: Vector = new Vector([1, 2, 3.0000001]);
        expect(a.allClose(a.copy())).toBe(true);
        expect(a.allClose(b)).toBe(true);
        expect(() => a.allClose(new Vector([1, 2]))).toThrowError(RangeError);
    });

    it('computes Euclidean distance between vectors', () => {
        const a: Vector = new Vector([0, 0]);
        const b: Vector = new Vector([3, 4]);
        expect(a.dist(b)).toBe(5);
        expect(() => a.dist(new Vector(3))).toThrowError(RangeError);
    });

    it('performs addSelf/subSelf/multSelf/addScaled/fill in place', () => {
        const v: Vector = new Vector([1, 2, 3]);
        v.subSelf(new Vector([1, 1, 1]));
        expect(v.toArray()).toEqual([0, 1, 2]);

        v.multSelf(3);
        expect(v.toArray()).toEqual([0, 3, 6]);

        v.addScaled(new Vector([1, 1, 1]), 2);
        expect(v.toArray()).toEqual([2, 5, 8]);

        v.fill(0);
        expect(v.toArray()).toEqual([0, 0, 0]);
    });

    it('fills every component with a given value in place, and returns this', () => {
        const v: Vector = new Vector(3);
        const result = v.fill(7);
        expect(result).toBe(v);
        expect(v.toArray()).toEqual([7, 7, 7]);
    });

    it('sets this to a - b via subVectors without allocating a new vector', () => {
        const scratch: Vector = new Vector(2);
        const a: Vector = new Vector([5, 5]);
        const b: Vector = new Vector([2, 1]);
        const result = scratch.subVectors(a, b);
        expect(result).toBe(scratch); // returns this, for chaining
        expect(scratch.toArray()).toEqual([3, 4]);
    });

    it('throws when computing min/max of an empty vector', () => {
        const empty: Vector = new Vector(0);
        expect(() => empty.min()).toThrowError(RangeError);
        expect(() => empty.max()).toThrowError(RangeError);
    });

    it('propagates NaN in min/max instead of silently skipping it', () => {
        const v: Vector = new Vector([1, NaN, 3]);
        expect(Number.isNaN(v.min())).toBe(true);
        expect(Number.isNaN(v.max())).toBe(true);
    });

    it('supports iteration and destructuring', () => {
        const v: Vector = new Vector([7, 8, 9]);
        expect([...v]).toEqual([7, 8, 9]);
        const [a, b, c] = v;
        expect([a, b, c]).toEqual([7, 8, 9]);
    });

    it('builds vectors via static zero() and from()', () => {
        expect(Vector.zero(3).toArray()).toEqual([0, 0, 0]);
        expect(Vector.from([1, 2, 3]).toArray()).toEqual([1, 2, 3]);
    });

    it('computes the arithmetic mean', () => {
        const v: Vector = new Vector([1, 2, 3, 4]);
        expect(v.mean()).toBe(2.5);
    });

    it('returns NaN for the mean of an empty vector', () => {
        const empty: Vector = new Vector(0);
        expect(Number.isNaN(empty.mean())).toBe(true);
    });

    it('builds evenly spaced samples via linspace()', () => {
        const v: Vector = Vector.linspace(0, 10, 5);
        expect(v.toArray()).toEqual([0, 2.5, 5, 7.5, 10]);
    });

    it('excludes the stop value in linspace() when endpoint is false', () => {
        const v: Vector = Vector.linspace(0, 10, 5, false);
        expect(v.toArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('handles edge cases of num=0 and num=1 in linspace()', () => {
        expect(Vector.linspace(0, 10, 0).size).toBe(0);
        expect(Vector.linspace(3, 99, 1).toArray()).toEqual([3]);
    });

    it('defaults linspace() num to 50', () => {
        expect(Vector.linspace(0, 1).size).toBe(50);
    });

    it('rejects an invalid num in linspace()', () => {
        expect(() => Vector.linspace(0, 1, -1)).toThrowError(RangeError);
        expect(() => Vector.linspace(0, 1, 1.5)).toThrowError(RangeError);
    });

    it('builds log-spaced samples via logspace()', () => {
        const v: Vector = Vector.logspace(0, 3, 4);
        expect(v.toArray()).toEqual([1, 10, 100, 1000]);
    });

    it('supports a custom base in logspace()', () => {
        const v: Vector = Vector.logspace(0, 3, 4, true, 2);
        expect(v.toArray()).toEqual([1, 2, 4, 8]);
    });

    it('defaults logspace() num to 50 and respects endpoint', () => {
        expect(Vector.logspace(0, 1).size).toBe(50);
        const v: Vector = Vector.logspace(0, 3, 3, false);
        expect(v.isClose(new Vector([1, 10, 100]))).toEqual([true, true, true]);
    });

    it('multiplies and divides elementwise via mult/div with a vector argument', () => {
        const a: Vector = new Vector([2, 3, 4]);
        const b: Vector = new Vector([1, 2, 4]);
        expect(a.mult(b).toArray()).toEqual([2, 6, 16]);
        expect(a.div(b).toArray()).toEqual([2, 1.5, 1]);
        expect(() => a.mult(new Vector(2))).toThrowError(RangeError);
        expect(() => a.div(new Vector(2))).toThrowError(RangeError);
    });

    it('divides by zero following standard float semantics', () => {
        const a: Vector = new Vector([1, -1, 0]);
        const zero: Vector = new Vector([0, 0, 0]);
        const result = a.div(zero).toArray();
        expect(result[0]).toBe(Infinity);
        expect(result[1]).toBe(-Infinity);
        expect(Number.isNaN(result[2])).toBe(true);
    });

    it('add/sub/mult/div all accept either a vector or a scalar', () => {
        const a: Vector = new Vector([2, 4, 6]);
        const b: Vector = new Vector([1, 1, 1]);

        expect(a.add(b).toArray()).toEqual([3, 5, 7]);
        expect(a.add(10).toArray()).toEqual([12, 14, 16]);

        expect(a.sub(b).toArray()).toEqual([1, 3, 5]);
        expect(a.sub(1).toArray()).toEqual([1, 3, 5]);

        expect(a.mult(b).toArray()).toEqual([2, 4, 6]);
        expect(a.mult(2).toArray()).toEqual([4, 8, 12]);

        expect(a.div(b).toArray()).toEqual([2, 4, 6]);
        expect(a.div(2).toArray()).toEqual([1, 2, 3]);

        // none of these mutate the original
        expect(a.toArray()).toEqual([2, 4, 6]);
    });

    it('addSelf/subSelf/multSelf/divSelf accept either a vector or a scalar, in place', () => {
        const v: Vector = new Vector([2, 4, 6]);

        let result = v.addSelf(1);
        expect(result).toBe(v);
        expect(v.toArray()).toEqual([3, 5, 7]);

        result = v.addSelf(new Vector([1, 1, 1]));
        expect(v.toArray()).toEqual([4, 6, 8]);

        result = v.subSelf(2);
        expect(v.toArray()).toEqual([2, 4, 6]);

        result = v.subSelf(new Vector([1, 1, 1]));
        expect(v.toArray()).toEqual([1, 3, 5]);

        result = v.multSelf(2);
        expect(v.toArray()).toEqual([2, 6, 10]);

        result = v.multSelf(new Vector([1, 2, 1]));
        expect(v.toArray()).toEqual([2, 12, 10]);

        result = v.divSelf(2);
        expect(v.toArray()).toEqual([1, 6, 5]);

        result = v.divSelf(new Vector([1, 2, 1]));
        expect(v.toArray()).toEqual([1, 3, 5]);
        expect(result).toBe(v);
    });

    it('throws a dimension-mismatch RangeError from the in-place vector-argument overloads', () => {
        const v: Vector = new Vector(3);
        const wrongSize: Vector = new Vector(2);
        expect(() => v.addSelf(wrongSize)).toThrowError(RangeError);
        expect(() => v.subSelf(wrongSize)).toThrowError(RangeError);
        expect(() => v.multSelf(wrongSize)).toThrowError(RangeError);
        expect(() => v.divSelf(wrongSize)).toThrowError(RangeError);
    });

    it('computes elementwise abs, pow, and sqrt', () => {
        const v: Vector = new Vector([-4, 9, -1]);
        expect(v.abs().toArray()).toEqual([4, 9, 1]);
        expect(new Vector([2, 3]).pow(3).toArray()).toEqual([8, 27]);
        expect(new Vector([4, 9]).sqrt().toArray()).toEqual([2, 3]);
        expect(Number.isNaN(new Vector([-1]).sqrt().get(0))).toBe(true);
    });

    it('clips components to a range, elementwise', () => {
        const v: Vector = new Vector([-5, 0, 5, 10]);
        expect(v.clip(0, 5).toArray()).toEqual([0, 0, 5, 5]);
        expect(() => v.clip(5, 0)).toThrowError(RangeError);
    });

    it('finds argmin and argmax, breaking ties at the first occurrence', () => {
        const v: Vector = new Vector([3, -1, 5, -1, 5]);
        expect(v.argmin()).toBe(1);
        expect(v.argmax()).toBe(2);
    });

    it('throws when computing argmin/argmax of an empty vector', () => {
        const empty: Vector = new Vector(0);
        expect(() => empty.argmin()).toThrowError(RangeError);
        expect(() => empty.argmax()).toThrowError(RangeError);
    });

    it('propagates NaN in argmin/argmax, pointing at its index', () => {
        const v: Vector = new Vector([1, NaN, 3]);
        expect(v.argmin()).toBe(1);
        expect(v.argmax()).toBe(1);

        const leadingNaN: Vector = new Vector([NaN, 1, 2]);
        expect(leadingNaN.argmin()).toBe(0);
        expect(leadingNaN.argmax()).toBe(0);
    });

    it('computes population variance and std by default (ddof=0)', () => {
        const v: Vector = new Vector([2, 4, 4, 4, 5, 5, 7, 9]);
        expect(v.variance()).toBeCloseTo(4);
        expect(v.std()).toBeCloseTo(2);
    });

    it('supports sample variance and std via ddof', () => {
        const v: Vector = new Vector([2, 4, 4, 4, 5, 5, 7, 9]);
        expect(v.variance(1)).toBeCloseTo(4.571428571);
        expect(v.std(1)).toBeCloseTo(Math.sqrt(4.571428571));
    });

    it('computes the cumulative sum', () => {
        const v: Vector = new Vector([1, 2, 3, 4]);
        expect(v.cumsum().toArray()).toEqual([1, 3, 6, 10]);
        expect(new Vector(0).cumsum().toArray()).toEqual([]);
    });

    it('builds vectors via static ones() and full()', () => {
        expect(Vector.ones(3).toArray()).toEqual([1, 1, 1]);
        expect(Vector.full(3, 9).toArray()).toEqual([9, 9, 9]);
    });

    it('sorts into a new vector, ascending numeric by default, leaving the original unchanged', () => {
        const v: Vector = new Vector([10, 2, 33, 4]);
        const sorted = v.sort();
        expect(sorted.toArray()).toEqual([2, 4, 10, 33]);
        expect(v.toArray()).toEqual([10, 2, 33, 4]); // unchanged
        expect(sorted).not.toBe(v);
    });

    it('sorts with a custom comparator', () => {
        const v: Vector = new Vector([1, 2, 3, 4]);
        const descending = v.sort((a, b) => b - a);
        expect(descending.toArray()).toEqual([4, 3, 2, 1]);
    });

    it('slices a sub-vector with Array.prototype.slice semantics', () => {
        const v: Vector = new Vector([0, 1, 2, 3, 4]);
        expect(v.slice(1, 3).toArray()).toEqual([1, 2]);
        expect(v.slice(2).toArray()).toEqual([2, 3, 4]);
        expect(v.slice().toArray()).toEqual([0, 1, 2, 3, 4]);
        expect(v.slice(-2).toArray()).toEqual([3, 4]);
    });

    it('slice() returns an independent copy', () => {
        const v: Vector = new Vector([0, 1, 2]);
        const s = v.slice(0, 2);
        s.set(0, 999);
        expect(v.get(0)).toBe(0);
    });

    it('maps each component with its index via map()', () => {
        const v: Vector = new Vector([1, 2, 3]);
        expect(v.map(x => x * 10).toArray()).toEqual([10, 20, 30]);
        expect(v.map((x, i) => x + i).toArray()).toEqual([1, 3, 5]);
    });

    it('builds a range via arange(stop)', () => {
        expect(Vector.arange(5).toArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('builds a range via arange(start, stop)', () => {
        expect(Vector.arange(2, 7).toArray()).toEqual([2, 3, 4, 5, 6]);
    });

    it('builds a range via arange(start, stop, step), including negative steps', () => {
        expect(Vector.arange(0, 10, 2).toArray()).toEqual([0, 2, 4, 6, 8]);
        expect(Vector.arange(5, 0, -1).toArray()).toEqual([5, 4, 3, 2, 1]);
    });

    it('returns an empty vector from arange() when the range is empty', () => {
        expect(Vector.arange(5, 5).size).toBe(0);
        expect(Vector.arange(5, 0).size).toBe(0); // positive step, decreasing range
    });

    it('rejects a zero step in arange()', () => {
        expect(() => Vector.arange(0, 10, 0)).toThrowError(RangeError);
    });

    it('performs absSelf/powSelf/sqrtSelf/clipSelf in place, returning this', () => {
        const v: Vector = new Vector([-2, 3, -4]);

        let result = v.absSelf();
        expect(result).toBe(v);
        expect(v.toArray()).toEqual([2, 3, 4]);

        result = v.powSelf(2);
        expect(result).toBe(v);
        expect(v.toArray()).toEqual([4, 9, 16]);

        result = v.sqrtSelf();
        expect(result).toBe(v);
        expect(v.toArray()).toEqual([2, 3, 4]);

        result = v.clipSelf(3, 3.5);
        expect(result).toBe(v);
        expect(v.toArray()).toEqual([3, 3, 3.5]);
    });

    it('sqrtSelf produces NaN for negative components, matching Math.sqrt', () => {
        const v: Vector = new Vector([-1, 4]);
        v.sqrtSelf();
        expect(Number.isNaN(v.get(0))).toBe(true);
        expect(v.get(1)).toBe(2);
    });

    it('clipSelf rejects min > max and leaves the vector unchanged', () => {
        const v: Vector = new Vector([1, 2, 3]);
        expect(() => v.clipSelf(5, 0)).toThrowError(RangeError);
        expect(v.toArray()).toEqual([1, 2, 3]);
    });

    it('immutable abs/pow/sqrt/clip leave the original vector unchanged, unlike their Self counterparts', () => {
        const v: Vector = new Vector([-2, 3]);
        const a = v.abs();
        const p = v.pow(2);
        const c = v.clip(0, 2.5);
        expect(v.toArray()).toEqual([-2, 3]); // untouched by any of the above
        expect(a.toArray()).toEqual([2, 3]);
        expect(p.toArray()).toEqual([4, 9]);
        expect(c.toArray()).toEqual([0, 2.5]);
    });
});