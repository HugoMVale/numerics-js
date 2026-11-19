import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';

// Behavior implemented once in `ArrayND` and shared by `Vector` and `Matrix`.
describe('ArrayND (shared behavior)', () => {
  describe('variance() and std() with ddof >= size', () => {
    it('return NaN when ddof equals the number of elements', () => {
      expect(Vector.from([1, 2, 3]).variance(3)).toBeNaN();
      expect(Vector.from([1, 2, 3]).std(3)).toBeNaN();
      expect(Vector.from([5]).variance(1)).toBeNaN();
      expect(
        Matrix.from([
          [1, 2],
          [3, 4]
        ]).variance(undefined, 4)
      ).toBeNaN();
    });

    it('return NaN, not a negative variance, when ddof exceeds the number of elements', () => {
      expect(Vector.from([1, 2]).variance(3)).toBeNaN();
      expect(Vector.from([1, 2]).std(3)).toBeNaN();
      expect(
        Matrix.from([
          [1, 2],
          [3, 4]
        ]).std(undefined, 7)
      ).toBeNaN();
    });

    it('return NaN per slice for axis-wise Matrix reductions', () => {
      const A = Matrix.from([
        [1, 2],
        [3, 4]
      ]);
      // Each column (axis 0) and each row (axis 1) has 2 elements.
      expect((A.variance(0, 2) as Vector).toArray()).toEqual([NaN, NaN]);
      expect((A.std(1, 3) as Vector).toArray()).toEqual([NaN, NaN]);
      expect((A.variance(0, 1) as Vector).toArray()).toEqual([2, 2]);
    });

    it('still accept a valid ddof', () => {
      expect(Vector.from([1, 2, 3, 4]).variance(3)).toBeCloseTo(5 / 1);
      expect(Vector.from([1, 2, 3, 4]).variance(1)).toBeCloseTo(5 / 3);
    });
  });

  describe('isClose() and allClose() with non-finite values', () => {
    it('treat exactly equal infinities as close', () => {
      const a = Vector.from([Infinity, -Infinity, 1]);
      const b = Vector.from([Infinity, -Infinity, 1]);
      expect(a.isClose(b)).toEqual([true, true, true]);
      expect(a.allClose(b)).toBe(true);
      expect(Matrix.from([[Infinity, 1]]).allClose(Matrix.from([[Infinity, 1]]))).toBe(true);
    });

    it('treat opposite or finite-vs-infinite pairs as not close', () => {
      expect(Vector.from([Infinity]).isClose(Vector.from([-Infinity]))).toEqual([false]);
      expect(Vector.from([Infinity]).isClose(Vector.from([1e308]))).toEqual([false]);
      expect(Vector.from([1e308]).allClose(Vector.from([Infinity]))).toBe(false);
    });

    it('never treat NaN as close, even to another NaN', () => {
      expect(Vector.from([NaN]).isClose(Vector.from([NaN]))).toEqual([false]);
      expect(Vector.from([NaN]).allClose(Vector.from([NaN]))).toBe(false);
    });
  });

  describe('clip() and clipSelf() bound validation', () => {
    it('reject NaN bounds', () => {
      const v = Vector.from([1, 2, 3]);
      expect(() => v.clip(NaN, 2)).toThrowError(RangeError);
      expect(() => v.clip(0, NaN)).toThrowError(RangeError);
      expect(() => v.clipSelf(NaN, NaN)).toThrowError(RangeError);
      expect(v.toArray()).toEqual([1, 2, 3]);
      expect(() => Matrix.identity(2).clip(NaN, 1)).toThrowError(RangeError);
    });

    it('accept equal and infinite bounds', () => {
      expect(Vector.from([1, 2, 3]).clip(2, 2).toArray()).toEqual([2, 2, 2]);
      expect(Vector.from([-5, 5]).clip(-Infinity, Infinity).toArray()).toEqual([-5, 5]);
    });
  });

  describe('empty-input errors', () => {
    it('name the vector consistently across min, max, argmin and argmax', () => {
      const empty = new Vector(0);
      expect(() => empty.min()).toThrowError(/Vector\.min: .* empty vector\./);
      expect(() => empty.max()).toThrowError(/Vector\.max: .* empty vector\./);
      expect(() => empty.argmin()).toThrowError(/Vector\.argmin: .* empty vector\./);
      expect(() => empty.argmax()).toThrowError(/Vector\.argmax: .* empty vector\./);
    });
  });
});
