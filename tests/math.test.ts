import { describe, it, expect } from 'vitest';
import { clip, copysign, isClose, rand, type RandMethod } from '../src/math.js';

describe('clip()', () => {
    it('restricts values to the inclusive range', () => {
        expect(clip(-1, 0, 10)).toBe(0);
        expect(clip(5, 0, 10)).toBe(5);
        expect(clip(11, 0, 10)).toBe(10);
    });

    it('keeps values on the bounds unchanged', () => {
        expect(clip(0, 0, 10)).toBe(0);
        expect(clip(10, 0, 10)).toBe(10);
    });
});

describe('copysign()', () => {
    it('uses the magnitude of the first value and sign of the second', () => {
        expect(copysign(-3.5, 1)).toBe(3.5);
        expect(copysign(3.5, -1)).toBe(-3.5);
    });

    it('treats zero as a positive sign', () => {
        expect(copysign(-3.5, 0)).toBe(3.5);
    });
});

describe('isClose()', () => {
    it('checks absolute tolerance and rejects values outside it', () => {
        expect(isClose(1, 1.001, 0.001)).toBe(true);
        expect(isClose(1, 1.002, 0.001)).toBe(false);
    });

    it('includes the relative tolerance term', () => {
        expect(isClose(100, 101, 0, 0.01)).toBe(true);
        expect(isClose(100, 101.1, 0, 0.01)).toBe(false);
    });
});

describe('rand()', () => {
    it('reproduces known mulberry32 values for a fixed seed', () => {
        const next = rand(42);
        expect(next()).toBe(0.6011037519201636);
        expect(next()).toBe(0.44829055899754167);
    });

    it('defaults to mulberry32', () => {
        const a = rand(7);
        const b = rand(7, 'mulberry32');
        for (let i = 0; i < 10; i++) expect(a()).toBe(b());
    });

    it('is deterministic per seed and differs between seeds', () => {
        const a = rand(1), b = rand(1), c = rand(2);
        const seqA = Array.from({ length: 5 }, () => a());
        expect(Array.from({ length: 5 }, () => b())).toEqual(seqA);
        expect(Array.from({ length: 5 }, () => c())).not.toEqual(seqA);
    });

    it('keeps generators independent of one another', () => {
        const a = rand(5), b = rand(5);
        a(); a();
        const fresh = rand(5);
        fresh(); fresh();
        expect(a()).toBe(fresh());
        expect(b()).toBe(rand(5)());
    });

    it('yields values in [0, 1) with a mean near 0.5', () => {
        const next = rand(123);
        let sum = 0;
        const n = 10000;
        for (let i = 0; i < n; i++) {
            const x = next();
            expect(x).toBeGreaterThanOrEqual(0);
            expect(x).toBeLessThan(1);
            sum += x;
        }
        expect(Math.abs(sum / n - 0.5)).toBeLessThan(0.02);
    });

    it('accepts zero and negative seeds', () => {
        expect(rand(0)()).toBeGreaterThanOrEqual(0);
        expect(rand(-1)()).not.toBe(rand(1)());
    });

    it('rejects non-finite seeds', () => {
        expect(() => rand(NaN)).toThrowError(RangeError);
        expect(() => rand(Infinity)).toThrowError(RangeError);
    });

    it('rejects unknown methods', () => {
        expect(() => rand(1, 'nope' as RandMethod)).toThrowError(RangeError);
        expect(() => rand(1, 'toString' as RandMethod)).toThrowError(RangeError);
    });
});
