import { describe, it, expect } from 'vitest';
import { createRng, type RngMethod } from '../../src/math.js';

describe('createRng()', () => {
  it('reproduces known mulberry32 values for a fixed seed', () => {
    const next = createRng(42);
    expect(next()).toBe(0.6011037519201636);
    expect(next()).toBe(0.44829055899754167);
  });

  it('defaults to mulberry32', () => {
    const a = createRng(7);
    const b = createRng(7, 'mulberry32');
    for (let i = 0; i < 10; i++) expect(a()).toBe(b());
  });

  it('is deterministic per seed and differs between seeds', () => {
    const a = createRng(1),
      b = createRng(1),
      c = createRng(2);
    const seqA = Array.from({ length: 5 }, () => a());
    expect(Array.from({ length: 5 }, () => b())).toEqual(seqA);
    expect(Array.from({ length: 5 }, () => c())).not.toEqual(seqA);
  });

  it('keeps generators independent of one another', () => {
    const a = createRng(5),
      b = createRng(5);
    a();
    a();
    const fresh = createRng(5);
    fresh();
    fresh();
    expect(a()).toBe(fresh());
    expect(b()).toBe(createRng(5)());
  });

  it('yields values in [0, 1) with a mean near 0.5', () => {
    const next = createRng(123);
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
    expect(createRng(0)()).toBeGreaterThanOrEqual(0);
    expect(createRng(-1)()).not.toBe(createRng(1)());
  });

  it('converts the seed to a 32-bit integer: truncation and wrap-around modulo 2^32', () => {
    expect(createRng(1.9)()).toBe(createRng(1)());
    expect(createRng(-1.9)()).toBe(createRng(-1)());
    expect(createRng(2 ** 32)()).toBe(createRng(0)());
    expect(createRng(2 ** 32 + 5)()).toBe(createRng(5)());
    expect(createRng(2 ** 31)()).toBe(createRng(-(2 ** 31))());
    expect(() => createRng(1e300)).not.toThrow();
  });

  it('rejects non-finite seeds', () => {
    expect(() => createRng(NaN)).toThrowError(RangeError);
    expect(() => createRng(Infinity)).toThrowError(RangeError);
    expect(() => createRng(-Infinity)).toThrow('createRng: seed must be finite, got -Infinity.');
  });

  it('rejects seeds that are not numbers', () => {
    expect(() => createRng('7' as unknown as number)).toThrowError(TypeError);
    expect(() => createRng('7' as unknown as number)).toThrow(
      'createRng: seed must be a number, got string.'
    );
    expect(() => createRng(undefined as unknown as number)).toThrowError(TypeError);
    expect(() => createRng(null as unknown as number)).toThrowError(TypeError);
  });

  it('rejects unknown methods', () => {
    expect(() => createRng(1, 'nope' as RngMethod)).toThrowError(RangeError);
    expect(() => createRng(1, 'toString' as RngMethod)).toThrowError(RangeError);
    expect(() => createRng(1, 'nope' as RngMethod)).toThrow("createRng: unknown method 'nope'.");
  });
});
