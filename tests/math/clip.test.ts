import { describe, it, expect } from 'vitest';
import { clip } from '../../src/math.js';
import { Vector } from '../../src/linalg.js';

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

  it('handles infinite values and bounds', () => {
    expect(clip(Infinity, 0, 10)).toBe(10);
    expect(clip(-Infinity, 0, 10)).toBe(0);
    expect(clip(5, -Infinity, Infinity)).toBe(5);
    expect(clip(-1e300, -Infinity, 0)).toBe(-1e300);
  });

  it('returns NaN for a NaN value and keeps the sign of an in-range zero', () => {
    expect(clip(NaN, 0, 1)).toBeNaN();
    expect(Object.is(clip(-0, -1, 1), -0)).toBe(true);
  });

  it('does not validate the bounds: with min > max it yields min below min, else max', () => {
    expect(clip(5, 10, 0)).toBe(10);
    expect(clip(15, 10, 0)).toBe(0);
  });

  it('agrees with Vector.clip for valid bounds', () => {
    const xs = [-Infinity, -2, -0.5, 0, 0.5, 1, 3, Infinity];
    const expected = Vector.from(xs).clip(-1, 1);
    xs.forEach((x, i) => expect(clip(x, -1, 1)).toBe(expected.get(i)));
  });
});
