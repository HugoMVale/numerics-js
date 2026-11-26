import { describe, it, expect } from 'vitest';
import { copysign } from '../../src/math.js';

describe('copysign()', () => {
  it('uses the magnitude of the first value and sign of the second', () => {
    expect(copysign(-3.5, 1)).toBe(3.5);
    expect(copysign(3.5, -1)).toBe(-3.5);
    expect(copysign(-3.5, -1)).toBe(-3.5);
    expect(copysign(3.5, 1)).toBe(3.5);
  });

  it('follows the sign of a signed zero, as C, Python and Fortran SIGN do', () => {
    expect(copysign(3.5, 0)).toBe(3.5);
    expect(copysign(-3.5, 0)).toBe(3.5);
    expect(copysign(3.5, -0)).toBe(-3.5);
    expect(copysign(-3.5, -0)).toBe(-3.5);
  });

  it('handles infinities and a zero magnitude', () => {
    expect(copysign(2, -Infinity)).toBe(-2);
    expect(copysign(-2, Infinity)).toBe(2);
    expect(copysign(Infinity, -1)).toBe(-Infinity);
    expect(Object.is(copysign(0, -1), -0)).toBe(true);
    expect(Object.is(copysign(-0, 1), 0)).toBe(true);
  });

  it('propagates a NaN magnitude and treats a NaN sign as positive', () => {
    expect(copysign(NaN, -1)).toBeNaN();
    expect(copysign(-3.5, NaN)).toBe(3.5);
  });
});
