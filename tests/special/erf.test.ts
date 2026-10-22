import { describe, expect, it } from 'vitest';
import { erf } from '../../src/special/erf.js';

// Reference values computed with mpmath (60 digits), stored as the nearest doubles.
// One point (or more) per internal branch of the implementation.
const ERF_REFERENCE: [x: number, value: number][] = [
  [1e-10, 1.1283791670955126e-10],
  [0.001, 0.0011283787909692365],
  [0.1, 0.1124629160182849],
  [0.5, 0.5204998778130465],
  [0.84375, 0.7672256612323416],
  [1, 0.8427007929497149],
  [1.25, 0.9229001282564583],
  [1.5, 0.9661051464753108],
  [2, 0.9953222650189527],
  [2.857142857142857, 0.9999466876886117],
  [3, 0.9999779095030014],
  [4, 0.9999999845827421],
  [5, 0.9999999999984626],
  [5.9, 0.9999999999999999],
  [6, 1],
  [7, 1]
];

describe('erf', () => {
  it('matches high-precision reference values to within 2 ulp', () => {
    for (const [x, expected] of ERF_REFERENCE) {
      expect(Math.abs(erf(x) - expected), `erf(${x})`).toBeLessThanOrEqual(
        2 * Number.EPSILON * Math.abs(expected)
      );
    }
  });

  it('is odd, exactly', () => {
    for (const x of [1e-300, 1e-10, 0.3, 0.84375, 1, 1.25, 2, 2.9, 5.5, 6, 30]) {
      expect(erf(-x)).toBe(-erf(x));
    }
  });

  it('preserves the sign of zero and is exact at the origin', () => {
    expect(erf(0)).toBe(0);
    expect(Object.is(erf(-0), -0)).toBe(true);
  });

  it('handles non-finite arguments', () => {
    expect(erf(NaN)).toBeNaN();
    expect(erf(Infinity)).toBe(1);
    expect(erf(-Infinity)).toBe(-1);
  });

  it('saturates to exactly ±1 for |x| >= 6', () => {
    expect(erf(6)).toBe(1);
    expect(erf(-6)).toBe(-1);
    expect(erf(100)).toBe(1);
    expect(erf(-1e300)).toBe(-1);
  });

  it('is accurate for tiny arguments, including subnormal results', () => {
    const c = 2 / Math.sqrt(Math.PI);
    expect(erf(1e-300) / 1e-300).toBeCloseTo(c, 14);
    expect(erf(1e-10) / 1e-10).toBeCloseTo(c, 14);
    expect(erf(Number.MIN_VALUE)).toBeGreaterThan(0); // does not underflow to zero
    expect(erf(Number.MIN_VALUE)).toBeLessThanOrEqual(2 * Number.MIN_VALUE);
  });

  it('is continuous across the internal branch boundaries', () => {
    for (const b of [2 ** -28, 0.84375, 1.25, 1 / 0.35, 6]) {
      const below = b * (1 - 4 * Number.EPSILON);
      const above = b * (1 + 4 * Number.EPSILON);
      expect(Math.abs(erf(above) - erf(below)), `boundary ${b}`).toBeLessThanOrEqual(
        4 * Number.EPSILON
      );
    }
  });

  it('is monotonically non-decreasing', () => {
    let prev = erf(-7);
    for (let x = -7; x <= 7; x += 1 / 64) {
      const value = erf(x);
      expect(value, `erf(${x})`).toBeGreaterThanOrEqual(prev);
      prev = value;
    }
  });

  it('agrees with a numerical integral of the Gaussian', () => {
    // Composite Simpson on exp(-t^2), independent of the library's quadrature.
    const simpson = (b: number, n: number): number => {
      const h = b / n;
      let sum = 1 + Math.exp(-b * b);
      for (let i = 1; i < n; i++) sum += (i % 2 === 0 ? 2 : 4) * Math.exp(-((i * h) ** 2));
      return ((2 / Math.sqrt(Math.PI)) * sum * h) / 3;
    };
    for (const x of [0.2, 0.9, 1.7, 3.1]) {
      expect(erf(x)).toBeCloseTo(simpson(x, 2000), 12);
    }
  });
});
