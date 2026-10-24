import { describe, expect, it } from 'vitest';
import { erf, erfc } from '../../src/special/erf.js';

// Reference values from mpmath (60 digits) as double-double pairs: value = hi + lo.
// One point (or more) per internal branch of the implementation.
const ERF_REFERENCE: [x: number, hi: number, lo: number][] = [
  [1e-10, 1.1283791670955126e-10, 3.250270181699747e-27],
  [0.001, 0.0011283787909692365, -7.88735357945236e-20],
  [0.1, 0.1124629160182849, 9.255497413884101e-19],
  [0.5, 0.5204998778130465, 1.900077467916287e-17],
  [0.84375, 0.7672256612323416, 3.982733307777803e-17],
  [1, 0.8427007929497149, -2.4801011789118602e-17],
  [1.25, 0.9229001282564583, -5.51775442986392e-17],
  [1.5, 0.9661051464753108, -3.3867031441680696e-17],
  [2, 0.9953222650189527, 2.20719858329765e-17],
  [2.857142857142857, 0.9999466876886117, -4.6715199063846973e-17],
  [3, 0.9999779095030014, 5.363397058636269e-17],
  [4, 0.9999999845827421, 1.44826531920025e-17],
  [5, 0.9999999999984626, -2.294992711807301e-17],
  [5.9, 0.9999999999999999, 3.911820462701088e-17],
  [6, 1, -2.1519736712498913e-17],
  [7, 1, -4.183825607779414e-23]
];

// Unit in the last place of a double; 2^-1074 for subnormal values.
const ulp = (value: number): number =>
  Math.max(2 ** (Math.floor(Math.log2(Math.abs(value))) - 52), 2 ** -1074);

describe('erf', () => {
  it('matches high-precision reference values to within 1 ulp', () => {
    for (const [x, hi, lo] of ERF_REFERENCE) {
      const error = Math.abs(erf(x) - hi - lo) / ulp(hi);
      expect(error, `erf(${x})`).toBeLessThanOrEqual(1);
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
    // Compare the jump across each boundary with the first-order prediction
    // erf'(x) = 2/sqrt(pi) * exp(-x^2); any mismatch beyond rounding exposes a bad branch.
    for (const b of [2 ** -28, 0.84375, 1.25, 1 / 0.35, 6]) {
      const below = b * (1 - 4 * Number.EPSILON);
      const above = b * (1 + 4 * Number.EPSILON);
      const slope = (2 / Math.sqrt(Math.PI)) * Math.exp(-b * b);
      const predicted = erf(below) + slope * (above - below);
      expect(Math.abs(erf(above) - predicted), `boundary ${b}`).toBeLessThanOrEqual(
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

// Reference values from mpmath (60 digits) as double-double pairs: value = hi + lo.
const ERFC_REFERENCE: [x: number, hi: number, lo: number][] = [
  [-5, 1.9999999999984626, -2.294992711807301e-17],
  [-1, 1.8427007929497148, 8.622129067339705e-17],
  [-0.5, 1.5204998778130465, 1.900077467916287e-17],
  [0.1, 0.887537083981715, 5.4585601489869416e-17],
  [0.25, 0.7236736098317631, -3.128407501007366e-17],
  [0.5, 0.4795001221869535, -1.900077467916287e-17],
  [0.84375, 0.23277433876765838, -1.207175746214912e-17],
  [1, 0.15729920705028513, -2.954563826510312e-18],
  [1.25, 0.07709987174354177, -3.3360693261863044e-19],
  [2, 0.004677734981047266, -3.8794238326641256e-19],
  [3, 2.209049699858544e-5, 1.5563377960343457e-22],
  [5, 1.537459794428035e-12, -8.569418222079096e-29],
  [6, 2.1519736712498913e-17, 3.1898197253599377e-34],
  [10, 2.088487583762545e-45, -1.2006565763501381e-61],
  [20, 5.395865611607901e-176, -2.4980975484348278e-192],
  [26.5, 2.2109076642637343e-307, 5e-324]
];

describe('erfc', () => {
  it('matches high-precision reference values to within 3 ulp', () => {
    for (const [x, hi, lo] of ERFC_REFERENCE) {
      const error = Math.abs(erfc(x) - hi - lo) / ulp(hi);
      expect(error, `erfc(${x})`).toBeLessThanOrEqual(3);
    }
  });

  it('keeps full relative accuracy in the tail, where 1 - erf(x) is useless', () => {
    expect(1 - erf(10)).toBe(0);
    expect(erfc(10) / 2.088487583762545e-45).toBeCloseTo(1, 14);
    expect(erfc(25) / 8.300172571196523e-274).toBeCloseTo(1, 14);
  });

  it('satisfies erfc(x) + erf(x) = 1 for moderate arguments', () => {
    for (const x of [-2, -0.7, -0.1, 0, 0.1, 0.7, 1.3, 2]) {
      expect(erfc(x) + erf(x)).toBeCloseTo(1, 15);
    }
  });

  it('satisfies the reflection formula erfc(-x) = 2 - erfc(x)', () => {
    for (const x of [1e-10, 0.1, 0.5, 0.84375, 1, 1.25, 2, 3, 5.5, 6]) {
      expect(erfc(-x)).toBeCloseTo(2 - erfc(x), 15);
    }
  });

  it('is exact at the origin', () => {
    expect(erfc(0)).toBe(1);
    expect(erfc(-0)).toBe(1);
  });

  it('handles non-finite arguments', () => {
    expect(erfc(NaN)).toBeNaN();
    expect(erfc(Infinity)).toBe(0);
    expect(erfc(-Infinity)).toBe(2);
  });

  it('saturates for large |x|', () => {
    expect(erfc(28)).toBe(0);
    expect(erfc(1e10)).toBe(0);
    expect(erfc(-6)).toBe(2);
    expect(erfc(-1e10)).toBe(2);
  });

  it('degrades gracefully through the subnormal range without spurious zeros', () => {
    // erfc(x) is subnormal for x >= 26.54 and rounds to zero from x = 27.226.
    expect(erfc(26.6)).toBeGreaterThan(0);
    expect(erfc(26.6)).toBeLessThan(2.2250738585072014e-308);
    expect(erfc(27.2)).toBeGreaterThan(0);
    expect(erfc(27.3)).toBe(0);
  });

  it('is continuous across the internal branch boundaries', () => {
    // Same first-order check as for erf, in relative terms: erfc'(x) = -2/sqrt(pi) * exp(-x^2).
    for (const b of [2 ** -56, 0.25, 0.84375, 1.25, 1 / 0.35]) {
      const below = b * (1 - 4 * Number.EPSILON);
      const above = b * (1 + 4 * Number.EPSILON);
      const slope = (-2 / Math.sqrt(Math.PI)) * Math.exp(-b * b);
      const predicted = erfc(below) + slope * (above - below);
      expect(Math.abs(erfc(above) - predicted) / erfc(below), `boundary ${b}`).toBeLessThanOrEqual(
        8 * Number.EPSILON
      );
    }
  });

  it('is monotonically non-increasing', () => {
    let prev = erfc(-7);
    for (let x = -7; x <= 27; x += 1 / 64) {
      const value = erfc(x);
      expect(value, `erfc(${x})`).toBeLessThanOrEqual(prev);
      prev = value;
    }
  });
});
