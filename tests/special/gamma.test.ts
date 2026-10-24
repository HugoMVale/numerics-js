import { describe, expect, it } from 'vitest';
import { gamma } from '../../src/special/gamma.js';

const EPS = Number.EPSILON;
const MIN_NORMAL = 2.2250738585072014e-308;
const OVERFLOW_X = 171.6243769563027;

// Reference values from mpmath (60 digits) as double-double pairs: value = hi + lo
// (subnormals correctly rounded). The points cover every branch of the implementation:
// tiny x, the polynomial core, the Stirling series with and without the split power,
// the overflow edge, and the negative axis down to the subnormal range.
const GAMMA_REFERENCE: [x: number, hi: number, lo: number][] = [
  [1e-300, 9.999999999999999e299, 7.113783875736513e283],
  [1e-10, 9999999999.422785, -8.344224519785601e-7],
  [0.001, 999.4237724845955, -2.9154041355101375e-14],
  [0.1, 9.51350769866873, 7.024723067503411e-16],
  [0.5, 1.772453850905516, -7.666586499825799e-17],
  [0.9999, 1.0000577314579577, -5.513626775559614e-17],
  [1.2, 0.9181687423997607, -2.876954153043759e-17],
  [1.5, 0.886226925452758, -3.8332932499128993e-17],
  [1.9999, 0.9999577256848119, -6.40372259862714e-19],
  [2.5, 1.329340388179137, -1.9882475174356644e-18],
  [3.7, 4.170651783796604, 7.928234708325672e-17],
  [5.5, 52.34277778455352, 1.2539803835511586e-15],
  [9.99, 354802.0170198311, -1.222307590117912e-11],
  [10.0, 362880.0, 0.0],
  [10.5, 1133278.3889487856, -3.9437133369201146e-11],
  [20.3, 2.9724610752355725e17, -23.68228593711868],
  [50.5, 4.29046291235196e63, 5.889040643606666e46],
  [100.1, 1.478454494651475e156, -7.950233463574864e139],
  [143.0, 2.695364137888163e245, -7.64415879271572e228],
  [143.5, 3.2203704817308086e246, -1.0975650293983117e230],
  [150.25, 1.3321507761951635e261, 1.5104307427870924e243],
  [170.5, 5.56209241456e305, -2.6853104074749465e289],
  [171.5, 9.4833675668248e307, 3.2177969643752153e291],
  [171.6243769563027, 1.7976931348622299e308, -1.6923339316180877e291],
  [-1e-10, -10000000000.577215, -1.0597631686767099e-7],
  [-0.001, -1000.5782056293586, 2.2102236557379066e-14],
  [-0.5, -3.544907701811032, 1.5333172999651597e-16],
  [-0.9, -10.570564109631926, -6.173843454459702e-16],
  [-1.5, 2.363271801207355, -1.0222115333101065e-16],
  [-2.5, -0.9453087204829419, 1.868400083990113e-17],
  [-3.7, 0.2516439959024227, -2.0041489571361624e-17],
  [-10.5, -2.640121820547716e-7, -2.5798508816562404e-23],
  [-20.25, -8.569032663885128e-19, 4.0292507727416115e-35],
  [-50.5, -1.449954393907748e-65, -2.6388151023057596e-82],
  [-100.5, -3.3536908198076787e-159, 6.410630702960576e-176],
  [-150.3, -1.5097598047749159e-263, 6.339343818319325e-280],
  [-170.5, -3.3127395215386074e-308, 0.0],
  [-170.5674972726612, -2.3949604663495347e-308, -0.0],
  [-171.5, 1.9316265431712e-310, 0.0],
  [-175.5, 2.1075e-319, -0.0],
  [-180.25, -0.0, -0.0],
  [-183.5, 0.0, 0.0]
];

// Unit in the last place of a double; 2^-1074 for subnormal values.
const ulp = (value: number): number => {
  const a = Math.abs(value);
  if (a < MIN_NORMAL) return 2 ** -1074;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--; // Math.log2 may round up just below a power of two
  else if (2 ** (e + 1) <= a) e++;
  return 2 ** (e - 52);
};

describe('gamma', () => {
  it('matches high-precision reference values to within 6 ulp', () => {
    for (const [x, hi, lo] of GAMMA_REFERENCE) {
      const actual = gamma(x);
      const error = Math.abs(actual - hi - lo) / ulp(hi);
      expect(error, `gamma(${x}): got ${actual}, expected ${hi}`).toBeLessThanOrEqual(6);
      if (hi === 0) expect(Object.is(actual, hi), `sign of gamma(${x})`).toBe(true);
    }
  });

  it('returns the factorial exactly at positive integers', () => {
    let factorial = 1;
    for (let n = 1; n <= 22; n++) {
      expect(gamma(n)).toBe(factorial); // (n - 1)! is exact in a double up to 22!
      factorial *= n;
    }
    // Beyond that the table holds the correctly rounded factorial.
    expect(gamma(171)).toBe(7.257415615307999e306);
    expect(gamma(100)).toBe(9.332621544394415e155);
  });

  it('satisfies the recurrence gamma(x + 1) = x * gamma(x)', () => {
    for (const x of [1e-5, 0.3, 0.77, 1.4, 2.9, 7.3, 9.6, 12.2, 55.5, 140.1, -0.3, -1.7, -8.4]) {
      const lhs = gamma(x + 1);
      expect(Math.abs(lhs - x * gamma(x)), `x = ${x}`).toBeLessThanOrEqual(
        12 * EPS * Math.abs(lhs)
      );
    }
  });

  it('satisfies the reflection formula gamma(x) * gamma(1 - x) = pi / sin(pi x)', () => {
    for (const x of [0.01, 0.2, 0.5, 0.73, 0.999, 1.5, 2.3, 7.7, -0.4, -3.2, -12.9]) {
      const lhs = gamma(x) * gamma(1 - x);
      const rhs = Math.PI / Math.sin(Math.PI * x);
      expect(Math.abs(lhs - rhs) / Math.abs(rhs), `x = ${x}`).toBeLessThanOrEqual(1e-13);
    }
  });

  it('has gamma(1/2) = sqrt(pi) and the half-integer closed form', () => {
    expect(Math.abs(gamma(0.5) - Math.sqrt(Math.PI))).toBeLessThanOrEqual(2 * EPS * gamma(0.5));
    // gamma(n + 1/2) = (2n)! sqrt(pi) / (4^n n!)
    expect(Math.abs(gamma(4.5) - 11.631728396567448)).toBeLessThanOrEqual(4 * EPS * 11.63);
  });

  it('handles NaN, zeros and infinities', () => {
    expect(gamma(NaN)).toBeNaN();
    expect(gamma(0)).toBe(Infinity);
    expect(gamma(-0)).toBe(-Infinity);
    expect(gamma(Infinity)).toBe(Infinity);
    expect(gamma(-Infinity)).toBeNaN();
  });

  it('returns NaN at the poles on the negative axis', () => {
    for (const x of [-1, -2, -3, -10, -170, -171, -184, -1e6, -(2 ** 53), -1e300]) {
      expect(gamma(x), `gamma(${x})`).toBeNaN();
    }
  });

  it('is accurate for tiny arguments, including subnormal ones', () => {
    expect(gamma(Number.MIN_VALUE)).toBe(Infinity); // 1/x overflows
    expect(gamma(1e-310)).toBe(Infinity);
    expect(gamma(5e-309)).toBe(Infinity);
    expect(gamma(1e-300)).toBeCloseTo(1e300, -285); // 1/x - gamma_Euler, to 15 digits
    expect(gamma(-Number.MIN_VALUE)).toBe(-Infinity);
    expect(gamma(2 ** -60) * 2 ** -60).toBeCloseTo(1 - 0.5772156649015329 * 2 ** -60, 15);
  });

  it('overflows exactly at the documented threshold', () => {
    expect(Number.isFinite(gamma(OVERFLOW_X))).toBe(true);
    expect(gamma(OVERFLOW_X)).toBeGreaterThan(1.797e308);
    expect(gamma(172)).toBe(Infinity);
    expect(gamma(1e10)).toBe(Infinity);
    // the next double above the threshold already overflows
    expect(gamma(171.62437695630274)).toBe(Infinity);
  });

  it('keeps the tiny non-zero values on the negative axis near the underflow limit', () => {
    // Between -184 and -170.57 Gamma is not zero, but subnormal or tiny; gamma must not flush it.
    expect(gamma(-171.5)).toBeGreaterThan(0);
    expect(gamma(-171.5)).toBeLessThan(MIN_NORMAL);
    expect(gamma(-175.5)).toBeGreaterThan(0);
    // one ulp to the left of the pole at -183 the value is about 2.9e-323, i.e. a few subnormal units
    expect(gamma(-183 - 2 ** -45)).toBeGreaterThan(0);
    expect(gamma(-183 - 2 ** -45)).toBeLessThan(1e-322);
    expect(gamma(-170.5)).toBeLessThan(0);
  });

  it('flushes to a signed zero where the result is below half the smallest subnormal', () => {
    expect(Object.is(gamma(-180.25), -0)).toBe(true); // (-181, -180): negative
    expect(Object.is(gamma(-183.5), 0)).toBe(true); // (-184, -183): positive
    expect(Object.is(gamma(-184.5), -0)).toBe(true);
    expect(Object.is(gamma(-185.5), 0)).toBe(true);
    expect(Object.is(gamma(-1e15 - 0.5), -0)).toBe(true);
    expect(Object.is(gamma(-1e300), NaN)).toBe(true);
  });

  it('alternates in sign on the negative axis', () => {
    for (let n = 0; n < 170; n++) {
      const value = gamma(-n - 0.5);
      expect(Math.sign(value), `gamma(${-n - 0.5})`).toBe(n % 2 === 0 ? -1 : 1);
    }
  });

  it('is continuous across the internal branch boundaries', () => {
    // first-order check with the derivative gamma'(x) = gamma(x) * digamma(x), digamma ~ ln(x) - 1/(2x)
    for (const b of [1, 10, 143, 170, -0.5, -1.5]) {
      const below = b - Math.abs(b) * 4 * EPS;
      const above = b + Math.abs(b) * 4 * EPS;
      const slope = gamma(b) * (Math.log(Math.abs(b)) - 1 / (2 * b));
      const predicted = gamma(below) + slope * (above - below);
      expect(Math.abs(gamma(above) - predicted) / Math.abs(gamma(b)), `boundary ${b}`).toBeLessThan(
        1e-14
      );
    }
  });

  it('is monotonically increasing for x >= 1.4616321449683623 and decreasing below', () => {
    let prev = gamma(1.47);
    for (let x = 1.47; x <= 30; x += 1 / 16) {
      const value = gamma(x);
      expect(value, `gamma(${x})`).toBeGreaterThanOrEqual(prev);
      prev = value;
    }
    prev = gamma(1.46);
    for (let x = 1.46; x >= 0.05; x -= 1 / 16) {
      const value = gamma(x);
      expect(value, `gamma(${x})`).toBeGreaterThanOrEqual(prev);
      prev = value;
    }
  });
});
