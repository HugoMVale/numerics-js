import { describe, expect, it } from 'vitest';
import { gamma } from '../../src/special/gamma.js';
import { lgamma } from '../../src/special/lgamma.js';

const EPS = Number.EPSILON;

// Reference values computed with mpmath (80 digits, `re(loggamma(x))`), stored as the nearest doubles.
// One point (or more) per internal branch, plus the negative axis (reflection formula).
const LGAMMA_REFERENCE: [x: number, value: number][] = [
  [1e-300, 690.7755278982137],
  [1e-10, 23.025850929882736],
  [0.001, 6.907178885383853],
  [0.1, 2.252712651734206],
  [0.2, 1.5240638224307845],
  [0.5, 0.5723649429247001],
  [0.7, 0.26086724653166654],
  [0.9, 0.06637623973474295],
  [1.0000001, -5.772155829918507e-8],
  [1.1, -0.049872441259839764],
  [1.5, -0.12078223763524522],
  [1.9, -0.03898427592308336],
  [1.9999999, -4.2278430309861296e-8],
  [2.0000001, 4.227843666532498e-8],
  [2.5, 0.2846828704729192],
  [3.5, 1.2009736023470743],
  [5.5, 3.9578139676187165],
  [7.9, 8.32426586800881],
  [8, 8.525161361065415],
  [8.5, 9.549267257300997],
  [20, 39.339884187199495],
  [100.5, 361.4355404677776],
  [1000, 5905.220423209181],
  [1000000.0, 12815504.569147611],
  [10000000000.0, 220258509288.81058],
  [1e17, 3.8143946580898775e18],
  [1e20, 4.505170185988091e21],
  [1e100, 2.2925850929940456e102],
  [1e300, 6.897755278982137e302],
  [2.5599833278516383e305, 1.7976931348623157e308],
  [-1e-10, 23.025850929998178],
  [-0.001, 6.908333317515028],
  [-0.5, 1.2655121234846454],
  [-1.5, 0.860047015376481],
  [-3.5, -1.309006684993042],
  [-10.5, -15.147270590717842],
  [-50.25, -147.96757866714194],
  [-100.5, -364.90096830942736],
  [-1000.5, -5914.437701116852],
  [-1000000.5, -12815524.147683937]
];

describe('lgamma', () => {
  it('matches high-precision reference values to within 4 ulp', () => {
    for (const [x, expected] of LGAMMA_REFERENCE) {
      const actual = lgamma(x);
      expect(
        Math.abs(actual - expected),
        `lgamma(${x}): got ${actual}, expected ${expected}`
      ).toBeLessThanOrEqual(4 * EPS * Math.abs(expected));
    }
  });

  it('keeps full relative accuracy next to its zeros at x = 1 and x = 2', () => {
    const EULER = 0.5772156649015329;
    const ZETA2 = Math.PI ** 2 / 6;
    const ZETA3 = 1.2020569031595942;
    const taylor = (d: number) => -EULER * d + (ZETA2 / 2) * d ** 2 - (ZETA3 / 3) * d ** 3; // lgamma(1 + d) up to O(d^4)
    for (const delta of [1e-3, 1e-6, 1e-9, 1e-12]) {
      // The argument is rounded, so d is the offset actually represented.
      const d1 = 1 + delta - 1;
      expect(
        Math.abs(lgamma(1 + delta) - taylor(d1)) / Math.abs(taylor(d1)),
        `1 + ${delta}`
      ).toBeLessThan(1e-9);
      // Gamma(2 + d) = (1 + d) Gamma(1 + d)
      const d2 = 2 + delta - 2;
      const expected2 = taylor(d2) + Math.log1p(d2);
      expect(
        Math.abs(lgamma(2 + delta) - expected2) / Math.abs(expected2),
        `2 + ${delta}`
      ).toBeLessThan(1e-9);
    }
    expect(lgamma(1)).toBe(0);
    expect(lgamma(2)).toBe(0);
  });

  it('agrees with log(gamma(x)) where gamma does not over- or underflow', () => {
    for (const x of [
      0.01, 0.3, 0.9, 1.3, 1.7, 3.2, 7.5, 9.1, 25.5, 80.3, 150.2, -0.7, -2.2, -9.5, -30.3
    ]) {
      const expected = Math.log(Math.abs(gamma(x)));
      expect(Math.abs(lgamma(x) - expected), `x = ${x}`).toBeLessThanOrEqual(
        1e-14 * Math.max(1, Math.abs(expected))
      );
    }
  });

  it('satisfies the recurrence lgamma(x + 1) = lgamma(x) + log(x)', () => {
    for (const x of [0.2, 0.9, 1.6, 2.5, 7.7, 8.3, 40.5, 1234.5]) {
      const lhs = lgamma(x + 1);
      expect(Math.abs(lhs - (lgamma(x) + Math.log(x))), `x = ${x}`).toBeLessThanOrEqual(
        8 * EPS * Math.max(Math.abs(lhs), 1)
      );
    }
  });

  it('is finite where gamma overflows, up to the documented threshold', () => {
    expect(Math.abs(lgamma(200) - 857.9336698258575)).toBeLessThanOrEqual(4 * EPS * 857.93);
    expect(Math.abs(lgamma(1e5) - 1051287.7089736569)).toBeLessThanOrEqual(4 * EPS * 1051287.71);
    expect(Number.isFinite(lgamma(2.5599833278516383e305))).toBe(true);
    expect(lgamma(2.56e305)).toBe(Infinity);
    expect(lgamma(1e308)).toBe(Infinity);
  });

  it('handles NaN, zeros, infinities and poles', () => {
    expect(lgamma(NaN)).toBeNaN();
    expect(lgamma(0)).toBe(Infinity);
    expect(lgamma(-0)).toBe(Infinity);
    expect(lgamma(Infinity)).toBe(Infinity);
    expect(lgamma(-Infinity)).toBe(Infinity);
    for (const x of [-1, -2, -3, -50, -(2 ** 52), -(2 ** 60), -1e300]) {
      expect(lgamma(x), `lgamma(${x})`).toBe(Infinity);
    }
  });

  it('is accurate for tiny arguments, including subnormal ones', () => {
    expect(lgamma(1e-310)).toBeCloseTo(713.8013788281542, 10);
    expect(lgamma(-1e-310)).toBe(lgamma(1e-310));
    expect(lgamma(Number.MIN_VALUE)).toBeCloseTo(744.4400719213812, 10);
    expect(lgamma(2 ** -60)).toBeCloseTo(60 * Math.LN2, 12);
  });

  it('locates the zeros of ln|Gamma| on the negative axis with small absolute error', () => {
    // Near a zero the relative error is meaningless; the absolute error stays at a few 1e-16.
    // The arguments are the doubles nearest to the true zeros (mpmath); the values there are
    // 5.6e-17, 1.7e-16 and 1.7e-15, i.e. the slope times the rounding of the argument.
    expect(Math.abs(lgamma(-2.4570247382208006))).toBeLessThan(2e-15);
    expect(Math.abs(lgamma(-2.7476826467274127))).toBeLessThan(2e-15);
    expect(Math.abs(lgamma(-3.14358088834998))).toBeLessThan(1e-14);
  });

  it('matches the reflection formula on the negative axis', () => {
    // lgamma(x) + lgamma(1 - x) = ln(pi / |sin(pi x)|)
    for (const x of [-0.3, -1.4, -2.5, -7.7, -20.1, -100.4]) {
      const lhs = lgamma(x) + lgamma(1 - x);
      const rhs = Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x)));
      expect(Math.abs(lhs - rhs), `x = ${x}`).toBeLessThanOrEqual(
        1e-12 * Math.max(1, Math.abs(rhs), Math.abs(lgamma(1 - x)))
      );
    }
  });

  it('is continuous across the internal branch boundaries', () => {
    // fdlibm switches representation at these points; the jump must match the first-order prediction.
    // The slope is the digamma function, here from a central difference of the function itself.
    const bounds = [
      0.23163998126983643, 0.7315998077392578, 0.9000000953674316, 1.2316322326660156,
      1.7316312789916992, 2, 3, 4, 5, 6, 7, 8
    ];
    for (const b of bounds) {
      const h = 1e-6;
      const slope = (lgamma(b + h) - lgamma(b - h)) / (2 * h);
      const below = b - 4 * EPS * b;
      const above = b + 4 * EPS * b;
      const predicted = lgamma(below) + slope * (above - below);
      expect(Math.abs(lgamma(above) - predicted), `boundary ${b}`).toBeLessThanOrEqual(1e-14);
    }
  });

  it('is monotone on each side of the minimum at x = 1.4616...', () => {
    let prev = lgamma(1.4617);
    for (let x = 1.4617; x <= 50; x += 1 / 32) {
      const value = lgamma(x);
      expect(value, `lgamma(${x})`).toBeGreaterThanOrEqual(prev);
      prev = value;
    }
    prev = lgamma(1.4616);
    for (let x = 1.4616; x >= 0.02; x -= 1 / 32) {
      const value = lgamma(x);
      expect(value, `lgamma(${x})`).toBeGreaterThanOrEqual(prev);
      prev = value;
    }
  });

  it('grows like x (ln x - 1) for very large arguments', () => {
    for (const x of [1e20, 1e100, 1e300]) {
      expect(lgamma(x) / (x * (Math.log(x) - 1))).toBeCloseTo(1, 12);
    }
  });
});
