import { beforeEach, describe, expect, it } from 'vitest';
import {
  besselI,
  besselIScaled,
  besselJ,
  besselJZero,
  besselY,
  clearBesselCache
} from '../../src/special/bessel.js';

// Reference values computed with mpmath (40 digits), stored as the nearest doubles.
const J_REFERENCE: [n: number, x: number, value: number][] = [
  [0, 1, 0.7651976865579666],
  [1, 1, 0.4400505857449335],
  [2, 1, 0.11490348493190047],
  [3, 0.5, 0.002563729994587244],
  [5, 2.5, 0.01950162513450322],
  [20, 5, 2.7703300521289416e-11],
  [5, 0.001, 2.6041665581597246e-19],
  [10, 9.5, 0.16502640472619115],
  [10, 10, 0.20748610663335887],
  [10, 10.5, 0.24774553753592743],
  [30, 30, 0.1439358500103072],
  [0, 50, 0.055812327669251816],
  [100, 50, 1.1159273690838094e-21],
  [2, 100, -0.021528757344505364],
  [50, 100, -0.038698339728525384],
  [1, 1000, 0.004728311907089524]
];

const ZERO_REFERENCE: [n: number, m: number, value: number][] = [
  [0, 1, 2.404825557695773],
  [0, 2, 5.5200781102863106],
  [0, 3, 8.653727912911013],
  [1, 1, 3.8317059702075125],
  [1, 2, 7.0155866698156188],
  [2, 1, 5.135622301840683],
  [2, 20, 65.1592731907578],
  [5, 3, 15.700174079711671],
  [30, 2, 41.092778663153425],
  [100, 1, 108.83616589840977]
];

describe('besselJ', () => {
  it('matches high-precision reference values', () => {
    for (const [n, x, expected] of J_REFERENCE) {
      const actual = besselJ(n, x);
      // Absolute accuracy ~1e-15 where J is O(0.1); relative accuracy where it decays.
      expect(Math.abs(actual - expected), `J_${n}(${x})`).toBeLessThanOrEqual(
        1e-15 + 1e-13 * Math.abs(expected)
      );
    }
  });

  it('is exact at x = 0', () => {
    expect(besselJ(0, 0)).toBe(1);
    expect(besselJ(1, 0)).toBe(0);
    expect(besselJ(7, 0)).toBe(0);
    expect(besselJ(0, -0)).toBe(1);
  });

  it('applies the parity identity J_n(-x) = (-1)^n J_n(x)', () => {
    for (const n of [0, 1, 2, 3, 10, 11]) {
      for (const x of [0.3, 2.5, 25, 250]) {
        expect(besselJ(n, -x)).toBe((n % 2 === 0 ? 1 : -1) * besselJ(n, x));
      }
    }
  });

  it('handles non-finite arguments', () => {
    expect(besselJ(0, NaN)).toBeNaN();
    expect(besselJ(3, NaN)).toBeNaN();
    expect(besselJ(0, Infinity)).toBe(0);
    expect(besselJ(2, -Infinity)).toBe(0);
    expect(besselJ(3, -Infinity)).toBe(-0); // odd order: limit approached from the negative side
  });

  it('is finite and accurate for tiny arguments (no overflow in the recurrence)', () => {
    expect(besselJ(0, 1e-300)).toBe(1);
    expect(besselJ(1, 1e-300) / 5e-301).toBeCloseTo(1, 14);
    expect(besselJ(2, 1e-100) / 1.25e-201).toBeCloseTo(1, 14);
    expect(besselJ(2, 1e-300)).toBe(0); // true value 1.25e-601 underflows
    expect(besselJ(5, 1e-310)).toBe(0);
  });

  it('is continuous across the internal method boundaries', () => {
    // x = 1 (series -> recurrence), n = x (forward -> Miller).
    const below = besselJ(3, 1 - 1e-12);
    const above = besselJ(3, 1 + 1e-12);
    expect(Math.abs(below - above)).toBeLessThan(1e-11);
    for (const n of [2, 10, 100, 500]) {
      const lo = besselJ(n, n - 1e-9);
      const at = besselJ(n, n);
      const hi = besselJ(n, n + 1e-9);
      expect(Math.abs(lo - at)).toBeLessThan(1e-8);
      expect(Math.abs(hi - at)).toBeLessThan(1e-8);
    }
  });

  it('satisfies the three-term recurrence J_{n-1} + J_{n+1} = (2n/x) J_n', () => {
    for (const x of [0.7, 3.3, 15, 80, 400, 3000]) {
      for (const n of [1, 2, 5, 12, 40, 150]) {
        const lhs = besselJ(n - 1, x) + besselJ(n + 1, x);
        const rhs = ((2 * n) / x) * besselJ(n, x);
        expect(Math.abs(lhs - rhs), `n=${n}, x=${x}`).toBeLessThan(
          1e-13 * Math.max(1, (2 * n) / x)
        );
      }
    }
  });

  it('satisfies the Neumann sum rule J_0^2 + 2 sum_k J_k^2 = 1, including large x', () => {
    for (const x of [0.5, 7, 60, 1000]) {
      let sum = besselJ(0, x) ** 2;
      for (let k = 1; k < x + 20 * Math.cbrt(x) + 60; k++) sum += 2 * besselJ(k, x) ** 2;
      expect(Math.abs(sum - 1), `x=${x}`).toBeLessThan(1e-12);
    }
  });

  it('satisfies the Jacobi-Anger expansion cos(x) = J_0 + 2 sum (-1)^k J_{2k}', () => {
    for (const x of [2, 30, 200]) {
      let sum = besselJ(0, x);
      for (let k = 1; 2 * k < x + 20 * Math.cbrt(x) + 60; k++) {
        sum += 2 * (k % 2 === 0 ? 1 : -1) * besselJ(2 * k, x);
      }
      expect(Math.abs(sum - Math.cos(x)), `x=${x}`).toBeLessThan(1e-12);
    }
  });

  it('stays accurate for n >= 2 at large x (regression: Miller start order)', () => {
    // The old implementation was off by O(1) here.
    expect(besselJ(2, 100)).toBeCloseTo(-0.021528757344505364, 13);
    expect(besselJ(5, 1000)).toBeCloseTo(0.005025406945233186, 13);
  });

  it('stays accurate for J_0 and J_1 beyond the old sampling cap (regression)', () => {
    // x = 1e4 exceeded the old fixed cap of 4000 trapezoid steps.
    expect(besselJ(0, 1e4)).toBeCloseTo(-0.007096160353388801, 12);
    expect(besselJ(1, 1e4)).toBeCloseTo(0.00364745075552958, 12);
  });

  it('works for large orders', () => {
    expect(besselJ(1000, 1)).toBe(0); // underflows
    expect(besselJ(1000, 1000)).toBeCloseTo(0.04473067294796404, 13);
    expect(Number.isFinite(besselJ(1e6, 1e6))).toBe(true);
  });

  it('throws RangeError for invalid orders', () => {
    for (const n of [-1, 1.5, NaN, Infinity, -Infinity, 1e6 + 1, 1e9]) {
      expect(() => besselJ(n, 5), `n=${n}`).toThrow(RangeError);
    }
    expect(() => besselJ(-1, NaN)).toThrow(RangeError);
  });
});

describe('besselJZero', () => {
  beforeEach(() => {
    clearBesselCache();
  });

  it('matches high-precision reference values', () => {
    for (const [n, m, expected] of ZERO_REFERENCE) {
      const actual = besselJZero(n, m);
      expect(Math.abs(actual - expected) / expected, `j_{${n},${m}}`).toBeLessThan(1e-14);
    }
  });

  it('returns zeros that are roots of J_n', () => {
    for (const n of [0, 1, 2, 7, 25]) {
      for (const m of [1, 2, 5, 17]) {
        const z = besselJZero(n, m);
        const slope = Math.abs(besselJ(n - 1 >= 0 ? n - 1 : 1, z) - (n / z) * besselJ(n, z));
        expect(Math.abs(besselJ(n, z)) / Math.max(slope, 1e-3), `n=${n}, m=${m}`).toBeLessThan(
          5e-14
        );
      }
    }
  });

  it('returns strictly increasing zeros with no missed index (large m, n >= 2)', () => {
    // The old implementation skipped zeros beyond x ~ 60 for n >= 2.
    for (const n of [0, 2, 5, 30]) {
      let previous = besselJZero(n, 1);
      for (let m = 2; m <= 120; m++) {
        const z = besselJZero(n, m);
        expect(z).toBeGreaterThan(previous + 3.11); // gaps between zeros exceed 3.116
        previous = z;
      }
    }
    expect(besselJZero(2, 20)).toBeCloseTo(65.1592731907578, 12);
    expect(besselJZero(2, 50)).toBeCloseTo(159.42406617141825, 10);
  });

  it('gives the same value regardless of request order and cache state', () => {
    const direct = besselJZero(3, 9);
    clearBesselCache();
    const sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((m) => besselJZero(3, m));
    expect(sequence[8]).toBe(direct);
    clearBesselCache();
    const outOfOrder = [
      besselJZero(3, 9),
      besselJZero(3, 2),
      besselJZero(3, 20),
      besselJZero(3, 2)
    ];
    expect(outOfOrder[0]).toBe(direct);
    expect(outOfOrder[1]).toBe(sequence[1]);
    expect(outOfOrder[3]).toBe(outOfOrder[1]);
  });

  it('keeps orders independent', () => {
    const j0 = besselJZero(0, 3);
    const j1 = besselJZero(1, 3);
    expect(besselJZero(0, 3)).toBe(j0);
    expect(besselJZero(1, 3)).toBe(j1);
    expect(j0).not.toBe(j1);
  });

  it('throws RangeError for invalid order or index, leaving the cache usable', () => {
    for (const n of [-1, 1.5, NaN, Infinity, 1e6 + 1]) {
      expect(() => besselJZero(n, 1), `n=${n}`).toThrow(RangeError);
    }
    for (const m of [0, -3, 1.5, NaN, Infinity]) {
      expect(() => besselJZero(0, m), `m=${m}`).toThrow(RangeError);
    }
    expect(besselJZero(0, 1)).toBeCloseTo(2.404825557695773, 14);
  });
});

// Reference values computed with mpmath (40 digits), stored as the nearest doubles.
const I_REFERENCE: [n: number, x: number, value: number][] = [
  [0, 1, 1.2660658777520084],
  [1, 1, 0.565159103992485],
  [2, 1, 0.13574766976703828],
  [3, 0.5, 0.002645111968990286],
  [0, 10, 2815.7166284662544],
  [5, 2.5, 0.03284347517202321],
  [20, 5, 5.024239357971806e-11],
  [2, 100, 1.0523843193243106e42],
  [100, 50, 2.7278879470966917e-16],
  [0, 700, 1.5295933476718737e302],
  [100, 720, 7.098312758841151e307]
];

// Reference values computed with mpmath (60 digits), stored as the nearest doubles.
const Y_REFERENCE: [n: number, x: number, value: number][] = [
  [0, 1, 0.08825696421567696],
  [1, 1, -0.7812128213002887],
  [2, 1, -1.6506826068162543],
  [3, 0.5, -42.059494304723884],
  [5, 2.5, -3.8301760007407517],
  [20, 5, -593396529.6914321],
  [5, 0.001, -2.4446200786802637e17],
  [0, 2.5, 0.4980703596152319],
  [1, 2.9, 0.29594005460767475],
  [0, 3.5, 0.1890219439208265],
  [1, 10, 0.24901542420695388],
  [0, 19.9, 0.04576209415938548],
  [1, 25, -0.09882996478323741],
  [10, 9.5, -0.4439065532932459],
  [10, 10, -0.35981415218340274],
  [10, 10.5, -0.28022819155866036],
  [30, 30, -0.24937439396697417],
  [0, 50, -0.09806499547007708],
  [100, 50, -3.2938001882026665e18],
  [2, 100, 0.07683686712502795],
  [50, 100, 0.07650526394480305],
  [1, 1000, -0.024784331292351778],
  [1000, 1000, -0.07747600152072075],
  [100, 0.5, -4.776690378041764e215]
];

const Y0_ZEROS = [
  0.8935769662791675, 3.957678419314858, 7.086051060301773, 10.222345043496418, 13.361097473872764,
  16.50092244152809
];
const Y1_ZEROS = [
  2.197141326031017, 5.429681040794135, 8.596005868331169, 11.749154830839881, 14.897442128336726
];

describe('besselY', () => {
  it('matches high-precision reference values', () => {
    for (const [n, x, expected] of Y_REFERENCE) {
      const actual = besselY(n, x);
      expect(Math.abs(actual - expected), `Y_${n}(${x})`).toBeLessThanOrEqual(
        1e-15 + 2e-14 * Math.abs(expected)
      );
    }
  });

  it('vanishes at the known zeros of Y_0, Y_1 and Y_5', () => {
    for (const z of Y0_ZEROS) expect(Math.abs(besselY(0, z)), `y_{0}(${z})`).toBeLessThan(1e-15);
    for (const z of Y1_ZEROS) expect(Math.abs(besselY(1, z)), `y_{1}(${z})`).toBeLessThan(1e-15);
    for (const z of [6.747183824871022, 10.597176726782031]) {
      expect(Math.abs(besselY(5, z)), `y_{5}(${z})`).toBeLessThan(1e-14);
    }
  });

  it('handles the singularity, the domain and non-finite arguments', () => {
    for (const n of [0, 1, 2, 7]) {
      expect(besselY(n, 0)).toBe(-Infinity);
      expect(besselY(n, -0)).toBe(-Infinity);
      expect(besselY(n, -1e-300)).toBeNaN(); // not real for x < 0
      expect(besselY(n, -5)).toBeNaN();
      expect(besselY(n, -Infinity)).toBeNaN();
      expect(besselY(n, NaN)).toBeNaN();
      expect(besselY(n, Infinity)).toBe(0);
    }
  });

  it('is accurate for tiny arguments, including subnormals', () => {
    // Y_0 ~ (2/pi) (ln(x/2) + gamma) must not hit ln(0) for subnormal x.
    expect(besselY(0, 1e-300) / -439.8351636227653).toBeCloseTo(1, 14);
    expect(besselY(0, 1e-308) / -451.5621332048362).toBeCloseTo(1, 14);
    expect(besselY(0, 5e-324) / -473.9990734230043).toBeCloseTo(1, 14);
    // Y_1 ~ -2 / (pi x), finite until it exceeds the double range.
    expect(besselY(1, 1e-300) / -6.366197723675813e299).toBeCloseTo(1, 14);
    expect(besselY(1, 1e-308) / -6.366197723675814e307).toBeCloseTo(1, 14);
    expect(besselY(1, 5e-324)).toBe(-Infinity);
    // Y_n ~ -((n-1)! / pi) (2/x)^n for small x.
    expect(besselY(2, 1e-100) / -1.2732395447351626e200).toBeCloseTo(1, 14);
    expect(besselY(3, 1e-100) / -5.0929581789406506e300).toBeCloseTo(1, 14);
  });

  it('does not lose accuracy at huge arguments (phase is not rounded)', () => {
    // sin and cos see the exact x, so there is no x * eps phase error.
    expect(Math.abs(besselY(0, 1e8) - 7.306391165521707e-5)).toBeLessThan(1e-19);
    expect(Math.abs(besselY(1, 1e8) - -3.2060294975092524e-5)).toBeLessThan(1e-19);
    expect(Math.abs(besselY(0, 1e15) - 2.4468665123771324e-8)).toBeLessThan(1e-23);
    expect(Math.abs(besselY(1, 1e15) - -6.15663864688501e-9)).toBeLessThan(2e-24);
    expect(Math.abs(besselY(2, 1e15) - -2.4468665123771334e-8)).toBeLessThan(1e-23);
    // sqrt(pi x) would overflow here; the 1/sqrt(x) form does not.
    expect(besselY(0, 1e300)).toBeCloseTo(-1.3681360450342481e-151, 165);
    expect(besselY(1, 1e300)).toBeCloseTo(7.860673062724093e-151, 165);
    expect(besselY(0, Number.MAX_VALUE)).not.toBe(0);
  });

  it('is continuous across the internal method boundaries', () => {
    // x = 3 (series -> Neumann series) and x = 20 (Neumann series -> Hankel expansion).
    for (const boundary of [3, 20]) {
      for (const n of [0, 1, 4]) {
        const below = besselY(n, boundary - 1e-12);
        const at = besselY(n, boundary);
        const above = besselY(n, boundary + 1e-12);
        const slope = Math.abs(besselY(n + 1, boundary)) + Math.abs(besselY(n, boundary));
        expect(Math.abs(below - at), `n=${n}, x=${boundary}-`).toBeLessThan(1e-14 + 1e-11 * slope);
        expect(Math.abs(above - at), `n=${n}, x=${boundary}+`).toBeLessThan(1e-14 + 1e-11 * slope);
      }
    }
  });

  it('satisfies the Wronskian J_{n+1} Y_n - J_n Y_{n+1} = 2 / (pi x) against besselJ', () => {
    // Independent of the Y code paths: besselJ uses a different algorithm.
    for (const x of [0.01, 0.3, 0.99, 1.5, 2.99, 3, 3.5, 9, 19.99, 20, 33, 150, 800]) {
      for (const n of [0, 1, 2, 5, 12]) {
        const a = besselJ(n + 1, x) * besselY(n, x);
        const b = besselJ(n, x) * besselY(n + 1, x);
        const scale = Math.max(Math.abs(a), Math.abs(b), 2 / (Math.PI * x));
        // besselJ has an absolute error of a few 1e-16 (growing slowly with x), which
        // enters multiplied by |Y|; allow for it.
        const allowed =
          2e-14 * scale + 3e-15 * (Math.abs(besselY(n, x)) + Math.abs(besselY(n + 1, x)));
        expect(Math.abs(a - b - 2 / (Math.PI * x)), `n=${n}, x=${x}`).toBeLessThan(allowed);
      }
    }
  });

  it("satisfies Y_0' = -Y_1 (central differences, also across the method boundaries)", () => {
    const h = 1e-4;
    for (const x of [0.4, 1.7, 3, 3 + 5e-5, 6.2, 12, 19.99995, 20, 20.00004, 45, 300]) {
      const slope = (besselY(0, x + h) - besselY(0, x - h)) / (2 * h);
      // Truncation error ~ h^2 |Y_0'''| / 6; round-off ~ 3e-16 / h.
      expect(Math.abs(slope + besselY(1, x)), `x=${x}`).toBeLessThan(
        2e-8 * Math.max(1, 1 / x ** 3)
      );
    }
  });

  it('has no jumps in x over a fine scan through all three regions', () => {
    // Step 0.01; relative allowance for n > x, where |Y_n| is large and varies fast.
    for (const n of [0, 1, 3, 25]) {
      let previous = besselY(n, 4);
      for (let x = 4.01; x < 60; x += 0.01) {
        const value = besselY(n, x);
        const allowed = 0.02 + 0.1 * Math.max(Math.abs(value), Math.abs(previous));
        expect(Math.abs(value - previous), `n=${n}, x=${x}`).toBeLessThan(allowed);
        previous = value;
      }
    }
  });

  it('overflows to -Infinity (never NaN) once |Y_n| exceeds the double range', () => {
    expect(besselY(200, 5)).toBeCloseTo(-3.3446295e292, -285); // still representable
    expect(besselY(250, 5)).toBe(-Infinity); // true value -1.4e390
    expect(besselY(1e6, 1)).toBe(-Infinity);
    expect(besselY(2, 1e-200)).toBe(-Infinity);
    expect(Number.isFinite(besselY(100, 0.5))).toBe(true); // -4.8e215
    for (const x of [1e-3, 0.5, 3, 20]) {
      let overflowed = false;
      for (let n = 0; n <= 400; n++) {
        const value = besselY(n, x);
        expect(Number.isNaN(value), `Y_${n}(${x})`).toBe(false);
        if (overflowed) expect(value, `Y_${n}(${x})`).toBe(-Infinity);
        if (value === -Infinity) overflowed = true;
      }
    }
  });

  it('is negative and strictly decreasing in n beyond the turning point', () => {
    for (const x of [0.2, 1, 4, 10]) {
      let previous = Infinity;
      for (let n = Math.ceil(x); n < 40; n++) {
        const value = besselY(n, x);
        if (!Number.isFinite(value)) break;
        expect(value, `Y_${n}(${x})`).toBeLessThan(0);
        expect(value, `Y_${n}(${x})`).toBeLessThan(previous);
        previous = value;
      }
    }
  });

  it('works for large orders', () => {
    expect(Math.abs(besselY(1000, 1000) - -0.07747600152072075)).toBeLessThan(1e-14);
    expect(besselY(1000, 1)).toBe(-Infinity);
    const huge = besselY(1e6, 1e6);
    expect(Number.isFinite(huge)).toBe(true);
    expect(huge).toBeLessThan(0);
  });

  it('throws RangeError for invalid orders', () => {
    for (const n of [-1, 1.5, NaN, Infinity, -Infinity, 1e6 + 1, 1e9]) {
      expect(() => besselY(n, 5), `n=${n}`).toThrow(RangeError);
    }
    expect(() => besselY(-1, NaN)).toThrow(RangeError);
    expect(() => besselY(-1, 0)).toThrow(RangeError);
  });
});

describe('besselI', () => {
  it('matches high-precision reference values', () => {
    for (const [n, x, expected] of I_REFERENCE) {
      expect(Math.abs(besselI(n, x) / expected - 1), `I_${n}(${x})`).toBeLessThan(1e-14);
    }
  });

  it('is exact at x = 0', () => {
    expect(besselI(0, 0)).toBe(1);
    expect(besselI(1, 0)).toBe(0);
    expect(besselI(7, 0)).toBe(0);
    expect(besselIScaled(0, -0)).toBe(1);
    expect(besselIScaled(4, 0)).toBe(0);
  });

  it('applies the parity identity I_n(-x) = (-1)^n I_n(x)', () => {
    for (const n of [0, 1, 2, 3, 10, 11]) {
      for (const x of [0.3, 2.5, 25, 250]) {
        const sign = n % 2 === 0 ? 1 : -1;
        expect(besselI(n, -x)).toBe(sign * besselI(n, x));
        expect(besselIScaled(n, -x)).toBe(sign * besselIScaled(n, x));
      }
    }
  });

  it('handles non-finite arguments', () => {
    expect(besselI(0, NaN)).toBeNaN();
    expect(besselIScaled(3, NaN)).toBeNaN();
    expect(besselI(0, Infinity)).toBe(Infinity);
    expect(besselI(2, -Infinity)).toBe(Infinity);
    expect(besselI(3, -Infinity)).toBe(-Infinity);
    expect(besselIScaled(0, Infinity)).toBe(0);
    expect(besselIScaled(2, -Infinity)).toBe(0);
  });

  it('is finite and accurate for tiny arguments', () => {
    expect(besselI(0, 1e-300)).toBe(1);
    expect(besselI(1, 1e-300) / 5e-301).toBeCloseTo(1, 14);
    expect(besselI(2, 1e-100) / 1.25e-201).toBeCloseTo(1, 14);
    expect(besselI(2, 1e-300)).toBe(0); // true value 1.25e-601 underflows
    expect(besselI(5, 1e-310)).toBe(0);
  });

  it('is continuous across the series / recurrence boundary at x = 1', () => {
    for (const n of [0, 1, 3, 20]) {
      const below = besselI(n, 1 - 1e-12);
      const above = besselI(n, 1 + 1e-12);
      // d ln I_n / dx = n / x at x = 1, so the two sides differ by about 2e-12 * n.
      expect(Math.abs(below / above - 1)).toBeLessThan(3e-12 * (n + 1));
    }
  });

  it('is positive and strictly increasing in x for x > 0 (no cancellation)', () => {
    for (const n of [0, 1, 4, 30]) {
      let previous = 0;
      for (const x of [0.01, 0.5, 0.999, 1, 1.001, 3, 10, 40, 200, 600]) {
        const value = besselI(n, x);
        expect(value, `I_${n}(${x})`).toBeGreaterThan(previous);
        previous = value;
      }
    }
  });

  it('satisfies the three-term recurrence I_{n-1} - I_{n+1} = (2n/x) I_n', () => {
    for (const x of [0.7, 3.3, 15, 80, 400]) {
      for (const n of [1, 2, 5, 12, 40, 150]) {
        const lhs = besselIScaled(n - 1, x) - besselIScaled(n + 1, x);
        const rhs = ((2 * n) / x) * besselIScaled(n, x);
        // The difference cancels; the error scales with the largest operand.
        expect(Math.abs(lhs - rhs), `n=${n}, x=${x}`).toBeLessThanOrEqual(
          1e-14 * besselIScaled(n - 1, x)
        );
      }
    }
  });

  it('satisfies the generating-function identity e^x = I_0 + 2 sum I_k', () => {
    for (const x of [0.5, 7, 60, 500, 5000]) {
      let sum = besselIScaled(0, x);
      for (let k = 1; k < 12 * Math.sqrt(x) + 60; k++) sum += 2 * besselIScaled(k, x);
      expect(Math.abs(sum - 1), `x=${x}`).toBeLessThan(2e-13);
    }
  });

  it('satisfies the alternating identity e^{-x} = I_0 + 2 sum (-1)^k I_k', () => {
    const x = 3;
    let sum = besselI(0, x);
    for (let k = 1; k < 80; k++) sum += 2 * (k % 2 === 0 ? 1 : -1) * besselI(k, x);
    expect(Math.abs(sum - Math.exp(-x))).toBeLessThan(1e-14);
  });

  it('besselIScaled never overflows and follows the asymptotic law at huge x', () => {
    for (const x of [1e3, 1e5, 1e8]) {
      const approx = 1 / Math.sqrt(2 * Math.PI * x);
      const value = besselIScaled(0, x);
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value / approx - 1)).toBeLessThan(2 / (8 * x) + 1e-12);
    }
  });

  it('overflows to Infinity only when I_n(x) exceeds the double range', () => {
    expect(besselI(0, 720)).toBe(Infinity);
    expect(besselI(100, 720)).toBeLessThan(Infinity); // 7.1e307, still representable
    expect(besselI(100, 2000)).toBe(Infinity);
    expect(besselI(1e6, 1e6)).toBe(Infinity); // I ~ e^(5e5); scaled value underflows too
  });

  it('keeps representable values that underflow when scaled', () => {
    // e^{-100} I_500(100) ~ 1.3e-326 underflows, while I_500(100) ~ 3.6e-283 does not.
    expect(besselIScaled(500, 100)).toBe(0);
    expect(besselI(500, 100) / 3.5900514314696126e-283 - 1).toBeLessThan(2e-13);
  });

  it('works for large orders', () => {
    expect(besselI(1000, 1)).toBe(0); // underflows
    expect(besselIScaled(1e6, 1e6)).toBe(0); // e^(-4.7e5) underflows, no NaN or hang
    // Uniform asymptotics for n << x: e^{-n^2/(2x)} / sqrt(2 pi x) ~ 2.5e-27 here.
    const v = besselIScaled(1e6, 1e10);
    expect(Math.abs(v / (Math.exp(-50) / Math.sqrt(2 * Math.PI * 1e10)) - 1)).toBeLessThan(1e-3);
  });

  it('throws RangeError for invalid orders', () => {
    for (const n of [-1, 1.5, NaN, Infinity, -Infinity, 1e6 + 1, 1e9]) {
      expect(() => besselI(n, 5), `n=${n}`).toThrow(RangeError);
      expect(() => besselIScaled(n, 5), `n=${n}`).toThrow(RangeError);
    }
    expect(() => besselI(-1, NaN)).toThrow(RangeError);
  });
});
