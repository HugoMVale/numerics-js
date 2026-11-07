import { beforeEach, describe, expect, it } from 'vitest';
import {
  besselI,
  besselIScaled,
  besselJ,
  besselJZero,
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
