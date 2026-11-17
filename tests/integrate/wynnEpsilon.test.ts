import { describe, expect, it } from 'vitest';
import { EpsilonTable } from '../../src/integrate/wynnEpsilon.js';

/** Partial sums `s_k = a_0 + ... + a_k` for `k = 0, ..., n - 1`. */
function partialSums(n: number, term: (k: number) => number): number[] {
  const sums: number[] = [];
  let s = 0;
  for (let k = 0; k < n; k++) {
    s += term(k);
    sums.push(s);
  }
  return sums;
}

describe('EpsilonTable', () => {
  it('has no estimate with fewer than three elements', () => {
    const table = new EpsilonTable();
    expect(table.length).toBe(0);
    table.push(3);
    expect(table.extrapolate()).toEqual({ result: 3, error: Number.MAX_VALUE });
    table.push(4);
    expect(table.length).toBe(2);
    expect(table.extrapolate()).toEqual({ result: 4, error: Number.MAX_VALUE });
  });

  it('is exact for the partial sums of a geometric series', () => {
    // Shanks transform of order 1 (Aitken's delta-squared) is exact for s_k = L + c r^k.
    const r = 0.5;
    const table = new EpsilonTable();
    for (const s of partialSums(3, (k) => r ** k)) table.push(s);
    const { result } = table.extrapolate();
    expect(result).toBeCloseTo(1 / (1 - r), 14);
  });

  it('accelerates the Leibniz series for pi/4', () => {
    // Raw partial sums converge like 1/n: after 10 terms the error is about 2.5e-2.
    const sums = partialSums(10, (k) => (-1) ** k / (2 * k + 1));
    expect(Math.abs(sums[9] - Math.PI / 4)).toBeGreaterThan(2e-2);

    const table = new EpsilonTable();
    let estimate = { result: NaN, error: NaN };
    for (const s of sums) {
      table.push(s);
      if (table.length >= 3) estimate = table.extrapolate();
    }
    expect(Math.abs(estimate.result - Math.PI / 4)).toBeLessThan(1e-7);
    // The error estimate is honest: it is not smaller than the true error.
    expect(estimate.error).toBeGreaterThan(Math.abs(estimate.result - Math.PI / 4));
    expect(estimate.error).toBeLessThan(1e-4);
  });

  it('withholds the error estimate until it has three previous results', () => {
    const table = new EpsilonTable();
    const errors: number[] = [];
    for (const s of partialSums(8, (k) => (-1) ** k / (2 * k + 1))) {
      table.push(s);
      if (table.length >= 3) errors.push(table.extrapolate().error);
    }
    expect(errors.slice(0, 3)).toEqual([Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE]);
    for (const error of errors.slice(3)) expect(error).toBeLessThan(1);
  });

  it('detects convergence when three elements agree to machine precision', () => {
    const table = new EpsilonTable();
    for (const s of [1, 1, 1]) table.push(s);
    const { result, error } = table.extrapolate();
    expect(result).toBe(1);
    // Floor of the error estimate: 5 eps |result|.
    expect(error).toBeLessThanOrEqual(5 * Number.EPSILON);
    expect(error).toBeGreaterThan(0);
  });

  it('detects convergence of a sequence that has converged by itself', () => {
    // A geometric series with ratio 1/2: after 5 elements the Aitken estimate is exactly 2, and
    // the previous one agrees to rounding.
    const table = new EpsilonTable();
    let estimate = { result: NaN, error: NaN };
    for (const s of partialSums(5, (k) => 0.5 ** k)) {
      table.push(s);
      if (table.length >= 3) estimate = table.extrapolate();
    }
    expect(estimate.result).toBeCloseTo(2, 14);
    expect(estimate.error).toBeLessThan(1e-13);
  });

  it('keeps the table bounded for long sequences', () => {
    // Basel series: slowly (algebraically) convergent, the epsilon table keeps improving it.
    const table = new EpsilonTable();
    let maxLength = 0;
    let result = NaN;
    for (const s of partialSums(400, (k) => 1 / (k + 1) ** 2)) {
      table.push(s);
      if (table.length > 2) result = table.extrapolate().result;
      maxLength = Math.max(maxLength, table.length);
    }
    expect(maxLength).toBeLessThanOrEqual(50);
    expect(result).toBeCloseTo(Math.PI ** 2 / 6, 2);
  });

  it('stays finite on a divergent, oscillating sequence', () => {
    // Divergent (alternating) sequence: the result must stay finite and the table well-defined.
    const table = new EpsilonTable();
    for (let k = 0; k < 40; k++) {
      table.push(k % 2 === 0 ? 1 : -1);
      if (table.length > 2) {
        const { result, error } = table.extrapolate();
        expect(Number.isFinite(result)).toBe(true);
        expect(Number.isNaN(error)).toBe(false);
      }
    }
    expect(table.length).toBeLessThanOrEqual(50);
  });
});
