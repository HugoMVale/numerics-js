import { describe, expect, it } from 'vitest';
import { gaussKronrod15, gaussKronrod15Scaled } from '../../src/integrate/kronrod.js';
import { fPolynomial } from './testFunctions.js';

describe('gaussKronrod15', () => {
  it('computes a single-subinterval estimate for polynomials accurately', () => {
    let evaluations = 0;
    const evalF = (x: number) => {
      evaluations++;
      return fPolynomial(x);
    };

    const result = gaussKronrod15(evalF, 0, 1);
    expect(result.kronrod).toBeCloseTo(1 / 3, 10);
    expect(evaluations).toBe(15);
  });

  it('is exact up to degree 22 (Kronrod) and 13 (embedded Gauss) on [-1, 1]', () => {
    const exact = (n: number) => (n % 2 === 0 ? 2 / (n + 1) : 0);

    for (const n of [0, 1, 2, 7, 12, 13, 22]) {
      expect(gaussKronrod15((x) => x ** n, -1, 1).kronrod).toBeCloseTo(exact(n), 14);
    }
    for (const n of [0, 1, 2, 7, 12, 13]) {
      expect(gaussKronrod15((x) => x ** n, -1, 1).gauss).toBeCloseTo(exact(n), 14);
    }

    // One degree beyond each rule's exactness must show up in the estimates.
    expect(Math.abs(gaussKronrod15((x) => x ** 14, -1, 1).gauss - exact(14))).toBeGreaterThan(1e-8);
    expect(Math.abs(gaussKronrod15((x) => x ** 24, -1, 1).kronrod - exact(24))).toBeGreaterThan(
      1e-12
    );
  });

  it('estimates the error as |kronrod - gauss| and scales with the interval', () => {
    const result = gaussKronrod15((x) => x ** 14, -1, 1);
    expect(result.error).toBe(Math.abs(result.kronrod - result.gauss));
    expect(result.error).toBeGreaterThan(0);

    // Integration over a shifted, stretched interval: ∫_2^6 (x - 4)^2 dx = 2 * 2^3 / 3.
    expect(gaussKronrod15((x) => (x - 4) ** 2, 2, 6).kronrod).toBeCloseTo(16 / 3, 13);
  });
});

describe('gaussKronrod15Scaled', () => {
  it('has the same Kronrod value and node count as the unscaled rule', () => {
    let evaluations = 0;
    const evalF = (x: number) => {
      evaluations++;
      return Math.exp(x);
    };
    const scaled = gaussKronrod15Scaled(evalF, 0, 1);
    const plain = gaussKronrod15(Math.exp, 0, 1);
    expect(evaluations).toBe(15);
    expect(scaled.value).toBeCloseTo(plain.kronrod, 14);
  });

  it('bounds the error below by roundoff and reports resabs and resasc', () => {
    const r = gaussKronrod15Scaled((x) => x, 0, 1);
    expect(r.value).toBeCloseTo(0.5, 15);
    expect(r.resabs).toBeCloseTo(0.5, 15);
    expect(r.error).toBeGreaterThanOrEqual(50 * Number.EPSILON * r.resabs);
    expect(r.resasc).toBeCloseTo(0.25, 2);
  });

  it('is symmetric under reversal of orientation', () => {
    const f = (x: number) => Math.cos(3 * x);
    const forward = gaussKronrod15Scaled(f, 0, 2);
    const backward = gaussKronrod15Scaled(f, 2, 0);
    expect(backward.value).toBeCloseTo(-forward.value, 14);
  });
});
