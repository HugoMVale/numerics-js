import { describe, expect, it } from 'vitest';
import { gaussKronrod, gaussKronrod15 } from '../../src/integrate/gaussKronrod.js';
import {
  fPolynomial,
  fSin,
  fArctanDeriv,
  fIdentity,
  fOscillatory,
  fBlowupMidpoint
} from './testFunctions.js';

describe('gaussKronrod15', () => {
  it('computes a single-panel estimate for polynomials accurately', () => {
    let evaluations = 0;
    const evalF = (x: number) => {
      evaluations++;
      return fPolynomial(x);
    };

    const result = gaussKronrod15(evalF, 0, 1);
    expect(result.kronrod).toBeCloseTo(1 / 3, 10);
    expect(evaluations).toBe(15);
  });
});

describe('gaussKronrod', () => {
  it('integrates standard functions accurately', () => {
    const r1 = gaussKronrod(fSin, 0, Math.PI, { atol: 1e-10, rtol: 1e-10 });
    expect(r1.value).toBeCloseTo(2, 10);
    expect(r1.converged).toBe(true);

    const r2 = gaussKronrod(fArctanDeriv, 0, 1, { atol: 1e-12, rtol: 1e-12 });
    expect(r2.value).toBeCloseTo(Math.PI, 12);
    expect(r2.converged).toBe(true);
  });

  it('handles identical integration bounds', () => {
    const res = gaussKronrod(fIdentity, 2, 2);
    expect(res).toEqual({
      value: 0,
      error: 0,
      evaluations: 0,
      converged: true,
      subintervals: 0
    });
  });

  it('supports reversed bounds (a > b)', () => {
    const forward = gaussKronrod(fSin, 0, Math.PI);
    const reverse = gaussKronrod(fSin, Math.PI, 0);

    expect(reverse.value).toBeCloseTo(-forward.value, 10);
    expect(reverse.evaluations).toBe(forward.evaluations);
  });

  it('integrates over large finite bounds without overflowing panel centers', () => {
    const scale = 1e308;
    const result = gaussKronrod(
      (x) => Math.exp(-Math.pow((x / scale - 1.05) / 0.01, 2)),
      scale,
      1.1 * scale,
      { atol: 0, rtol: 1e-12 }
    );
    const expected = Math.sqrt(Math.PI) * 1e306;

    expect(result.converged).toBe(true);
    expect(Math.abs(result.value - expected) / expected).toBeLessThan(1e-8);
  });

  it('handles finite bounds whose difference overflows', () => {
    const scale = 1e308;
    const result = gaussKronrod((x) => Math.exp(-Math.pow(x / scale, 2)), -scale, scale, {
      atol: 0,
      rtol: 1e-12
    });

    expect(result.converged).toBe(true);
    expect(Number.isFinite(result.value)).toBe(true);
    expect(result.value).toBeGreaterThan(0);
  });

  it('respects maxSubintervals and sets converged to false when exceeded', () => {
    const res = gaussKronrod(fOscillatory, 0, Math.PI, {
      atol: 1e-15,
      rtol: 0,
      maxSubintervals: 3
    });

    expect(res.converged).toBe(false);
    expect(res.subintervals).toBe(3);
  });

  it('rejects invalid tolerances and panel limits before evaluating the integrand', () => {
    let evaluations = 0;
    const f = (x: number) => {
      evaluations++;
      return x;
    };

    for (const bad of [NaN, Infinity, -Infinity, -1]) {
      expect(() => gaussKronrod(f, 0, 1, { atol: bad })).toThrow(RangeError);
      expect(() => gaussKronrod(f, 0, 1, { rtol: bad })).toThrow(RangeError);
    }
    for (const maxSubintervals of [NaN, Infinity, -Infinity, 0, -1, 1.5]) {
      expect(() => gaussKronrod(f, 0, 1, { maxSubintervals })).toThrow(RangeError);
    }

    expect(evaluations).toBe(0);
  });

  it('accepts zero tolerances when the embedded estimates agree exactly', () => {
    const res = gaussKronrod(fPolynomial, 0, 1, { atol: 0, rtol: 0 });

    expect(res.error).toBe(0);
    expect(res.converged).toBe(true);
  });

  it('converges once error <= atol + rtol * |value|, with either term alone sufficient', () => {
    const f = (x: number) => 1e6 * Math.exp(Math.sin(30 * x));
    const absOnly = gaussKronrod(f, 0, 1, { atol: 1e-3, rtol: 0 });
    const relOnly = gaussKronrod(f, 0, 1, { atol: 0, rtol: 1e-9 });
    for (const res of [absOnly, relOnly]) {
      expect(res.converged).toBe(true);
    }
    expect(absOnly.error).toBeLessThanOrEqual(1e-3);
    expect(relOnly.error).toBeLessThanOrEqual(1e-9 * Math.abs(relOnly.value));
  });

  it('is invariant under rescaling of the integrand when atol = 0', () => {
    const g = (x: number) => Math.exp(Math.sin(30 * x));
    const reference = gaussKronrod(g, 0, 1, { atol: 0, rtol: 1e-8 });
    expect(reference.converged).toBe(true);

    for (const c of [1e-12, 1e-3, 1e3, 1e12]) {
      const res = gaussKronrod((x) => c * g(x), 0, 1, { atol: 0, rtol: 1e-8 });
      expect(res.converged).toBe(true);
      expect(res.subintervals).toBe(reference.subintervals);
      expect(res.evaluations).toBe(reference.evaluations);
      expect(res.value / c).toBeCloseTo(reference.value, 12);
    }
  });

  it('resolves integrals far below the default absolute tolerance via rtol', () => {
    // Integral of 1e-12 * sin(100 x) over [0, 1]; the default atol alone would accept
    // the first panel (error ~1e-13 << 1e-8) with a completely wrong value.
    const exact = (1e-12 * (1 - Math.cos(100))) / 100;
    const res = gaussKronrod((x) => 1e-12 * Math.sin(100 * x), 0, 1, { atol: 0 });

    expect(res.converged).toBe(true);
    expect(Math.abs(res.value - exact) / Math.abs(exact)).toBeLessThan(1e-12);
  });

  it('does not chase unreachable accuracy for large integrals with the default tolerances', () => {
    // Integral ~1e10: the old absolute-only criterion needed 1e-18 relative accuracy.
    const res = gaussKronrod((x) => 1e10 * Math.exp(-x), 0, 50);
    expect(res.converged).toBe(true);
    expect(res.subintervals).toBeLessThan(10);
    expect(Math.abs(res.value / (1e10 * (1 - Math.exp(-50))) - 1)).toBeLessThan(1e-12);
  });

  it('throws TypeError for invalid integrand input', () => {
    expect(() => gaussKronrod('not a function' as unknown as (x: number) => number, 0, 1)).toThrow(
      TypeError
    );
  });

  it('throws RangeError for infinite or NaN integration limits', () => {
    expect(() => gaussKronrod(fSin, 0, Infinity)).toThrow(
      'gaussKronrod: a and b must be finite numbers'
    );
    expect(() => gaussKronrod(fSin, NaN, 1)).toThrow(
      'gaussKronrod: a and b must be finite numbers'
    );
  });

  it('throws an Error when the function produces non-finite values', () => {
    expect(() => gaussKronrod(fBlowupMidpoint, 0, 1)).toThrow(
      /gaussKronrod: f\(0\.5\) returned a non-finite value \(Infinity\); cannot integrate\./
    );
  });
});
