import { describe, expect, it } from 'vitest';
import { gaussKronrod } from '../../src/integrate/gaussKronrod.js';
import {
  fPolynomial,
  fSin,
  fArctanDeriv,
  fIdentity,
  fOscillatory,
  fBlowupMidpoint
} from './testFunctions.js';

describe('gaussKronrod', () => {
  it('integrates standard functions accurately', () => {
    const r1 = gaussKronrod(fSin, 0, Math.PI, { atol: 1e-10, rtol: 1e-10 });
    expect(r1.value).toBeCloseTo(2, 10);
    expect(r1.success).toBe(true);

    const r2 = gaussKronrod(fArctanDeriv, 0, 1, { atol: 1e-12, rtol: 1e-12 });
    expect(r2.value).toBeCloseTo(Math.PI, 12);
    expect(r2.success).toBe(true);
  });

  it('handles identical integration bounds', () => {
    const res = gaussKronrod(fIdentity, 2, 2);
    expect(res).toEqual({
      success: true,
      message: 'Empty integration interval (a === b).',
      evaluations: 0,
      subintervals: 0,
      value: 0,
      error: 0
    });
  });

  it('supports reversed bounds (a > b)', () => {
    const forward = gaussKronrod(fSin, 0, Math.PI);
    const reverse = gaussKronrod(fSin, Math.PI, 0);

    expect(reverse.value).toBeCloseTo(-forward.value, 10);
    expect(reverse.evaluations).toBe(forward.evaluations);
  });

  it('integrates over large finite bounds without overflowing subinterval centers', () => {
    const scale = 1e308;
    const result = gaussKronrod(
      (x) => Math.exp(-Math.pow((x / scale - 1.05) / 0.01, 2)),
      scale,
      1.1 * scale,
      { atol: 0, rtol: 1e-12 }
    );
    const expected = Math.sqrt(Math.PI) * 1e306;

    expect(result.success).toBe(true);
    expect(Math.abs(result.value - expected) / expected).toBeLessThan(1e-8);
  });

  it('handles finite bounds whose difference overflows', () => {
    const scale = 1e308;
    const result = gaussKronrod((x) => Math.exp(-Math.pow(x / scale, 2)), -scale, scale, {
      atol: 0,
      rtol: 1e-12
    });

    expect(result.success).toBe(true);
    expect(Number.isFinite(result.value)).toBe(true);
    expect(result.value).toBeGreaterThan(0);
  });

  it('respects maxSubintervals and sets success to false when exceeded', () => {
    const res = gaussKronrod(fOscillatory, 0, Math.PI, {
      atol: 1e-15,
      rtol: 0,
      maxSubintervals: 3
    });

    expect(res.success).toBe(false);
    expect(res.message).toBe('Maximum number of subintervals (3) reached.');
    expect(res.subintervals).toBe(3);
  });

  it('describes the reason for success in the message', () => {
    const res = gaussKronrod(fSin, 0, Math.PI);
    expect(res.success).toBe(true);
    expect(res.message).toBe('Estimated error is at most `atol + rtol*|value|`.');
  });

  it('keeps the reported error non-negative and consistent with the stopping criterion', () => {
    // A narrow Lorentzian needs many subdivisions. Updating the totals incrementally let
    // them drift: after 54 subintervals the error estimate was -1.4e-11 (true sum: 6e-10),
    // so the loop stopped early and reported convergence with a meaningless error.
    const f = (x: number) => 1 / ((x - 0.5) ** 2 + 1e-8);
    const exact = (2 * Math.atan(0.5 / 1e-4)) / 1e-4;
    const rtol = 1e-14;
    const res = gaussKronrod(f, 0, 1, { atol: 0, rtol });

    expect(res.error).toBeGreaterThanOrEqual(0);
    expect(res.success).toBe(true);
    expect(res.error).toBeLessThanOrEqual(rtol * Math.abs(res.value));
    // The reported error must bound the true error.
    expect(Math.abs(res.value - exact)).toBeLessThanOrEqual(res.error);
    expect(res.subintervals).toBeGreaterThan(54);
  });

  it('rejects invalid tolerances and subinterval limits before evaluating the integrand', () => {
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
    expect(res.success).toBe(true);
  });

  it('converges once error <= atol + rtol * |value|, with either term alone sufficient', () => {
    const f = (x: number) => 1e6 * Math.exp(Math.sin(30 * x));
    const absOnly = gaussKronrod(f, 0, 1, { atol: 1e-3, rtol: 0 });
    const relOnly = gaussKronrod(f, 0, 1, { atol: 0, rtol: 1e-9 });
    for (const res of [absOnly, relOnly]) {
      expect(res.success).toBe(true);
    }
    expect(absOnly.error).toBeLessThanOrEqual(1e-3);
    expect(relOnly.error).toBeLessThanOrEqual(1e-9 * Math.abs(relOnly.value));
  });

  it('is invariant under rescaling of the integrand when atol = 0', () => {
    const g = (x: number) => Math.exp(Math.sin(30 * x));
    const reference = gaussKronrod(g, 0, 1, { atol: 0, rtol: 1e-8 });
    expect(reference.success).toBe(true);

    for (const c of [1e-12, 1e-3, 1e3, 1e12]) {
      const res = gaussKronrod((x) => c * g(x), 0, 1, { atol: 0, rtol: 1e-8 });
      expect(res.success).toBe(true);
      expect(res.subintervals).toBe(reference.subintervals);
      expect(res.evaluations).toBe(reference.evaluations);
      expect(res.value / c).toBeCloseTo(reference.value, 12);
    }
  });

  it('resolves integrals far below the default absolute tolerance via rtol', () => {
    // Integral of 1e-12 * sin(100 x) over [0, 1]; the default atol alone would accept
    // the first subinterval (error ~1e-13 << 1e-8) with a completely wrong value.
    const exact = (1e-12 * (1 - Math.cos(100))) / 100;
    const res = gaussKronrod((x) => 1e-12 * Math.sin(100 * x), 0, 1, { atol: 0 });

    expect(res.success).toBe(true);
    expect(Math.abs(res.value - exact) / Math.abs(exact)).toBeLessThan(1e-12);
  });

  it('does not chase unreachable accuracy for large integrals with the default tolerances', () => {
    // Integral ~1e10: with a purely absolute criterion, atol = 1e-8 would demand a relative accuracy of 1e-18.
    const res = gaussKronrod((x) => 1e10 * Math.exp(-x), 0, 50);
    expect(res.success).toBe(true);
    expect(res.subintervals).toBeLessThan(10);
    expect(Math.abs(res.value / (1e10 * (1 - Math.exp(-50))) - 1)).toBeLessThan(1e-12);
  });

  it('throws TypeError for invalid integrand input, also when a === b', () => {
    const notAFunction = 'not a function' as unknown as (x: number) => number;
    expect(() => gaussKronrod(notAFunction, 0, 1)).toThrow(TypeError);
    expect(() => gaussKronrod(notAFunction, 0, 1)).toThrow('gaussKronrod: f must be a function');
    expect(() => gaussKronrod(notAFunction, 1, 1)).toThrow(TypeError);
  });

  it('validates the options also when a === b', () => {
    expect(() => gaussKronrod(fSin, 1, 1, { atol: -1 })).toThrow(RangeError);
    expect(() => gaussKronrod(fSin, 1, 1, { maxSubintervals: 0 })).toThrow(RangeError);
    expect(() => gaussKronrod(fSin, 1, Infinity, { atol: -1 })).toThrow(/finite numbers/);
  });

  it('throws RangeError for infinite or NaN integration limits', () => {
    expect(() => gaussKronrod(fSin, 0, Infinity)).toThrow(
      'gaussKronrod: a and b must be finite numbers'
    );
    expect(() => gaussKronrod(fSin, NaN, 1)).toThrow(
      'gaussKronrod: a and b must be finite numbers'
    );
  });

  it('stops with success = false when bisection reaches an endpoint where f is not finite', () => {
    // (1 - x)^-0.9 is singular at x = 1. Bisecting toward it eventually places a node exactly at
    // x = 1, where f is not finite; that must end the bisection, not throw.
    const f = (x: number) => (1 - x) ** -0.9;
    const res = gaussKronrod(f, 0, 1, { atol: 0, rtol: 1e-12, maxSubintervals: 1000 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Bisection reached floating-point resolution at a point where/);
    expect(Number.isFinite(res.value)).toBe(true);
    expect(res.value).toBeGreaterThan(9);
    expect(res.value).toBeLessThan(10);
    expect(res.subintervals).toBeLessThan(100);
  });

  it('does not stop early when nodes round onto endpoints where f is still finite', () => {
    // The singularity at 1/3 is interior, so bisection ends up with nodes that coincide with the
    // endpoints of tiny subintervals next to it, but f is finite there and the run converges.
    const f = (x: number) => Math.abs(x - 1 / 3) ** -0.5;
    const res = gaussKronrod(f, 0, 1, { atol: 0, rtol: 1e-8 });
    expect(res.success).toBe(true);
    expect(res.value).toBeCloseTo(2 * (Math.sqrt(1 / 3) + Math.sqrt(2 / 3)), 7);
  });

  it('still throws for a non-finite value at an abscissa that is not an endpoint', () => {
    // The blow-up is at the centre of the left half, which is an interior node, not an endpoint.
    const f = (x: number) => (x === 0.25 ? Infinity : Math.sin(40 * x));
    expect(() => gaussKronrod(f, 0, 1)).toThrow('gaussKronrod: f(0.25) returned a non-finite');
  });

  it('throws an Error when the function produces non-finite values', () => {
    expect(() => gaussKronrod(fBlowupMidpoint, 0, 1)).toThrow(
      /gaussKronrod: f\(0\.5\) returned a non-finite value \(Infinity\); cannot integrate/
    );
  });
});
