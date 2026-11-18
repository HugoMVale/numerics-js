import type { GaussKronrodOptions, GaussKronrodResult } from './types.js';

// Machine constants of QUADPACK, shared by the Gauss-Kronrod rules and the extrapolation.

/**
 * Largest relative spacing of double precision numbers (QUADPACK's `epmach`).
 *
 * @internal
 */
export const EPSILON = Number.EPSILON;

/**
 * Smallest positive normalized double (QUADPACK's `uflow`).
 *
 * @internal
 */
export const UNDERFLOW = 2.2250738585072014e-308;

/**
 * Largest finite double (QUADPACK's `oflow`), used as "no error estimate available yet".
 *
 * @internal
 */
export const OVERFLOW = Number.MAX_VALUE;

/**
 * Message of a result that met the requested tolerance. Shared by {@link gaussKronrod} and
 * {@link quad}, so that both report it identically.
 *
 * @internal
 */
export const CONVERGED_MESSAGE = 'Estimated error is at most `atol + rtol*|value|`.';

/**
 * Message of a result that reached the subinterval limit.
 *
 * @internal
 */
export function subintervalLimitMessage(limit: number): string {
  return `Maximum number of subintervals (${limit}) reached.`;
}

/**
 * Message of a result whose bisection reached floating-point resolution at a point where the
 * integrand is not finite.
 *
 * @internal
 */
export const ENDPOINT_SINGULARITY_MESSAGE =
  'Bisection reached floating-point resolution at a point where the integrand is not finite (probable endpoint singularity).';

/**
 * Error thrown by the integrand wrappers of {@link gaussKronrod} and {@link quad} when the
 * integrand (or a transform) is not finite at `point`, the abscissa that was being evaluated in
 * the variable the engine works in.
 *
 * The engines use `point` to tell the harmless-looking failure of an integrable endpoint
 * singularity apart from a genuinely bad integrand: see `integrateAdaptive` and
 * `extrapolatedIntegral`. It is an ordinary `Error` for callers, so it is documented as such.
 *
 * @internal
 */
export class NonFiniteIntegrandError extends Error {
  /** Abscissa, in the variable the engine integrates over, where the evaluation failed. */
  readonly point: number;

  constructor(message: string, point: number) {
    super(message);
    this.point = point;
  }
}

/**
 * An integrand wrapped by {@link countedIntegrand}.
 *
 * @internal
 */
export interface CountedIntegrand {
  /**
   * Evaluates the integrand at `x`.
   * @throws {NonFiniteIntegrandError} If the integrand is not finite at `x`.
   */
  readonly evaluate: (x: number) => number;
  /** Number of calls to {@link CountedIntegrand.evaluate} so far. */
  readonly evaluations: number;
}

/**
 * Wraps the integrand `f` so that its evaluations are counted and a non-finite value is reported
 * as a {@link NonFiniteIntegrandError}. The counter belongs to the returned object, not to the
 * module, so integrands that integrate in turn (nested integrals) do not interfere.
 *
 * @internal
 *
 * @param name Public function name, used as prefix of error messages.
 * @param f Integrand.
 * @returns The instrumented integrand.
 */
export function countedIntegrand(name: string, f: (x: number) => number): CountedIntegrand {
  let evaluations = 0;
  return {
    evaluate: (x: number): number => {
      evaluations++;
      const y = f(x);
      if (!Number.isFinite(y)) {
        throw new NonFiniteIntegrandError(
          `${name}: f(${x}) returned a non-finite value (${y}); cannot integrate.`,
          x
        );
      }
      return y;
    },
    get evaluations() {
      return evaluations;
    }
  };
}

/**
 * Midpoint of `[a, b]` that cannot overflow and always lies within the interval: the sum is used
 * only when the limits have opposite signs (so it cannot overflow), otherwise `a + (b - a) / 2`.
 *
 * This is the midpoint of the plain engine of {@link gaussKronrod}. The extrapolating engine of
 * {@link quad} uses {@link quadpackMidpoint} instead, to reproduce QUADPACK's results to the last bit.
 *
 * @internal
 */
export function midpoint(a: number, b: number): number {
  return (a < 0 && b > 0) || (a > 0 && b < 0) ? (a + b) / 2 : a + (b - a) / 2;
}

/**
 * Midpoint of `[a, b]` computed as in QUADPACK (`0.5 * (a + b)`), which is exact for all finite
 * inputs except when the sum overflows; in that case the halves are added instead.
 *
 * @internal
 */
export function quadpackMidpoint(a: number, b: number): number {
  const c = 0.5 * (a + b);
  return Number.isFinite(c) ? c : a / 2 + b / 2;
}

/**
 * Half-length of `[a, b]` (QUADPACK's `0.5 * (b - a)`), safe against overflow of `b - a`.
 *
 * @internal
 */
export function halfLength(a: number, b: number): number {
  const h = 0.5 * (b - a);
  return Number.isFinite(h) ? h : b / 2 - a / 2;
}

/**
 * Validates `options` and fills in the defaults.
 *
 * @internal
 *
 * @param name Public function name, used as prefix of error messages.
 * @param options User options.
 * @returns Validated options with defaults applied.
 * @throws {RangeError} If `atol` or `rtol` is not finite and non-negative, or if
 * `maxSubintervals` is not a finite positive integer.
 */
export function resolveGaussKronrodOptions(
  name: string,
  options: GaussKronrodOptions
): Required<GaussKronrodOptions> {
  const { atol = 1e-8, rtol = 1e-8, maxSubintervals = 200 } = options;

  if (!Number.isFinite(atol) || !Number.isFinite(rtol) || atol < 0 || rtol < 0) {
    throw new RangeError(
      `${name}: atol (${atol}) and rtol (${rtol}) must be finite and non-negative.`
    );
  }
  if (!Number.isInteger(maxSubintervals) || maxSubintervals < 1) {
    throw new RangeError(
      `${name}: maxSubintervals must be a finite positive integer, got ${maxSubintervals}.`
    );
  }
  return { atol, rtol, maxSubintervals };
}

/**
 * Result of integrating over an empty interval (`a === b`), which needs no evaluations.
 *
 * @internal
 */
export function emptyIntervalResult(): GaussKronrodResult {
  return {
    success: true,
    message: 'Empty integration interval (a === b).',
    evaluations: 0,
    subintervals: 0,
    value: 0,
    error: 0
  };
}

/**
 * Checks that the integrand is a function.
 *
 * @internal
 *
 * @param name Public function name, used as prefix of error messages.
 * @param f Value to check.
 * @throws {TypeError} If `f` is not a function.
 */
export function validateIntegrand(name: string, f: unknown): void {
  if (typeof f !== 'function') {
    throw new TypeError(`${name}: f must be a function.`);
  }
}
