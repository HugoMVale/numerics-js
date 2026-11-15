import { copysign } from '../math.js';

/**
 * Options for {@link brent}.
 */
export interface BrentOptions {
  /**
   * Stop when the search interval shrinks to approximately this width.
   * Must be finite and non-negative.
   * @default 1e-8
   */
  tolx?: number;
  /**
   * Maximum number of iterations. Must be a strictly positive integer.
   * @default 50
   */
  maxIter?: number;
}

/**
 * Result of a {@link brent} optimization.
 */
export interface BrentResult {
  /** Whether the optimization terminated successfully. */
  success: boolean;
  /** Human-readable description of the termination reason. */
  message: string;
  /** Number of function evaluations performed. */
  evaluations: number;
  /** Best point found. */
  x: number;
  /** Function value at `x`. */
  fx: number;
}

/**
 * Finds a local minimum of a scalar function `f(x)` using Brent's method, given
 * a bracketing interval.
 *
 * Brent's method is a derivative-free optimization algorithm that combines
 * golden-section search with inverse parabolic interpolation. It maintains a
 * bracketing interval known to contain a local minimum and iteratively
 * narrows it: when the function behaves smoothly it attempts a fast
 * parabolic step, and otherwise falls back to the more robust golden-section
 * step.
 *
 * Results are validated against `scipy.optimize.minimize_scalar` with `method="brent"`.
 * With the same `tolx`, the number of function calls is essentially identical to SciPy's
 * across the benchmark problems;
 * see the [comparison table](https://github.com/HugoMVale/numerics-js/blob/main/benchmarks/optimize/brent.scipy.benchmark.md).
 *
 * @param f Function to minimize.
 * @param xa One end of the bracketing interval.
 * @param xb Other end of the bracketing interval.
 * @param options Optional settings; see {@link BrentOptions}.
 * @returns A {@link BrentResult} with the minimizer, the function value and convergence information.
 * @throws {RangeError} If `xa` or `xb` is not finite, if `tolx` is not finite and
 * non-negative, or if `maxIter` is not a strictly positive integer.
 *
 * @example
 * ```ts
 * import { brent } from 'numerics-js/optimize';
 *
 * const f = (x: number): number => x ** 4 - x + 1;
 * const result = brent(f, -3.0, 3.0);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: '|dx| <= tolx',
 *   evaluations: 19,
 *   x: 0.6299605196128919,
 *   fx: 0.5275296062894226
 * }
 * ```
 */
export function brent(
  f: (x: number) => number,
  xa: number,
  xb: number,
  options: BrentOptions = {}
): BrentResult {
  const { tolx = 1e-8, maxIter = 50 } = options;

  if (!Number.isFinite(xa) || !Number.isFinite(xb)) {
    throw new RangeError(`brent: xa and xb must be finite, got ${xa} and ${xb}`);
  }
  if (!(Number.isFinite(tolx) && tolx >= 0)) {
    throw new RangeError(`brent: tolx must be finite and non-negative, got ${tolx}`);
  }
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new RangeError(`brent: maxIter must be a strictly positive integer, got ${maxIter}`);
  }

  // Golden ratio constant: (3 - sqrt(5)) / 2
  const GOLDEN = 0.38196601125010515;
  const EPS = Number.EPSILON;

  let a = Math.min(xa, xb);
  let b = Math.max(xa, xb);

  let x = a + GOLDEN * (b - a);
  let v = x;
  let w = x;

  let fx = f(x);
  let fv = fx;
  let fw = fx;
  let nFev = 1;

  let d = 0;
  let e = 0;

  for (let k = 0; k < maxIter; k++) {
    const xm = 0.5 * (a + b);
    const tol1 = EPS * Math.abs(x) + tolx / 3.0;
    const tol2 = 2.0 * tol1;

    if (Math.abs(x - xm) <= tol2 - 0.5 * (b - a)) {
      return {
        success: true,
        message: '|dx| <= tolx',
        evaluations: nFev,
        x,
        fx
      };
    }

    let p: number;
    let q: number;
    let r: number;
    let parabolic = false;

    if (Math.abs(e) > tol1) {
      // Fit parabola through (v, fv), (w, fw), (x, fx)
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) {
        p = -p;
      }
      q = Math.abs(q);
      r = e;
      e = d;

      // Is the parabolic step acceptable?
      if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (a - x) && p < q * (b - x)) {
        d = p / q;
        const u = x + d;
        // Convergence check for u
        if (u - a < tol2 || b - u < tol2) {
          d = copysign(tol1, xm - x);
        }
        parabolic = true;
      }
    }

    if (!parabolic) {
      // Golden-section step
      e = x < xm ? b - x : a - x;
      d = GOLDEN * e;
    }

    // Numerical safety: ensure step is at least tol1
    const u = Math.abs(d) >= tol1 ? x + d : x + copysign(tol1, d);
    const fu = f(u);
    nFev++;

    // Update points
    if (fu <= fx) {
      if (u >= x) {
        a = x;
      } else {
        b = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      if (fu <= fw || w === x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v === x || v === w) {
        v = u;
        fv = fu;
      }
    }
  }

  return {
    success: false,
    message: `Maximum number of iterations (${maxIter}) reached.`,
    evaluations: nFev,
    x,
    fx
  };
}
