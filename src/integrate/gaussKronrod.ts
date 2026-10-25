/** Result of a {@link gaussKronrod} quadrature. */
export interface GaussKronrodResult {
  /** Estimated value of the definite integral. */
  value: number;
  /** Estimated absolute integration error. */
  error: number;
  /** Number of integrand evaluations performed. */
  evaluations: number;
  /** Whether the requested error tolerance was reached. */
  converged: boolean;
  /** Number of final subintervals used by the adaptive algorithm. */
  subintervals: number;
}

/**
 * Options for {@link gaussKronrod} and, through {@link QuadOptions}, for {@link quad}.
 *
 * The adaptive algorithm stops as soon as the estimated error of the whole
 * integral satisfies `error <= atol + rtol * |value|`, the same convention used
 * by the ODE solvers.
 */
export interface GaussKronrodOptions {
  /**
   * Absolute error tolerance. It dominates when the integral is close to zero.
   * @default 1e-8
   */
  atol?: number;
  /**
   * Relative error tolerance, with respect to the magnitude of the estimated
   * integral. With `atol = 0` the stopping criterion is purely relative, hence
   * invariant under a rescaling of the integrand.
   * @default 1e-8
   */
  rtol?: number;
  /**
   * Safety limit on the total number of subintervals (panels) of the whole
   * integral, including the initial ones.
   * @default 200
   */
  maxSubintervals?: number;
}

interface LocalGaussKronrodResult {
  kronrod: number;
  gauss: number;
  error: number;
}

const KRONROD_NODES: readonly number[] = [
  0.0, 0.20778495500789848, 0.4058451513773972, 0.5860872354676911, 0.7415311855993945,
  0.8648644233597691, 0.9491079123427585, 0.9914553711208126
];

const KRONROD_WEIGHTS: readonly number[] = [
  0.20948214108472782, 0.20443294007529889, 0.19035057806478542, 0.1690047266392679,
  0.14065325971552592, 0.10479001032225019, 0.06309209262997856, 0.022935322010529224
];

const GAUSS_WEIGHTS: readonly number[] = [
  0.4179591836734694, 0, 0.3818300505051189, 0, 0.27970539148927664, 0, 0.1294849661688697, 0
];

interface Panel {
  a: number;
  b: number;
  kronrod: number;
  error: number;
}

function midpoint(a: number, b: number): number {
  return (a < 0 && b > 0) || (a > 0 && b < 0) ? (a + b) / 2 : a + (b - a) / 2;
}

function halfLength(a: number, b: number): number {
  const length = b - a;
  return Number.isFinite(length) ? length / 2 : b / 2 - a / 2;
}

/**
 * Applies the 7-15 Gauss-Kronrod rule to one panel and returns the
 * embedded Gauss and Kronrod estimates.
 *
 * @internal
 *
 * @param evalF Instrumented function evaluator.
 * @param a Lower limit of integration `a`.
 * @param b Upper limit of integration `b`.
 * @returns Local quadrature estimates and estimated local error.
 */
export function gaussKronrod15(
  evalF: (x: number) => number,
  a: number,
  b: number
): LocalGaussKronrodResult {
  const panelHalfLength = halfLength(a, b);
  const center = midpoint(a, b);

  let kronrodSum = 0;
  let gaussSum = 0;

  for (let i = 0; i < KRONROD_NODES.length; i++) {
    const dx = panelHalfLength * KRONROD_NODES[i];
    const kw = KRONROD_WEIGHTS[i];
    const gw = GAUSS_WEIGHTS[i];

    if (i === 0) {
      const f0 = evalF(center);
      kronrodSum += kw * f0;
      gaussSum += gw * f0;
    } else {
      const fPlus = evalF(center + dx);
      const fMinus = evalF(center - dx);
      kronrodSum += kw * (fPlus + fMinus);
      gaussSum += gw * (fPlus + fMinus);
    }
  }

  const kronrod = panelHalfLength * kronrodSum;
  const gauss = panelHalfLength * gaussSum;

  return { kronrod, gauss, error: Math.abs(kronrod - gauss) };
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
    throw new RangeError(`${name}: maxSubintervals must be a finite positive integer`);
  }
  return { atol, rtol, maxSubintervals };
}

/**
 * Adaptive G7-K15 engine shared by {@link gaussKronrod} and {@link quad}.
 *
 * All initial panels (`points[i]`, `points[i + 1]`) share one global error budget
 * and one panel limit: at each step the panel with the largest error estimate is
 * bisected, until `totalError <= atol + rtol * |totalValue|` for the whole integral.
 *
 * @internal
 *
 * @param name Public function name, used as prefix of error messages.
 * @param f Integrand.
 * @param points Strictly increasing finite integration points, at least two.
 * @param options Validated options, see {@link resolveGaussKronrodOptions}.
 * @returns Integral over `[points[0], points[points.length - 1]]` (ascending orientation).
 */
export function integrateAdaptive(
  name: string,
  f: (x: number) => number,
  points: readonly number[],
  options: Required<GaussKronrodOptions>
): GaussKronrodResult {
  const { atol, rtol, maxSubintervals } = options;

  let evaluations = 0;
  const evalF = (x: number): number => {
    evaluations++;
    const y = f(x);
    if (!Number.isFinite(y)) {
      throw new Error(`${name}: f(${x}) returned a non-finite value (${y}); cannot integrate.`);
    }
    return y;
  };

  const panels: Panel[] = [];
  let totalValue = 0;
  let totalError = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const local = gaussKronrod15(evalF, points[i], points[i + 1]);
    panels.push({ a: points[i], b: points[i + 1], kronrod: local.kronrod, error: local.error });
    totalValue += local.kronrod;
    totalError += local.error;
  }

  let converged = true;
  while (totalError > atol + rtol * Math.abs(totalValue)) {
    if (panels.length >= maxSubintervals) {
      converged = false;
      break;
    }

    let worstIdx = 0;
    for (let i = 1; i < panels.length; i++) {
      if (panels[i].error > panels[worstIdx].error) worstIdx = i;
    }
    const worst = panels[worstIdx];

    const mid = midpoint(worst.a, worst.b);
    const left = gaussKronrod15(evalF, worst.a, mid);
    const right = gaussKronrod15(evalF, mid, worst.b);

    totalValue += left.kronrod + right.kronrod - worst.kronrod;
    totalError += left.error + right.error - worst.error;

    panels[worstIdx] = { a: worst.a, b: mid, kronrod: left.kronrod, error: left.error };
    panels.push({ a: mid, b: worst.b, kronrod: right.kronrod, error: right.error });
  }

  return {
    value: totalValue,
    error: totalError,
    evaluations,
    converged,
    subintervals: panels.length
  };
}

/**
 * Numerically integrate `f` over `[a, b]` using adaptive Gauss-Kronrod
 * (G7-K15) quadrature.
 *
 * This function is suitable for numerically integrating smooth functions over finite
 * intervals. For functions with discontinuities or sharp features, or infinite domains,
 * use {@link quad} instead.
 *
 * The interval is bisected adaptively until the estimated error of the whole integral
 * satisfies `error <= atol + rtol * |value|`.
 *
 * @param f Scalar function `f` to integrate.
 * @param a Lower bound of integration `a`.
 * @param b Upper bound of integration `b`.
 * @param options Optional settings, see {@link GaussKronrodOptions}.
 * @returns Quadrature output including final value, estimated error, and diagnostics.
 * @throws {RangeError} If `a` or `b` are non-finite numbers, `atol` or `rtol` is not a
 * finite non-negative number, or `maxSubintervals` is not a finite positive integer.
 * @throws {Error} If `f` evaluates to a non-finite value.
 *
 * @example
 * ```ts
 * import { gaussKronrod } from 'numerics-js/integrate';
 *
 * // Integrate an oscillatory function that requires adaptive subdivision.
 * const result = gaussKronrod(x => Math.sin(100 * x), 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   value: 0.0013768112771228931,
 *   error: 2.0058021368820277e-9,
 *   evaluations: 465,
 *   converged: true,
 *   subintervals: 16
 * }
 * ```
 *
 * @example
 * ```ts
 * // A purely relative tolerance is invariant under rescaling of the integrand:
 * // the same panels are used whether the integral is of order 1e-12 or 1e12.
 * const small = gaussKronrod(x => 1e-12 * Math.exp(Math.sin(30 * x)), 0, 1, { atol: 0, rtol: 1e-6 });
 * const large = gaussKronrod(x => 1e12 * Math.exp(Math.sin(30 * x)), 0, 1, { atol: 0, rtol: 1e-6 });
 * console.log(small.subintervals, large.subintervals);
 * ```
 *
 * Output:
 * ```text
 * 10 10
 * ```
 */
export function gaussKronrod(
  f: (x: number) => number,
  a: number,
  b: number,
  options: GaussKronrodOptions = {}
): GaussKronrodResult {
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    throw new RangeError('gaussKronrod: a and b must be finite numbers');
  }
  const resolved = resolveGaussKronrodOptions('gaussKronrod', options);
  if (a === b) {
    return { value: 0, error: 0, evaluations: 0, converged: true, subintervals: 0 };
  }

  const sign = a > b ? -1 : 1;
  const result = integrateAdaptive('gaussKronrod', f, [Math.min(a, b), Math.max(a, b)], resolved);
  result.value *= sign;
  return result;
}
