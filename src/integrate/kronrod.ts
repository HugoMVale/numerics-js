import { EPSILON, UNDERFLOW, halfLength, midpoint, quadpackMidpoint } from './common.js';

/**
 * Non-negative abscissae of the 15-point Kronrod rule on `[-1, 1]`, in ascending order
 * (the full rule is symmetric about 0). The nodes at even indices (0, 2, 4, 6) are those
 * of the embedded 7-point Gauss-Legendre rule. Same values as QUADPACK's `qk15`.
 */
const KRONROD_NODES: readonly number[] = [
  0.0, 0.20778495500789848, 0.4058451513773972, 0.5860872354676911, 0.7415311855993945,
  0.8648644233597691, 0.9491079123427585, 0.9914553711208126
];

/** Weights of the 15-point Kronrod rule, one per entry of {@link KRONROD_NODES}. */
const KRONROD_WEIGHTS: readonly number[] = [
  0.20948214108472782, 0.20443294007529889, 0.19035057806478542, 0.1690047266392679,
  0.14065325971552592, 0.10479001032225019, 0.06309209262997856, 0.022935322010529224
];

/**
 * Weights of the embedded 7-point Gauss rule, aligned with {@link KRONROD_NODES}. They are
 * zero at the odd indices, which are Kronrod-only nodes.
 */
const GAUSS_WEIGHTS: readonly number[] = [
  0.4179591836734694, 0, 0.3818300505051189, 0, 0.27970539148927664, 0, 0.1294849661688697, 0
];

/** Indices into {@link KRONROD_NODES} of the Gauss nodes, outermost first (QUADPACK's order). */
const GAUSS_NODES = [6, 4, 2] as const;

/** Indices into {@link KRONROD_NODES} of the Kronrod-only nodes, outermost first. */
const KRONROD_ONLY_NODES = [7, 5, 3, 1] as const;

/**
 * Estimates produced by one application of the G7-K15 rule to a single subinterval, see
 * {@link gaussKronrod15}.
 *
 * @internal
 */
export interface LocalGaussKronrodResult {
  /** 15-point Kronrod estimate of the integral. */
  kronrod: number;
  /** Embedded 7-point Gauss estimate of the integral. */
  gauss: number;
  /** Estimated absolute error, `|kronrod - gauss|`. */
  error: number;
}

/**
 * Estimates produced by one application of the G7-K15 rule with QUADPACK's error scaling, see
 * {@link gaussKronrod15Scaled}.
 *
 * @internal
 */
export interface ScaledGaussKronrodResult {
  /** 15-point Kronrod estimate of the integral. */
  value: number;
  /** Estimated absolute error, scaled and bounded below by roundoff, as in QUADPACK's `qk15`. */
  error: number;
  /** Estimate of `∫|f|` over the subinterval (`resabs`). */
  resabs: number;
  /** Estimate of `∫|f - mean(f)|` over the subinterval (`resasc`). */
  resasc: number;
}

/**
 * Applies the 7-15 Gauss-Kronrod rule to one subinterval and returns the
 * embedded Gauss and Kronrod estimates.
 *
 * The error estimate is the plain difference `|kronrod - gauss|`, which approximates the
 * error of the (less accurate) Gauss estimate. It therefore overestimates the error of the
 * Kronrod value by a wide margin for smooth integrands, where Kronrod converges much faster,
 * and is close to it, or even below it for strong singularities, where both rules converge
 * slowly.
 * No further scaling is applied (as, e.g., in QUADPACK's `qk15`, see
 * {@link gaussKronrod15Scaled}), because that inflates the estimate for such hard integrands,
 * and its roundoff floor can make tight tolerances unreachable.
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
  const half = halfLength(a, b);
  const center = midpoint(a, b);

  // The center node is shared by both rules and is not mirrored.
  const fCenter = evalF(center);
  let kronrodSum = KRONROD_WEIGHTS[0] * fCenter;
  let gaussSum = GAUSS_WEIGHTS[0] * fCenter;

  for (let i = 1; i < KRONROD_NODES.length; i++) {
    const dx = half * KRONROD_NODES[i];
    const fPair = evalF(center + dx) + evalF(center - dx);
    kronrodSum += KRONROD_WEIGHTS[i] * fPair;
    gaussSum += GAUSS_WEIGHTS[i] * fPair;
  }

  const kronrod = half * kronrodSum;
  const gauss = half * gaussSum;

  return { kronrod, gauss, error: Math.abs(kronrod - gauss) };
}

/**
 * G7-K15 rule on `[a, b]` with QUADPACK's (`qk15`) error estimate, which the roundoff and
 * divergence logic of the extrapolation relies on.
 *
 * The raw difference `d = |K15 - G7|` is a pessimistic estimate for smooth integrands (the
 * Kronrod rule is far more accurate than the Gauss rule), but too optimistic for hard ones. It is
 * rescaled with `resasc = ∫|f - mean(f)|`, which measures how rough the integrand is:
 * `error = resasc * min(1, (200 d / resasc)^1.5)`. For a smooth integrand `d` is much smaller than
 * `resasc` and the error drops faster than `d` (the 1.5 power); for a rough integrand it saturates
 * at `resasc`, the largest error that can be made. The error is finally never allowed below
 * `50 eps ∫|f|`, the roundoff level, so that asking for more accuracy than is representable can be
 * detected instead of chased.
 *
 * Sums are accumulated in QUADPACK's order (Gauss pairs from the outermost node inwards, then the
 * Kronrod-only pairs), so that results agree with QUADPACK's to the last bit.
 *
 * @internal
 *
 * @param evalF Instrumented function evaluator.
 * @param a Lower limit of integration `a`.
 * @param b Upper limit of integration `b`.
 * @returns Kronrod estimate, scaled error estimate and the auxiliary integrals `resabs`, `resasc`.
 */
export function gaussKronrod15Scaled(
  evalF: (x: number) => number,
  a: number,
  b: number
): ScaledGaussKronrodResult {
  const mid = quadpackMidpoint(a, b);
  const half = halfLength(a, b);
  const absHalf = Math.abs(half);
  const fv1 = new Float64Array(KRONROD_NODES.length);
  const fv2 = new Float64Array(KRONROD_NODES.length);

  const fCenter = evalF(mid);
  let resg = fCenter * GAUSS_WEIGHTS[0];
  let resk = fCenter * KRONROD_WEIGHTS[0];
  let resabs = Math.abs(resk);

  for (const i of GAUSS_NODES) {
    const dx = half * KRONROD_NODES[i];
    const f1 = evalF(mid - dx);
    const f2 = evalF(mid + dx);
    fv1[i] = f1;
    fv2[i] = f2;
    const fsum = f1 + f2;
    resg += GAUSS_WEIGHTS[i] * fsum;
    resk += KRONROD_WEIGHTS[i] * fsum;
    resabs += KRONROD_WEIGHTS[i] * (Math.abs(f1) + Math.abs(f2));
  }
  for (const i of KRONROD_ONLY_NODES) {
    const dx = half * KRONROD_NODES[i];
    const f1 = evalF(mid - dx);
    const f2 = evalF(mid + dx);
    fv1[i] = f1;
    fv2[i] = f2;
    const fsum = f1 + f2;
    resk += KRONROD_WEIGHTS[i] * fsum;
    resabs += KRONROD_WEIGHTS[i] * (Math.abs(f1) + Math.abs(f2));
  }

  // resk / 2 is the mean value of f (the integral divided by the length).
  const reskh = resk * 0.5;
  let resasc = KRONROD_WEIGHTS[0] * Math.abs(fCenter - reskh);
  for (let i = KRONROD_NODES.length - 1; i >= 1; i--) {
    resasc += KRONROD_WEIGHTS[i] * (Math.abs(fv1[i] - reskh) + Math.abs(fv2[i] - reskh));
  }

  const value = resk * half;
  resabs *= absHalf;
  resasc *= absHalf;
  let error = Math.abs((resk - resg) * half);
  if (resasc !== 0 && error !== 0) {
    error = resasc * Math.min(1, ((200 * error) / resasc) ** 1.5);
  }
  if (resabs > UNDERFLOW / (50 * EPSILON)) {
    error = Math.max(EPSILON * 50 * resabs, error);
  }
  return { value, error, resabs, resasc };
}
