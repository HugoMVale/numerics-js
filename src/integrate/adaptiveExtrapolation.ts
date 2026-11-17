import {
  GAUSS_WEIGHTS,
  KRONROD_NODES,
  KRONROD_WEIGHTS,
  NonFiniteIntegrandError,
  type GaussKronrodOptions,
  type GaussKronrodResult
} from './gaussKronrod.js';
import { EpsilonTable } from './wynnEpsilon.js';

// This module is a port of QUADPACK's `dqagpe` (R. Piessens, E. de Doncker-Kapenga,
// C. Ueberhuber, D. Kahaner, "QUADPACK: A Subroutine Package for Automatic Integration",
// Springer, 1983) with the G7-K15 rule, i.e. adaptive bisection with extrapolation by the
// epsilon algorithm and with user-supplied break points. Without break points it is the
// algorithm of `dqagse`, which SciPy's `quad` uses. The control flow is the original one;
// only the indexing is zero-based, and the arrays grow as needed instead of being allocated
// with the limit.

/** Largest relative spacing of doubles (QUADPACK's `epmach`). */
const EPSILON = Number.EPSILON;

/** Smallest positive normalized double (QUADPACK's `uflow`). */
const UNDERFLOW = 2.2250738585072014e-308;

/** Largest finite double (QUADPACK's `oflow`): "no error estimate yet". */
const OVERFLOW = Number.MAX_VALUE;

/**
 * Midpoint of `[a, b]`, computed as in QUADPACK (`0.5 * (a + b)`), which is exact for all
 * finite inputs except when the sum overflows; in that case the halves are added instead.
 */
function center(a: number, b: number): number {
  const c = 0.5 * (a + b);
  return Number.isFinite(c) ? c : a / 2 + b / 2;
}

/** Half-length of `[a, b]`, computed as in QUADPACK (`0.5 * (b - a)`), safe against overflow. */
function halfLength(a: number, b: number): number {
  const h = 0.5 * (b - a);
  return Number.isFinite(h) ? h : b / 2 - a / 2;
}

/** Result of one application of the G7-K15 rule with QUADPACK's error scaling. */
interface ScaledRuleResult {
  /** 15-point Kronrod estimate of the integral. */
  value: number;
  /** Estimated absolute error, scaled and bounded below by roundoff, as in QUADPACK's `qk15`. */
  error: number;
  /** Estimate of `∫|f|` over the subinterval (`resabs`). */
  resabs: number;
  /** Estimate of `∫|f - mean(f)|` over the subinterval (`resasc`). */
  resasc: number;
}

/** Indices into {@link KRONROD_NODES} of the Gauss nodes, outermost first (QUADPACK's order). */
const GAUSS_NODES = [6, 4, 2] as const;

/** Indices into {@link KRONROD_NODES} of the Kronrod-only nodes, outermost first. */
const KRONROD_ONLY_NODES = [7, 5, 3, 1] as const;

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
 */
function kronrod15Scaled(evalF: (x: number) => number, a: number, b: number): ScaledRuleResult {
  const mid = center(a, b);
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

/**
 * Keeps `iord` (QUADPACK's `dqpsrt`), the pointers to the subintervals, ordered by decreasing
 * error estimate after the subinterval `maxerr` has been replaced by its two halves, and returns
 * the new position `nrmax` of the next subinterval to bisect.
 *
 * Only the first `jupbn` pointers are kept sorted, where `jupbn` is `last` while that is at most
 * `limit / 2 + 2` and `limit + 3 - last` afterwards. Subintervals with the smallest errors are
 * hardly ever selected, and this bound limits the cost of an insertion; those that are left out
 * are not needed as long as the limit is not exceeded, since each bisection adds one subinterval.
 *
 * `nrmax` is the position in `iord` of the subinterval that was just bisected, i.e. the
 * `nrmax`-th largest error (zero-based). It is normally 0. It is larger when the extrapolation
 * skipped the larger subintervals (see {@link extrapolatedIntegral}) and, if the bisection
 * increased the error of the subinterval (difficult integrand), it is first moved back up to its
 * place. Then the two new errors are inserted by walking down from `nrmax` for the larger one and
 * up from the bottom of the ordered part for the smaller one.
 *
 * @param limit Maximum number of subintervals.
 * @param last Number of subintervals, including the new one (at index `last - 1`).
 * @param maxerr Index of the subinterval that has just been bisected, holding the larger new error.
 * @param elist Error estimates of the subintervals, already updated.
 * @param iord Pointers to the subintervals; updated.
 * @param nrmax Position in `iord` of `maxerr` before the update.
 * @returns The position in `iord` of the subinterval to bisect next.
 */
function updateOrdering(
  limit: number,
  last: number,
  maxerr: number,
  elist: number[],
  iord: number[],
  nrmax: number
): number {
  if (last <= 2) {
    iord[0] = 0;
    iord[1] = 1;
    return nrmax;
  }

  const errmax = elist[maxerr];
  if (nrmax !== 0) {
    const ido = nrmax;
    for (let i = 0; i < ido; i++) {
      const isucc = iord[nrmax - 1];
      if (errmax <= elist[isucc]) break;
      iord[nrmax] = isucc;
      nrmax--;
    }
  }

  // Number of pointers that are kept sorted.
  let jupbn = last;
  if (last > Math.floor(limit / 2) + 2) jupbn = limit + 3 - last;
  const errmin = elist[last - 1];

  // Insert errmax by walking down from just after nrmax.
  const jbnd = jupbn - 2;
  const ibeg = nrmax + 1;
  let i = ibeg;
  let inserted = false;
  for (; i <= jbnd; i++) {
    const isucc = iord[i];
    if (errmax >= elist[isucc]) {
      inserted = true;
      break;
    }
    iord[i - 1] = isucc;
  }

  if (!inserted) {
    iord[jbnd] = maxerr;
    iord[jupbn - 1] = last - 1;
    return nrmax;
  }

  // Insert errmin by walking up from the bottom of the sorted part.
  iord[i - 1] = maxerr;
  let k = jbnd;
  for (let j = i; j <= jbnd; j++) {
    const isucc = iord[k];
    if (errmin < elist[isucc]) {
      iord[k + 1] = last - 1;
      return nrmax;
    }
    iord[k + 1] = isucc;
    k--;
  }
  iord[i] = last - 1;
  return nrmax;
}

/**
 * Termination status of {@link extrapolatedIntegral}, QUADPACK's `ier` after its final
 * renumbering:
 * - 0: the requested accuracy was achieved;
 * - 1: the maximum number of subintervals was reached;
 * - 2: roundoff error was detected in the bisection, which prevents the requested accuracy;
 * - 3: bad integrand behaviour (singularity or discontinuity) at a point of the range, or
 *   bisection reached floating-point resolution at a point where the integrand is not finite;
 * - 4: roundoff error was detected in the extrapolation table: the requested accuracy cannot be
 *   achieved;
 * - 5: the integral is probably divergent, or converges too slowly.
 *
 * For a status other than 0 the integral estimate is the best one found; for 5 it is meaningless.
 */
export type ExtrapolationStatus = 0 | 1 | 2 | 3 | 4 | 5;

/** Result of {@link extrapolatedIntegral}. */
export interface ExtrapolatedIntegral {
  /** Estimate of the integral. */
  value: number;
  /** Estimated absolute error of `value`. */
  error: number;
  /** Termination status. */
  status: ExtrapolationStatus;
  /** Number of integrand evaluations. */
  evaluations: number;
  /** Number of subintervals of the final partition. */
  subintervals: number;
  /**
   * Whether status 3 was caused by bisection reaching floating-point resolution at a point where
   * the integrand is not finite (an endpoint singularity), instead of by the bad-behaviour test.
   */
  endpointSingularity: boolean;
}

/**
 * Adaptive Gauss-Kronrod (G7-K15) integration with extrapolation by the epsilon algorithm, which
 * is QUADPACK's `dqagpe`.
 *
 * The integral is estimated over the initial subintervals defined by `points`, then the subinterval
 * with the largest error estimate is bisected repeatedly, as in a plain adaptive scheme. The
 * additional ingredient is the sequence of the approximations to the integral after every
 * "level": the sums of the integrals over all subintervals, obtained at the moments at which all the
 * subintervals that are larger than the smallest ones considered so far have been dealt with.
 * Local singularities (and discontinuities) make the errors of the subintervals that contain them
 * decrease slowly and in a regular way as these are bisected, so that the sequence of the
 * approximations is extrapolated with the epsilon algorithm ({@link EpsilonTable}) to
 * obtain a better estimate of the limit and of its error, often with a small fraction of the
 * subdivisions that the plain scheme needs.
 *
 * To be able to extrapolate, the algorithm must be allowed to bisect the smallest subintervals
 * first. At any step it therefore distinguishes
 * - the normal mode, in which the subinterval with the largest error is bisected, unless it is the
 *   next to be bisected at the current `levmax` level, in which case
 * - the extrapolation mode (`extrap`) begins: before bisecting the smallest subintervals, the
 *   errors of the larger ones (`erlarg`, the sum of their errors) must have been decreased until
 *   they are below the tolerance of the extrapolation (`ertest`); only then is the approximation
 *   added to the table and extrapolated, and the level `levmax` is increased.
 *
 * Each subinterval has a `level`: if `[aa, bb]` was obtained from the initial subinterval
 * `[p1, p2]` by `l` bisections, its level is `l` and `bb - aa = (p2 - p1) / 2^l`. The "smallest"
 * subintervals of the algorithm are those with level `levmax - 1` or more. This replaces the test
 * on the length, `small`, of the QUADPACK routine without break points, which assumed that there was
 * one initial subinterval.
 *
 * The estimated result and error returned are the extrapolated ones if the extrapolation was
 * decisive; otherwise they are the plain sums over the subintervals.
 *
 * Several tests guard against the failure modes of the extrapolation:
 * - `iroff1`, `iroff2`, `iroff3` count bisections for which the integral and the error estimate
 *   hardly change (a symptom of roundoff, which can not be fixed by further bisection), or for
 *   which the sum of the errors increases (an integrand that is too hard for the rule);
 *   if they are too many the loop ends with status 2. `ierro` (set when `iroff2` is large,
 *   i.e. in extrapolation mode) additionally disables the wait for `erlarg` to fall below `ertest`.
 * - The bisection is stopped, with status 3, when the subinterval to bisect is so small that
 *   its endpoints are nearly equal to the point itself (the relative width is below
 *   `100 eps`): that is the sign of a singularity or of a discontinuity that the algorithm can
 *   not resolve. The same status is given if the integrand is not finite at a node that
 *   coincides with an endpoint of the subinterval (the endpoint singularity of
 *   `endpointSingularity`).
 * - If, after five extrapolations in a row without improvement of the estimate, the extrapolated
 *   error is much smaller than the sum of the errors, the table is not trusted any more and the
 *   loop ends with status 4 (roundoff in the extrapolation table). The same status is assigned,
 *   at the end, if the extrapolated and the summed estimates are inconsistent.
 * - A final test on divergence: unless the integrand is oscillatory (the result is a small part of
 *   `∫|f|`), a result that is not within a factor of 100 of the sum, or an error larger than
 *   the integral, gives status 5. This does not detect divergence with certainty, only gives up
 *   on an integral that does not look like converging.
 * - Subintervals whose error estimate is equal to `resasc` (`ndin`: the rule saw nothing but
 *   noise) are given, after the first integration over the initial subintervals, the sum of
 *   the errors of all initial subintervals, so that they are subdivided first.
 *
 * Deviations from QUADPACK (both concern a `limit` equal to the number of initial subintervals,
 * i.e. no room left to bisect; `dqagpe` requires a larger one):
 * - `dqagpe` reports "limit reached" even if the first integration already met the accuracy, and,
 *   with a single initial subinterval, goes on to read an uninitialized subinterval. Here the
 *   first integration is returned with status 0 if it met the accuracy and status 1 otherwise.
 *
 * @param name Public function name, used as prefix of error messages.
 * @param f Integrand.
 * @param points Strictly increasing finite integration points, at least two: the limits of
 * integration and the break points between them.
 * @param options Validated options, see `resolveGaussKronrodOptions`.
 * @returns Integral over `[points[0], points[points.length - 1]]` (ascending orientation).
 */
export function extrapolatedIntegral(
  name: string,
  f: (x: number) => number,
  points: readonly number[],
  options: Required<GaussKronrodOptions>
): ExtrapolatedIntegral {
  const { atol: epsabs, rtol: epsrel, maxSubintervals: limit } = options;
  const nint = points.length - 1;
  // `dqagpe`'s `npts2`: the number of limits and break points.
  const npts2 = nint + 1;

  let evaluations = 0;
  const evalF = (x: number): number => {
    evaluations++;
    const y = f(x);
    if (!Number.isFinite(y)) {
      throw new NonFiniteIntegrandError(
        `${name}: f(${x}) returned a non-finite value (${y}); cannot integrate.`,
        x
      );
    }
    return y;
  };

  // Left endpoints, right endpoints, integrals, errors and levels of the subintervals, and the
  // pointers to the subintervals in order of decreasing error.
  const alist: number[] = [];
  const blist: number[] = [];
  const rlist: number[] = [];
  const elist: number[] = [];
  const level: number[] = [];
  const iord: number[] = [];

  // First integration over the initial subintervals.
  let result = 0;
  let abserr = 0;
  let resabs = 0;
  const ndin: boolean[] = [];
  for (let i = 0; i < nint; i++) {
    const q = kronrod15Scaled(evalF, points[i], points[i + 1]);
    abserr += q.error;
    result += q.value;
    ndin.push(q.error === q.resasc && q.error !== 0);
    resabs += q.resabs;
    level.push(0);
    elist.push(q.error);
    alist.push(points[i]);
    blist.push(points[i + 1]);
    rlist.push(q.value);
    iord.push(i);
  }
  let errsum = 0;
  for (let i = 0; i < nint; i++) {
    if (ndin[i]) elist[i] = abserr;
    errsum += elist[i];
  }

  const finish = (
    value: number,
    error: number,
    status: number,
    subintervals: number,
    endpointSingularity = false
  ): ExtrapolatedIntegral => ({
    value,
    error,
    status: status as ExtrapolationStatus,
    evaluations,
    subintervals,
    endpointSingularity
  });

  // Test on accuracy.
  let last = nint;
  const dres = Math.abs(result);
  let errbnd = Math.max(epsabs, epsrel * dres);
  let ier = 0;
  if (abserr <= 100 * EPSILON * resabs && abserr > errbnd) ier = 2;
  if (nint > 1) {
    // Selection sort of the pointers by decreasing error.
    for (let i = 0; i < nint - 1; i++) {
      let ind1 = iord[i];
      let k = i;
      for (let j = i + 1; j < nint; j++) {
        const ind2 = iord[j];
        if (elist[ind1] > elist[ind2]) continue;
        ind1 = ind2;
        k = j;
      }
      if (ind1 === iord[i]) continue;
      iord[k] = iord[i];
      iord[i] = ind1;
    }
  }
  // No room to bisect: the limit is reached unless the first integration already met the accuracy.
  if (limit < npts2 && abserr > errbnd) ier = 1;
  if (ier !== 0 || abserr <= errbnd) return finish(result, abserr, ier, last);

  // Initialization of the extrapolation.
  const table = new EpsilonTable();
  table.push(result);
  let maxerr = iord[0];
  let errmax = elist[maxerr];
  let area = result;
  let nrmax = 0;
  let ktmin = 0;
  let extrap = false;
  let noext = false;
  let erlarg = errsum;
  let ertest = errbnd;
  let levmax = 1;
  let iroff1 = 0;
  let iroff2 = 0;
  let iroff3 = 0;
  let ierro = 0;
  let correc = 0;
  abserr = OVERFLOW;
  // Whether the integrand is not oscillating: the integral is not much smaller than ∫|f|.
  const ksgn = dres >= (1 - 50 * EPSILON) * resabs ? 1 : -1;
  let useSum = false;
  let endpointSingularity = false;

  // Main loop: bisect the subinterval with the nrmax-th largest error estimate.
  for (last = npts2; last <= limit; last++) {
    const levcur = level[maxerr] + 1;
    const a1 = alist[maxerr];
    const b1 = center(alist[maxerr], blist[maxerr]);
    const a2 = b1;
    const b2 = blist[maxerr];
    const erlast = errmax;

    let q1: ScaledRuleResult;
    let q2: ScaledRuleResult;
    try {
      q1 = kronrod15Scaled(evalF, a1, b1);
      q2 = kronrod15Scaled(evalF, a2, b2);
    } catch (e) {
      // Bisection toward a singular endpoint reached floating-point resolution: a node rounded
      // onto the endpoint, where the integrand is not finite. The bisection has not been
      // recorded, so the data are those of the last complete step.
      const atEndpoint =
        e instanceof NonFiniteIntegrandError &&
        (e.point === a1 || e.point === b1 || e.point === b2);
      if (!atEndpoint) throw e;
      ier = 4;
      endpointSingularity = true;
      last--;
      break;
    }
    const area1 = q1.value;
    const error1 = q1.error;
    const defab1 = q1.resasc;
    const area2 = q2.value;
    const error2 = q2.error;
    const defab2 = q2.resasc;

    // Improve the previous approximations to the integral and the error, and test for accuracy.
    const area12 = area1 + area2;
    const erro12 = error1 + error2;
    errsum = errsum + erro12 - errmax;
    area = area + area12 - rlist[maxerr];
    if (defab1 !== error1 && defab2 !== error2) {
      if (!(Math.abs(rlist[maxerr] - area12) > 1e-5 * Math.abs(area12) || erro12 < 0.99 * errmax)) {
        if (extrap) iroff2++;
        else iroff1++;
      }
      if (last > 10 && erro12 > errmax) iroff3++;
    }
    const added = last - 1;
    level[maxerr] = levcur;
    level[added] = levcur;
    rlist[maxerr] = area1;
    rlist[added] = area2;
    errbnd = Math.max(epsabs, epsrel * Math.abs(area));

    // Test for roundoff error and eventually set the error flag.
    if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
    if (iroff2 >= 5) ierro = 3;

    // Set the error flag in the case that the number of subintervals equals the limit.
    if (last === limit) ier = 1;

    // Set the error flag in the case of bad integrand behaviour at a point of the range.
    if (
      Math.max(Math.abs(a1), Math.abs(b2)) <=
      (1 + 100 * EPSILON) * (Math.abs(a2) + 1000 * UNDERFLOW)
    ) {
      ier = 4;
    }

    // Append the newly created subintervals to the list.
    if (error2 > error1) {
      alist[maxerr] = a2;
      alist[added] = a1;
      blist[added] = b1;
      rlist[maxerr] = area2;
      rlist[added] = area1;
      elist[maxerr] = error2;
      elist[added] = error1;
    } else {
      alist[added] = a2;
      blist[maxerr] = b1;
      blist[added] = b2;
      elist[maxerr] = error1;
      elist[added] = error2;
    }

    // Maintain the descending order of the errors and select the next subinterval to bisect.
    if (iord.length < last) iord.push(0);
    nrmax = updateOrdering(limit, last, maxerr, elist, iord, nrmax);
    maxerr = iord[nrmax];
    errmax = elist[maxerr];

    if (errsum <= errbnd) {
      useSum = true;
      break;
    }
    if (ier !== 0) break;
    if (noext) continue;
    erlarg -= erlast;
    if (levcur + 1 <= levmax) erlarg += erro12;
    if (!extrap) {
      // Test whether the subinterval to be bisected next is the smallest one.
      if (level[maxerr] + 1 <= levmax) continue;
      extrap = true;
      nrmax = 1;
    }
    if (ierro !== 3 && erlarg > ertest) {
      // The smallest subinterval has the largest error. Before bisecting it, decrease the sum of
      // the errors over the larger subintervals, i.e. look for one of them to bisect.
      let jupbnd = last;
      if (last > 2 + Math.floor(limit / 2)) jupbnd = limit + 3 - last;
      let larger = false;
      for (let k = nrmax; k < jupbnd; k++) {
        maxerr = iord[nrmax];
        errmax = elist[maxerr];
        if (level[maxerr] + 1 <= levmax) {
          larger = true;
          break;
        }
        nrmax++;
      }
      if (larger) continue;
    }

    // Perform the extrapolation.
    table.push(area);
    if (table.length > 2) {
      const { result: reseps, error: abseps } = table.extrapolate();
      ktmin++;
      if (ktmin > 5 && abserr < 1e-3 * errsum) ier = 5;
      if (abseps < abserr) {
        ktmin = 0;
        abserr = abseps;
        result = reseps;
        correc = erlarg;
        ertest = Math.max(epsabs, epsrel * Math.abs(reseps));
        if (abserr < ertest) break;
      }
      // Prepare the bisection of the smallest subinterval.
      if (table.length === 1) noext = true;
      if (ier >= 5) break;
    }
    maxerr = iord[0];
    errmax = elist[maxerr];
    nrmax = 0;
    extrap = false;
    levmax++;
    erlarg = errsum;
  }
  if (last > limit) last = limit;

  // Set the final result.
  if (!useSum) {
    if (abserr === OVERFLOW) {
      useSum = true;
    } else {
      let testDivergence = true;
      if (ier + ierro !== 0) {
        if (ierro === 3) abserr += correc;
        if (ier === 0) ier = 3;
        if (result !== 0 && area !== 0) {
          if (abserr / Math.abs(result) > errsum / Math.abs(area)) useSum = true;
        } else if (abserr > errsum) {
          useSum = true;
        } else if (area === 0) {
          testDivergence = false;
        }
      }
      if (!useSum && testDivergence) {
        // Test on divergence.
        const oscillatory =
          ksgn === -1 && Math.max(Math.abs(result), Math.abs(area)) <= resabs * 0.01;
        if (!oscillatory) {
          const ratio = result / area;
          if (ratio < 0.01 || ratio > 100 || errsum > Math.abs(area)) ier = 6;
        }
      }
    }
  }
  if (useSum) {
    // Compute the global integral sum.
    result = 0;
    for (let k = 0; k < last; k++) result += rlist[k];
    abserr = errsum;
  }
  if (ier > 2) ier--;
  return finish(result, abserr, ier, last, endpointSingularity && ier === 3);
}

/**
 * Message describing the termination status of {@link extrapolatedIntegral}.
 *
 * @param status Termination status.
 * @param limit Maximum number of subintervals.
 * @param endpointSingularity Whether status 3 was due to an endpoint singularity.
 */
function describeStatus(
  status: ExtrapolationStatus,
  limit: number,
  endpointSingularity: boolean
): string {
  switch (status) {
    case 0:
      return 'Estimated error is at most `atol + rtol*|value|`.';
    case 1:
      return `Maximum number of subintervals (${limit}) reached.`;
    case 2:
      return 'Roundoff error detected: `atol + rtol*|value|` is probably not achievable.';
    case 3:
      return endpointSingularity
        ? 'Bisection reached floating-point resolution at a point where the integrand is not finite (probable endpoint singularity).'
        : 'Bad integrand behaviour (probable singularity or discontinuity) at a point of the integration range; try `breakpoints`.';
    case 4:
      return 'Roundoff error detected in the extrapolation: `atol + rtol*|value|` is not achievable.';
    case 5:
      return 'The integral is probably divergent, or converges too slowly.';
  }
}

/**
 * Adaptive G7-K15 engine with extrapolation, for {@link quad}: the same contract as
 * `integrateAdaptive` (all initial subintervals share one error budget and one subinterval
 * limit), but with QUADPACK's extrapolation by the epsilon algorithm and its error estimate
 * and termination logic; see {@link extrapolatedIntegral}.
 *
 * The `value` and `error` of the result are those of the extrapolation, not the sum over the
 * subintervals, unless the extrapolation was not used. If `success` is `false` they are only the
 * best estimates found, and for a probably divergent integral the value is meaningless.
 *
 * @internal
 *
 * @param name Public function name, used as prefix of error messages.
 * @param f Integrand.
 * @param points Strictly increasing finite integration points, at least two.
 * @param options Validated options, see `resolveGaussKronrodOptions`.
 * @returns Integral over `[points[0], points[points.length - 1]]` (ascending orientation).
 */
export function integrateExtrapolated(
  name: string,
  f: (x: number) => number,
  points: readonly number[],
  options: Required<GaussKronrodOptions>
): GaussKronrodResult {
  const r = extrapolatedIntegral(name, f, points, options);
  return {
    success: r.status === 0,
    message: describeStatus(r.status, options.maxSubintervals, r.endpointSingularity),
    evaluations: r.evaluations,
    subintervals: r.subintervals,
    value: r.value,
    error: r.error
  };
}
