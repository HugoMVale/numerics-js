import { brent } from '../roots/brent.js';

/** Largest supported order `n`; guards against unbounded work for absurd orders. */
const MAX_ORDER = 1e6;

/**
 * Scan increment used to bracket zeros. It must stay below the smallest gap
 * between consecutive positive zeros of any `J_n` (about 3.116, between
 * `j_{0,1}` and `j_{0,2}`), so that no scan interval can contain two zeros.
 */
const ZERO_SCAN_STEP = 2.5;

/** Cache of the positive zeros found so far, keyed by order `n` (dense, ascending, 0-based). */
const zerosCache = new Map<number, number[]>();

/** Same as {@link zerosCache}, for the positive zeros of `Y_n`. */
const yZerosCache = new Map<number, number[]>();

function assertOrder(n: number): void {
  if (!Number.isInteger(n) || n < 0 || n > MAX_ORDER) {
    throw new RangeError(`Bessel order n must be an integer in [0, ${MAX_ORDER}], got ${n}.`);
  }
}

/**
 * Shared engine of {@link besselJZero} and {@link besselYZero}: returns the
 * `m`-th positive zero (1-based) of a Bessel function `f` of order `n`,
 * extending `cache` as needed.
 *
 * Zeros are bracketed by a fixed-step scan for sign changes, refined with
 * Brent's method (tolerance `1e-15`) and polished with up to two Newton
 * steps. The scan step ({@link ZERO_SCAN_STEP}) is below the minimum spacing
 * of consecutive zeros, so the scan can neither skip nor duplicate a zero.
 *
 * @param name Public function name, used in error messages.
 * @param cache Per-order cache of the zeros found so far; mutated.
 * @param f The Bessel function of order `n`.
 * @param df Derivative of `f` at `x`, given `fx = f(x)`.
 * @param start Scan origin for the first zero; `f(start)` must be nonzero and
 * no zero may lie in `(0, start]`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]` or `m` is not a
 * positive integer.
 */
function scanBesselZero(
  name: string,
  cache: Map<number, number[]>,
  n: number,
  m: number,
  f: (x: number) => number,
  df: (x: number, fx: number) => number,
  start: number
): number {
  assertOrder(n);
  if (!Number.isInteger(m) || m < 1) {
    throw new RangeError(`Zero index m must be a positive integer (1-based), got ${m}.`);
  }

  let zeros = cache.get(n);
  if (zeros === undefined) {
    zeros = [];
    cache.set(n, zeros);
  }
  if (zeros.length >= m) return zeros[m - 1];

  // After a zero z, the next one lies more than ZERO_SCAN_STEP beyond z + 1,
  // so z + 1 is a safe restart.
  let a = zeros.length > 0 ? zeros[zeros.length - 1] + 1 : start;
  let fa = f(a);

  while (zeros.length < m) {
    const b = a + ZERO_SCAN_STEP;
    const fb = f(b);

    let root: number | undefined;
    if (fb === 0) {
      root = b;
    } else if (Math.sign(fa) !== Math.sign(fb)) {
      const result = brent(f, a, b, { tolx: 1e-15, tolf: 0, maxIter: 100 });
      if (!result.success) {
        throw new Error(
          `${name}: root refinement failed for n=${n} in [${a}, ${b}]: ${result.message}`
        );
      }
      let r: number = result.x;
      for (let i = 0; i < 2; i++) {
        const fr = f(r);
        const next: number = r - fr / df(r, fr);
        if (!(next >= a && next <= b)) break;
        r = next;
      }
      root = r;
    }

    if (root !== undefined) {
      zeros.push(root);
      a = root + 1;
      fa = f(a);
    } else {
      a = b;
      fa = fb;
    }
  }

  return zeros[m - 1];
}

/**
 * Power series `J_n(x) = (x/2)^n / n! * sum_k (-x^2/4)^k / (k! (n+1)_k)`.
 *
 * Used for `0 <= x < 1`, where the series is alternating with terms
 * decreasing by at least a factor 4 and therefore free of cancellation.
 * Underflow of the `(x/2)^n / n!` prefactor correctly yields 0.
 */
function seriesJ(n: number, x: number): number {
  const h = 0.5 * x;
  let prefactor = 1;
  for (let i = 1; i <= n; i++) {
    prefactor *= h / i;
    if (prefactor === 0) return 0;
  }
  const q = -h * h;
  let term = 1;
  let sum = 1;
  for (let k = 1; k < 40; k++) {
    term *= q / (k * (n + k));
    sum += term;
    if (Math.abs(term) <= 0.5 * Number.EPSILON * Math.abs(sum)) break;
  }
  return prefactor * sum;
}

/**
 * `J_n(x)` for `n < x`, `x >= 1`: `J_0` and `J_1` from the trapezoidal rule
 * applied to the integral representation
 * `J_n(x) = (1/pi) * integral_0^pi cos(n t - x sin t) dt`, then the forward
 * recurrence `J_{k+1} = (2k/x) J_k - J_{k-1}`, which is stable while `k < x`.
 *
 * The integrand is smooth and periodic, so the trapezoidal rule converges
 * geometrically. The aliasing error is of the order of `J_{2N-1}(x)`, which
 * is negligible once `2N` exceeds `x` by a few times `x^(1/3)` (the width of
 * the transition region) plus a fixed margin.
 */
function forwardJ(n: number, x: number): number {
  const steps = Math.ceil(0.5 * (x + 8 * Math.cbrt(x) + 40)) + 4;
  let sum0 = 0;
  let sum1 = 0;
  for (let i = 0; i <= steps; i++) {
    const t = (Math.PI * i) / steps;
    const arg = x * Math.sin(t);
    const weight = i === 0 || i === steps ? 0.5 : 1;
    sum0 += weight * Math.cos(arg);
    sum1 += weight * Math.cos(t - arg);
  }
  let jPrev = sum0 / steps;
  if (n === 0) return jPrev;
  let jCurr = sum1 / steps;
  for (let k = 1; k < n; k++) {
    const jNext = ((2 * k) / x) * jCurr - jPrev;
    jPrev = jCurr;
    jCurr = jNext;
  }
  return jCurr;
}

/**
 * `J_n(x)` for `n >= x >= 1` using Miller's backward recurrence.
 *
 * The recurrence `J_{k-1} = (2k/x) J_k - J_{k+1}` is run downward from an
 * order above `n` where `J_k` is negligible, from an arbitrary seed. The
 * unnormalized sequence is then scaled with the identity
 * `J_0 + 2 * sum_{k>=1} J_{2k} = 1`. The starting order exceeds `n` by
 * `sqrt(160 n)` plus a margin, which suppresses the unwanted solution to
 * below round-off for every `x <= n`.
 */
function millerJ(n: number, x: number): number {
  let start = n + Math.ceil(Math.sqrt(160 * n)) + 10;
  if (start % 2 !== 0) start++;

  let uNext = 0;
  let uCurr = 1;
  let uN = 0;
  let sum = 0;

  for (let k = start; k >= 1; k--) {
    const uPrev = ((2 * k) / x) * uCurr - uNext;
    uNext = uCurr;
    uCurr = uPrev;

    const idx = k - 1;
    if (idx === n) uN = uCurr;
    if (idx % 2 === 0) sum += (idx === 0 ? 1 : 2) * uCurr;

    // Rescale to avoid overflow during the descent.
    if (Math.abs(uCurr) > 1e250) {
      const scale = 1e-250;
      uCurr *= scale;
      uNext *= scale;
      sum *= scale;
      uN *= scale;
    }
  }

  return uN / sum;
}

/**
 * Computes the Bessel function of the first kind, `J_n(x)`, for a
 * non-negative integer order `n` and real `x`.
 *
 * The method depends on the region:
 *
 * - `|x| < 1`: power series.
 * - `n < |x|`: trapezoidal rule for `J_0` and `J_1`, then stable forward
 *   recurrence.
 * - `n >= |x| >= 1`: Miller's backward recurrence.
 * - Negative `x` uses `J_n(-x) = (-1)^n J_n(x)`; `J_n(±Infinity)` is 0.
 *
 * Accuracy: absolute error of a few `1e-16` for `|x| <= 1e3`, growing slowly
 * with `|x|` because the phase `x sin(t)` is rounded (about `2e-15` at
 * `|x| = 5e3` and `7e-14` at `|x| = 1e5`); relative error is at the
 * round-off level when `n > |x|`, where `J_n` decays. The cost is
 * proportional to `max(n, |x|)`, so very large arguments are slow.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param x Argument; any real number. `NaN` returns `NaN`.
 * @returns The value of `J_n(x)`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]`.
 *
 * @example
 * ```ts
 * import { besselJ } from 'numerics-js/special';
 *
 * console.log(besselJ(0, 1));
 * console.log(besselJ(5, 2.5));
 * console.log(besselJ(2, 100));
 * ```
 *
 * Output:
 * ```text
 * 0.7651976865579665
 * 0.01950162513450322
 * -0.02152875734450559
 * ```
 */
export function besselJ(n: number, x: number): number {
  assertOrder(n);
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return (n % 2 === 0 ? 1 : -1) * besselJ(n, -x);
  if (x === Infinity) return 0;
  if (x < 1) return seriesJ(n, x);
  return n < x ? forwardJ(n, x) : millerJ(n, x);
}

/**
 * Computes the `m`-th positive zero of `J_n(x)`, counted from the smallest
 * (`m = 1` is the first zero).
 *
 * Zeros are bracketed by scanning for sign changes of `J_n` starting at
 * `x = n` (the first zero always exceeds `n`), refined with Brent's method to
 * a tolerance of `1e-15`, and polished with Newton steps using
 * `J_n' = J_{n-1} - (n/x) J_n`. The relative error is at the round-off level
 * for the orders and indices tested against SciPy.
 *
 * Zeros are cached per order, so later calls with the same or a smaller `m`
 * are free, and larger `m` resume from the last cached zero. The cost grows
 * roughly quadratically with `m`.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param m Index of the zero, 1-based; a positive integer.
 * @returns The `m`-th positive zero of `J_n`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]` or `m` is not a
 * positive integer.
 *
 * @example
 * ```ts
 * import { besselJZero } from 'numerics-js/special';
 *
 * console.log(besselJZero(0, 1));
 * console.log(besselJZero(2, 20));
 * ```
 *
 * Output:
 * ```text
 * 2.4048255576957724
 * 65.1592731907578
 * ```
 */
export function besselJZero(n: number, m: number): number {
  // J_n(n) > 0 because the first zero exceeds n.
  return scanBesselZero(
    'besselJZero',
    zerosCache,
    n,
    m,
    (x) => besselJ(n, x),
    // J_n' = J_{n-1} - (n/x) J_n, and J_0' = -J_1.
    (x, jn) => (n === 0 ? -besselJ(1, x) : besselJ(n - 1, x) - (n / x) * jn),
    n
  );
}

/** Euler-Mascheroni constant. */
const EULER_GAMMA = 0.5772156649015329;

/** `1 / sqrt(pi)`; applied as `INV_SQRT_PI / sqrt(x)` so that `pi * x` cannot overflow. */
const INV_SQRT_PI = 0.5641895835477563;

/** Below this `x`, `Y_0` and `Y_1` come from the ascending series; from here, from the Neumann series. */
const Y_SERIES_MAX = 3;

/** From this `x` on, `Y_0` and `Y_1` come from the Hankel asymptotic expansions. */
const Y_ASYMPTOTIC_MIN = 20;

/**
 * Pair `[Y_0(x), Y_1(x)]` left by the last {@link seriesY01}, {@link neumannY01}
 * or {@link hankelY01} call. Safe as module state: calls are synchronous.
 */
const y01 = new Float64Array(2);

/**
 * `Y_0(x)` and `Y_1(x)` from the ascending series (Abramowitz & Stegun 9.1.11),
 *
 * - `Y_0 = (2/pi) [(ln(x/2) + gamma) J_0 + sum_{k>=1} (-1)^(k+1) H_k (x^2/4)^k / (k!)^2]`,
 * - `Y_1 = -2/(pi x) + (2/pi) ln(x/2) J_1
 *   - (x/(2 pi)) sum_{k>=0} (H_k + H_{k+1} - 2 gamma) (-x^2/4)^k / (k! (k+1)!)`,
 *
 * with `H_k` the harmonic numbers. Used for `0 < x < 3`: the terms decrease
 * like `(x^2/4)^k / (k!)^2` and stay `O(1)`, so cancellation costs at most a
 * fraction of a digit; beyond `x ~ 4` it grows steadily. The logarithm is
 * taken as `ln(x) - ln(2)` so that subnormal `x` does not hit `ln(0)`.
 */
function seriesY01(x: number): void {
  const h = 0.5 * x;
  const q = h * h;
  const lnh = Math.log(x) - Math.LN2;
  const tol = 1e-17;

  let a = 1; // (-q)^k / (k!)^2
  let b = 1; // (-q)^k / (k! (k+1)!)
  let hk = 0; // H_k
  let hk1 = 1; // H_{k+1}
  let j0 = 1; // J_0
  let j1 = 1; // J_1 / h
  let s0 = 0;
  let s1 = 1 - 2 * EULER_GAMMA;
  for (let k = 1; k < 40; k++) {
    a *= -q / (k * k);
    b *= -q / (k * (k + 1));
    hk += 1 / k;
    hk1 += 1 / (k + 1);
    j0 += a;
    j1 += b;
    const d0 = hk * a;
    const d1 = (hk + hk1 - 2 * EULER_GAMMA) * b;
    s0 -= d0;
    s1 += d1;
    if (Math.abs(d0) <= tol && Math.abs(d1) <= tol) break;
  }

  y01[0] = (2 / Math.PI) * ((lnh + EULER_GAMMA) * j0 + s0);
  y01[1] = -2 / (Math.PI * x) + (2 / Math.PI) * lnh * (h * j1) - (h * s1) / Math.PI;
}

/**
 * `Y_0(x)` and `Y_1(x)` for `3 <= x < 20` from the Neumann-type series
 *
 * - `Y_0 = (2/pi) [(ln(x/2) + gamma) J_0 - 2 sum_{k>=1} (-1)^k J_{2k} / k]`,
 * - `Y_1 = -Y_0' = (2/pi) [(ln(x/2) + gamma) J_1 - J_0 / x
 *   + sum_{k>=1} (-1)^k (J_{2k-1} - J_{2k+1}) / k]`
 *
 * (Abramowitz & Stegun 9.1.88, differentiated using `J_0' = -J_1` and
 * `2 J_k' = J_{k-1} - J_{k+1}`). Every `J_k` is produced at once by Miller's
 * backward recurrence, normalized with `J_0 + 2 sum_{k>=1} J_{2k} = 1`, exactly
 * as in {@link millerJ}. All terms are bounded by 1 and `J_k` is negligible
 * beyond `k ~ x + 12 x^(1/3)`, so there is no cancellation beyond the `O(1)`
 * level and the absolute error stays at a few `1e-16`. The sums are
 * accumulated on the way down, so no `J_k` is stored.
 */
function neumannY01(x: number): void {
  // Order beyond which J_k(x) is negligible, then the Miller margin (as in millerJ).
  const nn = Math.ceil(x + 12 * Math.cbrt(x) + 10);
  let start = nn + Math.ceil(Math.sqrt(160 * nn)) + 10;
  if (start % 2 !== 0) start++;

  let uNext = 0;
  let uCurr = 1;
  let norm = 0; // u_0 + 2 sum u_{2k}
  let sum0 = 0; // sum (-1)^k u_{2k} / k
  let sum1 = 0; // sum (-1)^k (u_{2k-1} - u_{2k+1}) / k, regrouped by odd index
  let u0 = 0;
  let u1 = 0;

  for (let k = start; k >= 1; k--) {
    const uPrev = ((2 * k) / x) * uCurr - uNext;
    uNext = uCurr;
    uCurr = uPrev;

    const idx = k - 1;
    if (idx % 2 === 0) {
      if (idx === 0) {
        u0 = uCurr;
        norm += uCurr;
      } else {
        const m = idx / 2;
        norm += 2 * uCurr;
        sum0 += (m % 2 === 0 ? uCurr : -uCurr) / m;
      }
    } else {
      // u_idx appears in the terms m = (idx + 1) / 2 and, for idx >= 3, m = (idx - 1) / 2.
      const m = (idx + 1) / 2;
      const c = 2 / (idx + 1) + (idx >= 3 ? 2 / (idx - 1) : 0);
      sum1 += (m % 2 === 0 ? c : -c) * uCurr;
      if (idx === 1) u1 = uCurr;
    }

    // Rescale to avoid overflow during the descent.
    if (Math.abs(uCurr) > 1e250) {
      const scale = 1e-250;
      uCurr *= scale;
      uNext *= scale;
      norm *= scale;
      sum0 *= scale;
      sum1 *= scale;
      u0 *= scale;
      u1 *= scale;
    }
  }

  const l = Math.log(x) - Math.LN2 + EULER_GAMMA;
  const j0 = u0 / norm;
  const j1 = u1 / norm;
  y01[0] = (2 / Math.PI) * (l * j0 - (2 * sum0) / norm);
  y01[1] = (2 / Math.PI) * (l * j1 - j0 / x + sum1 / norm);
}

/**
 * `Y_0(x)` and `Y_1(x)` for `x >= 20` from the Hankel asymptotic expansions
 * (Abramowitz & Stegun 9.2.6, 9.2.9-9.2.10),
 * `Y_nu = sqrt(2 / (pi x)) [P sin(chi) + Q cos(chi)]` with
 * `chi = x - (nu/2 + 1/4) pi`, where `P = t_0 - t_2 + t_4 - ...` and
 * `Q = t_1 - t_3 + ...` are built from `t_k = t_{k-1} (4 nu^2 - (2k-1)^2) / (8 k x)`.
 *
 * The expansion is asymptotic: its smallest term is of order `e^(-2x)`, i.e.
 * below `1e-17` from `x = 20`, so it reaches full double precision (already at
 * about `x = 16`). It costs `O(1)` for any `x`.
 *
 * The phase is never formed: `sin(chi)` and `cos(chi)` are expanded in
 * `sin(x)` and `cos(x)` (`chi` differs from `x` by the fixed `pi/4` or
 * `3 pi/4`), so the only argument reduction is the one done by `Math.sin` and
 * `Math.cos` on the exact input `x`, however large.
 */
function hankelY01(x: number): void {
  const s = Math.sin(x);
  const c = Math.cos(x);
  const pref = INV_SQRT_PI / Math.sqrt(x);

  for (let nu = 0; nu <= 1; nu++) {
    const mu = 4 * nu * nu;
    let t = 1;
    let p = 1;
    let q = 0;
    for (let k = 1; k < 60; k++) {
      const next = (t * (mu - (2 * k - 1) ** 2)) / (8 * k * x);
      if (Math.abs(next) >= Math.abs(t)) break; // smallest term reached
      t = next;
      if (k % 2 === 1) q += k % 4 === 1 ? t : -t;
      else p += k % 4 === 2 ? -t : t;
      if (Math.abs(t) < 1e-17) break;
    }
    // sqrt(2/(pi x)) (P sin(chi) + Q cos(chi)) with chi = x - pi/4 (nu = 0) or x - 3 pi/4 (nu = 1).
    y01[nu] = nu === 0 ? pref * ((p + q) * s + (q - p) * c) : pref * ((q - p) * s - (p + q) * c);
  }
}

/**
 * Computes the Bessel function of the second kind, `Y_n(x)` (also called the
 * Neumann or Weber function), for a non-negative integer order `n` and
 * `x > 0`.
 *
 * `Y_0` and `Y_1` are evaluated by region and `Y_n` follows from the forward
 * recurrence `Y_{k+1} = (2k/x) Y_k - Y_{k-1}` (Abramowitz & Stegun 9.1.27),
 * which is stable because `Y_n` is the dominant solution:
 *
 * - `0 < x < 3`: ascending series with the logarithmic term.
 * - `3 <= x < 20`: Neumann-type series in `J_{2k}`, with the `J_k` generated
 *   by Miller's backward recurrence.
 * - `x >= 20`: Hankel asymptotic expansions, `O(1)` for any `x`.
 *
 * Special values: `Y_n(0) = -Infinity`; `Y_n(x)` is not real for `x < 0`, so
 * the result is `NaN`; `Y_n(Infinity) = 0`. The result is `-Infinity` once
 * `|Y_n(x)|` exceeds the double range (e.g. `n = 250` at `x = 5`).
 *
 * Accuracy (measured against mpmath): for `n <= 1` the absolute error is at
 * most `4e-16` where `|Y_n| <= 1` and the relative error at most `4e-16`
 * where `|Y_n| > 1`; as for any oscillating function, the relative error is
 * not meaningful next to a zero. For `n >= 2` the forward recurrence adds
 * round-off that grows with `n`: up to `n = 1000` the error is at most `2e-15`
 * (absolute, where `|Y_n| <= 1`) or `1e-14` (relative, where `|Y_n| > 1`),
 * the largest values occurring near the turning point `x ~ n`. The error does
 * not grow with `x`, which was tested up to `1e15` (the phase is not rounded;
 * see the Hankel expansion). The cost is proportional to `n` and independent of
 * `x`.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param x Argument; a real number. `NaN` returns `NaN`.
 * @returns The value of `Y_n(x)`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]`.
 *
 * @example
 * ```ts
 * import { besselY } from 'numerics-js/special';
 *
 * console.log(besselY(0, 1));
 * console.log(besselY(5, 2.5));
 * console.log(besselY(2, 100));
 * ```
 *
 * Output:
 * ```text
 * 0.088256964215677
 * -3.830176000740752
 * 0.07683686712502796
 * ```
 */
export function besselY(n: number, x: number): number {
  assertOrder(n);
  if (Number.isNaN(x) || x < 0) return NaN;
  if (x === 0) return -Infinity;
  if (x === Infinity) return 0;

  if (x < Y_SERIES_MAX) seriesY01(x);
  else if (x < Y_ASYMPTOTIC_MIN) neumannY01(x);
  else hankelY01(x);

  if (n === 0) return y01[0];

  let yPrev = y01[0];
  let yCurr = y01[1];
  for (let k = 1; k < n; k++) {
    const yNext = ((2 * k) / x) * yCurr - yPrev;
    // |Y_k| grows without bound beyond the turning point, so an infinite step means overflow.
    if (!Number.isFinite(yNext)) return -Infinity;
    yPrev = yCurr;
    yCurr = yNext;
  }
  return yCurr;
}

/**
 * Computes the `m`-th positive zero of `Y_n(x)`, counted from the smallest
 * (`m = 1` is the first zero).
 *
 * The method mirrors {@link besselJZero}: zeros are bracketed by scanning for
 * sign changes of `Y_n`, refined with Brent's method to a tolerance of
 * `1e-15`, and polished with Newton steps using `Y_n' = Y_{n-1} - (n/x) Y_n`
 * (`Y_0' = -Y_1`). The scan is exhaustive and cannot skip or duplicate a
 * zero, because consecutive zeros of `Y_n` are always more than 3.06 apart
 * (the minimum, `y_{0,2} - y_{0,1}`, occurs for `n = 0`; for `n >= 1` the
 * spacing exceeds pi) while the scan step is 2.5. The scan starts at
 * `x = max(n, 0.5)`, where `Y_n < 0`, because the first zero always exceeds
 * `n`. Zeros of `Y_n` and `J_n` interlace: `y_{n,m} < j_{n,m} < y_{n,m+1}`.
 *
 * Zeros are cached per order, so later calls with the same or a smaller `m`
 * are free, and larger `m` resume from the last cached zero. Since the cost
 * of {@link besselY} does not depend on `x`, the cost grows linearly with `m`
 * and with `n`.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param m Index of the zero, 1-based; a positive integer.
 * @returns The `m`-th positive zero of `Y_n`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]` or `m` is not a
 * positive integer.
 *
 * @example
 * ```ts
 * import { besselYZero } from 'numerics-js/special';
 *
 * console.log(besselYZero(0, 1));
 * console.log(besselYZero(2, 20));
 * ```
 *
 * Output:
 * ```text
 * 0.8935769662791675
 * 63.587765799737596
 * ```
 */
export function besselYZero(n: number, m: number): number {
  // Y_n < 0 on (0, y_{n,1}) and y_{n,1} > n, so Y_n(start) < 0.
  return scanBesselZero(
    'besselYZero',
    yZerosCache,
    n,
    m,
    (x) => besselY(n, x),
    // Y_n' = Y_{n-1} - (n/x) Y_n, and Y_0' = -Y_1.
    (x, yn) => (n === 0 ? -besselY(1, x) : besselY(n - 1, x) - (n / x) * yn),
    Math.max(n, 0.5)
  );
}

/**
 * Power series `I_n(x) = (x/2)^n / n! * sum_k (x^2/4)^k / (k! (n+1)_k)`.
 *
 * Used for `0 <= x < 1`. All terms are positive, so there is no cancellation,
 * and they decrease by at least a factor 4. Underflow of the `(x/2)^n / n!`
 * prefactor correctly yields 0.
 */
function seriesI(n: number, x: number): number {
  const h = 0.5 * x;
  let prefactor = 1;
  for (let i = 1; i <= n; i++) {
    prefactor *= h / i;
    if (prefactor === 0) return 0;
  }
  const q = h * h;
  let term = 1;
  let sum = 1;
  for (let k = 1; k < 40; k++) {
    term *= q / (k * (n + k));
    sum += term;
    if (term <= 0.5 * Number.EPSILON * sum) break;
  }
  return prefactor * sum;
}

/**
 * Pieces left by the last {@link millerIScaled} call, so that
 * `I_n = u_n 1e-250^shifts e^x / sum` can be evaluated in the log domain when
 * the scaled value underflows. `u_n` is deliberately not rescaled after it is
 * captured (that would underflow it to 0 for large `n`); `shifts` counts the
 * rescalings of `sum` since. Safe as module state: calls are synchronous.
 */
const millerParts = { uN: 0, shifts: 0, sum: 0 };

const LOG_RESCALE = 250 * Math.LN10;

/**
 * `e^{-x} I_n(x)` for `x >= 1` using Miller's backward recurrence.
 *
 * `I_n` is the minimal solution of `I_{k-1} = (2k/x) I_k + I_{k+1}` for every
 * `x > 0`, so the downward recurrence is stable and, since all terms are
 * positive, free of cancellation. It is run from an arbitrary seed at an order
 * `start` where `I_k` is negligible, then normalized with the identity
 * `I_0 + 2 * sum_{k>=1} I_k = e^x`, which directly yields the scaled value
 * `u_n / (u_0 + 2 * sum u_k)`.
 *
 * Because `I_k(x) / I_0(x)` falls off like `exp(-k^2 / (2x))` for `k < x` and
 * factorially beyond, `start` exceeds `n` only by `sqrt(100 max(n, x))` plus a
 * margin; the cost is therefore `O(n + sqrt(x))`, not `O(x)`.
 */
function millerIScaled(n: number, x: number): number {
  const start = n + Math.ceil(Math.sqrt(100 * Math.max(n, x))) + 10;

  let uNext = 0;
  let uCurr = 1;
  let uN = 0;
  let shifts = 0;
  let sum = 0;

  for (let k = start; k >= 1; k--) {
    const uPrev = ((2 * k) / x) * uCurr + uNext;
    uNext = uCurr;
    uCurr = uPrev;

    const idx = k - 1;
    if (idx === n) {
      uN = uCurr;
      shifts = 0;
    }
    sum += (idx === 0 ? 1 : 2) * uCurr;

    // Rescale to avoid overflow during the descent.
    if (uCurr > 1e250) {
      const scale = 1e-250;
      uCurr *= scale;
      uNext *= scale;
      sum *= scale;
      shifts++;
    }
  }

  millerParts.uN = uN;
  millerParts.shifts = shifts;
  millerParts.sum = sum;
  return (uN / sum) * 1e-250 ** shifts;
}

/**
 * Computes the modified Bessel function of the first kind, `I_n(x)`, for a
 * non-negative integer order `n` and real `x`.
 *
 * Uses the power series for `|x| < 1` and otherwise Miller's backward
 * recurrence, normalized with `e^x = I_0 + 2 sum_{k>=1} I_k`. `I_n` is the
 * minimal solution of the recurrence, so this is stable for every `x`, and all
 * terms are positive, so there is no cancellation. The cost is proportional
 * to `n + sqrt(max(n, |x|))`, i.e. large arguments are cheap.
 * Negative `x` uses `I_n(-x) = (-1)^n I_n(x)`.
 *
 * Accuracy: the relative error is at the round-off level (a few `1e-15`,
 * measured against mpmath) wherever the result is representable. The result
 * overflows to `Infinity` for `|x|` beyond about `713` at small `n`; use
 * {@link besselIScaled} there. Where `e^{-x} I_n(x)` would underflow, or for
 * `|x| > 1400`, the final exponential is evaluated as `exp(x + ln r)`, which
 * costs roughly `|x| * 1e-16` of relative accuracy. `I_n(±Infinity)` is `±Infinity` (sign `(-1)^n` for `-Infinity`).
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param x Argument; any real number. `NaN` returns `NaN`.
 * @returns The value of `I_n(x)`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]`.
 *
 * @example
 * ```ts
 * import { besselI } from 'numerics-js/special';
 *
 * console.log(besselI(0, 1));
 * console.log(besselI(5, 2.5));
 * console.log(besselI(2, 100));
 * ```
 *
 * Output:
 * ```text
 * 1.2660658777520082
 * 0.032843475172023205
 * 1.0523843193243107e+42
 * ```
 */
export function besselI(n: number, x: number): number {
  assertOrder(n);
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return (n % 2 === 0 ? 1 : -1) * besselI(n, -x);
  if (x === Infinity) return Infinity;
  if (x < 1) return seriesI(n, x);
  const r = millerIScaled(n, x);
  // Avoid overflow of e^x and underflow of the scaled value (I_n representable
  // although e^{-x} I_n is not) by switching to the log domain.
  if (x > 1400 || r < 1e-290) {
    const { uN, shifts, sum } = millerParts;
    return Math.exp(x + Math.log(uN) - shifts * LOG_RESCALE - Math.log(sum));
  }
  if (x <= 700) return r * Math.exp(x);
  return r * Math.exp(0.5 * x) * Math.exp(0.5 * x);
}

/**
 * Computes the exponentially scaled modified Bessel function of the first
 * kind, `e^{-|x|} I_n(x)`, for a non-negative integer order `n` and real `x`.
 *
 * Unlike {@link besselI}, this never overflows and keeps full relative
 * accuracy for any `x`: `I_n(x)` exceeds the double range for `|x|` above
 * about `713`, whereas `e^{-|x|} I_n(x) ~ 1 / sqrt(2 pi |x|)` stays moderate.
 * It uses the same algorithm as {@link besselI}. Negative `x` uses
 * `I_n(-x) = (-1)^n I_n(x)`, so the result is `(-1)^n e^{-|x|} I_n(|x|)`;
 * the limit at `±Infinity` is `0`.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param x Argument; any real number. `NaN` returns `NaN`.
 * @returns The value of `e^{-|x|} I_n(x)`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]`.
 *
 * @example
 * ```ts
 * import { besselIScaled } from 'numerics-js/special';
 *
 * console.log(besselIScaled(0, 1));
 * console.log(besselIScaled(1, 1000));
 * ```
 *
 * Output:
 * ```text
 * 0.46575960759364043
 * 0.01261093025692863
 * ```
 */
export function besselIScaled(n: number, x: number): number {
  assertOrder(n);
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return (n % 2 === 0 ? 1 : -1) * besselIScaled(n, -x);
  if (x === Infinity) return 0;
  return x < 1 ? seriesI(n, x) * Math.exp(-x) : millerIScaled(n, x);
}

/** Below this `x`, `K_0` and `K_1` come from the ascending series; from here, from the trapezoidal rule. */
const K_SERIES_MAX = 1;

/**
 * Terms of the trapezoidal sum are dropped once the exponent `2 x sinh^2(t/2)`
 * of the integrand exceeds this value (`e^-40 ~ 4e-18`, below round-off).
 */
const K_TAIL_EXPONENT = 40;

/**
 * From this `x` on, `K_n(x)` underflows to 0 for every supported order: for
 * `n <= 1e6` it is bounded by `e^(-0.53 x)`. It also keeps the argument
 * reduction in {@link unscaleK} exact.
 */
const K_UNDERFLOW_X = 1e6;

/** Binary exponent of the rescaling applied to the working values of the `K_n` recurrence on overflow. */
const K_RESCALE_EXP = 800;

/** Exact (a power of two) rescaling factor, `2^-K_RESCALE_EXP`. */
const K_RESCALE = 2 ** -K_RESCALE_EXP;

/**
 * `ln 2` split as `LN2_HI + LN2_LO`, with `LN2_HI` having only 32 significant
 * bits so that `m * LN2_HI` is exact for every integer `|m| < 2^21` (as in fdlibm's `exp`).
 */
const LN2_HI = 6.9314718036912382e-1;
const LN2_LO = 1.9082149292705877e-10;

/**
 * Pair `[K_0(x), K_1(x)]` left by the last {@link seriesK01} or
 * {@link trapezoidK01} call. The values are plain for `x < K_SERIES_MAX` and
 * carry a factor `e^x` otherwise. Safe as module state: calls are synchronous.
 */
const k01 = new Float64Array(2);

/**
 * Pieces left by the last {@link evaluateK} call: the working value of `K_n`
 * (in the convention of {@link k01}) is `value * 2^(K_RESCALE_EXP * shifts)`,
 * with `shifts` counting the rescalings applied to avoid overflow.
 * `shifts > 0` therefore means that the working value exceeds the double range.
 */
const kParts = { value: 0, shifts: 0 };

/**
 * `K_0(x)` and `K_1(x)` from the ascending series (Abramowitz & Stegun 9.6.11,
 * 9.6.13; the `mu = 0` case of Temme's series),
 *
 * - `K_0 = -(ln(x/2) + gamma) I_0 + sum_{k>=1} H_k (x^2/4)^k / (k!)^2`,
 * - `K_1 = 1/x + ln(x/2) I_1 - (x/4) sum_{k>=0} (H_k + H_{k+1} - 2 gamma) (x^2/4)^k / (k! (k+1)!)`,
 *
 * with `H_k` the harmonic numbers. Used for `0 < x < 1`: `ln(x/2) + gamma` is
 * negative there, so all contributions to `K_0` are positive and nothing
 * cancels; the terms decrease by at least a factor 4. Beyond `x ~ 1.1` the sign
 * changes and cancellation grows steadily. The logarithm is taken as
 * `ln(x) - ln(2)` so that subnormal `x` does not hit `ln(0)`.
 */
function seriesK01(x: number): void {
  const h = 0.5 * x;
  const q = h * h;
  const lnh = Math.log(x) - Math.LN2;

  let a = 1; // (x^2/4)^k / (k!)^2
  let b = 1; // (x^2/4)^k / (k! (k+1)!)
  let hk = 0; // H_k
  let hk1 = 1; // H_{k+1}
  let i0 = 1; // I_0
  let i1 = 1; // I_1 / h
  let s0 = 0;
  let s1 = 1 - 2 * EULER_GAMMA;
  for (let k = 1; k < 40; k++) {
    a *= q / (k * k);
    b *= q / (k * (k + 1));
    hk += 1 / k;
    hk1 += 1 / (k + 1);
    i0 += a;
    i1 += b;
    s0 += hk * a;
    s1 += (hk + hk1 - 2 * EULER_GAMMA) * b;
    if (a <= 1e-18 * i0) break;
  }

  k01[0] = -(lnh + EULER_GAMMA) * i0 + s0;
  k01[1] = 1 / x + lnh * (h * i1) - 0.5 * h * s1;
}

/**
 * `e^x K_0(x)` and `e^x K_1(x)` for `x >= 1` from the trapezoidal rule applied
 * to the integral representation
 * `e^x K_nu(x) = integral_0^inf exp(-2 x sinh^2(t/2)) cosh(nu t) dt`
 * (the representation `K_nu(x) = integral_0^inf exp(-x cosh t) cosh(nu t) dt`
 * with `cosh t - 1 = 2 sinh^2(t/2)`; see Gil, Segura & Temme, *Numerical
 * Methods for Special Functions*, SIAM, 2007, on trapezoidal quadrature).
 *
 * The integrand is positive, even in `t`, entire and double-exponentially
 * decaying, so the trapezoidal rule converges geometrically and, with no
 * cancellation, keeps full relative accuracy. By Poisson summation the
 * aliasing error is governed by `K_{i 2 pi / h}(x)`: it falls like
 * `exp(-pi^2 / h)` for moderate `x` and like `exp(-2 pi^2 / (h^2 x))` for large
 * `x`, which the step `h = 0.9 / sqrt(2x + 16)` covers with room to spare
 * (the quadrature error is below `1e-18` for `1 <= x <= 1e100`, checked against
 * 40-digit arithmetic). The sum is cut where the integrand falls below
 * `e^-40`. This takes at most 24 terms for any `x >= 1`, the count decreasing
 * to about 13 as `x` grows, so the cost is `O(1)`. The exponent is formed as
 * `2 (sqrt(x) sinh(t/2))^2` and `h` without forming `2x`, so that neither
 * overflows nor loses precision for huge `x`.
 */
function trapezoidK01(x: number): void {
  const sx = Math.sqrt(x);
  const h = 0.9 / (Math.SQRT2 * Math.sqrt(x + 8));
  const steps = Math.ceil((2 * Math.asinh(Math.sqrt(K_TAIL_EXPONENT / 2) / sx)) / h);

  let sum0 = 0.5;
  let sum1 = 0.5;
  for (let j = 1; j <= steps; j++) {
    const s = Math.sinh(0.5 * h * j);
    const w = Math.exp(-2 * (sx * s) ** 2);
    sum0 += w;
    sum1 += w * (1 + 2 * s * s); // cosh(t) = 1 + 2 sinh^2(t/2)
  }
  k01[0] = h * sum0;
  k01[1] = h * sum1;
}

/**
 * Evaluates `K_n(x)` for `0 < x < Infinity` and leaves it in {@link kParts}.
 *
 * `K_0` and `K_1` come from {@link seriesK01} or {@link trapezoidK01}; higher
 * orders follow from the forward recurrence `K_{k+1} = K_{k-1} + (2k/x) K_k`
 * (Abramowitz & Stegun 9.6.26). `K_n` is the dominant solution of this
 * recurrence and all terms are positive, so it is stable and free of
 * cancellation. If the working values would overflow, both are rescaled by
 * the exact factor `2^-800` and the rescalings are counted, so that `K_n` is
 * still recovered when only the factor `e^x` in the working value pushes it
 * out of range. A value beyond the double range even after rescaling is left
 * as `Infinity`.
 */
function evaluateK(n: number, x: number): void {
  if (x < K_SERIES_MAX) seriesK01(x);
  else trapezoidK01(x);

  let kPrev = k01[0];
  let kCurr = n === 0 ? k01[0] : k01[1];
  let shifts = 0;
  if (n >= 2) {
    for (let k = 1; k < n; k++) {
      const f = (2 * k) / x;
      let kNext = kPrev + f * kCurr;
      if (kNext === Infinity) {
        kPrev *= K_RESCALE;
        kCurr *= K_RESCALE;
        shifts++;
        kNext = kPrev + f * kCurr;
        if (kNext === Infinity) {
          kCurr = Infinity;
          shifts = 0;
          break;
        }
      }
      kPrev = kCurr;
      kCurr = kNext;
    }
  }
  kParts.value = kCurr;
  kParts.shifts = shifts;
}

/**
 * Removes the factor `e^x` from a working value, i.e. returns
 * `value * 2^(K_RESCALE_EXP * shifts) * e^-x` for `1 <= x <= K_UNDERFLOW_X`.
 *
 * Writing `e^-x = 2^-m e^-r` with `m = round(x / ln 2)` and `|r| <= ln(2) / 2`
 * (the reduction is exact thanks to the split `ln 2`), everything but `e^-r` is
 * a power of two, so the result carries the rounding error of a single `exp`
 * of a small argument however far `x` or the rescaling are from zero. The
 * power of two is applied in two halves, the first before the `exp`, so that no
 * intermediate result overflows or underflows unless the final one does.
 */
function unscaleK(value: number, shifts: number, x: number): number {
  const m = Math.round(x / Math.LN2);
  const r = x - m * LN2_HI - m * LN2_LO;
  const k = K_RESCALE_EXP * shifts - m;
  const k1 = Math.trunc(k / 2);
  return value * 2 ** k1 * Math.exp(-r) * 2 ** (k - k1);
}

/**
 * Computes the modified Bessel function of the second kind, `K_n(x)` (also
 * called the Macdonald function), for a non-negative integer order `n` and
 * `x > 0`.
 *
 * `K_0` and `K_1` are evaluated by region and `K_n` follows from the forward
 * recurrence `K_{k+1} = K_{k-1} + (2k/x) K_k` (Abramowitz & Stegun 9.6.26),
 * which is stable because `K_n` is the dominant solution and every term is
 * positive:
 *
 * - `0 < x < 1`: ascending series with the logarithmic term.
 * - `x >= 1`: trapezoidal rule on `K_nu(x) = integral_0^inf exp(-x cosh t)
 *   cosh(nu t) dt`, whose positive, double-exponentially decaying integrand
 *   makes it converge geometrically without cancellation. It needs at most 24
 *   terms for any `x`, i.e. `O(1)` work.
 *
 * Special values: `K_n(0) = Infinity`; `K_n(x)` is not real for `x < 0`, so the
 * result is `NaN`; `K_n(Infinity) = 0`. The result overflows to `Infinity` once
 * `K_n(x)` exceeds the double range (e.g. `n = 152` at `x = 1`) and underflows
 * to 0 for `x` beyond about `745` at small `n`; use {@link besselKScaled}
 * there. A representable `K_n(x)` is returned even where `e^x K_n(x)` would
 * overflow (large `n` and large `x`), with no loss of accuracy.
 *
 * Accuracy (measured against mpmath): the relative error is at most `3e-15`
 * for every tested `n <= 5000` and every `x` where the result is a normal
 * double, i.e. at the round-off level (SciPy's `kv`/`kve` show `2e-14` to
 * `3e-13` on the same points). The forward recurrence adds only a few ulp up
 * to `n ~ 1e3` because every term is positive. In the subnormal range the
 * result is limited by the format. The cost is `O(1)` for `n <= 1` and
 * `O(n)` otherwise, independent of `x`.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param x Argument; a real number. `NaN` returns `NaN`.
 * @returns The value of `K_n(x)`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]`.
 *
 * @example
 * ```ts
 * import { besselK } from 'numerics-js/special';
 *
 * console.log(besselK(0, 1));
 * console.log(besselK(5, 2.5));
 * console.log(besselK(2, 100));
 * ```
 *
 * Output:
 * ```text
 * 0.4210244382407083
 * 2.7168842907865423
 * 4.75022530388864e-45
 * ```
 */
export function besselK(n: number, x: number): number {
  assertOrder(n);
  if (Number.isNaN(x) || x < 0) return NaN;
  if (x === 0) return Infinity;
  if (x > K_UNDERFLOW_X) return 0; // includes Infinity

  evaluateK(n, x);
  const { value, shifts } = kParts;
  if (value === Infinity) return Infinity;
  // Below K_SERIES_MAX the working value is K_n itself, so an overflow of it is one of K_n.
  if (x < K_SERIES_MAX) return shifts === 0 ? value : Infinity;
  // Otherwise it carries a factor e^x, which can overflow while K_n(x) is still representable.
  if (shifts === 0 && x <= 700) return value * Math.exp(-x);
  return unscaleK(value, shifts, x);
}

/**
 * Computes the exponentially scaled modified Bessel function of the second
 * kind, `e^x K_n(x)`, for a non-negative integer order `n` and `x > 0`.
 *
 * Unlike {@link besselK}, this does not underflow for large `x` and keeps full
 * relative accuracy there: `K_n(x)` falls below the smallest double for `x`
 * above about `745`, whereas `e^x K_n(x) ~ sqrt(pi / (2x))` stays moderate. It
 * uses the same algorithm as {@link besselK}, so the relative error is at
 * the same round-off level (at most `3e-15` in the tests) and the cost is
 * `O(1)` for `n <= 1` and `O(n)` otherwise, independent of `x`. The result is `Infinity` only once `e^x K_n(x)` itself
 * exceeds the double range, i.e. for `n` large compared to `x`.
 *
 * Special values: `e^x K_n(x) = Infinity` at `x = 0`; it is not real for
 * `x < 0`, so the result is `NaN`; the limit at `Infinity` is `0`.
 *
 * @param n Order; a non-negative integer not larger than `1e6`.
 * @param x Argument; a real number. `NaN` returns `NaN`.
 * @returns The value of `e^x K_n(x)`.
 * @throws {RangeError} If `n` is not an integer in `[0, 1e6]`.
 *
 * @example
 * ```ts
 * import { besselKScaled } from 'numerics-js/special';
 *
 * console.log(besselKScaled(0, 1));
 * console.log(besselKScaled(1, 1000));
 * ```
 *
 * Output:
 * ```text
 * 1.1444630798068949
 * 0.03964813081296022
 * ```
 */
export function besselKScaled(n: number, x: number): number {
  assertOrder(n);
  if (Number.isNaN(x) || x < 0) return NaN;
  if (x === 0) return Infinity;
  if (x === Infinity) return 0;

  evaluateK(n, x);
  const { value, shifts } = kParts;
  if (shifts > 0) return Infinity;
  return x < K_SERIES_MAX ? value * Math.exp(x) : value;
}

/**
 * Empties the cache of computed Bessel zeros.
 *
 * @internal
 */
export function clearBesselCache(): void {
  zerosCache.clear();
  yZerosCache.clear();
}
