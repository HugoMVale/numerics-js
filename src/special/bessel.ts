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

function assertOrder(n: number): void {
  if (!Number.isInteger(n) || n < 0 || n > MAX_ORDER) {
    throw new RangeError(`Bessel order n must be an integer in [0, ${MAX_ORDER}], got ${n}.`);
  }
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
  assertOrder(n);
  if (!Number.isInteger(m) || m < 1) {
    throw new RangeError(`Zero index m must be a positive integer (1-based), got ${m}.`);
  }

  let zeros = zerosCache.get(n);
  if (zeros === undefined) {
    zeros = [];
    zerosCache.set(n, zeros);
  }
  if (zeros.length >= m) return zeros[m - 1];

  const f = (x: number): number => besselJ(n, x);

  // J_n(n) > 0 because the first zero exceeds n; after a zero z, the next one
  // lies more than ZERO_SCAN_STEP beyond z + 1, so z + 1 is a safe restart.
  let a = zeros.length > 0 ? zeros[zeros.length - 1] + 1 : n;
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
          `besselJZero: root refinement failed for n=${n} in [${a}, ${b}]: ${result.message}`
        );
      }
      let r: number = result.x;
      // Newton polish; J_n' = J_{n-1} - (n/x) J_n, and J_0' = -J_1.
      for (let i = 0; i < 2; i++) {
        const jn = f(r);
        const dj: number = n === 0 ? -besselJ(1, r) : besselJ(n - 1, r) - (n / r) * jn;
        const next: number = r - jn / dj;
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

/**
 * Empties the cache of computed Bessel zeros.
 *
 * @internal
 */
export function clearBesselCache(): void {
  zerosCache.clear();
}
