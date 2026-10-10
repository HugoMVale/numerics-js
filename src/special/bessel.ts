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
                throw new Error(`besselJZero: root refinement failed for n=${n} in [${a}, ${b}]: ${result.message}`);
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

/**
 * Empties the cache of computed Bessel zeros.
 *
 * @internal
 */
export function clearBesselCache(): void {
    zerosCache.clear();
}
