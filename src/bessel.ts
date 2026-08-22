import { bisection } from './roots.js';

export interface BesselUtility {
    _integralJ(n: number, x: number): number;
    _besselMiller(n: number, x: number): number;
    J(n: number, x: number): number;
    zerosCache: number[][];
    getZero(n: number, m: number): number;
}

/**
 * Numerical utilities for the Bessel function of the first kind, J_n(x),
 * restricted to non-negative integer orders n.
 *
 * Two independent techniques are combined:
 *  - A direct trapezoidal-rule evaluation of the integral representation,
 *    used for the base orders n = 0 and n = 1.
 *  - Miller's backward-recurrence algorithm, used for all n >= 2, which is
 *    numerically stable regardless of how n compares to x (the naive
 *    forward recurrence is only stable while n < x and blows up otherwise).
 */
export const bessel: BesselUtility = {
    /**
     * Evaluates J_n(x) via its integral representation
     *   J_n(x) = (1/pi) * integral_0^pi cos(n*tau - x*sin(tau)) d(tau)
     * using the composite trapezoidal rule.
     *
     * Only accurate for small, fixed orders (n = 0 or n = 1) — for larger n
     * use `J`, which routes through the stable recurrence instead.
     */
    _integralJ(n: number, x: number): number {
        // The integrand oscillates faster as x grows, so scale the sample
        // count with x to keep the quadrature error bounded. The cap keeps
        // pathologically large x from causing an unbounded amount of work.
        const steps = Math.min(4000, Math.max(60, Math.ceil(15 * x)));
        let sum = 0;
        for (let i = 0; i <= steps; i++) {
            const tau = (Math.PI * i) / steps;
            const weight = (i === 0 || i === steps) ? 0.5 : 1;
            sum += weight * Math.cos(n * tau - x * Math.sin(tau));
        }
        return sum / steps;
    },

    /**
     * Computes J_n(x) for n >= 2 using Miller's backward-recurrence
     * algorithm.
     *
     * The recurrence J_{k-1}(x) = (2k/x) J_k(x) - J_{k+1}(x) is stable when
     * run *downward* from an order well above both n and x, starting from
     * an arbitrary (unnormalized) seed. The resulting unnormalized sequence
     * is then rescaled using the identity
     *   J_0(x) + 2 * sum_{k=1..inf} J_{2k}(x) = 1
     * to recover the true, normalized values.
     */
    _besselMiller(n: number, x: number): number {
        // Start comfortably above both n and x; the extra margin controls
        // how quickly the unwanted (growing) solution is suppressed by the
        // time the recurrence reaches order n.
        const start = n + 15 + Math.ceil(Math.sqrt(40 * Math.max(n, x, 1)));

        let uNext = 0; // u_{k+1}
        let uCurr = 1; // u_k, arbitrary nonzero seed (unnormalized)
        let uN = 0;    // will hold the unnormalized value at order n
        let sum = 0;   // running normalization sum

        for (let k = start; k >= 1; k--) {
            const uPrev = ((2 * k) / x) * uCurr - uNext;
            uNext = uCurr;
            uCurr = uPrev;

            const idx = k - 1;
            if (idx === n) uN = uCurr;
            if (idx % 2 === 0) sum += (idx === 0 ? 1 : 2) * uCurr;

            // Rescale periodically to avoid overflow during the descent.
            if (Math.abs(uCurr) > 1e150) {
                const scale = 1e-150;
                uCurr *= scale;
                uNext *= scale;
                sum *= scale;
                uN *= scale;
            }
        }

        return uN / sum;
    },

    /**
     * Computes the Bessel function of the first kind, J_n(x), for a
     * non-negative integer order n and real x.
     *
     * - n = 0, 1: evaluated directly via the integral representation.
     * - n >= 2: evaluated via Miller's stable backward recurrence.
     * - Negative x is handled through the identity J_n(-x) = (-1)^n J_n(x).
     */
    J(n: number, x: number): number {
        if (!Number.isInteger(n) || n < 0) {
            throw new RangeError(`Bessel order n must be a non-negative integer, got ${n}.`);
        }
        if (x === 0) return n === 0 ? 1 : 0;
        if (x < 0) return (n % 2 === 0 ? 1 : -1) * this.J(n, -x);

        if (n === 0) return this._integralJ(0, x);
        if (n === 1) return this._integralJ(1, x);

        return this._besselMiller(n, x);
    },

    /**
     * Cache of previously computed positive zeros of J_n, keyed as
     * `zerosCache[n][m]` (1-indexed: entry 0 of each inner array is unused).
     */
    zerosCache: [],

    /**
     * Retrieves the m-th positive zero of J_n(x), i.e. the m-th root of
     * J_n on (0, infinity), counted from the smallest.
     *
     * Zeros are located by scanning x in fixed increments for sign changes
     * of J_n and refining each bracket with bisection. Results are cached
     * per order n, and repeated calls with increasing m resume the scan
     * from the last cached zero rather than starting over.
     */
    getZero(n: number, m: number): number {
        if (!Number.isInteger(m) || m < 1) {
            throw new RangeError(`m must be a positive integer (1-indexed), got ${m}.`);
        }

        if (!this.zerosCache[n]) this.zerosCache[n] = [];
        const cache = this.zerosCache[n];

        if (cache[m] !== undefined) return cache[m];

        const fn = (v: number): number => this.J(n, v);
        const step = 0.5;

        // Resume from the highest already-cached zero instead of
        // rescanning from x = 0.1 every time.
        let rootsFound = 0;
        while (cache[rootsFound + 1] !== undefined) rootsFound++;
        let x = rootsFound > 0 ? cache[rootsFound] + step : 0.1;

        let lastSign = Math.sign(fn(x)) || 1;

        while (rootsFound < m) {
            x += step;
            const sign = Math.sign(fn(x));
            if (sign !== 0) {
                if (sign !== lastSign) {
                    rootsFound++;
                    cache[rootsFound] = bisection(fn, x - step, x);
                }
                lastSign = sign;
            }
        }

        return cache[m];
    }
};