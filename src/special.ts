/**
 * Special functions.
 *
 * - Use {@link besselJ} to evaluate the Bessel function of the first kind,
 *   `J_n(x)`, for non-negative integer orders.
 * - Use {@link besselJZero} to obtain its positive zeros.
 * - Use {@link erf} to evaluate the error function.
 *
 * @module special
 */
export { besselJ, besselJZero } from './special/bessel.js';
export { erf } from './special/erf.js';
