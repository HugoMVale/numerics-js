/**
 * Special functions.
 *
 * - Use {@link besselJ} to evaluate the Bessel function of the first kind,
 *   `J_n(x)`, for non-negative integer orders.
 * - Use {@link besselJZero} to obtain its positive zeros.
 * - Use {@link erf} and {@link erfc} to evaluate the error function and its complement.
 * - Use {@link gamma} and {@link lgamma} to evaluate the gamma function and the logarithm of its absolute value.
 *
 * @module special
 */
export { besselJ, besselJZero } from './special/bessel.js';
export { erf, erfc } from './special/erf.js';
export { gamma } from './special/gamma.js';
export { lgamma } from './special/lgamma.js';
