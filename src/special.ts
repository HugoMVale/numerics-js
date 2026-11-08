/**
 * Special functions.
 *
 * - Use {@link besselJ} to evaluate the Bessel function of the first kind,
 *   `J_n(x)`, for non-negative integer orders.
 * - Use {@link besselJZero} to obtain its positive zeros.
 * - Use {@link besselY} to evaluate the Bessel function of the second kind,
 *   `Y_n(x)`, for non-negative integer orders and `x > 0`.
 * - Use {@link besselI} to evaluate the modified Bessel function of the first kind,
 *   `I_n(x)`, and {@link besselIScaled} for its exponentially scaled form `e^{-|x|} I_n(x)`.
 * - Use {@link besselK} to evaluate the modified Bessel function of the second kind,
 *   `K_n(x)`, for non-negative integer orders and `x > 0`, and {@link besselKScaled}
 *   for its exponentially scaled form `e^x K_n(x)`.
 * - Use {@link erf} and {@link erfc} to evaluate the error function and its complement.
 * - Use {@link gamma} and {@link lgamma} to evaluate the gamma function and the logarithm of its absolute value.
 *
 * @module special
 */
export {
  besselI,
  besselIScaled,
  besselJ,
  besselJZero,
  besselK,
  besselKScaled,
  besselY
} from './special/bessel.js';
export { erf, erfc } from './special/erf.js';
export { gamma } from './special/gamma.js';
export { lgamma } from './special/lgamma.js';
