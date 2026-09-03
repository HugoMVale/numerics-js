/**
 * Derivative-free optimization methods.
 *
 * Choose a method based on the dimensionality of the objective:
 *
 * - Use {@link brent} to minimize a scalar function within a bounded
 *   search interval.
 * - Use {@link nelderMead} to minimize an unconstrained multivariate
 *   function when derivatives are unavailable or inconvenient to provide.
 *
 * @module optimize
 */
export { brent } from './optimize/brent.js';
export { nelderMead } from './optimize/nelderMead.js';
export type { BrentResult, BrentOptions } from './optimize/brent.js';
export type { NelderMeadResult, NelderMeadOptions } from './optimize/nelderMead.js';