/**
 * Scalar root-finding methods.
 *
 * Choose a method based on whether a root can be bracketed:
 *
 * - Use {@link brent} as the general default when the endpoints have opposite
 *   signs; it combines bracketing reliability with fast convergence.
 * - Use {@link bisection} when a bracket is available and a simple,
 *   predictable convergence guarantee is more important than speed.
 * - Use {@link secant} with good initial guesses when no bracket is available
 *   and faster convergence is worth the risk of non-convergence.
 *
 * @module roots
 */
export { bisection } from './roots/bisection.js';
export { brent } from './roots/brent.js';
export { secant } from './roots/secant.js';
export type { BisectionOptions, BrentOptions, SecantOptions, RootResult } from './roots/types.js';
