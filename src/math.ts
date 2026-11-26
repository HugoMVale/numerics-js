/**
 * Scalar mathematical utilities.
 *
 * - Use {@link clip} to restrict a number to a range.
 * - Use {@link copysign} to transfer the sign of one number to the magnitude of another.
 * - Use {@link isClose} to compare two numbers with absolute and relative tolerances.
 * - Use {@link createRng} to create a deterministic, seedable pseudo-random number generator.
 *
 * @module math
 */
export { clip } from './math/clip.js';
export { copysign } from './math/copysign.js';
export { isClose, type CloseOptions } from './math/isClose.js';
export { createRng, type RngMethod } from './math/createRng.js';
