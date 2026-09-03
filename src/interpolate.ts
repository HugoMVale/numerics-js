/**
 * One-dimensional interpolation methods.
 *
 * Choose a method based on the required accuracy and use pattern:
 *
 * - Use {@link interp} for simple, one-off linear interpolation.
 * - Use {@link LinearInterpolator} when evaluating the same data repeatedly
 *   with piecewise-linear interpolation.
 * - Use {@link PchipInterpolator} when smooth, shape-preserving interpolation
 *   is needed without overshooting monotonic data.
 *
 * @module interpolate
 */
export { LinearInterpolator, interp } from './interpolate/interp1.js';
export { PchipInterpolator } from './interpolate/pchip.js';
export type { InterpOptions } from './interpolate/common.js';