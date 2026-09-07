/**
 * Interpolation methods for scalar (1D) or vector-valued (ND) functions that share a common
 * set of x-coordinates.
 *
 * Choose a method based on the required accuracy and use pattern:
 *
 * - Use {@link interp1D} or {@link interpND} for simple, one-off linear
 *   interpolation of scalar or vector-valued data.
 * - Use {@link LinearInterpolator1D} or {@link LinearInterpolatorND} when
 *   evaluating the same scalar or vector-valued data repeatedly with
 *   piecewise-linear interpolation.
 * - Use {@link PchipInterpolator1D} or {@link PchipInterpolatorND} when smooth,
 *   shape-preserving interpolation is needed without overshooting monotonic
 *   data.
 *
 * @module interpolate
 */
export { LinearInterpolator1D, LinearInterpolatorND, interp1D, interpND } from './interpolate/interp1.js';
export { PchipInterpolator1D, PchipInterpolatorND } from './interpolate/pchip.js';
export type { Interp1DOptions, InterpNDOptions } from './interpolate/common.js';