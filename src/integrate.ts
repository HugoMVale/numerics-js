/**
 * Numerical integration (quadrature) methods.
 *
 * Choose a method based on the form of the input:
 *
 * - Use {@link quad} for a general function, possibly with infinite bounds,
 *   known discontinuities, or sharp features supplied as breakpoints.
 * - Use {@link gaussKronrod} for a smooth function on a finite interval 
 *   (otherwise use {@link quad}).
 * - Use {@link simpson} for sampled data when a higher-order rule is useful.
 * - Use {@link trapezoid} for sampled data when a simple, robust estimate is
 *   sufficient.
 *
 * @module integrate
 */
export { trapezoid } from './integrate/trapezoid.js';
export { simpson } from './integrate/simpson.js';
export { gaussKronrod } from './integrate/gaussKronrod.js';
export { quad } from './integrate/quad.js';
export type { GaussKronrodResult } from './integrate/gaussKronrod.js';
export type { QuadResult, QuadOptions } from './integrate/quad.js';