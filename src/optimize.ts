/**
 * Optimization methods.
 *
 * Choose a method based on the type of problem:
 *
 * - Use {@link brent} to minimize a scalar function within a bounded
 *   search interval.
 * - Use {@link nelderMead} to minimize an unconstrained multivariate
 *   function when derivatives are unavailable or inconvenient to provide.
 * - Use {@link levenbergMarquardt} to minimize the sum of squares of a vector
 *   function (nonlinear least squares); derivatives are supplied or
 *   approximated by finite differences.
 * - Use {@link curveFit} to fit a model function to data (nonlinear least
 *   squares), with parameter uncertainties.
 *
 * @module optimize
 */
export { brent } from './optimize/brent.js';
export { nelderMead } from './optimize/nelderMead.js';
export { levenbergMarquardt } from './optimize/levenbergMarquardt.js';
export { curveFit } from './optimize/curveFit.js';
export type { BrentResult } from './optimize/brent.js';
export type { NelderMeadResult, NelderMeadOptions } from './optimize/nelderMead.js';
export type {
  LevenbergMarquardtResult,
  LevenbergMarquardtOptions
} from './optimize/levenbergMarquardt.js';
export type { CurveFitResult, CurveFitOptions } from './optimize/curveFit.js';
