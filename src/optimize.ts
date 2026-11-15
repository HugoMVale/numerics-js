/**
 * Optimization methods.
 *
 * Choose a method based on the type of problem:
 *
 * - Use {@link brent} to minimize a scalar function within a bounded
 *   search interval.
 * - Use {@link nelderMead} to minimize an unconstrained multivariate
 *   function when derivatives are unavailable or inconvenient to provide.
 * - Use {@link quasiNewton} to minimize an unconstrained multivariate
 *   function using gradient information (supplied or approximated by finite
 *   differences); it is typically faster and more accurate than
 *   {@link nelderMead} for smooth functions.
 * - Use {@link levenbergMarquardt} to minimize the sum of squares of a vector
 *   function (nonlinear least squares); derivatives are supplied or
 *   approximated by finite differences.
 * - Use {@link curveFit} to fit a model function to data (nonlinear least
 *   squares), with parameter uncertainties.
 *
 * @module optimize
 */
export { brent, type BrentOptions, type BrentResult } from './optimize/brent.js';
export {
  nelderMead,
  type NelderMeadOptions,
  type NelderMeadResult
} from './optimize/nelderMead.js';
export {
  quasiNewton,
  type BfgsMethod,
  type QuasiNewtonOptions,
  type QuasiNewtonResult
} from './optimize/quasiNewton.js';
export {
  levenbergMarquardt,
  type LevenbergMarquardtOptions,
  type LevenbergMarquardtResult
} from './optimize/levenbergMarquardt.js';
export { curveFit, type CurveFitOptions, type CurveFitResult } from './optimize/curveFit.js';
