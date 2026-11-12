import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import {
  prepareInterp1D,
  prepareInterpND,
  checkStrictlyIncreasing,
  type Interp1DOptions,
  type InterpNDOptions
} from './common.js';
import {
  hermiteOne1D,
  hermiteMany1D,
  hermiteOneND,
  hermiteManyND,
  hermiteIntegrate1D,
  hermiteIntegrateND
} from './hermite.js';

/**
 * Computes the PCHIP slopes at the knots (the first derivative of the
 * interpolant at each knot) for one scalar series `y` over the knots `x`:
 * a weighted harmonic mean of the adjacent secant slopes at interior knots
 * (zero at local extrema) and a shape-preserving three-point formula at the
 * ends, which together keep the interpolant monotone wherever the data are.
 * Shared by `PchipInterpolator1D` (called once, over `fp`) and
 * `PchipInterpolatorND` (called once per column of `fp`), since the
 * slope-fitting algorithm only ever sees one series at a time. The
 * counterpart of `computeSplineSlopes` in `cubic.ts`.
 * @param x The strictly increasing x-coordinates of the data points (length `n`).
 * @param y The y-coordinates of the data points (length `n`).
 * @returns The slope at each knot (length `n`).
 */
function computePchipSlopes(x: Float64Array, y: Float64Array): Float64Array {
  const n = x.length;
  const slopes = new Float64Array(n);

  if (n === 1) return slopes;

  if (n === 2) {
    slopes[0] = slopes[1] = (y[1] - y[0]) / (x[1] - x[0]);
    return slopes;
  }

  const hk = new Float64Array(n - 1);
  const mk = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    hk[i] = x[i + 1] - x[i];
    mk[i] = (y[i + 1] - y[i]) / hk[i];
  }

  // Interior derivatives (harmonic mean). The sign of each secant slope is tested instead of the
  // sign of their product, which would underflow to 0 for very small slopes (e.g. 1e-200 * 1e-200).
  for (let i = 1; i < n - 1; i++) {
    if (Math.sign(mk[i - 1]) * Math.sign(mk[i]) <= 0) {
      slopes[i] = 0;
    } else {
      const w1 = 2 * hk[i] + hk[i - 1];
      const w2 = hk[i] + 2 * hk[i - 1];
      slopes[i] = (w1 + w2) / (w1 / mk[i - 1] + w2 / mk[i]);
    }
  }

  // Endpoint derivative (left)
  slopes[0] = ((2 * hk[0] + hk[1]) * mk[0] - hk[0] * mk[1]) / (hk[0] + hk[1]);
  if (Math.sign(slopes[0]) !== Math.sign(mk[0])) {
    slopes[0] = 0;
  } else if (Math.sign(mk[0]) !== Math.sign(mk[1]) && Math.abs(slopes[0]) > 3 * Math.abs(mk[0])) {
    slopes[0] = 3 * mk[0];
  }

  // Endpoint derivative (right)
  const mN1 = mk[n - 2];
  const mN2 = mk[n - 3];
  const hN1 = hk[n - 2];
  const hN2 = hk[n - 3];
  slopes[n - 1] = ((2 * hN1 + hN2) * mN1 - hN1 * mN2) / (hN1 + hN2);
  if (Math.sign(slopes[n - 1]) !== Math.sign(mN1)) {
    slopes[n - 1] = 0;
  } else if (Math.sign(mN1) !== Math.sign(mN2) && Math.abs(slopes[n - 1]) > 3 * Math.abs(mN1)) {
    slopes[n - 1] = 3 * mN1;
  }

  return slopes;
}

/**
 * Builds the per-knot, per-component slope Matrix (`n x m`) for
 * `PchipInterpolatorND`, by running `computePchipSlopes` independently on
 * each column of `fp` (each output component is its own scalar PCHIP problem
 * sharing `xp`). The counterpart of `computeSplineSlopesND` in `cubic.ts`.
 */
function computePchipSlopesND(xp: Vector, fp: Matrix): Matrix {
  const n = xp.size;
  const m = fp.cols;
  const slopes = new Matrix(n, m);

  for (let j = 0; j < m; j++) {
    slopes.setCol(j, computePchipSlopes(xp.data, fp.col(j).data));
  }
  return slopes;
}

/**
 * A shape-preserving cubic Hermite interpolant (PCHIP) for a
 * scalar-valued function (`number -> number`).
 *
 * Given the discrete data points `(xp[i], fp[i])`, with finite, strictly increasing `xp`,
 * `eval(x)` returns a shape-preserving cubic interpolation at `x`. Unlike a
 * standard cubic spline, PCHIP does not introduce overshoots between monotonic
 * data points and has a continuous first derivative. For `x` outside the range
 * of `xp`, the result is clamped to the boundary value of `fp` unless
 * `options.left`/`options.right` are given.
 *
 * For interpolating a vector-valued function (`number -> Vector`) against
 * the same shared `xp` knots, see {@link PchipInterpolatorND}.
 *
 * @example
 * ```ts
 * import { PchipInterpolator1D } from 'numerics-js/interpolate';
 *
 * const f = new PchipInterpolator1D([1, 2, 3], [3, 2, 0]);
 * console.log(f.eval(2.5));
 * console.log(f.eval([0, 1.5, 3.14]).toString());
 * ```
 *
 * Output:
 * ```
 * 1.1458333333333335
 * Vector(3, 2.6041666666666665, 0)
 * ```
 */
export class PchipInterpolator1D {
  private readonly xp: Vector;
  private readonly fp: Vector;
  private readonly leftVal: number;
  private readonly rightVal: number;
  private readonly slopes: Float64Array; // Precomputed slopes at knots

  /**
   * @param xp The `x`-coordinates of the data points: a `Vector`, a plain `number[]` or a
   * `Float64Array`. Must be strictly increasing, finite, and non-empty. A single point defines a
   * constant interpolant. The data is copied, so later changes to the input do not affect the
   * interpolator.
   * @param fp The `y`-coordinates of the data points: a `Vector`, a plain `number[]` or a
   * `Float64Array`, with the same length as `xp`. Copied like `xp`.
   * @param options Optional settings; see {@link Interp1DOptions}. Validation
   * (matching lengths, non-empty, and, unless `options.checkSorted` is
   * `false`, strictly increasing) happens once, here in the constructor.
   * If `options.checkSorted` is `false` and `xp` is not actually sorted,
   * `eval`, `derivative`, and `integrate` results are unspecified.
   * @throws {TypeError} If `xp` or `fp` is not a `VectorLike`.
   * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different
   * lengths, if `xp` contains non-finite values, or (when
   * `options.checkSorted` is `true`) if `xp` is not strictly increasing.
   */
  constructor(xp: VectorLike, fp: VectorLike, options: Interp1DOptions = {}) {
    const { left, right, checkSorted = true } = options;
    const p = prepareInterp1D(xp, fp, left, right, checkSorted, 'PchipInterpolator1D', true);
    if (checkSorted) {
      checkStrictlyIncreasing(p.xp, 'PchipInterpolator1D');
    }
    this.xp = p.xp;
    this.fp = p.fp;
    this.leftVal = p.leftVal;
    this.rightVal = p.rightVal;
    this.slopes = computePchipSlopes(this.xp.data, this.fp.data);
  }

  /**
   * The number of data points backing this interpolant.
   */
  get size(): number {
    return this.xp.size;
  }

  /**
   * Evaluates the shape-preserving cubic interpolant at `x`.
   * @param x The x-coordinate(s) at which to evaluate. A single `number` returns a `number`;
   * a `VectorLike` (a `Vector`,
   * plain `number[]` or `Float64Array`) returns a `Vector`.
   * @returns The interpolated or clamped value(s), matching the shape of `x`.
   * @throws {TypeError} If `x` is neither a `number` nor a `VectorLike`.
   */
  eval(x: number): number;
  eval(x: VectorLike): Vector;
  eval(x: number | VectorLike): number | Vector {
    if (typeof x === 'number') {
      return hermiteOne1D(x, this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, false);
    }
    return hermiteMany1D(x, this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, false);
  }

  /**
   * Evaluates the first derivative of the interpolant at `x`.
   * Values outside the data range return `0`, the derivative of the
   * constant clamping regions.
   * @param x The `x`-coordinate(s) at which to evaluate. A single
   * `number` returns a `number`; a `VectorLike` (a `Vector`,
   * plain `number[]` or `Float64Array`) returns an `Vector`.
   * @returns The derivative value(s), matching the shape of `x`.
   * @throws {TypeError} If `x` is neither a `number` nor a `VectorLike`.
   */
  derivative(x: number): number;
  derivative(x: VectorLike): Vector;
  derivative(x: number | VectorLike): number | Vector {
    if (typeof x === 'number') {
      return hermiteOne1D(x, this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, true);
    }
    return hermiteMany1D(x, this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, true);
  }

  /**
   * Integrates the interpolant from `a` to `b` analytically.
   *
   * Evaluates the exact piecewise cubic polynomial within the data bounds,
   * with `left`/`right` clamp values forming constant rectangular areas
   * outside the `xp` range.
   *
   * @param a Lower integration bound `a`.
   * @param b Upper integration bound `b`.
   * @returns The integral of the interpolant over `[a, b]`.
   * @throws {RangeError} If either bound is `NaN`.
   */
  integrate(a: number, b: number): number {
    if (Number.isNaN(a) || Number.isNaN(b)) {
      throw new RangeError('PchipInterpolator1D.integrate: bounds must not be NaN');
    }
    return hermiteIntegrate1D(this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, a, b);
  }
}

/**
 * A shape-preserving cubic Hermite interpolant (PCHIP) for a
 * vector-valued function (`number -> Vector`).
 *
 * Given the discrete data points `(xp[i], fp.row(i))`, with finite, strictly
 * increasing `xp`, `eval(x)` returns the shape-preserving cubic interpolation
 * at `x`, applied independently to each output component (each column of
 * `fp` gets its own PCHIP fit, all sharing the same `xp` knots). For `x`
 * outside the range of `xp`, the result is clamped to the boundary row of
 * `fp` unless `options.left`/`options.right` are given.
 *
 * For interpolating a scalar-valued function (`number -> number`),
 * see {@link PchipInterpolator1D}.
 *
 * @example
 * ```ts
 * import { Matrix } from 'numerics-js/linalg';
 * import { PchipInterpolatorND } from 'numerics-js/interpolate';
 *
 * const f = new PchipInterpolatorND([1, 2, 3], Matrix.from([[3, 0], [2, 1], [0, 4]]));
 * console.log(f.eval(2.5).toString());
 * ```
 *
 * Output:
 * ```
 * Vector(1.1458333333333335, 2.1875)
 * ```
 */
export class PchipInterpolatorND {
  private readonly xp: Vector;
  private readonly fp: Matrix;
  private readonly leftVal: Vector;
  private readonly rightVal: Vector;
  private readonly slopes: Matrix; // Precomputed slopes at knots, n x dim

  /**
   * @param xp The `x`-coordinates of the data points, shared by every
   * output component: a `Vector`, a plain `number[]` or a `Float64Array`. Must be strictly
   * increasing, finite, and non-empty. A single point defines a constant interpolant. Copied, like
   * any `Vector` passed as `options.left`/`options.right`, so later changes to the inputs do not
   * affect the interpolator.
   * @param fp The vector-valued data points: row `i` is the value at
   * `xp[i]`. Must have `fp.rows === xp.length`. Copied, like `xp`.
   * @param options Optional settings; see {@link InterpNDOptions}.
   * @throws {TypeError} If `xp` (or an explicit `options.left`/`options.right`) is not a `VectorLike`.
   * @throws {RangeError} If `xp` is empty, if `xp.length !== fp.rows`,
   * if an explicit `options.left` or `options.right` `VectorLike` does
   * not have length `fp.cols`, if `xp` contains non-finite values, or
   * (when `options.checkSorted` is `true`) if `xp` is not strictly
   * increasing.
   */
  constructor(xp: VectorLike, fp: Matrix, options: InterpNDOptions = {}) {
    const { left, right, checkSorted = true } = options;
    const p = prepareInterpND(xp, fp, left, right, checkSorted, 'PchipInterpolatorND', true);
    if (checkSorted) {
      checkStrictlyIncreasing(p.xp, 'PchipInterpolatorND');
    }
    this.xp = p.xp;
    this.fp = p.fp;
    this.leftVal = p.leftVal;
    this.rightVal = p.rightVal;
    this.slopes = computePchipSlopesND(this.xp, this.fp);
  }

  /**
   * The number of data points (knots) backing this interpolant.
   */
  get size(): number {
    return this.xp.size;
  }

  /**
   * The dimension of the interpolated output vector (`fp.cols`).
   */
  get dim(): number {
    return this.fp.cols;
  }

  /**
   * Evaluates the shape-preserving cubic interpolant at `x`.
   * @param x The x-coordinate(s) at which to evaluate. A single `number`
   * returns a `Vector` (length `dim`); a `VectorLike` (a `Vector`,
   * plain `number[]` or `Float64Array`) returns
   * a `Matrix` (one row per evaluation point, `dim` columns).
   * @returns The interpolated or clamped value(s), matching the shape of `x`.
   * @throws {TypeError} If `x` is neither a `number` nor a `VectorLike`.
   * @throws {RangeError} If `x` is a `VectorLike` of length `0`
   * (`Matrix` cannot represent a result with `0` rows).
   */
  eval(x: number): Vector;
  eval(x: VectorLike): Matrix;
  eval(x: number | VectorLike): Vector | Matrix {
    if (typeof x === 'number') {
      return hermiteOneND(x, this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, false);
    }
    return hermiteManyND(
      x,
      this.xp,
      this.fp,
      this.slopes,
      this.leftVal,
      this.rightVal,
      false,
      'PchipInterpolatorND.eval'
    );
  }

  /**
   * Evaluates the first derivative of the interpolant at `x`, component-wise.
   * Values outside the data range return the zero vector, the derivative
   * of the constant clamping regions.
   * @param x The `x`-coordinate(s) at which to evaluate. A single
   * `number` returns a `Vector`; a `VectorLike` (a `Vector`,
   * plain `number[]` or `Float64Array`) returns a `Matrix`.
   * @returns The derivative value(s), matching the shape of `x`.
   * @throws {TypeError} If `x` is neither a `number` nor a `VectorLike`.
   * @throws {RangeError} If `x` is a `VectorLike` of length `0`
   * (`Matrix` cannot represent a result with `0` rows).
   */
  derivative(x: number): Vector;
  derivative(x: VectorLike): Matrix;
  derivative(x: number | VectorLike): Vector | Matrix {
    if (typeof x === 'number') {
      return hermiteOneND(x, this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, true);
    }
    return hermiteManyND(
      x,
      this.xp,
      this.fp,
      this.slopes,
      this.leftVal,
      this.rightVal,
      true,
      'PchipInterpolatorND.derivative'
    );
  }

  /**
   * Integrates the interpolant from `a` to `b`, component-wise,
   * analytically. See `PchipInterpolator1D.integrate` for the
   * algorithm; this applies it independently to each output component.
   * @param a Lower integration bound `a`.
   * @param b Upper integration bound `b`.
   * @returns A `Vector` of length `dim`, the integral of each component over `[a, b]`.
   * @throws {RangeError} If either bound is `NaN`.
   */
  integrate(a: number, b: number): Vector {
    if (Number.isNaN(a) || Number.isNaN(b)) {
      throw new RangeError('PchipInterpolatorND.integrate: bounds must not be NaN');
    }
    return hermiteIntegrateND(this.xp, this.fp, this.slopes, this.leftVal, this.rightVal, a, b);
  }
}
