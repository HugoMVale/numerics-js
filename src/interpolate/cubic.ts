import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import {
  prepareInterp1D,
  prepareInterpND,
  checkStrictlyIncreasing,
  Interp1DOptions,
  InterpNDOptions
} from './common.js';
import {
  hermiteOne1D,
  hermiteMany1D,
  hermiteOneND,
  hermiteManyND,
  hermiteIntegrate1D,
  hermiteIntegrateND
} from './hermite.js';
import { factorTridiagonal, solveTridiagonal, type TridiagonalLU } from './tridiagonal.js';

/**
 * End conditions of a cubic spline.
 *
 * - `'not-a-knot'`: the third derivative is continuous at the second and
 *   second-to-last knots, i.e. the first two (and the last two) polynomial
 *   pieces coincide. Needs no extra information and is the usual best choice
 *   when nothing is known about the end behavior.
 * - `'natural'`: the second derivative is zero at both ends.
 * - `'clamped'`: the first derivative is prescribed at both ends, see
 *   {@link CubicSplineOptions1D.leftDerivative}.
 */
export type CubicSplineBoundary = 'not-a-knot' | 'natural' | 'clamped';

/**
 * Options for {@link CubicSplineInterpolator1D}, on top of the clamping and
 * validation options of {@link Interp1DOptions}.
 */
export interface CubicSplineOptions1D extends Interp1DOptions {
  /**
   * End conditions of the spline; see {@link CubicSplineBoundary}.
   * @default 'not-a-knot'
   */
  bc?: CubicSplineBoundary;
  /**
   * First derivative of the spline at `xp[0]`. Only valid with
   * `bc: 'clamped'`.
   * @default 0
   */
  leftDerivative?: number;
  /**
   * First derivative of the spline at `xp[xp.length - 1]`. Only valid with
   * `bc: 'clamped'`.
   * @default 0
   */
  rightDerivative?: number;
}

/**
 * Options for {@link CubicSplineInterpolatorND}, on top of the clamping and
 * validation options of {@link InterpNDOptions}. The vector counterpart of
 * {@link CubicSplineOptions1D}: the end derivatives are per-component, but a
 * bare `number` broadcasts to every component.
 */
export interface CubicSplineOptionsND extends InterpNDOptions {
  /**
   * End conditions of the spline, shared by every component; see
   * {@link CubicSplineBoundary}.
   * @default 'not-a-knot'
   */
  bc?: CubicSplineBoundary;
  /**
   * First derivative(s) at `xp[0]`. A single `number` broadcasts to every
   * component; a `VectorLike` must have length `fp.cols`. Only valid with
   * `bc: 'clamped'`.
   * @default 0
   */
  leftDerivative?: number | VectorLike;
  /**
   * First derivative(s) at `xp[xp.length - 1]`; same forms as
   * `leftDerivative`. Only valid with `bc: 'clamped'`.
   * @default 0
   */
  rightDerivative?: number | VectorLike;
}

const BOUNDARIES: readonly CubicSplineBoundary[] = ['not-a-knot', 'natural', 'clamped'];

/**
 * Checks `bc` and the end-derivative options shared by the 1D and ND classes.
 * @throws {RangeError} If `bc` is unknown or end derivatives are given
 * without `bc: 'clamped'`.
 */
function checkBoundary(
  bc: CubicSplineBoundary,
  leftDerivative: unknown,
  rightDerivative: unknown,
  caller: string
): void {
  if (!BOUNDARIES.includes(bc)) {
    throw new RangeError(`${caller}: bc must be one of ${BOUNDARIES.join(', ')}; got '${bc}'`);
  }
  if (bc !== 'clamped' && (leftDerivative !== undefined || rightDerivative !== undefined)) {
    throw new RangeError(
      `${caller}: leftDerivative and rightDerivative require bc to be 'clamped'`
    );
  }
}

/** Rejects a non-finite end derivative. */
function checkDerivative(value: number, name: string, caller: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${caller}: ${name} must be finite`);
  }
}

/**
 * Resolves a vector-valued end derivative to a `Float64Array` of length
 * `dim`: `undefined` is `0`, a `number` broadcasts, a `VectorLike` must have
 * length `dim`.
 */
function resolveDerivativeND(
  val: number | VectorLike | undefined,
  dim: number,
  name: string,
  caller: string
): Float64Array {
  const res = new Float64Array(dim);
  if (val === undefined) return res;
  if (typeof val === 'number') {
    checkDerivative(val, name, caller);
    res.fill(val);
    return res;
  }
  const v = Vector.asVector(val);
  if (v.size !== dim) {
    throw new RangeError(`${caller}: ${name} must have length ${dim} (fp.cols), got ${v.size}`);
  }
  for (let j = 0; j < dim; j++) {
    checkDerivative(v.data[j], name, caller);
    res[j] = v.data[j];
  }
  return res;
}

/**
 * The part of the spline's slope system that depends only on the knots and the
 * end conditions, so that it is factored once and reused for every output
 * component of a vector-valued spline.
 *
 * The slopes `s[i]` at the knots of a C2 cubic spline satisfy, with
 * `h[i] = x[i+1] - x[i]` and `delta[i] = (y[i+1] - y[i]) / h[i]`,
 *
 * `h[i] s[i-1] + 2 (h[i-1] + h[i]) s[i] + h[i-1] s[i+1] = 3 (h[i] delta[i-1] + h[i-1] delta[i])`
 *
 * for every interior knot (continuity of the second derivative), plus one
 * condition at each end. Together this is a tridiagonal system.
 */
interface SplineSystem {
  n: number;
  bc: CubicSplineBoundary;
  /** Knot spacings, length `n - 1`. */
  h: Float64Array;
  /** Factored system; `null` for the closed-form cases `n <= 2` and `n === 3` with not-a-knot. */
  lu: TridiagonalLU | null;
}

/** Assembles and factors the tridiagonal slope system for the knots `x` and end conditions `bc`. */
function buildSplineSystem(x: Float64Array, bc: CubicSplineBoundary): SplineSystem {
  const n = x.length;
  const h = new Float64Array(Math.max(n - 1, 0));
  for (let i = 0; i < n - 1; i++) h[i] = x[i + 1] - x[i];

  // Closed-form cases: constant, line/Hermite cubic, and the parabola through
  // three points (not-a-knot with n = 3 is under-determined otherwise).
  if (n <= 2 || (n === 3 && bc === 'not-a-knot')) {
    return { n, bc, h, lu: null };
  }

  const dl = new Float64Array(n - 1); // dl[i - 1]: row i, column i - 1
  const d = new Float64Array(n);
  const du = new Float64Array(n - 1); // du[i]: row i, column i + 1

  for (let i = 1; i < n - 1; i++) {
    dl[i - 1] = h[i];
    d[i] = 2 * (h[i - 1] + h[i]);
    du[i] = h[i - 1];
  }

  switch (bc) {
    case 'natural':
      d[0] = 2;
      du[0] = 1;
      dl[n - 2] = 1;
      d[n - 1] = 2;
      break;
    case 'clamped':
      d[0] = 1;
      du[0] = 0;
      dl[n - 2] = 0;
      d[n - 1] = 1;
      break;
    case 'not-a-knot':
      d[0] = h[1];
      du[0] = h[0] + h[1];
      dl[n - 2] = h[n - 2] + h[n - 3];
      d[n - 1] = h[n - 3];
      break;
  }

  return { n, bc, h, lu: factorTridiagonal(dl, d, du) };
}

/**
 * Computes the spline slopes at the knots for one scalar series `y`, given
 * the (already factored) system for its knots. Shared by
 * `CubicSplineInterpolator1D` (once) and `CubicSplineInterpolatorND` (once per
 * column of `fp`).
 * @param y The y-coordinates of the data points (length `sys.n`).
 * @param leftD End derivative at the first knot (used only when `bc` is `'clamped'`).
 * @param rightD End derivative at the last knot (used only when `bc` is `'clamped'`).
 * @returns The slope at each knot.
 */
function computeSplineSlopes(
  sys: SplineSystem,
  y: Float64Array,
  leftD: number,
  rightD: number
): Float64Array {
  const { n, bc, h, lu } = sys;

  if (n === 1) return new Float64Array(1);

  const delta = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) delta[i] = (y[i + 1] - y[i]) / h[i];

  if (n === 2) {
    return bc === 'clamped' ? Float64Array.of(leftD, rightD) : Float64Array.of(delta[0], delta[0]);
  }

  if (lu === null) {
    // n === 3, not-a-knot: derivatives of the parabola through the three points
    const c = (delta[1] - delta[0]) / (h[0] + h[1]);
    return Float64Array.of(
      delta[0] - c * h[0],
      delta[0] + c * h[0],
      delta[0] + c * (h[0] + 2 * h[1])
    );
  }

  const r = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) {
    r[i] = 3 * (h[i] * delta[i - 1] + h[i - 1] * delta[i]);
  }

  switch (bc) {
    case 'natural':
      r[0] = 3 * delta[0];
      r[n - 1] = 3 * delta[n - 2];
      break;
    case 'clamped':
      r[0] = leftD;
      r[n - 1] = rightD;
      break;
    case 'not-a-knot': {
      const s0 = h[0] + h[1];
      r[0] = ((h[0] + 2 * s0) * h[1] * delta[0] + h[0] * h[0] * delta[1]) / s0;
      const s1 = h[n - 2] + h[n - 3];
      r[n - 1] =
        (h[n - 2] * h[n - 2] * delta[n - 3] + (2 * s1 + h[n - 2]) * h[n - 3] * delta[n - 2]) / s1;
      break;
    }
  }

  solveTridiagonal(lu, r);
  return r;
}

/**
 * A C2 cubic spline interpolant for a scalar-valued function
 * (`number -> number`).
 *
 * Given the discrete data points `(xp[i], fp[i])`, with finite, strictly
 * increasing `xp`, the interpolant is the unique piecewise cubic polynomial
 * that passes through all points and has continuous first and second
 * derivatives, completed by the end conditions of `options.bc` (not-a-knot by
 * default; see {@link CubicSplineBoundary}). The slopes at the knots are
 * obtained from one tridiagonal linear system, solved by LU factorization with
 * partial pivoting in `O(n)` operations.
 *
 * Because the spline is a global fit, every knot value influences the whole
 * curve, and a non-finite value in `fp` makes the whole interpolant `NaN`.
 * Unlike {@link PchipInterpolator1D}, a cubic spline is smoother but not
 * shape-preserving: it can overshoot between monotonic data points.
 *
 * With a single point the interpolant is constant; with two points it is the
 * straight line through them (or the Hermite cubic with the prescribed end
 * derivatives for `bc: 'clamped'`); with three points and `'not-a-knot'` it is
 * the parabola through them. For `x` outside the range of `xp`, the result is
 * clamped to the boundary value of `fp` unless `options.left`/`options.right`
 * are given, as for the other interpolators of this module.
 *
 * For interpolating a vector-valued function (`number -> Vector`) against the
 * same shared `xp` knots, see {@link CubicSplineInterpolatorND}.
 *
 * @example
 * ```ts
 * import { CubicSplineInterpolator1D } from 'numerics-js/interpolate';
 *
 * // Samples of x^3, which a not-a-knot spline reproduces exactly
 * const f = new CubicSplineInterpolator1D([0, 1, 2, 3, 4], [0, 1, 8, 27, 64]);
 * console.log(f.eval(2.5));
 * console.log(f.derivative(2.5));
 * console.log(f.integrate(0, 4));
 * ```
 *
 * Output:
 * ```
 * 15.625
 * 18.75
 * 64
 * ```
 */
export class CubicSplineInterpolator1D {
  private readonly xp: Vector;
  private readonly fp: Vector;
  private readonly leftVal: number;
  private readonly rightVal: number;
  private readonly d: Float64Array; // Slopes at knots

  /**
   * @param xp The `x`-coordinates of the data points: a `Vector`, a plain `number[]` or a
   * `Float64Array`. Must be strictly increasing, finite, and non-empty. The data is copied, so
   * later changes to the input do not affect the interpolator.
   * @param fp The `y`-coordinates of the data points: a `Vector`, a plain `number[]` or a
   * `Float64Array`, with the same length as `xp`. Copied like `xp`.
   * @param options Optional settings; see {@link CubicSplineOptions1D}. Validation (matching
   * lengths, non-empty, and, unless `options.checkSorted` is `false`, strictly increasing) happens
   * once, here in the constructor. If `options.checkSorted` is `false` and `xp` is not actually
   * strictly increasing, `eval`, `derivative`, and `integrate` results are unspecified.
   * @throws {TypeError} If `xp` or `fp` is not a `VectorLike`.
   * @throws {RangeError} If `xp` is empty, if `xp` and `fp` have different lengths, if `xp`
   * contains non-finite values, if (when `options.checkSorted` is `true`) `xp` is not strictly
   * increasing, if `options.bc` is not a known boundary type, or if `options.leftDerivative` /
   * `options.rightDerivative` is non-finite or given without `bc: 'clamped'`.
   */
  constructor(xp: VectorLike, fp: VectorLike, options: CubicSplineOptions1D = {}) {
    const {
      left,
      right,
      checkSorted = true,
      bc = 'not-a-knot',
      leftDerivative,
      rightDerivative
    } = options;
    const caller = 'CubicSplineInterpolator1D';
    checkBoundary(bc, leftDerivative, rightDerivative, caller);
    const leftD = leftDerivative ?? 0;
    const rightD = rightDerivative ?? 0;
    checkDerivative(leftD, 'leftDerivative', caller);
    checkDerivative(rightD, 'rightDerivative', caller);

    const p = prepareInterp1D(xp, fp, left, right, checkSorted, caller, true);
    if (checkSorted) {
      checkStrictlyIncreasing(p.xp, caller);
    }
    this.xp = p.xp;
    this.fp = p.fp;
    this.leftVal = p.leftVal;
    this.rightVal = p.rightVal;
    this.d = computeSplineSlopes(buildSplineSystem(this.xp.data, bc), this.fp.data, leftD, rightD);
  }

  /**
   * The number of data points backing this interpolant.
   */
  get size(): number {
    return this.xp.size;
  }

  /**
   * Evaluates the cubic spline at `x`.
   * @param x The x-coordinate(s) at which to evaluate. A single `number` returns a `number`;
   * a `VectorLike` (a `Vector`, plain `number[]` or `Float64Array`) returns a `Vector`.
   * @returns The interpolated or clamped value(s), matching the shape of `x`.
   * @throws {TypeError} If `x` is neither a `number` nor a `VectorLike`.
   */
  eval(x: number): number;
  eval(x: VectorLike): Vector;
  eval(x: number | VectorLike): number | Vector {
    if (typeof x === 'number') {
      return hermiteOne1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
    }
    return hermiteMany1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
  }

  /**
   * Evaluates the first derivative of the spline at `x`. Values outside the
   * data range return `0`, the derivative of the constant clamping regions.
   * At an interior knot the derivative is continuous.
   * @param x The `x`-coordinate(s) at which to evaluate. A single
   * `number` returns a `number`; a `VectorLike` (a `Vector`,
   * plain `number[]` or `Float64Array`) returns a `Vector`.
   * @returns The derivative value(s), matching the shape of `x`.
   * @throws {TypeError} If `x` is neither a `number` nor a `VectorLike`.
   */
  derivative(x: number): number;
  derivative(x: VectorLike): Vector;
  derivative(x: number | VectorLike): number | Vector {
    if (typeof x === 'number') {
      return hermiteOne1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
    }
    return hermiteMany1D(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
  }

  /**
   * Integrates the spline from `a` to `b` analytically.
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
      throw new RangeError('CubicSplineInterpolator1D.integrate: bounds must not be NaN');
    }
    return hermiteIntegrate1D(this.xp, this.fp, this.d, this.leftVal, this.rightVal, a, b);
  }
}

/**
 * A C2 cubic spline interpolant for a vector-valued function
 * (`number -> Vector`).
 *
 * Given the discrete data points `(xp[i], fp.row(i))`, with finite, strictly
 * increasing `xp`, `eval(x)` returns the cubic spline interpolation at `x`,
 * applied independently to each output component (each column of `fp` gets its
 * own spline, all sharing the same `xp` knots and end conditions). Since the
 * slope system depends only on `xp` and `options.bc`, it is factored once and
 * reused for every column. For `x` outside the range of `xp`, the result is
 * clamped to the boundary row of `fp` unless `options.left`/`options.right` are
 * given.
 *
 * For interpolating a scalar-valued function (`number -> number`), see
 * {@link CubicSplineInterpolator1D}, which also describes the method and its
 * special cases in detail.
 *
 * @example
 * ```ts
 * import { Matrix } from 'numerics-js/linalg';
 * import { CubicSplineInterpolatorND } from 'numerics-js/interpolate';
 *
 * const fp = Matrix.from([[0, 0], [1, 1], [8, 0], [27, -1], [64, 0]]);
 * const f = new CubicSplineInterpolatorND([0, 1, 2, 3, 4], fp);
 * console.log(f.eval(2.5).toString());
 * ```
 *
 * Output:
 * ```
 * Vector(15.625, -0.6250000000000001)
 * ```
 */
export class CubicSplineInterpolatorND {
  private readonly xp: Vector;
  private readonly fp: Matrix;
  private readonly leftVal: Vector;
  private readonly rightVal: Vector;
  private readonly d: Matrix; // Slopes at knots, n x dim

  /**
   * @param xp The `x`-coordinates of the data points, shared by every output component: a
   * `Vector`, a plain `number[]` or a `Float64Array`. Must be strictly increasing, finite, and
   * non-empty. Copied, like any `Vector` passed as `options.left`/`options.right`, so later
   * changes to the inputs do not affect the interpolator.
   * @param fp The vector-valued data points: row `i` is the value at `xp[i]`. Must have
   * `fp.rows === xp.length`. Copied, like `xp`.
   * @param options Optional settings; see {@link CubicSplineOptionsND}.
   * @throws {TypeError} If `xp` (or an explicit `options.left`/`options.right`/end derivative) is
   * not a `VectorLike`.
   * @throws {RangeError} If `xp` is empty, if `xp.length !== fp.rows`, if an explicit
   * `options.left`, `options.right`, `options.leftDerivative` or `options.rightDerivative`
   * `VectorLike` does not have length `fp.cols`, if `xp` contains non-finite values, if (when
   * `options.checkSorted` is `true`) `xp` is not strictly increasing, if `options.bc` is not a
   * known boundary type, or if an end derivative is non-finite or given without `bc: 'clamped'`.
   */
  constructor(xp: VectorLike, fp: Matrix, options: CubicSplineOptionsND = {}) {
    const {
      left,
      right,
      checkSorted = true,
      bc = 'not-a-knot',
      leftDerivative,
      rightDerivative
    } = options;
    const caller = 'CubicSplineInterpolatorND';
    checkBoundary(bc, leftDerivative, rightDerivative, caller);
    const leftD = resolveDerivativeND(leftDerivative, fp.cols, 'leftDerivative', caller);
    const rightD = resolveDerivativeND(rightDerivative, fp.cols, 'rightDerivative', caller);

    const p = prepareInterpND(xp, fp, left, right, checkSorted, caller, true);
    if (checkSorted) {
      checkStrictlyIncreasing(p.xp, caller);
    }
    this.xp = p.xp;
    this.fp = p.fp;
    this.leftVal = p.leftVal;
    this.rightVal = p.rightVal;

    const sys = buildSplineSystem(this.xp.data, bc);
    const m = this.fp.cols;
    this.d = new Matrix(this.xp.size, m);
    for (let j = 0; j < m; j++) {
      this.d.setCol(j, computeSplineSlopes(sys, this.fp.col(j).data, leftD[j], rightD[j]));
    }
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
   * Evaluates the cubic spline at `x`.
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
      return hermiteOneND(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, false);
    }
    return hermiteManyND(
      x,
      this.xp,
      this.fp,
      this.d,
      this.leftVal,
      this.rightVal,
      false,
      'CubicSplineInterpolatorND.eval'
    );
  }

  /**
   * Evaluates the first derivative of the spline at `x`, component-wise.
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
      return hermiteOneND(x, this.xp, this.fp, this.d, this.leftVal, this.rightVal, true);
    }
    return hermiteManyND(
      x,
      this.xp,
      this.fp,
      this.d,
      this.leftVal,
      this.rightVal,
      true,
      'CubicSplineInterpolatorND.derivative'
    );
  }

  /**
   * Integrates the spline from `a` to `b`, component-wise, analytically. See
   * `CubicSplineInterpolator1D.integrate` for the algorithm; this applies it
   * independently to each output component.
   * @param a Lower integration bound `a`.
   * @param b Upper integration bound `b`.
   * @returns A `Vector` of length `dim`, the integral of each component over `[a, b]`.
   * @throws {RangeError} If either bound is `NaN`.
   */
  integrate(a: number, b: number): Vector {
    if (Number.isNaN(a) || Number.isNaN(b)) {
      throw new RangeError('CubicSplineInterpolatorND.integrate: bounds must not be NaN');
    }
    return hermiteIntegrateND(this.xp, this.fp, this.d, this.leftVal, this.rightVal, a, b);
  }
}
