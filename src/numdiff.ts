/**
 * Numerical differentiation methods using finite differences.
 *
 * @module numdiff
 */

import { Vector } from './linalg/Vector.js';
import { Matrix } from './linalg/Matrix.js';
import { BandedMatrix } from './linalg/BandedMatrix.js';
import { typeName } from './linalg/checks.js';

const EPS = Number.EPSILON;

/** Options shared by the forward-difference routines. */
export interface ForwardDifferenceOptions {
  /**
   * Scaling factors for `x`. Ideally, `x[i] * sclx[i]` is close to `1`. Only the magnitudes
   * are used, i.e. `abs(sclx)`, and only the first `x.size` components are read, which must
   * be finite and nonzero. By default,
   * the factors are set internally based on the magnitudes of `x`, see {@link scaleVector}.
   */
  sclx?: Vector;
  /**
   * Machine precision of the function values. If the number of reliable base-10 digits in the
   * results returned by the function is `n`, then `epsf` is approximately `10^(-n)`. Must be
   * finite; values below the machine precision of the 64-bit floating-point type are replaced
   * by it.
   * @default Number.EPSILON
   */
  epsf?: number;
}

/** Options for {@link jacobianForward}. */
export interface JacobianForwardOptions extends ForwardDifferenceOptions {
  /**
   * Function values at `x`, if available. If given, `f(x)` is not evaluated, which saves a
   * function evaluation. It must be consistent with `f`.
   */
  fx?: Vector;
}

/** Options for {@link jacobianForwardBanded}. */
export interface JacobianForwardBandedOptions extends JacobianForwardOptions {
  /**
   * Number of subdiagonals of the Jacobian: the function component `k` must not depend on
   * `x[j]` for `j < k - kl`. A non-negative integer; a value of `x.size` or more is clamped to
   * `x.size - 1`.
   */
  kl: number;
  /**
   * Number of superdiagonals of the Jacobian: the function component `k` must not depend on
   * `x[j]` for `j > k + ku`. A non-negative integer; a value of `x.size` or more is clamped to
   * `x.size - 1`.
   */
  ku: number;
}

/** Options for {@link gradientForward}. */
export interface GradientForwardOptions extends ForwardDifferenceOptions {
  /**
   * Function value at `x`, if available. If given, `f(x)` is not evaluated, which saves a
   * function evaluation. It must be consistent with `f`.
   */
  fx?: number;
}

/** Options for {@link hessianForward}. */
export interface HessianForwardOptions extends ForwardDifferenceOptions {
  /**
   * Function value at `x`, if available. If given, `f(x)` is not evaluated, which saves a
   * function evaluation. It must be consistent with `f`.
   */
  fx?: number;
}

/** Options for {@link derivativeCentered}. */
export interface DerivativeCenteredOptions {
  /**
   * Machine precision of the function values. Must be finite; values below the machine
   * precision of the 64-bit floating-point type are replaced by it. Ignored if `h` is given.
   * @default Number.EPSILON
   */
  epsf?: number;
  /**
   * Relative finite-difference step. The actual step is `h * max(1, |x|)`, so `h` is a
   * relative step for `|x| > 1` and an absolute one otherwise. Must be finite and nonzero; its sign is
   * irrelevant. If omitted, the theoretical optimum `(3*epsf)^(1/3)` is used.
   */
  h?: number;
}

/**
 * Returns the effective machine precision of the function values, i.e. `epsf` bounded
 * below by the machine precision of the 64-bit floating-point type.
 *
 * @param who Name of the calling routine, used as prefix of error messages.
 * @param epsf User-supplied machine precision of the function values, if any.
 * @throws {RangeError} If `epsf` is not finite.
 */
function effectiveEpsf(who: string, epsf: number | undefined): number {
  if (epsf === undefined) {
    return EPS;
  }
  if (!Number.isFinite(epsf)) {
    throw new RangeError(`${who}: epsf must be finite, got ${epsf}.`);
  }
  return Math.max(epsf, EPS);
}

/**
 * Returns the magnitudes of the user-supplied scaling factors, or the default scaling
 * factors of `x` if none are supplied.
 *
 * @param who Name of the calling routine, used as prefix of error messages.
 * @param x Differentiation point.
 * @param sclx User-supplied scaling factors, if any.
 * @throws {RangeError} If `sclx` has fewer components than `x`, or if any of its first
 * `x.size` components is zero or not finite.
 */
function resolveScaling(who: string, x: Vector, sclx: Vector | undefined): Vector {
  if (sclx === undefined) {
    return scaleVector(x);
  }
  if (sclx.size < x.size) {
    throw new RangeError(`${who}: sclx must have ${x.size} components, got ${sclx.size}.`);
  }
  const magnitudes = sclx.abs();
  for (let i = 0; i < x.size; i++) {
    const s = magnitudes.get(i);
    if (!(s > 0 && s < Infinity)) {
      throw new RangeError(
        `${who}: sclx must be finite and nonzero, got ${sclx.get(i)} at index ${i}.`
      );
    }
  }
  return magnitudes;
}

/**
 * Computes scaling factors for a vector.
 *
 * The scaling factors are computed according to the heuristic procedure implemented in
 * ODRPACK95. The goal is to obtain factors `sclx` such that the magnitude of `sclx[i] * x[i]`
 * is close to `1`:
 *
 * - If all components of `x` are zero, all factors are `1`.
 * - If the largest and smallest nonzero magnitudes of `x` differ by at least one order of
 *   magnitude, then `sclx[i] = 1/|x[i]|`.
 * - Otherwise, all nonzero components share the factor `1/max|x|`.
 * - A zero component gets the factor `10/min|x|`, where the minimum is taken over the nonzero
 *   components.
 *
 * @param x Vector `x` to be scaled. All components must be finite.
 * @returns A new Vector containing the computed scaling factors.
 * @throws {RangeError} If `x` contains a non-finite component.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { scaleVector } from 'numerics-js/numdiff';
 *
 * const x = Vector.from([1e-2, 0.0, 1.0, 1e3]);
 * const sclx = scaleVector(x);
 * console.log('sclx  :', sclx.toString());
 * console.log('sclx*x:', sclx.mult(x).toString());
 * ```
 *
 * Output:
 * ```text
 * sclx  : Vector(100, 1000, 1, 0.001)
 * sclx*x: Vector(1, 0, 1, 1)
 * ```
 */
export function scaleVector(x: Vector): Vector {
  const sclx = Vector.ones(x.size);

  let xmax = 0;
  let xmin = Infinity;
  let hasNonZero = false;

  // Single pass to find both xmax (max absolute value) and xmin (min non-zero absolute value)
  for (let i = 0; i < x.size; i++) {
    const val = Math.abs(x.get(i));

    if (!Number.isFinite(val)) {
      throw new RangeError(`scaleVector: x must be finite, got ${x.get(i)} at index ${i}.`);
    }

    if (val > xmax) {
      xmax = val;
    }

    if (val > 0) {
      hasNonZero = true;
      if (val < xmin) {
        xmin = val;
      }
    }
  }

  // If there are no non-zero elements, return the vector of ones
  if (!hasNonZero) {
    return sclx;
  }

  const logRatio = Math.log10(xmax / xmin);

  // Second pass to assign the appropriate scaling factors
  for (let i = 0; i < x.size; i++) {
    const val = x.get(i);

    if (val === 0.0) {
      sclx.set(i, 1e1 / xmin);
    } else {
      if (logRatio >= 1.0) {
        sclx.set(i, 1.0 / Math.abs(val));
      } else {
        sclx.set(i, 1.0 / xmax);
      }
    }
  }

  return sclx;
}

/**
 * Computes the Jacobian matrix of a vector function `f(x)` using the forward
 * finite-difference scheme.
 *
 * For a function `f: ℝⁿ → ℝᵐ`, the result is the `m × n` matrix with entries
 * `J[k][i] = ∂f_k/∂x_i`. The step size for the `i`-th component is
 * `sqrt(epsf) * max(|x_i|, 1/|sclx_i|)`, which is optimal with respect to the machine
 * precision of the function values. Typically, the Jacobian is accurate to about half the
 * number of reliable digits returned by the function.
 *
 * If the function value at `x` is provided, `N` function evaluations are
 * required to compute the Jacobian, where `N` is the dimension of `x`.
 *
 * The vector passed to `f` is either `x` itself or an internal work buffer that is reused
 * between calls. Therefore, `f` must neither modify it nor keep a reference to it (copy it
 * if needed). The argument `x` is never modified.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, p. 314.
 *
 * @param f Function `f` to be differentiated.
 * @param x Differentiation point `x`.
 * @param options Optional settings; see {@link JacobianForwardOptions}.
 * @returns Jacobian matrix, with as many rows as `f` has components and as many columns as
 * `x` has components.
 * @throws {RangeError} If provided `fx` has a different size from perturbed
 * function values, if `f` returns vectors of inconsistent sizes, if `sclx` has fewer
 * components than `x` or contains a zero or non-finite value, or if `epsf` is not finite.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { jacobianForward } from 'numerics-js/numdiff';
 *
 * // Evaluate the numerical Jacobian of `f(x, y) = [x**2 + y**2, x * y]` at `(1, 2)`.
 * const f = (x: Vector) => {
 *     const x0 = x.get(0);
 *     const x1 = x.get(1);
 *     return Vector.from([x0 ** 2 + x1 ** 2, x0 * x1]);
 * };
 * const x = Vector.from([1.0, 2.0]);
 * const jacobian = jacobianForward(f, x);
 * console.log(jacobian.toString());
 * ```
 *
 * Output:
 * ```text
 * Matrix[[2.0000000298023224, 4.000000029802322], [2, 1]]
 * ```
 */
export function jacobianForward(
  f: (x: Vector) => Vector,
  x: Vector,
  options: JacobianForwardOptions = {}
): Matrix {
  const sclx = resolveScaling('jacobianForward', x, options.sclx);
  const h0 = Math.sqrt(effectiveEpsf('jacobianForward', options.epsf));
  const fx = options.fx ?? f(x);

  const jacobian = new Matrix(fx.size, x.size);
  const xh = x.copy();

  for (let i = 0; i < x.size; i++) {
    const xi = xh.get(i);
    let h = h0 * Math.max(Math.abs(xi), 1.0 / sclx.get(i));

    xh.set(i, xi + h);

    // Recompute step size to minimize floating-point error
    h = xh.get(i) - xi;

    const fxh = f(xh);
    if (fxh.size !== fx.size) {
      throw new RangeError(
        `jacobianForward: f returned ${fxh.size} components at a perturbed point, expected ${fx.size}.`
      );
    }
    for (let k = 0; k < fx.size; k++) {
      jacobian.set(k, i, (fxh.get(k) - fx.get(k)) / h);
    }

    // Restore the modified component
    xh.set(i, xi);
  }

  return jacobian;
}

/**
 * Computes the Jacobian matrix of a vector function `f(x)` with a banded sparsity pattern
 * using the forward finite-difference scheme, as a {@link BandedMatrix}.
 *
 * The Jacobian of `f: ℝⁿ → ℝⁿ` is assumed to have `kl` subdiagonals and `ku` superdiagonals,
 * that is, `∂f_k/∂x_i = 0` unless `-ku <= k - i <= kl`, as in a method-of-lines discretization
 * of a PDE. Columns that are at least `kl + ku + 1` apart do not touch the same rows, so they
 * are perturbed together (Curtis, Powell and Reid, 1974): `min(kl + ku + 1, n)` evaluations of
 * `f` give the whole Jacobian, instead of the `n` of {@link jacobianForward}. The step of
 * each column is the one of {@link jacobianForward}, `sqrt(epsf) * max(|x_i|, 1/|sclx_i|)`,
 * and every entry inside the band is computed from the same expression, so that for a function
 * that honors the band the result equals the band of `jacobianForward`.
 *
 * **The band is a promise.** It is not verified: if a component of `f` depends on a variable
 * outside its band, the perturbations of the columns of a group overlap and the entries of the
 * result are silently wrong, not merely missing. Periodic boundary conditions, for instance,
 * add entries in the corners of the Jacobian and are not banded.
 *
 * If the function value at `x` is provided, `min(kl + 1 + ku, n)` function evaluations are
 * required, otherwise one more.
 *
 * The vector passed to `f` is either `x` itself or an internal work buffer that is reused
 * between calls. Therefore, `f` must neither modify it nor keep a reference to it (copy it
 * if needed). The vector returned by `f` is read before the next call, so `f` may return the
 * same buffer every time, but not the vector given as `fx`. The argument `x` is never modified.
 *
 * **References**
 *
 * - A.R. Curtis, M.J.D. Powell, J.K. Reid, "On the estimation of sparse Jacobian matrices",
 *   J. Inst. Maths Applics 13 (1974), 117-119.
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, p. 314.
 *
 * @param f Function `f` to be differentiated. It must return as many components as `x` has.
 * @param x Differentiation point `x`.
 * @param options Bandwidth `kl` and `ku` and optional settings; see
 * {@link JacobianForwardBandedOptions}.
 * @returns The `n x n` Jacobian as a new `BandedMatrix` with the given bandwidths (clamped to
 * `n - 1`), where `n` is the number of components of `x`.
 * @throws {TypeError} If `kl` or `ku` is not a number.
 * @throws {RangeError} If `kl` or `ku` is not a non-negative integer, if `f` or the provided
 * `fx` does not have as many components as `x`, if `sclx` has fewer components than `x` or
 * contains a zero or non-finite value, or if `epsf` is not finite.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { jacobianForwardBanded } from 'numerics-js/numdiff';
 *
 * // Nonlinear diffusion on 5 points: f_i = x_{i-1} - 2 x_i + x_{i+1} + x_i^2 (zero outside).
 * // The Jacobian is tridiagonal, so 3 evaluations of f (plus the one at x) are enough.
 * let calls = 0;
 * const f = (x: Vector) => {
 *     calls++;
 *     const n = x.size;
 *     const out = new Vector(n);
 *     for (let i = 0; i < n; i++) {
 *         const left = i > 0 ? x.get(i - 1) : 0;
 *         const right = i < n - 1 ? x.get(i + 1) : 0;
 *         out.set(i, left - 2 * x.get(i) + right + x.get(i) ** 2);
 *     }
 *     return out;
 * };
 * const jacobian = jacobianForwardBanded(f, Vector.from([1, 2, 3, 4, 5]), { kl: 1, ku: 1 });
 * for (const row of jacobian.toMatrix().toArray()) {
 *     console.log(row.map((v) => v.toFixed(3)).join(' '));
 * }
 * console.log(`evaluations: ${calls}`);
 * ```
 *
 * Output:
 * ```text
 * 0.000 1.000 0.000 0.000 0.000
 * 1.000 2.000 1.000 0.000 0.000
 * 0.000 1.000 4.000 1.000 0.000
 * 0.000 0.000 1.000 6.000 1.000
 * 0.000 0.000 0.000 1.000 8.000
 * evaluations: 4
 * ```
 */
export function jacobianForwardBanded(
  f: (x: Vector) => Vector,
  x: Vector,
  options: JacobianForwardBandedOptions
): BandedMatrix {
  if (typeof options !== 'object' || options === null) {
    throw new TypeError(
      `jacobianForwardBanded: options with kl and ku are required, got ${typeName(options)}.`
    );
  }
  const { kl, ku } = options;
  checkBandwidth(kl, 'kl');
  checkBandwidth(ku, 'ku');
  const n = x.size;
  // `BandedMatrix` clamps both bandwidths to `n - 1`.
  const jacobian = new BandedMatrix(n, kl, ku);
  const sclx = resolveScaling('jacobianForwardBanded', x, options.sclx);
  const h0 = Math.sqrt(effectiveEpsf('jacobianForwardBanded', options.epsf));
  // A copy when computed here: `f` may return the same buffer at every call.
  const fx = options.fx ?? f(x).copy();
  if (fx.size !== n) {
    throw new RangeError(
      `jacobianForwardBanded: f must return as many components as x (${n}), got ${fx.size}.`
    );
  }

  const lower = jacobian.kl;
  const upper = jacobian.ku;
  const ldab = jacobian.ldab;
  const band = jacobian.data;
  const fxd = fx.data;
  // Columns i, i + groups, i + 2 * groups, ... are perturbed together.
  const groups = Math.min(lower + upper + 1, n);
  const xh = x.copy();
  const xhd = xh.data;
  const xd = x.data;
  const steps = new Float64Array(n);

  for (let g = 0; g < groups; g++) {
    for (let i = g; i < n; i += groups) {
      const xi = xd[i];
      let h = h0 * Math.max(Math.abs(xi), 1.0 / sclx.get(i));
      xhd[i] = xi + h;
      // Recompute step size to minimize floating-point error
      h = xhd[i] - xi;
      steps[i] = h;
    }

    const fxh = f(xh);
    if (fxh.size !== n) {
      throw new RangeError(
        `jacobianForwardBanded: f returned ${fxh.size} components at a perturbed point, expected ${n}.`
      );
    }
    const fxhd = fxh.data;

    for (let i = g; i < n; i += groups) {
      const h = steps[i];
      const k0 = i > upper ? i - upper : 0;
      const k1 = i + lower < n ? i + lower : n - 1;
      // Element (k, i) of the band is at `upper + k - i + i * ldab`.
      const base = upper - i + i * ldab;
      for (let k = k0; k <= k1; k++) {
        band[base + k] = (fxhd[k] - fxd[k]) / h;
      }
      // Restore the modified component
      xhd[i] = xd[i];
    }
  }

  return jacobian;
}

/** Checks that a bandwidth option of `jacobianForwardBanded` is a non-negative integer. */
function checkBandwidth(value: unknown, name: string): void {
  if (typeof value !== 'number') {
    throw new TypeError(`jacobianForwardBanded: ${name} must be a number, got ${typeName(value)}.`);
  }
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(
      `jacobianForwardBanded: ${name} must be a non-negative integer, got ${value}.`
    );
  }
}

/**
 * Computes the gradient of a scalar function `f(x)` using the forward
 * finite-difference scheme.
 *
 * The step size for the `i`-th component is `sqrt(epsf) * max(|x_i|, 1/|sclx_i|)`, which is
 * optimal with respect to the machine precision of the function values. Typically, the
 * gradient is accurate to about half the number of reliable digits returned by the function.
 *
 * If the function value at `x` is provided, `N` function evaluations are
 * required to compute the gradient, where `N` is the dimension of `x`.
 *
 * The vector passed to `f` is either `x` itself or an internal work buffer that is reused
 * between calls. Therefore, `f` must neither modify it nor keep a reference to it (copy it
 * if needed). The argument `x` is never modified.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, p. 322.
 *
 * @param f Function `f` to be differentiated.
 * @param x Differentiation point `x`.
 * @param options Optional settings; see {@link GradientForwardOptions}.
 * @returns Gradient vector.
 * @throws {RangeError} If `sclx` has fewer components than `x` or contains a zero or
 * non-finite value, or if `epsf` is not finite.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { gradientForward } from 'numerics-js/numdiff';
 *
 * // Evaluate the numerical gradient of `f(x, y) = x**2 * y**3` at `(2, -2)`.
 * const f = (x: Vector) => x.get(0) ** 2 * x.get(1) ** 3;
 * const gradient = gradientForward(f, Vector.from([2.0, -2.0]));
 * console.log(gradient.toString());
 * ```
 *
 * Output:
 * ```text
 * Vector(-32.00000023841858, 47.99999928474426)
 * ```
 */
export function gradientForward(
  f: (x: Vector) => number,
  x: Vector,
  options: GradientForwardOptions = {}
): Vector {
  const sclx = resolveScaling('gradientForward', x, options.sclx);
  const h0 = Math.sqrt(effectiveEpsf('gradientForward', options.epsf));
  const fx = options.fx ?? f(x);

  const gradient = new Vector(x.size);
  const xh = x.copy();

  for (let i = 0; i < x.size; i++) {
    const xi = xh.get(i);
    let h = h0 * Math.max(Math.abs(xi), 1.0 / sclx.get(i));

    xh.set(i, xi + h);

    // Recompute step size to minimize floating-point error
    h = xh.get(i) - xi;

    gradient.set(i, (f(xh) - fx) / h);

    // Restore the modified component
    xh.set(i, xi);
  }

  return gradient;
}

/**
 * Computes the Hessian of a scalar function `f(x)` using the forward
 * finite-difference scheme.
 *
 * The step size for the `i`-th component is `cbrt(epsf) * max(|x_i|, 1/|sclx_i|)`, which is
 * optimal with respect to the machine precision of the function values. Typically, the
 * Hessian is accurate to about a third of the number of reliable digits returned by the
 * function. The returned matrix is exactly symmetric.
 *
 * If the function value at `x` is provided, `N*(N+3)/2` function evaluations are
 * required to compute the Hessian, where `N` is the dimension of `x`.
 *
 * The vector passed to `f` is either `x` itself or an internal work buffer that is reused
 * between calls. Therefore, `f` must neither modify it nor keep a reference to it (copy it
 * if needed). The argument `x` is never modified.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, p. 321.
 *
 * @param f Function `f` to be differentiated.
 * @param x Differentiation point `x`.
 * @param options Optional settings; see {@link HessianForwardOptions}.
 * @returns Hessian matrix.
 * @throws {RangeError} If `sclx` has fewer components than `x` or contains a zero or
 * non-finite value, or if `epsf` is not finite.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { hessianForward } from 'numerics-js/numdiff';
 *
 * // Evaluate the numerical Hessian of `f(x, y) = x**2 * y**3` at `(2, -2)`.
 * const f = (x: Vector) => x.get(0) ** 2 * x.get(1) ** 3;
 * const hessian = hessianForward(f, Vector.from([2.0, -2.0]));
 * console.log(hessian.toString());
 * ```
 *
 * Output:
 * ```text
 * Matrix[[-16.00001241634706, 47.99984347361866], [47.99984347361866, -47.999722363649575]]
 * ```
 */
export function hessianForward(
  f: (x: Vector) => number,
  x: Vector,
  options: HessianForwardOptions = {}
): Matrix {
  const sclx = resolveScaling('hessianForward', x, options.sclx);
  const h0 = Math.cbrt(effectiveEpsf('hessianForward', options.epsf));
  const fx = options.fx ?? f(x);

  const n = x.size;
  const hessian = new Matrix(n, n);
  const fh = new Float64Array(n);
  const h = new Float64Array(n);
  const xh = x.copy();

  // Function values at the single-perturbed points x + h_i*e_i
  for (let i = 0; i < n; i++) {
    const xi = xh.get(i);
    xh.set(i, xi + h0 * Math.max(Math.abs(xi), 1.0 / sclx.get(i)));

    // Recompute step size to minimize floating-point error
    h[i] = xh.get(i) - xi;

    fh[i] = f(xh);
    xh.set(i, xi);
  }

  for (let i = 0; i < n; i++) {
    const xi = xh.get(i);

    // Diagonal: uses x + 2*h_i*e_i
    xh.set(i, xi + 2 * h[i]);
    hessian.set(i, i, (fx - fh[i] + (f(xh) - fh[i])) / (h[i] * h[i]));

    // Off-diagonal: x_i stays perturbed by h_i while x_j is perturbed by h_j
    xh.set(i, xi + h[i]);
    for (let j = i + 1; j < n; j++) {
      const xj = xh.get(j);
      xh.set(j, xj + h[j]);
      const hij = (fx - fh[i] + (f(xh) - fh[j])) / (h[i] * h[j]);
      hessian.set(i, j, hij);
      hessian.set(j, i, hij);
      xh.set(j, xj);
    }
    xh.set(i, xi);
  }

  return hessian;
}

/**
 * Computes the derivative of a scalar function `f(x)` using the centered
 * finite-difference scheme.
 *
 * The derivative is `(f(x + h) - f(x - h)) / (2h)`, where the actual step is
 * `h * max(1, |x|)` and `h` defaults to `(3*epsf)^(1/3)`, which is optimal with respect to the
 * machine precision of the function values. The method requires two function evaluations.
 *
 * Since `f(x)` itself is not evaluated, the mean of the two perturbed values,
 * `(f(x + h) + f(x - h)) / 2`, is returned as an estimate of it. It differs from `f(x)` by a
 * term of order `h²`.
 *
 * @param f Function `f` to be differentiated.
 * @param x Differentiation point `x`.
 * @param options Optional settings; see {@link DerivativeCenteredOptions}.
 * @returns Tuple containing the derivative and the mean function value, `(f'(x), f(x))`.
 * @throws {RangeError} If `options.h` is zero or not finite, or if `options.epsf` is not finite.
 *
 * @example
 * ```ts
 * import { derivativeCentered } from 'numerics-js/numdiff';
 *
 * // Evaluate the numerical derivative of `f(x) = x**3` at `x = 1`.
 * const f = (x: number) => x ** 3;
 * const [df, fx] = derivativeCentered(f, 1.0);
 * console.log('df:', df);
 * console.log('fx:', fx);
 * ```
 *
 * Output:
 * ```text
 * df: 3.0000000000699174
 * fx: 1.0000000002288205
 * ```
 */
export function derivativeCentered(
  f: (x: number) => number,
  x: number,
  options: DerivativeCenteredOptions = {}
): [number, number] {
  let h: number;

  if (options.h === undefined) {
    h = Math.cbrt(3 * effectiveEpsf('derivativeCentered', options.epsf));
  } else {
    if (!Number.isFinite(options.h) || options.h === 0) {
      throw new RangeError(`derivativeCentered: h must be finite and nonzero, got ${options.h}.`);
    }
    h = options.h;
  }
  h *= Math.max(1.0, Math.abs(x));

  const xp = x + h;
  const xm = x - h;
  const fp = f(xp);
  const fm = f(xm);
  const df = (fp - fm) / (xp - xm);
  const fx = (fp + fm) / 2.0;

  return [df, fx];
}
