/**
 * Numerical differentiation methods using finite differences.
 *
 * @module numdiff
 */

import { Vector } from './linalg/Vector.js';
import { Matrix } from './linalg/Matrix.js';

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
 * Matrix[[2, 4], [2, 1]]
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
