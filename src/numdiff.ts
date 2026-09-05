import { Vector } from './array/Vector.js';
import { Matrix } from './array/Matrix.js';
import { ScalarFunction } from './types.js'

/**
 * Calculates scaling factors for a given vector.
 *
 * The scaling factors are computed according to the heuristic procedure
 * implemented in ODRPACK95.
 *
 * @param x Vector to be scaled.
 * @returns A new Vector containing the computed scaling factors.
 *
 * @example
 * ```ts
 * const x = Vector.from([1e-2, 0.0, 1.0, 1e3]);
 * const scaled = scaleVector(x);
 * console.log(scaled.toString());
 * ```
 *
 * Output:
 * ```text
 * Vector(100, 1000, 1, 0.001)
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
 * Calculate the numerical Jacobian of a vector function `f(x)` using the forward 
 * finite-difference scheme.

 * The step size is optimally determined according to the machine precision of the
 * function values. Typically, the Jacobian is accurate to about half the number of
 * reliable digits returned by the function.
 *
 * If the function value at `x` is provided, `N` function evaluations are
 * required to compute the Jacobian, where `N` is the dimension of `x`.
 *
 * @param f Function to be differentiated.
 * @param x Differentiation point.
 * @param options Configuration options for the Jacobian calculation.
 * @returns Jacobian matrix.
 *
 * @example
 * ```ts
 * // Evaluate the numerical jacobian of f(x) = [x0**2 * x1**3] at (2, -2).
 * const f = (x: Vector) => Vector.from([Math.pow(x.get(0), 2) * Math.pow(x.get(1), 3)]);
 * const x = Vector.from([2.0, -2.0]);
 * const jacobian = jacobianForward(f, x);
 * console.log(jacobian.toString());
 * ```
 *
 * Output:
 * ```text
 * Matrix[[-32.00000023841858, 47.99999928474426]]
 * ```
 */
export function jacobianForward(
    f: (x: Vector) => Vector,
    x: Vector,
    options: {
        /** Function values at `x`, if available. */
        fx?: Vector;
        /**
         * Scaling factors for `x`. Ideally, `x[i]*sclx[i]` is close to 1. By
         * default, the factors are set internally based on the magnitudes of `x`.
         */
        sclx?: Vector;
        /**
         * Machine precision of the function values. If undefined, machine precision of 64-bit
         * floating-point type is assumed. If the number of reliable base-10 digits in the
         * results returned by the function is `n`, then `epsf` is approximately `10^(-n)`.
         */
        epsf?: number;
    } = {}
): Matrix {
    const fx = options.fx ?? f(x);
    const sclx = options.sclx ? options.sclx.abs() : scaleVector(x);

    const eps = Number.EPSILON;
    const epsf = options.epsf !== undefined ? Math.max(options.epsf, eps) : eps;
    const h0 = Math.sqrt(epsf);

    const jacobian = new Matrix(fx.size, x.size);
    const xh = x.copy();

    for (let i = 0; i < x.size; i++) {
        let h = h0 * Math.max(Math.abs(x.get(i)), 1.0 / sclx.get(i));
        const xTemp = xh.get(i);

        xh.set(i, xTemp + h);

        // Recompute step size to minimize floating-point error
        h = xh.get(i) - xTemp;

        const fxh = f(xh);
        const col = fxh.sub(fx).div(h);

        jacobian.setCol(i, col);

        // Restore the modified component
        xh.set(i, xTemp);
    }

    return jacobian;
}
/**
 * Calculate the numerical derivative of a scalar function using the centered
 * finite-difference scheme.
 *
 * The step size is optimally determined according to the machine precision of
 * the function values.
 *
 * @param f Function to be differentiated.
 * @param x Differentiation point.
 * @param options Configuration options for the derivative calculation.
 * @returns Tuple containing the derivative and mean function value, `(f'(x), f(x))`.
 *
 * @example
 * ```ts
 * // Evaluate the numerical derivative of f(x) = x**3 at x = 1.
 * const f = (x: number) => x ** 3;
 * const [df, fx] = derivativeCentered(f, 1.0);
 * console.log(df, fx);
 * ```
 *
 * Output:
 * ```text
 * 3.0000000000699174 1.0000000002288205
 * ```
 */
export function derivativeCentered(
    f: ScalarFunction,
    x: number,
    options: {
        /** Machine precision of the function values. */
        epsf?: number;
        /** Finite-difference step. If 0, the theoretical optimum is used. */
        h?: number;
    } = {}
): [number, number] {
    const eps = Number.EPSILON;
    const epsf = options.epsf !== undefined ? Math.max(options.epsf, eps) : eps;
    const h0 = Math.cbrt(3 * epsf);
    let h = options.h ?? 0;

    h = h !== 0 ? Math.max(h, h0) : h0;
    h *= Math.max(1.0, Math.abs(x));

    const xp = x + h;
    const xm = x - h;
    const fp = f(xp);
    const fm = f(xm);
    const df = (fp - fm) / (xp - xm);
    const fx = (fp + fm) / 2.0;

    return [df, fx];
}
