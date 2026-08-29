import type { Array1D } from '../array/array1d';
import type { AllocatingDerivativeFunction, DerivativeFunction } from './types';

/**
 * Adapts a convenience-style derivative function that allocates and returns
 * a new `Array1D` - `(t, y) => Array1D` - into the writes-into-`dydt` style that
 * ODE steppers/integrators in this package use.
 *
 * The wrapped function allocates on every derivative evaluation. Prefer a
 * {@link DerivativeFunction} that writes directly into its output buffer in
 * performance-sensitive integrations.
 *
 * @param f Derivative function `dy/dt = f(t, y)` that returns a newly allocated
 * `Array1D`. Its result must have the same dimension as `y`.
 * @returns A {@link DerivativeFunction} that copies `f`'s result into the
 * supplied `dydt` buffer and returns that buffer.
 *
 * @example
 * ```ts
 * // Exponential decay: dy/dt = -y
 * const f = wrapAllocatingDerivative((t, y) => y.copy().multSelf(-1));
 * const out = new Array1D(1);
 * rungeKuttaStep('rk4', f, 0, new Array1D([1]), 0.25, out);
 * console.log(out);
 * ```
 *
 * Output:
 * ```text
 * Array1D [ 0.7788085937500001 ]
 * ```
 */
export function wrapAllocatingDerivative(
    f: AllocatingDerivativeFunction
): DerivativeFunction {
    return (t: number, y: Array1D, dydt: Array1D): Array1D =>
        dydt.set(f(t, y).data);
}