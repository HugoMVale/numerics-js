import type { Array1D } from '../array/array1d';
import type { AllocatingDerivativeFn, DerivativeFn } from './types';

/**
 * Adapts a convenience-style derivative function that allocates and returns
 * a new `Array1D` - `(t, y) => Array1D` - into the writes-into-`dydt` style that
 * ODE steppers/integrators in this package use.
 */
export function wrapAllocatingDerivative(
    f: AllocatingDerivativeFn
): DerivativeFn {
    return (t: number, y: Array1D, dydt: Array1D): Array1D =>
        dydt.set(f(t, y).data);
}