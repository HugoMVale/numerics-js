import type { Array1D } from '../array/array1d';
import type { Array2D } from '../array/array2d';

export type DerivativeFn = (t: number, y: Array1D, dydt: Array1D) => Array1D;
export type AllocatingDerivativeFn = (t: number, y: Array1D) => Array1D;

export interface ODEIntegrateResult {
    t: Array1D;
    y: Array2D;
}