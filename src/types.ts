import type { Array1D } from './array/array1d.js';

/** Scalar function of a single real variable, e.g. for root finding or integration. */
export type ScalarFunction = (x: number) => number;

/** Scalar-valued function of a real vector, e.g. for multivariate optimization. */
export type VectorFunction = (x: Array1D) => number;
