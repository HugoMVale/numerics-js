export { Vec3 } from './array/vec3';
export { Array1D } from './array/array1d';
export { Array2D } from './array/array2d';
export { bessel } from './bessel';
export { bisection, secant } from './roots';
export { rk4Step, rk4Integrate } from './ode/rk4';
export { dp54Step, dp54Integrate } from './ode/dp54';
export { createVelocityVerlet } from './ode/verlet';
export { wrapAllocatingDerivative } from './ode/adapters';
export type { DerivativeFn, AllocatingDerivativeFn, ODEIntegrateResult } from './ode/types';

export const version = '0.1.0';
