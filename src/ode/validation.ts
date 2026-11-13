import type { Vector } from '../linalg/Vector.js';

/**
 * Default for the maximum number of steps allowed by the ODE solvers.
 *
 * @internal
 */
export const DEFAULT_MAX_STEPS = 1e5;

/**
 * Validates the integration interval and the initial state, the arguments that all ODE solvers
 * have in common.
 *
 * @internal
 *
 * @param who Name of the calling solver, used as prefix of error messages.
 * @param t0 Initial time.
 * @param tEnd Final time.
 * @param y0 Initial state.
 * @throws {RangeError} If the state has no components or a non-finite component, or if `t0` or
 * `tEnd` is not finite.
 */
export function validateProblem(who: string, t0: number, tEnd: number, y0: Vector): void {
  const dim = y0.size;
  if (dim <= 0) {
    throw new RangeError(`${who}: y0 must have at least one component, got dimension ${dim}.`);
  }
  if (!Number.isFinite(t0) || !Number.isFinite(tEnd)) {
    throw new RangeError(`${who}: t0 (${t0}) and tEnd (${tEnd}) must be finite.`);
  }
  for (let i = 0; i < dim; i++) {
    if (!Number.isFinite(y0.data[i])) {
      throw new RangeError(`${who}: y0 must be finite, got y0[${i}] = ${y0.data[i]}.`);
    }
  }
}

/**
 * Validates the `maxSteps` option.
 *
 * @internal
 *
 * @param who Name of the calling solver, used as prefix of error messages.
 * @param maxSteps Value to validate.
 * @throws {RangeError} If `maxSteps` is not a positive integer.
 */
export function validateMaxSteps(who: string, maxSteps: number): void {
  if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
    throw new RangeError(`${who}: maxSteps (${maxSteps}) must be a positive integer.`);
  }
}
