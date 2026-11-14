/**
 * Argument validation shared by the root-finding routines.
 *
 * All helpers throw a `RangeError` whose message starts with the name of the calling routine.
 */

/**
 * Validates a `maxIter` option.
 *
 * @internal
 *
 * @param who Name of the calling routine, used as prefix of error messages.
 * @param maxIter Value to validate.
 * @throws {RangeError} If `maxIter` is not a finite, strictly positive integer.
 */
export function validateMaxIter(who: string, maxIter: number): void {
  if (!(Number.isInteger(maxIter) && maxIter >= 1)) {
    throw new RangeError(
      `${who}: maxIter must be a finite strictly positive integer, got ${maxIter}`
    );
  }
}

/**
 * Validates a tolerance-like option: it must be a finite number, and strictly positive if
 * `strict` is `true`, non-negative otherwise.
 *
 * @internal
 *
 * @param who Name of the calling routine, used as prefix of error messages.
 * @param name Name of the option.
 * @param value Value to validate. `undefined` (option not given) is accepted.
 * @param strict Whether zero is rejected.
 * @throws {RangeError} If `value` is defined and violates the constraint.
 */
export function validateTolerance(
  who: string,
  name: string,
  value: number | undefined,
  strict = false
): void {
  if (value === undefined) return;
  if (!(Number.isFinite(value) && (strict ? value > 0 : value >= 0))) {
    throw new RangeError(
      `${who}: ${name} must be a finite ${strict ? 'strictly positive' : 'non-negative'} number, got ${value}`
    );
  }
}

/**
 * Validates that two scalars (endpoints, initial guesses) are finite and distinct.
 *
 * @internal
 *
 * @param who Name of the calling routine, used as prefix of error messages.
 * @param nameA Name of the first scalar.
 * @param a First scalar.
 * @param nameB Name of the second scalar.
 * @param b Second scalar.
 * @throws {RangeError} If either scalar is not finite, or if they are equal.
 */
export function validateDistinctFinite(
  who: string,
  nameA: string,
  a: number,
  nameB: string,
  b: number
): void {
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    throw new RangeError(`${who}: ${nameA} and ${nameB} must be finite, got ${a} and ${b}`);
  }
  if (a === b) {
    throw new RangeError(`${who}: ${nameA} and ${nameB} must be different`);
  }
}
