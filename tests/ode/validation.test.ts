import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { DEFAULT_MAX_STEPS, validateMaxSteps, validateProblem } from '../../src/ode/validation.js';

describe('validateProblem', () => {
  it('accepts a finite problem, in either direction and for an empty interval', () => {
    expect(() => validateProblem('solver', 0, 1, new Vector([1, 2]))).not.toThrow();
    expect(() => validateProblem('solver', 1, 0, new Vector([1]))).not.toThrow();
    expect(() => validateProblem('solver', 2, 2, new Vector([1]))).not.toThrow();
  });

  it.each<[string, number, number, number[], RegExp]>([
    [
      'an empty state',
      0,
      1,
      [],
      /^solver: y0 must have at least one component, got dimension 0\.$/
    ],
    ['NaN t0', NaN, 1, [1], /^solver: t0 \(NaN\) and tEnd \(1\) must be finite\.$/],
    ['infinite tEnd', 0, Infinity, [1], /tEnd \(Infinity\) must be finite/],
    ['NaN in y0', 0, 1, [1, NaN], /^solver: y0 must be finite, got y0\[1\] = NaN\.$/],
    ['infinite y0', 0, 1, [-Infinity], /y0\[0\] = -Infinity/]
  ])('rejects %s with a RangeError', (_label, t0, tEnd, y0, message) => {
    const call = () => validateProblem('solver', t0, tEnd, new Vector(y0));
    expect(call).toThrow(RangeError);
    expect(call).toThrow(message);
  });
});

describe('validateMaxSteps', () => {
  it('accepts positive integers, including the default', () => {
    expect(() => validateMaxSteps('solver', 1)).not.toThrow();
    expect(() => validateMaxSteps('solver', DEFAULT_MAX_STEPS)).not.toThrow();
  });

  it.each([0, -1, 1.5, NaN, Infinity])('rejects %s', (value) => {
    expect(() => validateMaxSteps('solver', value)).toThrow(RangeError);
    expect(() => validateMaxSteps('solver', value)).toThrow(
      `solver: maxSteps (${value}) must be a positive integer.`
    );
  });
});
