import { describe, it, expect } from 'vitest';
import { isClose } from '../../src/math.js';
import { Vec3, Vector } from '../../src/linalg.js';

describe('isClose()', () => {
  it('checks absolute tolerance and rejects values outside it', () => {
    expect(isClose(1, 1.001, { atol: 0.001 })).toBe(true);
    expect(isClose(1, 1.002, { atol: 0.001 })).toBe(false);
  });

  it('includes the relative tolerance term', () => {
    expect(isClose(100, 101, { atol: 0, rtol: 0.01 })).toBe(true);
    expect(isClose(100, 101.1, { atol: 0, rtol: 0.01 })).toBe(false);
  });

  it('defaults to atol = 1e-8 and rtol = 1e-5', () => {
    expect(isClose(1, 1 + 5e-9)).toBe(true);
    expect(isClose(1, 1 + 5e-6)).toBe(true);
    expect(isClose(1, 1 + 5e-5)).toBe(false);
    expect(isClose(0, 5e-9)).toBe(true);
    expect(isClose(0, 5e-8)).toBe(false);
    expect(isClose(1, 1 + 5e-6, {})).toBe(true);
    expect(isClose(1, 1 + 5e-6, { atol: 1e-8, rtol: 1e-5 })).toBe(true);
  });

  it('applies the relative tolerance to the reference value b only', () => {
    expect(isClose(1, 1.1, { atol: 0, rtol: 0.1 })).toBe(true); // tolerance 0.1 * |1.1|
    expect(isClose(1.1, 1, { atol: 0, rtol: 0.1 })).toBe(false); // tolerance 0.1 * |1|
  });

  it('treats equal values, including equal infinities, as close', () => {
    expect(isClose(0, 0, { atol: 0, rtol: 0 })).toBe(true);
    expect(isClose(Infinity, Infinity)).toBe(true);
    expect(isClose(-Infinity, -Infinity)).toBe(true);
  });

  it('treats an infinite reference as close to nothing but itself', () => {
    expect(isClose(1, Infinity)).toBe(false);
    expect(isClose(Infinity, -Infinity)).toBe(false);
    expect(isClose(-Infinity, Infinity)).toBe(false);
    expect(isClose(1e308, Infinity, { rtol: 1 })).toBe(false);
    expect(isClose(Infinity, 1)).toBe(false);
  });

  it('never treats NaN as close to anything', () => {
    expect(isClose(NaN, NaN)).toBe(false);
    expect(isClose(NaN, 1)).toBe(false);
    expect(isClose(1, NaN)).toBe(false);
  });

  it('agrees with Vector.isClose and Vec3.allClose on every kind of input', () => {
    const values = [0, -0, 1, 1 + 5e-6, 1 + 1e-3, -2, 1e300, Infinity, -Infinity, NaN];
    const optionSets = [undefined, { atol: 0, rtol: 0 }, { atol: 1e-2 }, { rtol: 0.5 }];
    for (const options of optionSets) {
      for (const a of values) {
        for (const b of values) {
          const scalar = isClose(a, b, options);
          expect(Vector.from([a]).isClose(Vector.from([b]), options)[0]).toBe(scalar);
          expect(Vector.from([a]).allClose(Vector.from([b]), options)).toBe(scalar);
          expect(new Vec3(a, a, a).allClose(new Vec3(b, b, b), options)).toBe(scalar);
        }
      }
    }
  });
});
