import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { relativeStepLength } from '../../src/roots/common.js';

describe('relativeStepLength', () => {
  it('is the largest |s_i| / max(|x_i|, 1/sclx_i)', () => {
    const s = Vector.from([2, -3, 0.5]);
    const x = Vector.from([4, -1, 0]);
    const sclx = Vector.from([1, 1, 10]);
    // 2/4 = 0.5; 3/max(1, 1) = 3; 0.5/max(0, 0.1) = 5
    expect(relativeStepLength(s, x, sclx)).toBeCloseTo(5, 14);
  });

  it('is zero for a zero step and propagates NaN', () => {
    const x = Vector.from([1, 2]);
    const sclx = Vector.from([1, 1]);
    expect(relativeStepLength(Vector.from([0, 0]), x, sclx)).toBe(0);
    expect(relativeStepLength(Vector.from([1, NaN]), x, sclx)).toBeNaN();
  });
});
