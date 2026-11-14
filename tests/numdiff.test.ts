import { describe, it, expect } from 'vitest';
import { Vector } from '../src/linalg/Vector.js';
import { Matrix } from '../src/linalg/Matrix.js';
import {
  scaleVector,
  jacobianForward,
  gradientForward,
  hessianForward,
  derivativeCentered
} from '../src/numdiff.js';

describe('scaleX', () => {
  it('generates an array of ones for a zero vector', () => {
    const sclx = scaleVector(Vector.from([0.0, 0.0]));
    expect(sclx.allClose(Vector.from([1.0, 1.0]))).toBe(true);
  });

  it('scales accurately with zero and negative large values', () => {
    const sclx = scaleVector(Vector.from([0.0, -1e2]));
    expect(sclx.allClose(Vector.from([1e-1, 1e-2]))).toBe(true);
  });

  it('scales accurately with mixed zero, small positive, and negative large values', () => {
    const sclx = scaleVector(Vector.from([0.0, 0.1, -1e2]));
    expect(sclx.allClose(Vector.from([1e2, 1e1, 1e-2]))).toBe(true);
  });

  it('scales accurately with zero and standard positive values', () => {
    const sclx = scaleVector(Vector.from([0.0, 1.0, 5.0]));
    expect(sclx.allClose(Vector.from([10.0, 0.2, 0.2]))).toBe(true);
  });
});

describe('jacobianForward', () => {
  // Target function: fnc3
  const fnc3 = (x: Vector): Vector => {
    const x1 = x.get(0);
    const x2 = x.get(1);
    const f1 = 0.5 * Math.cos(x1) + 0.1 * x2 + 0.5;
    const f2 = Math.sin(x2) - 0.2 * x1 + 1.2;

    return Vector.from([f1, f2]);
  };

  it('calculates the numerical jacobian correctly within relative tolerance', () => {
    const x = Vector.from([3.0, -2.0]);
    const jacobian = jacobianForward(fnc3, x);

    const expectedMatrix = Matrix.from([
      [-0.07055999, 0.1],
      [-0.2, -0.41614682]
    ]);

    // Utilizing Matrix.allClose with a relative tolerance (rtol) of 1e-6
    expect(jacobian.allClose(expectedMatrix, 1e-6)).toBe(true);
  });
});

describe('gradientForward', () => {
  const f = (x: Vector): number => x.get(0) ** 2 * x.get(1) ** 3;
  const x = Vector.from([2.0, -2.0]);

  it('matches the analytical gradient and the original Python values', () => {
    const g = gradientForward(f, x);
    expect(g.allClose(Vector.from([-32, 48]), 1e-6, 1e-6)).toBe(true);
    // Values of the original Python implementation (polykin.math.gradient_forward).
    expect(g.allClose(Vector.from([-32.00000024, 47.99999928]), 1e-9, 1e-9)).toBe(true);
  });

  it('needs N evaluations if fx is given and N+1 otherwise', () => {
    let n = 0;
    const counted = (v: Vector) => (n++, f(v));
    gradientForward(counted, x, { fx: f(x) });
    expect(n).toBe(2);
    n = 0;
    gradientForward(counted, x);
    expect(n).toBe(3);
  });

  it('does not modify x', () => {
    const x0 = Vector.from([2.0, -2.0]);
    gradientForward(f, x0);
    expect(x0.toArray()).toEqual([2, -2]);
  });

  it('respects sclx, including at x = 0, and uses abs(sclx)', () => {
    const seen: number[] = [];
    gradientForward((v) => (seen.push(v.get(0)), v.get(0)), Vector.from([0.0]), {
      fx: 0,
      sclx: Vector.from([-1e2])
    });
    // h = sqrt(eps) * max(|x|, 1/|sclx|) = sqrt(eps)/100
    expect(seen[0]).toBeCloseTo(Math.sqrt(Number.EPSILON) / 100, 20);
  });

  it('increases the step with a larger epsf, and ignores epsf below machine precision', () => {
    const steps = (epsf: number) => {
      let xi = 0;
      gradientForward((v) => ((xi = v.get(0)), v.get(0)), Vector.from([1.0]), { fx: 1, epsf });
      return xi - 1;
    };
    expect(steps(1e-8)).toBeGreaterThan(steps(Number.EPSILON));
    expect(steps(1e-20)).toBe(steps(Number.EPSILON));
  });

  it('throws if sclx is too short', () => {
    expect(() => gradientForward(f, x, { sclx: Vector.from([1]) })).toThrow(RangeError);
  });
});

describe('hessianForward', () => {
  const f = (x: Vector): number => x.get(0) ** 2 * x.get(1) ** 3;
  const x = Vector.from([2.0, -2.0]);

  it('matches the analytical Hessian and the original Python values', () => {
    const H = hessianForward(f, x);
    expect(
      H.allClose(
        Matrix.from([
          [-16, 48],
          [48, -48]
        ]),
        1e-3,
        1e-3
      )
    ).toBe(true);
    // Values of the original Python implementation (polykin.math.hessian_forward).
    expect(
      H.allClose(
        Matrix.from([
          [-16.00001242, 47.99984347],
          [47.99984347, -47.99972236]
        ]),
        1e-9,
        1e-9
      )
    ).toBe(true);
  });

  it('is exactly symmetric', () => {
    const g = (v: Vector) => Math.exp(v.get(0) * v.get(1)) + v.get(2) ** 3 * v.get(0);
    const H = hessianForward(g, Vector.from([0.3, -0.7, 1.1]));
    expect(H.isSymmetric(0)).toBe(true);
  });

  it('needs N(N+3)/2 evaluations if fx is given', () => {
    let n = 0;
    const counted = (v: Vector) => (n++, f(v));
    hessianForward(counted, x, { fx: f(x) });
    expect(n).toBe(5);
    n = 0;
    hessianForward(counted, Vector.from([1, 2, 3, 4]), { fx: 0 });
    expect(n).toBe(14);
  });

  it('does not modify x', () => {
    const x0 = Vector.from([2.0, -2.0]);
    hessianForward(f, x0);
    expect(x0.toArray()).toEqual([2, -2]);
  });

  it('is exact for a quadratic function (up to rounding)', () => {
    const A = Matrix.from([
      [4, 1],
      [1, 3]
    ]);
    const q = (v: Vector) => 0.5 * v.dot(A.mulVec(v));
    const H = hessianForward(q, Vector.from([1.0, 2.0]));
    expect(H.allClose(A, 1e-4, 1e-4)).toBe(true);
  });

  it('throws if sclx is too short', () => {
    expect(() => hessianForward(f, x, { sclx: Vector.from([1]) })).toThrow(RangeError);
  });
});

describe('derivativeCentered', () => {
  const square = (x: number): number => x * x;

  it('uses the automatic step when h is omitted', () => {
    const [derivative] = derivativeCentered(square, 2.0);

    expect(derivative).toBeCloseTo(4.0, 8);
  });

  it('throws when h is explicitly zero', () => {
    expect(() => derivativeCentered(square, 2.0, { h: 0 })).toThrow(RangeError);
  });

  it('uses a supplied nonzero step without replacing it', () => {
    const evaluationPoints: number[] = [];
    derivativeCentered(
      (x) => {
        evaluationPoints.push(x);
        return square(x);
      },
      0.0,
      { h: 1e-12 }
    );

    expect(evaluationPoints).toEqual([1e-12, -1e-12]);
  });
});
