import { describe, it, expect } from 'vitest';
import { Vector } from '../src/linalg/Vector.js';
import { Matrix } from '../src/linalg/Matrix.js';
import {
  scaleVector,
  jacobianForward,
  jacobianForwardBanded,
  gradientForward,
  hessianForward,
  derivativeCentered
} from '../src/numdiff.js';

describe('scaleVector', () => {
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

  it('uses a common factor 1/max|x| if the magnitudes differ by less than a decade', () => {
    const sclx = scaleVector(Vector.from([2.0, -5.0, 0.0]));
    expect(sclx.allClose(Vector.from([0.2, 0.2, 5.0]))).toBe(true);
  });

  it('does not modify its argument', () => {
    const x = Vector.from([0.0, 3.0, -7.0]);
    scaleVector(x);
    expect(x.toArray()).toEqual([0, 3, -7]);
  });

  it('throws if x has a non-finite component', () => {
    expect(() => scaleVector(Vector.from([1.0, NaN]))).toThrow(RangeError);
    expect(() => scaleVector(Vector.from([Infinity, 1.0]))).toThrow(RangeError);
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
    expect(jacobian.allClose(expectedMatrix, { rtol: 1e-6 })).toBe(true);
  });

  it('returns a matrix with as many rows as f has components and as many columns as x', () => {
    const f = (v: Vector) => Vector.from([v.get(0) + v.get(1) + v.get(2), v.get(0) * v.get(2)]);
    const jacobian = jacobianForward(f, Vector.from([1.0, 2.0, 3.0]));
    expect(jacobian.rows).toBe(2);
    expect(jacobian.cols).toBe(3);
    expect(
      jacobian.allClose(
        Matrix.from([
          [1, 1, 1],
          [3, 0, 1]
        ]),
        { atol: 1e-6, rtol: 1e-6 }
      )
    ).toBe(true);
  });

  it('needs N evaluations if fx is given and N+1 otherwise', () => {
    let n = 0;
    const counted = (v: Vector) => (n++, fnc3(v));
    const x = Vector.from([3.0, -2.0]);
    jacobianForward(counted, x, { fx: fnc3(x) });
    expect(n).toBe(2);
    n = 0;
    jacobianForward(counted, x);
    expect(n).toBe(3);
  });

  it('does not modify x', () => {
    const x = Vector.from([3.0, -2.0]);
    jacobianForward(fnc3, x);
    expect(x.toArray()).toEqual([3, -2]);
  });

  it('respects sclx and uses abs(sclx)', () => {
    const seen: number[] = [];
    const f = (v: Vector) => (seen.push(v.get(0)), Vector.from([v.get(0)]));
    jacobianForward(f, Vector.from([0.0]), { fx: Vector.from([0.0]), sclx: Vector.from([-1e2]) });
    // h = sqrt(eps) * max(|x|, 1/|sclx|) = sqrt(eps)/100
    expect(seen[0]).toBeCloseTo(Math.sqrt(Number.EPSILON) / 100, 20);
  });

  it('increases the step with a larger epsf, and ignores epsf below machine precision', () => {
    const steps = (epsf: number) => {
      let xi = 0;
      const f = (v: Vector) => ((xi = v.get(0)), Vector.from([v.get(0)]));
      jacobianForward(f, Vector.from([1.0]), { fx: Vector.from([1.0]), epsf });
      return xi - 1;
    };
    expect(steps(1e-8)).toBeGreaterThan(steps(Number.EPSILON));
    expect(steps(1e-20)).toBe(steps(Number.EPSILON));
  });

  it('throws if sclx is too short, or has a zero or non-finite value', () => {
    const x = Vector.from([3.0, -2.0]);
    expect(() => jacobianForward(fnc3, x, { sclx: Vector.from([1]) })).toThrow(/sclx must have 2/);
    expect(() => jacobianForward(fnc3, x, { sclx: Vector.from([1, 0]) })).toThrow(RangeError);
    expect(() => jacobianForward(fnc3, x, { sclx: Vector.from([NaN, 1]) })).toThrow(RangeError);
  });

  it('throws if epsf is not finite', () => {
    const x = Vector.from([3.0, -2.0]);
    expect(() => jacobianForward(fnc3, x, { epsf: NaN })).toThrow(RangeError);
    expect(() => jacobianForward(fnc3, x, { epsf: Infinity })).toThrow(RangeError);
  });

  it('throws if fx or the perturbed function values have inconsistent sizes', () => {
    const x = Vector.from([3.0, -2.0]);
    expect(() => jacobianForward(fnc3, x, { fx: Vector.from([1, 2, 3]) })).toThrow(RangeError);
    let n = 0;
    const inconsistent = (v: Vector) => (n++ === 0 ? fnc3(v) : Vector.from([1, 2, 3]));
    expect(() => jacobianForward(inconsistent, x)).toThrow(/returned 3 components/);
  });
});

describe('jacobianForwardBanded', () => {
  // A nonlinear function whose component k depends only on x[k - kl .. k + ku].
  const bandedFunction =
    (kl: number, ku: number) =>
    (x: Vector): Vector => {
      const n = x.size;
      const out = new Vector(n);
      for (let k = 0; k < n; k++) {
        let s = x.get(k) ** 2;
        for (let j = Math.max(0, k - kl); j <= Math.min(n - 1, k + ku); j++) {
          s += (1 + 0.1 * (k - j)) * Math.sin(x.get(j) + 0.3 * k);
        }
        out.set(k, s);
      }
      return out;
    };
  const point = (n: number) => Vector.from(Array.from({ length: n }, (_, i) => 0.5 + 0.37 * i));

  it('gives exactly the Jacobian of jacobianForward for functions that honor the band', () => {
    for (const [kl, ku] of [
      [1, 1],
      [2, 1],
      [0, 3],
      [3, 2],
      [0, 0]
    ]) {
      const f = bandedFunction(kl, ku);
      const x = point(12);
      const dense = jacobianForward(f, x);
      const band = jacobianForwardBanded(f, x, { kl, ku });

      expect(band.kl).toBe(kl);
      expect(band.ku).toBe(ku);
      expect(Array.from(band.toMatrix().data)).toEqual(Array.from(dense.data));
    }
  });

  it('honors fx, sclx and epsf like jacobianForward', () => {
    const f = bandedFunction(1, 1);
    const x = point(9);
    const options = {
      fx: f(x),
      sclx: Vector.from(Array.from({ length: 9 }, (_, i) => 1 + i)),
      epsf: 1e-10
    };
    const dense = jacobianForward(f, x, options);
    const band = jacobianForwardBanded(f, x, { kl: 1, ku: 1, ...options });
    expect(Array.from(band.toMatrix().data)).toEqual(Array.from(dense.data));
  });

  it('approximates the analytic tridiagonal Jacobian', () => {
    // f_k = x_{k-1} - 2 x_k + x_{k+1} + x_k^2, zero outside.
    const n = 8;
    const f = (x: Vector): Vector => {
      const out = new Vector(n);
      for (let k = 0; k < n; k++) {
        const left = k > 0 ? x.get(k - 1) : 0;
        const right = k < n - 1 ? x.get(k + 1) : 0;
        out.set(k, left - 2 * x.get(k) + right + x.get(k) ** 2);
      }
      return out;
    };
    const x = point(n);
    const band = jacobianForwardBanded(f, x, { kl: 1, ku: 1 });
    for (let k = 0; k < n; k++) {
      expect(band.get(k, k)).toBeCloseTo(-2 + 2 * x.get(k), 6);
      if (k > 0) expect(band.get(k, k - 1)).toBeCloseTo(1, 6);
      if (k < n - 1) expect(band.get(k, k + 1)).toBeCloseTo(1, 6);
    }
  });

  it('needs min(kl + ku + 1, n) evaluations, plus one for f(x) unless fx is given', () => {
    const count = (n: number, kl: number, ku: number, withFx: boolean): number => {
      const f0 = bandedFunction(kl, ku);
      let calls = 0;
      const f = (x: Vector): Vector => {
        calls++;
        return f0(x);
      };
      const x = point(n);
      jacobianForwardBanded(f, x, withFx ? { kl, ku, fx: f0(x) } : { kl, ku });
      return calls;
    };
    expect(count(50, 1, 1, true)).toBe(3);
    expect(count(50, 1, 1, false)).toBe(4);
    expect(count(50, 3, 2, true)).toBe(6);
    expect(count(50, 0, 0, true)).toBe(1);
    // The band covers the whole matrix: every column is its own group, as in jacobianForward.
    expect(count(5, 4, 4, true)).toBe(5);
    expect(count(5, 2, 2, true)).toBe(5);
    expect(count(5, 2, 1, true)).toBe(4);
  });

  it('clamps a bandwidth of n or more to n - 1', () => {
    const f = bandedFunction(2, 2);
    const x = point(4);
    const band = jacobianForwardBanded(f, x, { kl: 10, ku: 4 });
    expect(band.kl).toBe(3);
    expect(band.ku).toBe(3);
    expect(Array.from(band.toMatrix().data)).toEqual(Array.from(jacobianForward(f, x).data));
  });

  it('handles a single component', () => {
    const band = jacobianForwardBanded((x) => Vector.from([x.get(0) ** 3]), Vector.from([2]), {
      kl: 0,
      ku: 0
    });
    expect(band.n).toBe(1);
    expect(band.get(0, 0)).toBeCloseTo(12, 5);
  });

  it('accepts a function that returns the same buffer every time and never modifies x', () => {
    const f0 = bandedFunction(1, 1);
    const shared = new Vector(10);
    const f = (x: Vector): Vector => shared.set(f0(x).data);
    const x = point(10);
    const xCopy = x.copy();
    const band = jacobianForwardBanded(f, x, { kl: 1, ku: 1 });
    expect(Array.from(x.data)).toEqual(Array.from(xCopy.data));
    expect(Array.from(band.toMatrix().data)).toEqual(Array.from(jacobianForward(f0, x).data));
  });

  it('returns silently wrong entries if the function does not honor the declared band', () => {
    // A full coupling declared as tridiagonal: the grouped perturbations overlap.
    const f = bandedFunction(5, 5);
    const x = point(8);
    const wrong = jacobianForwardBanded(f, x, { kl: 1, ku: 1 }).toMatrix();
    const right = jacobianForward(f, x);
    let maxDiff = 0;
    for (let k = 0; k < 8; k++) {
      for (let i = Math.max(0, k - 1); i <= Math.min(7, k + 1); i++) {
        maxDiff = Math.max(maxDiff, Math.abs(wrong.get(k, i) - right.get(k, i)));
      }
    }
    expect(maxDiff).toBeGreaterThan(1e-3);
  });

  it('rejects invalid bandwidths and options', () => {
    const f = bandedFunction(1, 1);
    const x = point(5);
    // @ts-expect-error missing options
    expect(() => jacobianForwardBanded(f, x)).toThrow(/options with kl and ku are required/);
    // @ts-expect-error missing ku
    expect(() => jacobianForwardBanded(f, x, { kl: 1 })).toThrow(TypeError);
    // @ts-expect-error string bandwidth
    expect(() => jacobianForwardBanded(f, x, { kl: '1', ku: 1 })).toThrow(/kl must be a number/);
    expect(() => jacobianForwardBanded(f, x, { kl: -1, ku: 1 })).toThrow(RangeError);
    expect(() => jacobianForwardBanded(f, x, { kl: 1, ku: 1.5 })).toThrow(
      /ku must be a non-negative integer, got 1.5\./
    );
    expect(() => jacobianForwardBanded(f, x, { kl: NaN, ku: 1 })).toThrow(RangeError);
    expect(() => jacobianForwardBanded(f, x, { kl: 1, ku: 1, sclx: Vector.from([1]) })).toThrow(
      /sclx must have 5/
    );
    expect(() =>
      jacobianForwardBanded(f, x, { kl: 1, ku: 1, sclx: Vector.from([1, 1, 0, 1, 1]) })
    ).toThrow(RangeError);
    expect(() => jacobianForwardBanded(f, x, { kl: 1, ku: 1, epsf: NaN })).toThrow(RangeError);
  });

  it('rejects function values of the wrong size', () => {
    const x = point(5);
    expect(() => jacobianForwardBanded(() => Vector.from([1, 2, 3]), x, { kl: 1, ku: 1 })).toThrow(
      /f must return as many components as x \(5\), got 3\./
    );
    expect(() =>
      jacobianForwardBanded(bandedFunction(1, 1), x, { kl: 1, ku: 1, fx: Vector.from([1, 2]) })
    ).toThrow(RangeError);
    const f0 = bandedFunction(1, 1);
    let calls = 0;
    const inconsistent = (v: Vector): Vector => (++calls === 1 ? f0(v) : Vector.from([1, 2, 3]));
    expect(() => jacobianForwardBanded(inconsistent, x, { kl: 1, ku: 1 })).toThrow(
      /returned 3 components/
    );
  });
});

describe('gradientForward', () => {
  const f = (x: Vector): number => x.get(0) ** 2 * x.get(1) ** 3;
  const x = Vector.from([2.0, -2.0]);

  it('matches the analytical gradient and the original Python values', () => {
    const g = gradientForward(f, x);
    expect(g.allClose(Vector.from([-32, 48]), { atol: 1e-6, rtol: 1e-6 })).toBe(true);
    // Values of the original Python implementation (polykin.math.gradient_forward).
    expect(g.allClose(Vector.from([-32.00000024, 47.99999928]), { atol: 1e-9, rtol: 1e-9 })).toBe(
      true
    );
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

  it('throws if sclx is too short, or has a zero or non-finite value', () => {
    expect(() => gradientForward(f, x, { sclx: Vector.from([1]) })).toThrow(RangeError);
    expect(() => gradientForward(f, x, { sclx: Vector.from([1, 0]) })).toThrow(RangeError);
    expect(() => gradientForward(f, x, { sclx: Vector.from([Infinity, 1]) })).toThrow(RangeError);
  });

  it('accepts an sclx with surplus components', () => {
    const a = gradientForward(f, x, { sclx: Vector.from([1, 1]) });
    const b = gradientForward(f, x, { sclx: Vector.from([1, 1, 0]) });
    expect(b.toString()).toBe(a.toString());
  });

  it('throws if epsf is not finite', () => {
    expect(() => gradientForward(f, x, { epsf: NaN })).toThrow(RangeError);
    expect(() => gradientForward(f, x, { epsf: Infinity })).toThrow(RangeError);
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
        { atol: 1e-3, rtol: 1e-3 }
      )
    ).toBe(true);
    // Values of the original Python implementation (polykin.math.hessian_forward).
    expect(
      H.allClose(
        Matrix.from([
          [-16.00001242, 47.99984347],
          [47.99984347, -47.99972236]
        ]),
        { atol: 1e-9, rtol: 1e-9 }
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
    expect(H.allClose(A, { atol: 1e-4, rtol: 1e-4 })).toBe(true);
  });

  it('throws if sclx is too short, or has a zero or non-finite value', () => {
    expect(() => hessianForward(f, x, { sclx: Vector.from([1]) })).toThrow(RangeError);
    expect(() => hessianForward(f, x, { sclx: Vector.from([1, 0]) })).toThrow(RangeError);
    expect(() => hessianForward(f, x, { sclx: Vector.from([Infinity, 1]) })).toThrow(RangeError);
  });

  it('accepts an sclx with surplus components', () => {
    const a = hessianForward(f, x, { sclx: Vector.from([1, 1]) });
    const b = hessianForward(f, x, { sclx: Vector.from([1, 1, 0]) });
    expect(b.toString()).toBe(a.toString());
  });

  it('throws if epsf is not finite', () => {
    expect(() => hessianForward(f, x, { epsf: NaN })).toThrow(RangeError);
    expect(() => hessianForward(f, x, { epsf: Infinity })).toThrow(RangeError);
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

  it('returns the derivative and the mean of the perturbed function values', () => {
    const [df, fx] = derivativeCentered((x) => x ** 3, 1.0);
    expect(df).toBeCloseTo(3.0, 8);
    expect(fx).toBeCloseTo(1.0, 8);
  });

  it('scales the step by max(1, |x|)', () => {
    const points: number[] = [];
    derivativeCentered((x) => (points.push(x), x), 100.0, { h: 1e-3 });
    expect(points[0]).toBeCloseTo(100.1, 12);
    expect(points[1]).toBeCloseTo(99.9, 12);
  });

  it('does not depend on the sign of h', () => {
    const [a] = derivativeCentered(Math.sin, 0.5, { h: 1e-4 });
    const [b] = derivativeCentered(Math.sin, 0.5, { h: -1e-4 });
    expect(b).toBeCloseTo(a, 12);
  });

  it('throws when h or epsf is not finite', () => {
    expect(() => derivativeCentered(square, 2.0, { h: NaN })).toThrow(RangeError);
    expect(() => derivativeCentered(square, 2.0, { h: Infinity })).toThrow(RangeError);
    expect(() => derivativeCentered(square, 2.0, { epsf: NaN })).toThrow(RangeError);
  });

  it('ignores epsf below machine precision', () => {
    expect(derivativeCentered(square, 2.0, { epsf: 1e-20 })).toEqual(
      derivativeCentered(square, 2.0)
    );
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
