import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';

/** Smooth test function with analytic derivatives, for gradient-based minimizers. */
export interface GradientTestFunction {
  description: string;
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess?: (x: Vector) => Matrix;
  x0: number[];
  /** Minimizer. */
  xs: number[];
  /** Minimum value. */
  fs: number;
}

const rosenbrock: GradientTestFunction = {
  description: 'Rosenbrock function: (1 - x)^2 + 100*(y - x^2)^2.',
  f: (x) => (1 - x.get(0)) ** 2 + 100 * (x.get(1) - x.get(0) ** 2) ** 2,
  grad: (x) =>
    Vector.from([
      -2 * (1 - x.get(0)) - 400 * x.get(0) * (x.get(1) - x.get(0) ** 2),
      200 * (x.get(1) - x.get(0) ** 2)
    ]),
  hess: (x) =>
    Matrix.from([
      [2 - 400 * (x.get(1) - 3 * x.get(0) ** 2), -400 * x.get(0)],
      [-400 * x.get(0), 200]
    ]),
  x0: [-1.2, 1],
  xs: [1, 1],
  fs: 0
};

const wood: GradientTestFunction = {
  description: 'Wood function (Dennis & Schnabel, 1996; Moré, Garbow & Hillstrom, 1981).',
  f: (v) => {
    const x = v.data;
    return (
      100 * (x[0] ** 2 - x[1]) ** 2 +
      (x[0] - 1) ** 2 +
      (x[2] - 1) ** 2 +
      90 * (x[2] ** 2 - x[3]) ** 2 +
      10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) +
      19.8 * (x[1] - 1) * (x[3] - 1)
    );
  },
  grad: (v) => {
    const x = v.data;
    return Vector.from([
      400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1),
      -200 * (x[0] ** 2 - x[1]) + 20.2 * (x[1] - 1) + 19.8 * (x[3] - 1),
      2 * (x[2] - 1) + 360 * x[2] * (x[2] ** 2 - x[3]),
      -180 * (x[2] ** 2 - x[3]) + 20.2 * (x[3] - 1) + 19.8 * (x[1] - 1)
    ]);
  },
  x0: [-3, -1, -3, -1],
  xs: [1, 1, 1, 1],
  fs: 0
};

const bealeTerms = (x: Vector): [number, number, number] => {
  const p = x.get(0);
  const q = x.get(1);
  return [1.5 - p + p * q, 2.25 - p + p * q ** 2, 2.625 - p + p * q ** 3];
};

const beale: GradientTestFunction = {
  description: 'Beale function.',
  f: (x) => {
    const [a, b, c] = bealeTerms(x);
    return a * a + b * b + c * c;
  },
  grad: (x) => {
    const p = x.get(0);
    const q = x.get(1);
    const [a, b, c] = bealeTerms(x);
    return Vector.from([
      2 * a * (q - 1) + 2 * b * (q ** 2 - 1) + 2 * c * (q ** 3 - 1),
      2 * a * p + 4 * b * p * q + 6 * c * p * q ** 2
    ]);
  },
  x0: [1, 1],
  xs: [3, 0.5],
  fs: 0
};

const quadA = Matrix.from([
  [10, 2, 0],
  [2, 5, 1],
  [0, 1, 1]
]);
const quadB = Vector.from([1, 2, 3]);

const quadratic: GradientTestFunction = {
  description:
    'Convex quadratic: 0.5*xᵀAx - bᵀx, with A = [[10, 2, 0], [2, 5, 1], [0, 1, 1]], b = [1, 2, 3].',
  f: (x) => 0.5 * x.dot(quadA.mulVec(x)) - quadB.dot(x),
  grad: (x) => quadA.mulVec(x).sub(quadB),
  hess: () => quadA.copy(),
  x0: [1, 1, 1],
  xs: quadA.solve(quadB).toArray(),
  fs: -0.5 * quadB.dot(quadA.solve(quadB))
};

export const GRADIENT_TEST_FUNCTIONS: Record<string, GradientTestFunction> = {
  rosenbrock,
  wood,
  beale,
  quadratic
};
