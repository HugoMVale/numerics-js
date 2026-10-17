import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';

/** A nonlinear least-squares test problem: minimize `0.5*||f(x)||²`. */
export interface LeastSquaresTestProblem {
  description: string;
  /** Residual function. */
  f: (x: Vector) => Vector;
  /** Analytic Jacobian (`m x n`), if available. */
  jac?: (x: Vector) => Matrix;
  /** Standard starting point. */
  x0: Vector;
  /** Known minimizer, if unique. */
  xs?: Vector;
  /** Known minimum of the sum of squares `||f(x)||²`. */
  minSumSquares: number;
}

const sqrt5 = Math.sqrt(5);
const sqrt10 = Math.sqrt(10);

/** Linear function, full rank (Moré, Garbow & Hillstrom 1981, problem 1), n = 5, m = 10. */
export const linearFullRank: LeastSquaresTestProblem = (() => {
  const n = 5;
  const m = 10;
  return {
    description: 'Linear function, full rank (MGH 1), n = 5, m = 10',
    f: (x) => {
      const s = (2 / m) * x.sum();
      return Vector.from(Array.from({ length: m }, (_, i) => (i < n ? x.get(i) : 0) - s - 1));
    },
    jac: () => new Matrix(m, n).map((_, i, j) => (i === j ? 1 : 0) - 2 / m),
    x0: Vector.ones(n),
    xs: Vector.full(n, -1),
    minSumSquares: m - n
  };
})();

/** Rosenbrock function (MGH 4), n = m = 2. */
export const rosenbrock: LeastSquaresTestProblem = {
  description: 'Rosenbrock function (MGH 4)',
  f: (x) => Vector.from([10 * (x.get(1) - x.get(0) ** 2), 1 - x.get(0)]),
  jac: (x) =>
    Matrix.from([
      [-20 * x.get(0), 10],
      [-1, 0]
    ]),
  x0: Vector.from([-1.2, 1]),
  xs: Vector.from([1, 1]),
  minSumSquares: 0
};

/** Powell singular function (MGH 13), n = m = 4. The Jacobian is singular at the solution. */
export const powellSingular: LeastSquaresTestProblem = {
  description: 'Powell singular function (MGH 13)',
  f: (x) =>
    Vector.from([
      x.get(0) + 10 * x.get(1),
      sqrt5 * (x.get(2) - x.get(3)),
      (x.get(1) - 2 * x.get(2)) ** 2,
      sqrt10 * (x.get(0) - x.get(3)) ** 2
    ]),
  jac: (x) => {
    const a = x.get(1) - 2 * x.get(2);
    const b = x.get(0) - x.get(3);
    return Matrix.from([
      [1, 10, 0, 0],
      [0, 0, sqrt5, -sqrt5],
      [0, 2 * a, -4 * a, 0],
      [2 * sqrt10 * b, 0, 0, -2 * sqrt10 * b]
    ]);
  },
  x0: Vector.from([3, -1, 0, 1]),
  xs: Vector.zero(4),
  minSumSquares: 0
};

/** Helical valley function (MGH 7), n = m = 3. */
export const helicalValley: LeastSquaresTestProblem = {
  description: 'Helical valley function (MGH 7)',
  f: (x) => {
    let theta = Math.atan2(x.get(1), x.get(0)) / (2 * Math.PI);
    if (theta < -0.25) theta += 1;
    return Vector.from([
      10 * (x.get(2) - 10 * theta),
      10 * (Math.hypot(x.get(0), x.get(1)) - 1),
      x.get(2)
    ]);
  },
  x0: Vector.from([-1, 0, 0]),
  xs: Vector.from([1, 0, 0]),
  minSumSquares: 0
};

/** Beale function (MGH 5), n = 2, m = 3. */
export const beale: LeastSquaresTestProblem = {
  description: 'Beale function (MGH 5)',
  f: (x) =>
    Vector.from([1, 2, 3].map((i) => [1.5, 2.25, 2.625][i - 1] - x.get(0) * (1 - x.get(1) ** i))),
  jac: (x) =>
    new Matrix(3, 2).map((_, r, c) => {
      const i = r + 1;
      return c === 0 ? -(1 - x.get(1) ** i) : x.get(0) * i * x.get(1) ** (i - 1);
    }),
  x0: Vector.from([1, 1]),
  xs: Vector.from([3, 0.5]),
  minSumSquares: 0
};

/** Jennrich and Sampson function (MGH 6), n = 2, m = 10. */
export const jennrichSampson: LeastSquaresTestProblem = {
  description: 'Jennrich and Sampson function (MGH 6), n = 2, m = 10',
  f: (x) =>
    Vector.from(
      Array.from({ length: 10 }, (_, k) => {
        const i = k + 1;
        return 2 + 2 * i - (Math.exp(i * x.get(0)) + Math.exp(i * x.get(1)));
      })
    ),
  jac: (x) =>
    new Matrix(10, 2).map((_, k, c) => {
      const i = k + 1;
      return -i * Math.exp(i * x.get(c));
    }),
  x0: Vector.from([0.3, 0.4]),
  minSumSquares: 124.3622 // MGH, 4 decimals
};

/** Brown badly scaled function (MGH 4), n = 2, m = 3. */
export const brownBadlyScaled: LeastSquaresTestProblem = {
  description: 'Brown badly scaled function (MGH 4), n = 2, m = 3',
  f: (x) => Vector.from([x.get(0) - 1e6, x.get(1) - 2e-6, x.get(0) * x.get(1) - 2]),
  jac: (x) =>
    Matrix.from([
      [1, 0],
      [0, 1],
      [x.get(1), x.get(0)]
    ]),
  x0: Vector.from([1, 1]),
  xs: Vector.from([1e6, 2e-6]),
  minSumSquares: 0
};

/**
 * Freudenstein and Roth function (MGH 2), n = m = 2. From the standard starting point, MINPACK-type
 * methods converge to the local minimum `(11.41248, -0.896828)` (`||f||² = 48.98425`), not to the
 * zero at `(5, 4)`; SciPy's `leastsq` does the same.
 */
export const freudensteinRoth: LeastSquaresTestProblem = {
  description: 'Freudenstein and Roth function (MGH 2), local minimum from the standard start',
  f: (x) => {
    const [a, b] = [x.get(0), x.get(1)];
    return Vector.from([-13 + a + ((5 - b) * b - 2) * b, -29 + a + ((b + 1) * b - 14) * b]);
  },
  jac: (x) => {
    const b = x.get(1);
    return Matrix.from([
      [1, 10 * b - 3 * b * b - 2],
      [1, 3 * b * b + 2 * b - 14]
    ]);
  },
  x0: Vector.from([0.5, -2]),
  xs: Vector.from([11.41248446, -0.89682791]),
  minSumSquares: 48.98425372707155
};

/** Watson function (MGH 20), n = 6, m = 31. */
export const watson: LeastSquaresTestProblem = {
  description: 'Watson function (MGH 20), n = 6, m = 31',
  f: (x) => {
    const n = x.size;
    const r = new Vector(31);
    for (let i = 1; i <= 29; i++) {
      const t = i / 29;
      let s1 = 0;
      let s2 = 0;
      for (let j = 1; j < n; j++) s1 += j * x.get(j) * t ** (j - 1);
      for (let j = 0; j < n; j++) s2 += x.get(j) * t ** j;
      r.set(i - 1, s1 - s2 * s2 - 1);
    }
    r.set(29, x.get(0));
    r.set(30, x.get(1) - x.get(0) ** 2 - 1);
    return r;
  },
  x0: Vector.zero(6),
  minSumSquares: 2.28767005e-3
};

/** A rank-deficient problem: the three residuals are multiples of `x0 + x1 - 2`. */
export const rankDeficient: LeastSquaresTestProblem = {
  description: 'Rank-deficient linear problem, solutions on the line x0 + x1 = 2',
  f: (x) => {
    const s = x.get(0) + x.get(1) - 2;
    return Vector.from([s, 2 * s, 3 * s]);
  },
  jac: () =>
    Matrix.from([
      [1, 1],
      [2, 2],
      [3, 3]
    ]),
  x0: Vector.from([0, 0]),
  minSumSquares: 0
};

export const LEAST_SQUARES_PROBLEMS: Record<string, LeastSquaresTestProblem> = {
  linearFullRank,
  rosenbrock,
  powellSingular,
  helicalValley,
  beale,
  jennrichSampson,
  brownBadlyScaled,
  freudensteinRoth,
  watson,
  rankDeficient
};
