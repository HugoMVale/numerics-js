import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';

export interface RootTestFunction {
    description: string;
    f: (x: Vector) => Vector;
    x0: Vector;
    xs: Vector;
    jac?: (x: Vector) => Matrix;
}

export const fPowellSingular: RootTestFunction = {
    description: 'Extended Powell singular function. Dennis & Schnabel (1996).',
    f: (x) => {
        const result = new Vector(x.size);
        result.set(0, x.get(0) - 10 * x.get(1));
        result.set(1, Math.sqrt(5) * (x.get(2) - x.get(3)));
        result.set(2, (x.get(1) - 2 * x.get(2)) ** 2);
        result.set(3, Math.sqrt(10) * (x.get(0) - x.get(3)) ** 2);
        return result;
    },
    x0: Vector.from([3, -1, 0, 1]),
    xs: Vector.from([0, 0, 0, 0]),
};

export const fRosenbrock: RootTestFunction = {
    description: 'Extended Rosenbrock function. Dennis & Schnabel (1996).',
    f: (x) => {
        const result = new Vector(x.size);
        for (let k = 0; k < x.size / 2; k++) {
            const i = 2 * k;
            result.set(i, 1 - x.get(i));
            result.set(i + 1, 10 * (x.get(i + 1) - x.get(i) ** 2));
        }
        return result;
    },
    x0: Vector.from([-1.2, 1, -1.2, 1]),
    xs: Vector.from([1, 1, 1, 1]),
};

export const fTrigonometric: RootTestFunction = {
    description: 'Trigonometric function. Dennis & Schnabel (1996); Moré, Garbow & Hillstrom (1981), problem 26.',
    f: (x) => {
        const n = x.size;
        let sumCos = 0;
        for (let j = 0; j < n; j++) {
            sumCos += Math.cos(x.get(j));
        }
        // f_i(x) = n - sum_j cos(x_j) + (i + 1) * (1 - cos(x_i)) - sin(x_i), with i = 0, ..., n - 1.
        return x.map((value, i) => n - sumCos + (i + 1) * (1 - Math.cos(value)) - Math.sin(value));
    },
    // The standard starting point 1/n = 0.1 converges (for all solver settings) to a different root
    // with |x|max ~ 0.18, and SciPy's hybr fails from it; x0 = 0.01 reaches xs = 0 reliably.
    x0: Vector.full(10, 0.01),
    xs: Vector.zero(10),
};

export const fCase10: RootTestFunction = {
    description: 'Case 10 of Broyden (1965). Very tough.',
    f: (x) => {
        const x1 = x.get(0);
        const x2 = x.get(1);
        const result = new Vector(2);
        result.set(0, -13 + x1 + ((-x2 + 5) * x2 - 2) * x2);
        result.set(1, -29 + x1 + ((x2 + 1) * x2 - 14) * x2);
        return result;
    },
    x0: Vector.from([15, -2]),
    xs: Vector.from([5, 4]),
};

export const fExample65: RootTestFunction = {
    description: 'Example 6.5 of Dennis & Schnabel (1996).',
    f: (x) => {
        const x1 = x.get(0);
        const x2 = x.get(1);
        const result = new Vector(2);
        result.set(0, x1 ** 2 + x2 ** 2 - 2);
        result.set(1, Math.exp(x1 - 1) + x2 ** 3 - 2);
        return result;
    },
    x0: Vector.from([2, 0.5]),
    xs: Vector.from([1, 1]),
    jac: (x) => {
        const x1 = x.get(0);
        const x2 = x.get(1);
        return Matrix.from([
            [2 * x1, 2 * x2],
            [Math.exp(x1 - 1), 3 * x2 ** 2],
        ]);
    },
};