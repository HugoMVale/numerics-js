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
        const values = new Array<number>(x.size).fill(0);
        for (let k = 0; k < x.size / 4; k++) {
            const i = 4 * k;
            values[i] = x.get(i) - 10 * x.get(i + 1);
            values[i + 1] = Math.sqrt(5) * (x.get(i + 2) - x.get(i + 3));
            values[i + 2] = (x.get(i + 1) - 2 * x.get(i + 2)) ** 2;
            values[i + 3] = Math.sqrt(10) * (x.get(i) - x.get(i + 3)) ** 2;
        }
        return Vector.from(values);
    },
    x0: Vector.from([3, -1, 0, 1]),
    xs: Vector.from([0, 0, 0, 0]),
};

export const fRosenbrock: RootTestFunction = {
    description: 'Extended Rosenbrock function. Dennis & Schnabel (1996).',
    f: (x) => {
        const values = new Array<number>(x.size).fill(0);
        for (let k = 0; k < x.size / 2; k++) {
            const i = 2 * k;
            values[i] = 1 - x.get(i);
            values[i + 1] = 10 * (x.get(i + 1) - x.get(i) ** 2);
        }
        return Vector.from(values);
    },
    x0: Vector.from([-1.2, 1, -1.2, 1]),
    xs: Vector.from([1, 1, 1, 1]),
};

export const fTrigonometric: RootTestFunction = {
    description: 'Trignometric function. Dennis & Schnabel (1996).',
    f: (x) => {
        const values = new Array<number>(x.size).fill(0);
        for (let k = 0; k < x.size; k++) {
            let sum = 0;
            for (let j = 0; j < x.size; j++) {
                sum += Math.cos(x.get(j)) + k * (1 - Math.cos(x.get(k))) - Math.sin(x.get(k));
            }
            values[k] = x.size - sum;
        }
        return Vector.from(values);
    },
    x0: Vector.from(new Array(10).fill(0.1)),
    xs: Vector.from(new Array(10).fill(0)),
};

export const fCase10: RootTestFunction = {
    description: 'Case 10 of Broyden (1965). Very tough.',
    f: (x) => {
        const x1 = x.get(0);
        const x2 = x.get(1);
        return Vector.from([
            -13 + x1 + ((-x2 + 5) * x2 - 2) * x2,
            -29 + x1 + ((x2 + 1) * x2 - 14) * x2,
        ]);
    },
    x0: Vector.from([15, -2]),
    xs: Vector.from([5, 4]),
};

export const fExample65: RootTestFunction = {
    description: 'Example 6.5 of Dennis & Schnabel (1996).',
    f: (x) => {
        const x1 = x.get(0);
        const x2 = x.get(1);
        return Vector.from([x1 ** 2 + x2 ** 2 - 2, Math.exp(x1 - 1) + x2 ** 3 - 2]);
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