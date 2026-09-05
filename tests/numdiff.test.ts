import { describe, it, expect } from 'vitest';
import { Vector } from '../src/array/Vector.js';
import { Matrix } from '../src/array/Matrix.js';
import { scaleVector, jacobianForward } from '../src/numdiff.js';

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