import { describe, it, expect } from 'vitest';
import { rk4Step, rk4Integrate } from '../src/ode/rk4';
import { wrapAllocatingDerivative } from '../src/ode/adapters';
import { Array1D } from '../src/array/array1d';

describe('RK4 Integration Module', () => {

    // Helper to create a derivative function dy/dt = y (Exponential growth)
    // Analytical solution: y(t) = y(0) * e^t
    const expDerivative = (t: number, y: Array1D, dydt: Array1D): Array1D => {
        dydt.set(y.data);
        return dydt;
    };

    describe('rk4Step', () => {
        it('should correctly advance the state by one step (dy/dt = y)', () => {
            const dim = 1;
            const y0 = new Array1D(dim);
            y0.data[0] = 1.0; // y(0) = 1
            const out = new Array1D(dim);
            const h = 1.0; // Step size

            rk4Step(expDerivative, 0, y0, h, out);

            // RK4 approximation of e^1 with one step of h=1 is exactly 1 + 1 + 1/2 + 1/6 + 1/24 = 2.708333...
            const expectedApprox = 1 + 1 + 1 / 2 + 1 / 6 + 1 / 24;
            expect(out.data[0]).toBeCloseTo(expectedApprox, 5);
        });

        it('should not mutate the input state vector', () => {
            const y0 = new Array1D(1);
            y0.data[0] = 5.0;
            const out = new Array1D(1);

            rk4Step(expDerivative, 0, y0, 0.1, out);

            expect(y0.data[0]).toBe(5.0); // Should remain untouched
        });
    });

    it('should integrate correctly over multiple full steps', () => {
        const y0 = new Array1D(1);
        y0.data[0] = 1.0;

        const { t, y } = rk4Integrate(expDerivative, 0, 2, y0, 1.0);

        expect(t.data[0]).toBe(0);
        expect(t.data[1]).toBe(1.0);
        expect(t.data[2]).toBe(2.0);

        // t is an Array1D, which we know has the .dim property.
        // Both t and y have the same number of records.
        expect(t.dim).toBe(3);
    });

    it('should handle a shorter final partial step to land on tEnd exactly', () => {
        const y0 = new Array1D(1);
        y0.data[0] = 1.0;

        // Integrating from 0 to 2.5 with h=1 -> two full steps, one partial step of 0.5
        const { t, y } = rk4Integrate(expDerivative, 0, 2.5, y0, 1.0);

        const lastIdx = t.dim - 1;
        expect(t.data[lastIdx]).toBe(2.5);
        expect(t.dim).toBe(4); // 0.0, 1.0, 2.0, 2.5 (4 records)
    });

    it('should integrate backward if tEnd < t0 and h is negative', () => {
        const y0 = new Array1D(1);
        y0.data[0] = 1.0;

        const { t } = rk4Integrate(expDerivative, 2, 0, y0, -1.0);

        expect(t.data[0]).toBe(2.0);
        expect(t.data[1]).toBe(1.0);
        expect(t.data[2]).toBe(0.0);
    });

    it('should throw RangeError if h is 0', () => {
        const y0 = new Array1D(1);
        expect(() => rk4Integrate(expDerivative, 0, 1, y0, 0)).toThrow(RangeError);
    });

    it('should throw RangeError if the sign of h does not match integration direction', () => {
        const y0 = new Array1D(1);
        // Positive step but integrating backwards
        expect(() => rk4Integrate(expDerivative, 2, 0, y0, 1.0)).toThrow(RangeError);
        // Negative step but integrating forwards
        expect(() => rk4Integrate(expDerivative, 0, 2, y0, -1.0)).toThrow(RangeError);
    });
});

describe('wrapAllocatingDerivative', () => {
    it('should adapt an allocating derivative function to a non-allocating signature', () => {
        // Allocating version: returns a newly constructed Array1D
        const allocatingF = (t: number, y: Array1D): Array1D => {
            const res = new Array1D(1);
            res.data[0] = y.data[0] * 2;
            return res;
        };

        const wrappedF = wrapAllocatingDerivative(allocatingF);

        const y = new Array1D(1);
        y.data[0] = 10;
        const dydt = new Array1D(1);

        wrappedF(0, y, dydt);

        expect(dydt.data[0]).toBe(20);
    });
});