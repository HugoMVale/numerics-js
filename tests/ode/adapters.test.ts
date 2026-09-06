import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { wrapAllocatingDerivative } from '../../src/ode/adapters.js';

describe('wrapAllocatingDerivative', () => {
    it('copies the allocating derivative result into and returns the supplied output buffer', () => {
        const y = new Vector([2, -3]);
        const dydt = new Vector([99, 99]);
        const derivative = wrapAllocatingDerivative((t, state) => {
            expect(t).toBe(1.5);
            expect(state).toBe(y);
            return new Vector([state.get(0) + t, state.get(1) - t]);
        });

        const returned = derivative(1.5, y, dydt);

        expect(returned).toBe(dydt);
        expect(Array.from(dydt.data)).toEqual([3.5, -4.5]);
        expect(Array.from(y.data)).toEqual([2, -3]);
    });
});