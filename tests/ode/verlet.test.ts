import { describe, expect, it } from 'vitest';
import { createVelocityVerlet, type VectorArray, type VerletState } from '../../src/ode/verlet.js';

describe('createVelocityVerlet', () => {
    it('updates positions, velocities, and acceleration in the documented order', () => {
        const state: VerletState = {
            u: [1, -2],
            v: [3, 4],
            a: [0.5, -1],
            aNext: [0, 0],
        };
        let callbackState: { u: number[]; v: number[]; aOut: VectorArray } | undefined;
        const step = createVelocityVerlet((u, v, aOut) => {
            callbackState = {
                u: Array.from(u),
                v: Array.from(v),
                aOut,
            };
            aOut[0] = u[0] + 2 * v[0];
            aOut[1] = u[1] + 2 * v[1];
        });

        step(state, 0.2);

        expect(callbackState?.u).toEqual([1.61, -1.22]);
        expect(callbackState?.v).toEqual([3, 4]);
        expect(callbackState?.aOut).toBe(state.aNext);
        expect(state.u).toEqual([1.61, -1.22]);
        expect(state.v).toEqual([3.811, 4.578]);
        expect(state.a).toEqual([7.61, 6.78]);
    });

    it('supports typed-array state and reuses the next-acceleration buffer', () => {
        const state: VerletState = {
            u: new Float64Array([0]),
            v: new Float64Array([1]),
            a: new Float64Array([0]),
            aNext: new Float64Array([0]),
        };
        const accelerationBuffers: Float64Array[] = [];
        const step = createVelocityVerlet((u, _v, aOut) => {
            accelerationBuffers.push(aOut as Float64Array);
            aOut[0] = -u[0];
        });

        step(state, 0.1);
        step(state, 0.1);

        expect(state.u[0]).toBeCloseTo(0.199, 12);
        expect(state.v[0]).toBeCloseTo(0.98005, 12);
        expect(state.a[0]).toBeCloseTo(-0.199, 12);
        expect(accelerationBuffers).toEqual([state.aNext, state.aNext]);
    });
});