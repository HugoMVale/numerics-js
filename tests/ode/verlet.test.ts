import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { createVelocityVerlet, type VerletState } from '../../src/ode/verlet.js';

describe('createVelocityVerlet', () => {
  it('updates positions, velocities, and acceleration in the documented order', () => {
    const state: VerletState = {
      u: new Vector([1, -2]),
      v: new Vector([3, 4]),
      a: new Vector([0.5, -1]),
      aNext: new Vector(2)
    };
    let callbackState: { u: number[]; v: number[]; aOut: Vector } | undefined;
    const step = createVelocityVerlet((u, v, aOut) => {
      callbackState = {
        u: Array.from(u.data),
        v: Array.from(v.data),
        aOut
      };
      aOut.data[0] = u.data[0] + 2 * v.data[0];
      aOut.data[1] = u.data[1] + 2 * v.data[1];
      return aOut;
    });

    step(state, 0.2);

    expect(callbackState?.u).toEqual([1.61, -1.22]);
    expect(callbackState?.v).toEqual([3, 4]);
    expect(callbackState?.aOut).toBe(state.aNext);
    expect(Array.from(state.u.data)).toEqual([1.61, -1.22]);
    expect(Array.from(state.v.data)).toEqual([3.811, 4.578]);
    expect(Array.from(state.a.data)).toEqual([7.61, 6.78]);
  });

  it('advances the state in place and reuses the next-acceleration buffer', () => {
    const state: VerletState = {
      u: new Vector([0]),
      v: new Vector([1]),
      a: new Vector([0]),
      aNext: new Vector(1)
    };
    const { u, v, a, aNext } = state;
    const accelerationBuffers: Vector[] = [];
    const step = createVelocityVerlet((u, _v, aOut) => {
      accelerationBuffers.push(aOut);
      return aOut.set(0, -u.get(0));
    });

    step(state, 0.1);
    step(state, 0.1);

    expect(state.u).toBe(u);
    expect(state.v).toBe(v);
    expect(state.a).toBe(a);
    expect(state.aNext).toBe(aNext);
    expect(u.get(0)).toBeCloseTo(0.199, 12);
    expect(v.get(0)).toBeCloseTo(0.98005, 12);
    expect(a.get(0)).toBeCloseTo(-0.199, 12);
    expect(accelerationBuffers).toEqual([aNext, aNext]);
    expect(accelerationBuffers[0]).toBe(aNext);
  });

  it('conserves the energy of a harmonic oscillator over many steps', () => {
    const state: VerletState = {
      u: new Vector([1]),
      v: new Vector([0]),
      a: new Vector([-1]),
      aNext: new Vector(1)
    };
    const step = createVelocityVerlet((u, _v, aOut) => aOut.set(0, -u.get(0)));
    const energy = () => 0.5 * state.v.get(0) ** 2 + 0.5 * state.u.get(0) ** 2;

    for (let i = 0; i < 10000; i++) step(state, 0.01);

    expect(Math.abs(energy() - 0.5)).toBeLessThan(1e-4);
  });
});
