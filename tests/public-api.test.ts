import { describe, expect, it } from 'vitest';
import {
    Array1D,
    Array2D,
    Vec3,
    bessel,
    bisection,
    secant,
    rk4Integrate,
    rk4Step,
    createVelocityVerlet,
} from '../src/index.js';

describe('public API', () => {
    it('exports the main numerical library modules', () => {
        expect(Vec3).toBeTypeOf('function');
        expect(Array1D).toBeTypeOf('function');
        expect(Array2D).toBeTypeOf('function');
        expect(typeof bessel.J).toBe('function');
        expect(typeof bisection).toBe('function');
        expect(typeof secant).toBe('function');
        expect(typeof rk4Step).toBe('function');
        expect(typeof rk4Integrate).toBe('function');
        expect(typeof createVelocityVerlet).toBe('function');
    });
});