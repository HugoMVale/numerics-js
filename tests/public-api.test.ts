import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { array, ode, optimize, roots, special } from '../src/index.js';

describe('public API', () => {
    it('exports the main numerical library modules', () => {
        expect(array.Vec3).toBeTypeOf('function');
        expect(array.Array1D).toBeTypeOf('function');
        expect(array.Array2D).toBeTypeOf('function');
        expect(typeof special.bessel.J).toBe('function');
        expect(typeof roots.bisection).toBe('function');
        expect(typeof roots.secant).toBe('function');
        expect(typeof ode.rk4Step).toBe('function');
        expect(typeof ode.rk4Integrate).toBe('function');
        expect(typeof ode.createVelocityVerlet).toBe('function');
        expect(typeof optimize.brent).toBe('function');
        expect(typeof optimize.nelderMead).toBe('function');
    });

    it('publishes a browser + node compatible export map', () => {
        const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
        const rootExport = pkg.exports['.'];

        expect(rootExport).toMatchObject({
            types: './dist/index.d.ts',
            browser: './dist/index.js',
            import: './dist/index.js',
            default: './dist/index.js',
        });
    });
});