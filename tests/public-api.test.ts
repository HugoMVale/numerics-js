import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { linalg, integrate, interpolate, math, numdiff, ode, optimize, roots, special } from '../src/index.js';

describe('public API', () => {
    it('exports the main numerical library modules', () => {
        expect(linalg.Vec3).toBeTypeOf('function');
        expect(linalg.Vector).toBeTypeOf('function');
        expect(linalg.Matrix).toBeTypeOf('function');
        expect(typeof integrate.trapezoid).toBe('function');
        expect(typeof integrate.simpson).toBe('function');
        expect(typeof integrate.gaussKronrod).toBe('function');
        expect(typeof interpolate.interp).toBe('function');
        expect(interpolate.LinearInterpolator).toBeTypeOf('function');
        expect(interpolate.PchipInterpolator).toBeTypeOf('function');
        expect(typeof math.clip).toBe('function');
        expect(typeof math.copysign).toBe('function');
        expect(typeof math.isClose).toBe('function');
        expect(typeof numdiff.derivativeCentered).toBe('function');
        expect(typeof numdiff.jacobianForward).toBe('function');
        expect(typeof numdiff.scaleVector).toBe('function');
        expect(typeof special.bessel.J).toBe('function');
        expect(typeof special.bessel.getZero).toBe('function');
        expect(typeof roots.bisection).toBe('function');
        expect(typeof roots.brent).toBe('function');
        expect(typeof roots.secant).toBe('function');
        expect(typeof ode.rungeKuttaFixed).toBe('function');
        expect(typeof ode.rungeKuttaAdaptive).toBe('function');
        expect(typeof ode.createVelocityVerlet).toBe('function');
        expect(typeof ode.wrapAllocatingDerivative).toBe('function');
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

        for (const moduleName of ['linalg', 'integrate', 'interpolate', 'math', 'numdiff', 'ode', 'optimize', 'roots', 'special']) {
            expect(pkg.exports[`./${moduleName}`]).toMatchObject({
                types: `./dist/${moduleName}.d.ts`,
                browser: `./dist/${moduleName}.js`,
                import: `./dist/${moduleName}.js`,
                default: `./dist/${moduleName}.js`,
            });
        }
    });
});