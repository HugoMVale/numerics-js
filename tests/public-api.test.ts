import { readFileSync } from 'node:fs';
import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  linalg,
  integrate,
  interpolate,
  math,
  numdiff,
  ode,
  optimize,
  roots,
  special
} from '../src/index.js';

describe('public API', () => {
  it('exports the main numerical library modules', () => {
    expect(linalg.Vec3).toBeTypeOf('function');
    expect(linalg.Vector).toBeTypeOf('function');
    expect(linalg.Matrix).toBeTypeOf('function');
    expect(linalg.Vector.asVector).toBeTypeOf('function');
    expectTypeOf<linalg.VectorLike>().toEqualTypeOf<
      linalg.Vector | readonly number[] | Float64Array
    >();
    expect(typeof integrate.trapezoid).toBe('function');
    expect(typeof integrate.simpson).toBe('function');
    expect(typeof integrate.gaussKronrod).toBe('function');
    expect(typeof interpolate.interp1D).toBe('function');
    expect(typeof interpolate.interpND).toBe('function');
    expect(interpolate.LinearInterpolator1D).toBeTypeOf('function');
    expect(interpolate.LinearInterpolatorND).toBeTypeOf('function');
    expect(interpolate.PchipInterpolator1D).toBeTypeOf('function');
    expect(interpolate.PchipInterpolatorND).toBeTypeOf('function');
    expect(typeof math.clip).toBe('function');
    expect(typeof math.copysign).toBe('function');
    expect(typeof math.isClose).toBe('function');
    expect(typeof math.rand).toBe('function');
    expect(typeof numdiff.derivativeCentered).toBe('function');
    expect(typeof numdiff.jacobianForward).toBe('function');
    expect(typeof numdiff.scaleVector).toBe('function');
    expect(typeof special.besselJ).toBe('function');
    expect(typeof special.besselJZero).toBe('function');
    expect(typeof special.erf).toBe('function');
    expect(typeof special.erfc).toBe('function');
    expect(typeof roots.bisection).toBe('function');
    expect(typeof roots.brent).toBe('function');
    expect(typeof roots.secant).toBe('function');
    expect(typeof roots.quasiNewton).toBe('function');
    expect(typeof ode.rungeKuttaFixed).toBe('function');
    expect(typeof ode.rungeKuttaAdaptive).toBe('function');
    expect(typeof ode.trBdf2Adaptive).toBe('function');
    expect(typeof ode.createVelocityVerlet).toBe('function');
    expect(typeof ode.wrapAllocatingDerivative).toBe('function');
    expect(typeof optimize.brent).toBe('function');
    expect(typeof optimize.nelderMead).toBe('function');
    expect(typeof optimize.levenbergMarquardt).toBe('function');
    expect(typeof optimize.curveFit).toBe('function');
  });

  it('exposes dense output through the adaptive ode solvers, as a type only', () => {
    const f: ode.DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);
    const y0 = new linalg.Vector([1]);

    for (const result of [
      ode.rungeKuttaAdaptive('rk45', f, 0, 1, y0, { denseOutput: true }),
      ode.trBdf2Adaptive(f, 0, 1, y0, { denseOutput: true })
    ]) {
      const dense: ode.OdeDenseOutput = result.denseOutput!;
      expect(dense.eval(0.5).get(0)).toBeCloseTo(Math.exp(-0.5), 2);
    }
    // The class is only a type: it is obtained from a result and never constructed by users.
    expect('OdeDenseOutput' in ode).toBe(false);
  });

  it('publishes a browser + node compatible export map', () => {
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
    const rootExport = pkg.exports['.'];

    expect(rootExport).toMatchObject({
      types: './dist/index.d.ts',
      browser: './dist/index.js',
      import: './dist/index.js',
      default: './dist/index.js'
    });

    for (const moduleName of [
      'linalg',
      'integrate',
      'interpolate',
      'math',
      'numdiff',
      'ode',
      'optimize',
      'roots',
      'special'
    ]) {
      expect(pkg.exports[`./${moduleName}`]).toMatchObject({
        types: `./dist/${moduleName}.d.ts`,
        browser: `./dist/${moduleName}.js`,
        import: `./dist/${moduleName}.js`,
        default: `./dist/${moduleName}.js`
      });
    }
  });
});
