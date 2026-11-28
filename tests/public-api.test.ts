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
    expect(typeof integrate.quad).toBe('function');
    expectTypeOf<integrate.QuadResult>().toEqualTypeOf<integrate.GaussKronrodResult>();
    expectTypeOf<integrate.QuadOptions>().toExtend<integrate.GaussKronrodOptions>();
    expectTypeOf<integrate.QuadTransform['map']>().toBeFunction();
    expect(typeof interpolate.interp1D).toBe('function');
    expect(typeof interpolate.interpND).toBe('function');
    expect(interpolate.LinearInterpolator1D).toBeTypeOf('function');
    expect(interpolate.LinearInterpolatorND).toBeTypeOf('function');
    expect(interpolate.PchipInterpolator1D).toBeTypeOf('function');
    expect(interpolate.PchipInterpolatorND).toBeTypeOf('function');
    expect(interpolate.CubicSplineInterpolator1D).toBeTypeOf('function');
    expect(interpolate.CubicSplineInterpolatorND).toBeTypeOf('function');
    expect(typeof math.clip).toBe('function');
    expect(typeof math.copysign).toBe('function');
    expect(typeof math.isClose).toBe('function');
    expect(typeof math.createRng).toBe('function');
    expect(Object.keys(math).sort()).toEqual(['clip', 'copysign', 'createRng', 'isClose']);
    expectTypeOf<math.CloseOptions>().toEqualTypeOf<{ atol?: number; rtol?: number }>();
    expectTypeOf<math.RngMethod>().toEqualTypeOf<'mulberry32'>();
    expect(typeof numdiff.derivativeCentered).toBe('function');
    expect(typeof numdiff.jacobianForward).toBe('function');
    expect(typeof numdiff.gradientForward).toBe('function');
    expect(typeof numdiff.hessianForward).toBe('function');
    expect(typeof numdiff.scaleVector).toBe('function');
    expect(typeof special.besselI).toBe('function');
    expect(typeof special.besselIScaled).toBe('function');
    expect(typeof special.besselJ).toBe('function');
    expect(typeof special.besselJZero).toBe('function');
    expect(typeof special.besselK).toBe('function');
    expect(typeof special.besselKScaled).toBe('function');
    expect(typeof special.besselY).toBe('function');
    expect(typeof special.besselYZero).toBe('function');
    expect(typeof special.erf).toBe('function');
    expect(typeof special.erfc).toBe('function');
    expect(typeof special.gamma).toBe('function');
    expect(typeof special.lgamma).toBe('function');
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
    expect(typeof optimize.quasiNewton).toBe('function');
    expect(typeof optimize.levenbergMarquardt).toBe('function');
    expect(typeof optimize.curveFit).toBe('function');
  });

  it('exposes the option and result types of every optimize solver', () => {
    const f = (x: linalg.Vector): number => x.get(0) ** 2 + x.get(1) ** 2;

    const brentOptions: optimize.BrentOptions = { tolx: 1e-6, maxIter: 100 };
    const brentResult: optimize.BrentResult = optimize.brent(
      (x) => (x - 1) ** 2,
      -3,
      3,
      brentOptions
    );
    expect(brentResult.success).toBe(true);

    const nelderMeadOptions: optimize.NelderMeadOptions = { tolx: 1e-6 };
    const nelderMeadResult: optimize.NelderMeadResult = optimize.nelderMead(
      f,
      [1, 1],
      nelderMeadOptions
    );
    expect(nelderMeadResult.success).toBe(true);

    const bfgsMethod: optimize.BfgsMethod = 'inverse';
    const quasiNewtonOptions: optimize.QuasiNewtonOptions = { bfgsMethod };
    const quasiNewtonResult: optimize.QuasiNewtonResult = optimize.quasiNewton(
      f,
      [1, 1],
      quasiNewtonOptions
    );
    expect(quasiNewtonResult.success).toBe(true);

    const lmOptions: optimize.LevenbergMarquardtOptions = { jacFinal: true };
    const lmResult: optimize.LevenbergMarquardtResult = optimize.levenbergMarquardt(
      (x) => x.add(linalg.Vector.from([-1])),
      [3],
      lmOptions
    );
    expect(lmResult.success).toBe(true);

    const curveFitOptions: optimize.CurveFitOptions = { absoluteSigma: true };
    const curveFitResult: optimize.CurveFitResult = optimize.curveFit(
      (t, p) => t.mult(p.get(0)),
      [1, 2, 3],
      [2, 4, 6],
      [1],
      curveFitOptions
    );
    expect(curveFitResult.success).toBe(true);
  });

  it('exports exactly the linalg classes and the tridiagonal solvers as values from linalg', () => {
    // The factorization kernels in Matrix.linalg.ts and the helpers in checks.ts and
    // scalar.ts are internal: they are reachable only through the Matrix methods.
    expect(Object.keys(linalg).sort()).toEqual([
      'BandedMatrix',
      'Matrix',
      'Vec3',
      'Vector',
      'solveTridiagonal',
      'solveTridiagonalLU',
      'tridiagonalLU'
    ]);
  });

  it('names the LU solve next to the other solves, and has no luSolve', () => {
    const A = linalg.Matrix.from([
      [1, 2],
      [3, 4]
    ]);
    const x = linalg.Matrix.solveLU(A.lu(), [5, 11]);
    expectTypeOf(x).toEqualTypeOf<linalg.Vector>();
    expect(x.toArray()).toEqual([1, 2]);
    expect('luSolve' in linalg.Matrix).toBe(false);
  });

  it('exposes the banded and tridiagonal solvers with their result types from linalg', () => {
    const B = linalg.BandedMatrix.fromMatrix(
      linalg.Matrix.from([
        [2, 1],
        [1, 3]
      ])
    );
    const dec: linalg.BandedLUDecomposition = B.lu();
    expectTypeOf(linalg.BandedMatrix.solveLU(dec, [1, 2])).toEqualTypeOf<linalg.Vector>();
    expectTypeOf(B.solve([1, 2])).toEqualTypeOf<linalg.Vector>();
    expectTypeOf(B.toMatrix()).toEqualTypeOf<linalg.Matrix>();
    const tri: linalg.TridiagonalLU = linalg.tridiagonalLU([1], [2, 3], [1]);
    expectTypeOf(linalg.solveTridiagonalLU(tri, [1, 2])).toEqualTypeOf<linalg.Vector>();
    expectTypeOf(linalg.solveTridiagonal([1], [2, 3], [1], [1, 2])).toEqualTypeOf<linalg.Vector>();
    // the in-place kernels used by the cubic splines are internal
    expect('factorTridiagonalInPlace' in linalg).toBe(false);
    expect('solveTridiagonalInPlace' in linalg).toBe(false);
  });

  it('exposes the SVD family on Matrix and its result types from linalg', () => {
    const A = linalg.Matrix.from([
      [3, 0],
      [4, 5]
    ]);
    const dec: linalg.SVDDecomposition = A.svd({ full: true } satisfies linalg.SVDOptions);
    expectTypeOf(dec.U).toEqualTypeOf<linalg.Matrix>();
    expectTypeOf(dec.s).toEqualTypeOf<linalg.Vector>();
    expectTypeOf(dec.V).toEqualTypeOf<linalg.Matrix>();

    const cutoff: linalg.SVDCutoffOptions = { atol: 0, rtol: 1e-12 };
    const fit: linalg.LstsqResult = A.lstsq([1, 2], cutoff);
    expectTypeOf(fit.x).toEqualTypeOf<linalg.Vector>();
    expect(fit.rank).toBe(2);
    expect(A.pinv(cutoff).rows).toBe(2);
    expect(A.singularValues().size).toBe(2);
    const rankOptions: linalg.RankOptions = { method: 'elimination', atol: 0 };
    expect(A.rank(rankOptions)).toBe(2);
    expect(A.rank(1e-12)).toBe(2);
    expect(A.cond2()).toBeGreaterThan(1);
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
