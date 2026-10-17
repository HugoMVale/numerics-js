import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { curveFit } from '../../src/optimize/curveFit.js';
import fixtures from './curveFit.scipy.json' with { type: 'json' };

// Must mirror the models in generate_curve_fit_fixtures.py.
const MODELS: Record<string, (x: Vector, p: Vector) => Vector> = {
  exp_decay: (x, p) => x.map((xi) => p.get(0) * Math.exp(-p.get(1) * xi) + p.get(2)),
  gauss: (x, p) => x.map((xi) => p.get(0) * Math.exp(-(((xi - p.get(1)) / p.get(2)) ** 2) / 2)),
  logistic_sigma: (x, p) => x.map((xi) => p.get(0) / (1 + Math.exp(-p.get(1) * (xi - p.get(2))))),
  line_absolute_sigma: (x, p) => x.map((xi) => p.get(0) + p.get(1) * xi),
  michaelis_menten_sigma: (x, p) => x.map((xi) => (p.get(0) * xi) / (p.get(1) + xi)),
  power_law: (x, p) => x.map((xi) => p.get(0) * xi ** p.get(1))
};

const MIRROR_TOLERANCE = 1e-12;
const P_TOLERANCE = 1e-6;
const RSS_TOLERANCE = 1e-9;
// Like SciPy, curveFit computes the covariance from the Jacobian of the last iteration by default
// (`jacFinal: false`), so the two agree to ~1e-7 or better on these cases (see the benchmark report).
// This is not the exact covariance, which deviates from SciPy's by up to 1.5e-5, scaled by
// sqrt(cov_ii*cov_jj), as SciPy's Jacobian is evaluated one step before the solution.
const COV_TOLERANCE = 1e-6;
// Both solvers stop within their tolerances (~1.5e-8) of the minimum, so the fitted curves agree to
// that level relative to the scale of the data.
const CURVE_TOLERANCE = 1e-7;

describe('models mirror the SciPy reference models', () => {
  for (const fixture of fixtures) {
    it(`${fixture.id}: the model agrees with the Python implementation at generic parameters`, () => {
      const model = MODELS[fixture.id];
      expect(model, `no matching model registered for id "${fixture.id}"`).toBeDefined();

      const xdata = Vector.from(fixture.xdata);
      for (const point of fixture.mirrorPoints) {
        const fx = model(xdata, Vector.from(point.p));
        expect(fx.size).toBe(point.fx.length);
        point.fx.forEach((expected, i) => {
          expect(Math.abs(fx.get(i) - expected)).toBeLessThanOrEqual(
            MIRROR_TOLERANCE * Math.max(Math.abs(expected), 1)
          );
        });
      }
    });
  }
});

describe('curveFit vs scipy.optimize.curve_fit (method="lm") reference values', () => {
  for (const fixture of fixtures) {
    it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
      const model = MODELS[fixture.id];
      expect(model, `no matching model registered for id "${fixture.id}"`).toBeDefined();

      const xdata = Vector.from(fixture.xdata);
      const ydata = Vector.from(fixture.ydata);
      const p0 = Vector.from(fixture.p0);
      const result = curveFit(model, xdata, ydata, p0, {
        tolf: fixture.tolf,
        tolx: fixture.tolx,
        tolg: fixture.tolg,
        absoluteSigma: fixture.absoluteSigma,
        ...(fixture.sigma === null ? {} : { sigma: Vector.from(fixture.sigma) })
      });
      const n = fixture.p0.length;

      expect(result.success, result.message).toBe(true);
      expect(result.dof).toBe(fixture.xdata.length - n);

      for (let i = 0; i < n; i++) {
        expect(Math.abs(result.p.get(i) - fixture.scipyP[i])).toBeLessThanOrEqual(
          P_TOLERANCE * Math.max(Math.abs(fixture.scipyP[i]), 1e-3)
        );
      }
      expect(Math.abs(result.rss - fixture.scipyRss)).toBeLessThanOrEqual(
        RSS_TOLERANCE * fixture.scipyRss
      );
      const dataScale = Math.max(...fixture.ydata.map(Math.abs));
      for (let i = 0; i < fixture.xdata.length; i++) {
        expect(Math.abs(result.residuals.get(i) - fixture.scipyResiduals[i])).toBeLessThanOrEqual(
          CURVE_TOLERANCE * dataScale
        );
      }

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const scale = Math.sqrt(fixture.scipyCov[i][i] * fixture.scipyCov[j][j]);
          expect(Math.abs(result.cov.get(i, j) - fixture.scipyCov[i][j])).toBeLessThanOrEqual(
            COV_TOLERANCE * scale
          );
        }
        expect(Math.abs(result.stderr.get(i) - fixture.scipyStderr[i])).toBeLessThanOrEqual(
          COV_TOLERANCE * fixture.scipyStderr[i]
        );
      }
    });
  }
});
