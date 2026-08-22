import { describe, it, expect } from 'vitest';
import { Array1D } from '../../src/array/array1d.js';
import { Array2D } from '../../src/array/array2d.js';
import { dp54Step, dp54Integrate } from '../../src/ode/dp54.js';
import { wrapAllocatingDerivative } from '../../src/ode/adapters.js';

// NOTE on paths: this assumes dp54.test.js sits next to dp54.js and rk4.js,
// with an index.js exporting Array1D/Array2D one directory up — mirroring
// rk4.js's own `import { Array1D, Array2D } from '../index.js'`. Adjust the
// three import paths above if your project lays files out differently.

/** Derivative function type for ODE solvers */
type DerivativeFn = (t: number, y: Array1D, dydt: Array1D) => Array1D;

/** dy/dt = -k*y, exact solution y(t) = y0 * exp(-k*t). */
function makeDecay(k: number = 1): DerivativeFn {
    return (t: number, y: Array1D, dydt: Array1D): Array1D => {
        dydt.data[0] = -k * y.data[0];
        return dydt;
    };
}

/** Harmonic oscillator as a first-order system: [y, v]' = [v, -y]. */
function oscillator(t: number, y: Array1D, dydt: Array1D): Array1D {
    dydt.data[0] = y.data[1];
    dydt.data[1] = -y.data[0];
    return dydt;
}

function closeTo(actual: number, expected: number, tol: number): void {
    expect(Math.abs(actual - expected)).toBeLessThan(tol);
}

describe('dp54Step', () => {
    it('matches the analytic solution of exponential decay to high accuracy over one step', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const out = new Array1D(1);
        dp54Step(f, 0, y0, 0.1, out);
        closeTo(out.data[0], Math.exp(-0.1), 1e-9);
    });

    it('does not mutate y', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const out = new Array1D(1);
        dp54Step(f, 0, y0, 0.1, out);
        expect(y0.data[0]).toBe(1);
    });

    it('works with an implicit (freshly allocated) scratch workspace when omitted', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const out = new Array1D(1);
        expect(() => dp54Step(f, 0, y0, 0.1, out)).not.toThrow();
        closeTo(out.data[0], Math.exp(-0.1), 1e-9);
    });

    it('produces an identical result whether k1 is recomputed or reused via k1Ready', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);

        const outFresh = new Array1D(1);
        dp54Step(f, 0, y0, 0.1, outFresh);

        const scratch = {
            k1: new Array1D(1),
            k2: new Array1D(1),
            k3: new Array1D(1),
            k4: new Array1D(1),
            k5: new Array1D(1),
            k6: new Array1D(1),
            k7: new Array1D(1),
            yTemp: new Array1D(1),
        };
        f(0, y0, scratch.k1); // pre-populate k1 = f(t, y) ourselves
        const outReused = new Array1D(1);
        dp54Step(f, 0, y0, 0.1, outReused, scratch, true);

        closeTo(outReused.data[0], outFresh.data[0], 1e-15);
    });

    it('leaves scratch.k7 equal to f(t + h, out) (FSAL contract)', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const out = new Array1D(1);
        const scratch = {
            k1: new Array1D(1),
            k2: new Array1D(1),
            k3: new Array1D(1),
            k4: new Array1D(1),
            k5: new Array1D(1),
            k6: new Array1D(1),
            k7: new Array1D(1),
            yTemp: new Array1D(1),
        };
        dp54Step(f, 0, y0, 0.1, out, scratch);

        const expected = new Array1D(1);
        f(0.1, out, expected);
        closeTo(scratch.k7.data[0], expected.data[0], 1e-15);
    });

    it('local error shrinks sharply (roughly h^6) when the step size is halved', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const exact = (t: number): number => Math.exp(-t);

        const h = 0.2;
        const outFull = new Array1D(1);
        dp54Step(f, 0, y0, h, outFull);
        const errFull = Math.abs(outFull.data[0] - exact(h));

        const outHalf = new Array1D(1);
        dp54Step(f, 0, y0, h / 2, outHalf);
        const errHalf = Math.abs(outHalf.data[0] - exact(h / 2));

        expect(errFull / errHalf).toBeGreaterThan(20);
    });
});

describe('dp54Integrate: accuracy', () => {
    it('matches exponential decay closely under tight tolerances', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const { t, y } = dp54Integrate(f, 0, 5, y0, { atol: 1e-10, rtol: 1e-10 });
        const last = y.row(t.dim - 1);
        closeTo(last.data[0], Math.exp(-5), 1e-8);
    });

    it('conserves amplitude for the harmonic oscillator over several periods', () => {
        const y0 = new Array1D([1, 0]);
        const { t, y } = dp54Integrate(oscillator, 0, 6 * Math.PI, y0, { atol: 1e-10, rtol: 1e-10 });
        const last = y.row(t.dim - 1);
        const amplitude = Math.hypot(last.data[0], last.data[1]);
        closeTo(amplitude, 1, 1e-6);
        closeTo(last.data[0], 1, 1e-6);
        closeTo(last.data[1], 0, 1e-6);
    });

    it('recovers the initial condition when integrating backward in time', () => {
        const f = makeDecay(1);
        const yAtFive = new Array1D([Math.exp(-5)]);
        const { t, y } = dp54Integrate(f, 5, 0, yAtFive, { atol: 1e-10, rtol: 1e-10 });
        expect(t.data[t.dim - 1]).toBe(0);
        closeTo(y.row(t.dim - 1).data[0], 1, 1e-7);
    });
});

describe('dp54Integrate: output shape and contracts', () => {
    it('returns t and y with matching, consistent dimensions', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const { t, y } = dp54Integrate(f, 0, 3, y0);
        expect(y.rows).toBe(t.dim);
        expect(y.cols).toBe(y0.dim);
    });

    it('records t0 exactly at the first recorded time', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const { t } = dp54Integrate(f, 0, 3, y0);
        expect(t.data[0]).toBe(0);
    });

    it('records tEnd exactly (bit-for-bit) as the final recorded time', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const { t } = dp54Integrate(f, 0, 1 / 3, y0);
        expect(t.data[t.dim - 1]).toBe(1 / 3);
    });

    it('stores the initial condition in row 0', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([2, -3]);
        const { y } = dp54Integrate(oscillator, 0, 1, new Array1D([2, -3]));
        const row0 = y.row(0);
        closeTo(row0.data[0], y0.data[0], 1e-15);
        closeTo(row0.data[1], y0.data[1], 1e-15);
    });

    it('does not mutate the caller-supplied y0', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        dp54Integrate(f, 0, 5, y0);
        expect(y0.data[0]).toBe(1);
    });

    it('handles t0 === tEnd by returning a single row equal to y0', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([3.14]);
        const { t, y } = dp54Integrate(f, 2, 2, y0);
        expect(t.dim).toBe(1);
        expect(t.data[0]).toBe(2);
        expect(y.rows).toBe(1);
        expect(y.row(0).data[0]).toBe(3.14);
    });
});

describe('dp54Integrate: adaptive step-size control', () => {
    it('takes more accepted steps, and achieves a smaller error, under a tighter tolerance', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const exact = Math.exp(-5);

        const loose = dp54Integrate(f, 0, 5, y0, { atol: 1e-3, rtol: 1e-3 });
        const tight = dp54Integrate(f, 0, 5, y0, { atol: 1e-11, rtol: 1e-11 });

        expect(tight.t.dim).toBeGreaterThan(loose.t.dim);

        const looseErr = Math.abs(loose.y.row(loose.t.dim - 1).data[0] - exact);
        const tightErr = Math.abs(tight.y.row(tight.t.dim - 1).data[0] - exact);
        expect(tightErr).toBeLessThan(looseErr);
    });

    it('honors an explicit h0 as the first attempted step size when it is readily accepted', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const { t } = dp54Integrate(f, 0, 10, y0, { h0: 0.05, atol: 1, rtol: 1 });
        closeTo(t.data[1] - t.data[0], 0.05, 1e-12);
    });

    it('respects hMax as a ceiling on step size', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        const { t } = dp54Integrate(f, 0, 100, y0, { atol: 1, rtol: 1, hMax: 0.5 });
        for (let i = 1; i < t.dim; i++) {
            expect(t.data[i] - t.data[i - 1]).toBeLessThanOrEqual(0.5 + 1e-9);
        }
    });
});

describe('dp54Integrate: error handling', () => {
    it('throws a RangeError when the required step size underflows below hMin', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        expect(() =>
            dp54Integrate(f, 0, 1, y0, { atol: 0, rtol: 0, h0: 0.5, hMin: 1e-6 })
        ).toThrow(RangeError);
    });

    it('throws when maxSteps is exceeded before reaching tEnd', () => {
        const f = makeDecay(1);
        const y0 = new Array1D([1]);
        expect(() =>
            dp54Integrate(f, 0, 100, y0, { hMax: 1e-3, maxSteps: 3 })
        ).toThrow(/maxSteps/);
    });

    it('rejects a mismatched initial-condition dimension via the underlying Array1D guards', () => {
        const badF: DerivativeFn = (t: number, y: Array1D, dydt: Array1D): Array1D => {
            dydt.set([1, 2]);
            return dydt;
        };
        const y0 = new Array1D([1]);
        expect(() => dp54Integrate(badF, 0, 1, y0)).toThrow();
    });
});

describe('dp54Integrate: multi-dimensional systems and composability', () => {
    it('integrates a linear system driven by Array2D#mulVec, matching the direct oscillator formulation', () => {
        const A = Array2D.from([
            [0, 1],
            [-1, 0],
        ]);
        const f = wrapAllocatingDerivative((t: number, y: Array1D) => A.mulVec(y));

        const y0 = new Array1D([1, 0]);
        const { t, y } = dp54Integrate(f, 0, 2 * Math.PI, y0, { atol: 1e-10, rtol: 1e-10 });
        const last = y.row(t.dim - 1);

        closeTo(last.data[0], 1, 1e-6);
        closeTo(last.data[1], 0, 1e-6);
    });

    it('agrees with a hand-written zero-allocation derivative for the same system', () => {
        const A = Array2D.from([
            [0, 1],
            [-1, 0],
        ]);
        const fAllocating = wrapAllocatingDerivative((t: number, y: Array1D) => A.mulVec(y));
        const fDirect = oscillator;

        const y0a = new Array1D([1, 0]);
        const y0b = new Array1D([1, 0]);
        const a = dp54Integrate(fAllocating, 0, 5, y0a, { atol: 1e-9, rtol: 1e-9 });
        const b = dp54Integrate(fDirect, 0, 5, y0b, { atol: 1e-9, rtol: 1e-9 });

        const rowA = a.y.row(a.t.dim - 1);
        const rowB = b.y.row(b.t.dim - 1);
        closeTo(rowA.data[0], rowB.data[0], 1e-6);
        closeTo(rowA.data[1], rowB.data[1], 1e-6);
    });

    it('handles a higher-dimensional decoupled system (independent exponential decays)', () => {
        const rates = [1, 2, 0.5, 3];
        const f: DerivativeFn = (t: number, y: Array1D, dydt: Array1D): Array1D => {
            for (let i = 0; i < y.dim; i++) dydt.data[i] = -rates[i] * y.data[i];
            return dydt;
        };
        const y0 = new Array1D([1, 1, 1, 1]);
        const { t, y } = dp54Integrate(f, 0, 2, y0, { atol: 1e-10, rtol: 1e-10 });
        const last = y.row(t.dim - 1);
        for (let i = 0; i < rates.length; i++) {
            closeTo(last.data[i], Math.exp(-rates[i] * 2), 1e-7);
        }
    });
});