import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { rungeKuttaAdaptive, rungeKuttaAdaptiveStep } from '../../src/ode/rungeKuttaAdaptive.js';
import type { DerivativeFunction, RungeKuttaAdaptiveMethod } from '../../src/ode/types.js';
import { vectorLikeForms, mutableVectorLikeForms } from '../vectorLikeForms.js';

const methods = ['rk23', 'rk45'] as const;

describe('rungeKuttaAdaptive additional coverage', () => {
  it('returns the initial state unchanged when tEnd === t0', () => {
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = -y.data[0];
      return out;
    };
    const y0 = new Vector([3, 4]);
    const result = rungeKuttaAdaptive('rk45', f, 2, 2, y0);

    expect(result.t.size).toBe(1);
    expect(result.t.data[0]).toBe(2);
    expect(result.y.rows).toBe(1);
    expect(result.y.cols).toBe(2);
    expect(Array.from(result.y.row(0).data)).toEqual([3, 4]);
    expect(result.evaluations).toBe(0);
    expect(result.success).toBe(true);
    expect(result.message).toBe('Integration successful.');
  });

  it.each(methods)('(%s) reports a successful integration with a message', (method) => {
    const f: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);
    const result = rungeKuttaAdaptive(method, f, 0, 1, new Vector([1]), { h0: 0.1 });

    expect(result.success).toBe(true);
    expect(result.message).toBe('Integration successful.');
  });

  it.each(methods)('(%s) reports zero Jacobian evaluations and LU decompositions', (method) => {
    const f: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);
    const result = rungeKuttaAdaptive(method, f, 0, 1, new Vector([1]));
    const trivial = rungeKuttaAdaptive(method, f, 1, 1, new Vector([1]));

    expect(result.evaluationsJacobian).toBe(0);
    expect(result.decompositions).toBe(0);
    expect(trivial.evaluationsJacobian).toBe(0);
    expect(trivial.decompositions).toBe(0);
  });

  it.each(methods)('(%s) accepts an explicit initial step size h0', (method) => {
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = -y.data[0];
      return out;
    };
    const y0 = new Vector([1]);
    const result = rungeKuttaAdaptive(method, f, 0, 1, y0, {
      atol: 1e-8,
      rtol: 1e-8,
      h0: 0.01
    });

    const finalY = result.y.row(result.y.rows - 1).data[0];
    expect(finalY).toBeCloseTo(Math.exp(-1), 6);
  });

  it.each(methods)('(%s) reports every derivative evaluation', (method) => {
    let calls = 0;
    const f: DerivativeFunction = (_t, y, out) => {
      calls++;
      out.data[0] = -y.data[0];
      return out;
    };
    const result = rungeKuttaAdaptive(method, f, 0, 1, new Vector([1]), {
      atol: 1e-8,
      rtol: 1e-8,
      h0: 0.01
    });

    expect(result.evaluations).toBe(calls);
    expect(result.evaluations).toBeGreaterThan(0);
  });

  it.each(methods)('(%s) rejects an overly large first step and shrinks h to succeed', (method) => {
    // Nonlinear logistic growth: dy/dt = 10*y*(1-y). A first step of 0.5
    // (half the whole integration domain) is far too coarse for this
    // tolerance and forces the controller through several reject+shrink
    // cycles before converging.
    //
    // Note: a linear f = -y is a poor choice for this kind of test --
    // for the exponential test equation the BS23 embedded error
    // estimator is an exact polynomial in h*lambda that can (and does,
    // e.g. at h*lambda = -1) pass through zero for a finite step, so a
    // wildly inaccurate single step can be "accepted" with err === 0.
    // That's a property of the estimator on the linear test problem,
    // not a bug, but it makes a flaky/misleading regression test.
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = 10 * y.data[0] * (1 - y.data[0]);
      return out;
    };
    const y0 = new Vector([0.01]);
    const result = rungeKuttaAdaptive(method, f, 0, 1, y0, {
      atol: 1e-8,
      rtol: 1e-8,
      h0: 0.5
    });

    // Reference solution via the closed-form logistic curve.
    const exact = 1 / (1 + (1 / 0.01 - 1) * Math.exp(-10));
    const finalY = result.y.row(result.y.rows - 1).data[0];

    expect(finalY).toBeCloseTo(exact, 6);
    // A handful of accepted points implies rejection+shrink cycles
    // happened along the way -- a single oversized step could not have
    // produced this many accepted intermediate points.
    expect(result.t.size).toBeGreaterThan(5);
  });

  it('returns an unsuccessful result when step size underflows below hMin', () => {
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = Math.sin(1 / (y.data[0] + 1e-3)) * 1e8;
      return out;
    };
    const y0 = new Vector([1]);

    const result = rungeKuttaAdaptive('rk45', f, 0, 1, y0, {
      atol: 1e-14,
      rtol: 1e-14,
      hMin: 1e-9
    });
    expect(result.success).toBe(false);
    expect(result.message).toMatch(/underflowed below hMin/);
    expect(result.t.size).toBe(result.y.rows);
  });

  it('returns an unsuccessful result when maxSteps is exceeded', () => {
    const f: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);
    const result = rungeKuttaAdaptive('rk45', f, 0, 1, new Vector([1]), {
      h0: 0.1,
      hMax: 0.1,
      maxSteps: 1
    });

    expect(result.success).toBe(false);
    expect(result.message).toBe('rungeKuttaAdaptive: exceeded maxSteps (1) without reaching tEnd.');
    expect(result.t.size).toBe(result.y.rows);
    expect(result.t.data[result.t.size - 1]).toBeCloseTo(0.1, 12);
  });

  it('succeeds when the final allowed attempt reaches tEnd', () => {
    const f: DerivativeFunction = (_t, _y, out) => out.fill(0);
    const result = rungeKuttaAdaptive('rk45', f, 0, 1, new Vector([1]), {
      h0: 1,
      maxSteps: 1
    });

    expect(result.success).toBe(true);
    expect(result.t.data[result.t.size - 1]).toBe(1);
  });

  describe('input validation', () => {
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = -y.data[0];
      return out;
    };
    const y0 = new Vector([1]);

    it('rejects h0 = 0', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, { atol: 1e-6, rtol: 1e-6, h0: 0 })
      ).toThrow(RangeError);
    });

    it.each([
      { atol: Number.NaN, rtol: 1e-6 },
      { atol: 1e-6, rtol: Number.NaN },
      { atol: Number.POSITIVE_INFINITY, rtol: 1e-6 },
      { atol: 1e-6, rtol: Number.POSITIVE_INFINITY }
    ])('rejects non-finite tolerances ($atol, $rtol)', (options) => {
      expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, options)).toThrow(RangeError);
    });

    it('rejects non-positive hMin', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, { atol: 1e-6, rtol: 1e-6, hMin: 0 })
      ).toThrow(RangeError);
    });

    it('rejects non-positive hMax', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, { atol: 1e-6, rtol: 1e-6, hMax: -1 })
      ).toThrow(RangeError);
    });

    it('rejects hMin > hMax', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, {
          atol: 1e-6,
          rtol: 1e-6,
          hMax: 1,
          hMin: 2
        })
      ).toThrow(RangeError);
    });

    it('rejects non-positive maxSteps', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, { atol: 1e-6, rtol: 1e-6, maxSteps: 0 })
      ).toThrow(RangeError);
    });

    it('rejects non-integer maxSteps', () => {
      expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, { maxSteps: 1.5 })).toThrow(RangeError);
    });

    it('rejects safety outside (0, 1]', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, { atol: 1e-6, rtol: 1e-6, safety: 1.5 })
      ).toThrow(RangeError);
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, { atol: 1e-6, rtol: 1e-6, safety: 0 })
      ).toThrow(RangeError);
    });

    it('rejects minScale > maxScale', () => {
      expect(() =>
        rungeKuttaAdaptive('rk45', f, 0, 1, y0, {
          atol: 1e-6,
          rtol: 1e-6,
          minScale: 5,
          maxScale: 2
        })
      ).toThrow(RangeError);
    });
  });

  it('rungeKuttaAdaptiveStep throws for an unrecognized method', () => {
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = -y.data[0];
      return out;
    };
    const y0 = new Vector([1]);
    const out = new Vector(1);
    const scratch = {
      k1: new Vector(1),
      k2: new Vector(1),
      k3: new Vector(1),
      k4: new Vector(1),
      k5: new Vector(1),
      k6: new Vector(1),
      k7: new Vector(1),
      yTemp: new Vector(1)
    };

    expect(() =>
      rungeKuttaAdaptiveStep(
        'bogus' as unknown as RungeKuttaAdaptiveMethod,
        f,
        0,
        y0,
        0.1,
        out,
        scratch
      )
    ).toThrow(RangeError);
  });

  it('throws for an unrecognized method at the top level', () => {
    const f: DerivativeFunction = (t, y, out) => {
      out.data[0] = -y.data[0];
      return out;
    };
    const y0 = new Vector([1]);

    expect(() =>
      rungeKuttaAdaptive('bogus' as unknown as RungeKuttaAdaptiveMethod, f, 0, 1, y0)
    ).toThrow(RangeError);
  });

  it.each(methods)(
    '(%s) produces a monotonically increasing t for forward integration',
    (method) => {
      const f: DerivativeFunction = (t, y, out) => {
        out.data[0] = -y.data[0];
        return out;
      };
      const y0 = new Vector([1]);
      const result = rungeKuttaAdaptive(method, f, 0, 5, y0, { atol: 1e-6, rtol: 1e-6 });

      for (let i = 1; i < result.t.size; i++) {
        expect(result.t.data[i]).toBeGreaterThan(result.t.data[i - 1]);
      }
      expect(result.y.rows).toBe(result.t.size);
      expect(result.y.cols).toBe(1);
    }
  );
});

describe('rungeKuttaAdaptive with a VectorLike initial state', () => {
  // Uncoupled decays with different rates, so that every component matters.
  const pair: DerivativeFunction = (_t, y, out) => {
    out.data[0] = -y.data[0];
    out.data[1] = -2 * y.data[1];
    return out;
  };
  const y0Raw = [1, 3];

  for (const method of methods) {
    const ref = rungeKuttaAdaptive(method, pair, 0, 1, Vector.from(y0Raw));

    for (const [name, make] of vectorLikeForms) {
      it(`(${method}) accepts y0 as ${name}`, () => {
        const res = rungeKuttaAdaptive(method, pair, 0, 1, make(y0Raw));
        expect(res.success).toBe(true);
        expect(res.t.toArray()).toEqual(ref.t.toArray());
        expect(res.y.toArray()).toEqual(ref.y.toArray());
        expect(res.evaluations).toBe(ref.evaluations);
      });
    }
  }

  for (const [name, make] of mutableVectorLikeForms) {
    it(`does not mutate y0 given as ${name}`, () => {
      const y0 = make(y0Raw);
      rungeKuttaAdaptive('rk45', pair, 0, 1, y0);
      expect(Array.from(y0)).toEqual(y0Raw);
    });
  }

  it('does not mutate y0 given as a Vector when the integration stops at a terminal event', () => {
    const y0 = Vector.from([1, 0]);
    const oscillator: DerivativeFunction = (_t, y, out) => {
      out.data[0] = y.data[1];
      out.data[1] = -y.data[0];
      return out;
    };
    const res = rungeKuttaAdaptive('rk45', oscillator, 0, 10, y0, {
      events: [{ fn: (_t, y) => y.data[0], terminal: true }]
    });
    expect(res.events).toHaveLength(1);
    expect(y0.toArray()).toEqual([1, 0]);
  });

  it('hands the derivative function a Vector, whatever the form of y0', () => {
    for (const [, make] of vectorLikeForms) {
      const seen: unknown[] = [];
      const f: DerivativeFunction = (t, y, out) => {
        seen.push(y);
        return pair(t, y, out);
      };
      rungeKuttaAdaptive('rk45', f, 0, 1, make(y0Raw));
      expect(seen.length).toBeGreaterThan(0);
      for (const y of seen) expect(y).toBeInstanceOf(Vector);
    }
  });

  for (const [name, make] of vectorLikeForms) {
    it(`returns the initial state for tEnd === t0 with y0 as ${name}`, () => {
      const res = rungeKuttaAdaptive('rk45', pair, 2, 2, make([3, 4]), { denseOutput: true });
      expect(res.t.toArray()).toEqual([2]);
      expect(res.y.toArray()).toEqual([[3, 4]]);
      expect(res.denseOutput!.eval(2).toArray()).toEqual([3, 4]);
    });

    it(`supports events and dense output with y0 as ${name}`, () => {
      const res = rungeKuttaAdaptive('rk45', pair, 0, 1, make(y0Raw), {
        denseOutput: true,
        events: [{ fn: (_t, y) => y.data[0] - 0.5 }]
      });
      expect(res.events).toHaveLength(1);
      expect(res.events![0].t).toBeCloseTo(Math.log(2), 3);
      expect(res.denseOutput!.eval(0.5).get(0)).toBeCloseTo(Math.exp(-0.5), 3);
    });
  }

  it('throws a TypeError for an unsupported y0', () => {
    expect(() => rungeKuttaAdaptive('rk45', pair, 0, 1, 'abc' as never)).toThrowError(TypeError);
    expect(() => rungeKuttaAdaptive('rk45', pair, 0, 1, null as never)).toThrowError(
      /Vector\.asVector/
    );
    expect(() => rungeKuttaAdaptive('rk45', pair, 0, 1, {} as never)).toThrowError(TypeError);
  });

  it('throws a RangeError for an empty y0, whatever the form', () => {
    for (const [, make] of vectorLikeForms) {
      expect(() => rungeKuttaAdaptive('rk45', pair, 0, 1, make([]))).toThrowError(RangeError);
    }
  });
});
