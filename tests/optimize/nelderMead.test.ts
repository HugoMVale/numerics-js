import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { nelderMead } from '../../src/optimize/nelderMead.js';
import { TEST_FUNCTIONS_MULTIVAR } from './testFunctions.js';
import { vectorLikeForms, mutableVectorLikeForms } from '../vectorLikeForms.js';

describe.each(Object.entries(TEST_FUNCTIONS_MULTIVAR))('nelderMead — %s', (_name, data) => {
  const N = 2; // test in 2D for simplicity, but should work in any dimension
  const tolx = 1e-6;

  it('converges to the known global minimum', () => {
    const x0 = data.initialPoint(N);
    const res = nelderMead(data.fn, x0, { tolx: tolx });

    expect(res.success, res.message).toBe(true);
    expect(Math.abs(res.fx - data.globalMinimum)).toBeLessThanOrEqual(2 * tolx);
  });

  it('stops early and reports success when the callback requests it', () => {
    const x0 = data.initialPoint(N);
    const res = nelderMead(data.fn, x0, {
      tolx: tolx,
      callback: (nIter) => ({ stop: nIter >= 3, success: true })
    });

    expect(res.success).toBe(true);
    expect(res.message).toMatch(/callback/i);
  });
});

describe('nelderMead — options and edge cases', () => {
  it('throws for a zero-dimensional initial guess', () => {
    expect(() => nelderMead(() => 0, Vector.from([]))).toThrow(RangeError);
  });

  it('accepts an Vector as the initial guess', () => {
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([3, -4]));
    expect(res.success).toBe(true);
    expect(res.fx).toBeCloseTo(0, 5);
  });

  it('shrinks the simplex when reflection and inside contraction both fail', () => {
    const objective = (x: Vector): number => {
      const u = x.get(0) / 0.05;
      const v = x.get(1) / 0.05;
      return u ** 2 + 16 * u ** 2 * (u - 1) ** 2 + v ** 2;
    };
    let shrunkenVertices: number[] = [];

    const res = nelderMead(objective, Vector.from([0, 0]), {
      maxIter: 1,
      callback: (nIter, simplex) => {
        if (nIter === 2) {
          shrunkenVertices = [
            simplex.get(1, 0),
            simplex.get(1, 1),
            simplex.get(2, 0),
            simplex.get(2, 1)
          ];
        }
        return { stop: false, success: false };
      }
    });

    expect(shrunkenVertices).toEqual([0.025, 0, 0, 0.025]);
    expect(res.evaluations).toBe(7);
  });

  it('shrinks the simplex when outside contraction fails on Rosenbrock', () => {
    const objective = (x: Vector): number =>
      (1 - x.get(0)) ** 2 + 100 * (x.get(1) - x.get(0) ** 2) ** 2;
    let shrunkenVertices: number[][] = [];

    const res = nelderMead(objective, Vector.from([-4, 0.5]), {
      callback: (nIter, simplex) => {
        if (nIter === 9) {
          shrunkenVertices = Array.from({ length: simplex.rows }, (_, i) => [
            simplex.get(i, 0),
            simplex.get(i, 1)
          ]);
        }
        return { stop: nIter >= 9, success: false };
      }
    });

    expect(shrunkenVertices).toHaveLength(3);
    expect(shrunkenVertices[0][0]).toBeCloseTo(0.0375, 8);
    expect(shrunkenVertices[0][1]).toBeCloseTo(0.678125, 8);
    expect(shrunkenVertices[1][0]).toBeCloseTo(0.80625, 8);
    expect(shrunkenVertices[1][1]).toBeCloseTo(0.7046875, 8);
    expect(shrunkenVertices[2][0]).toBeCloseTo(0.25, 8);
    expect(shrunkenVertices[2][1]).toBeCloseTo(0.7375, 8);
    expect(res.evaluations).toBe(21);
  });

  it('respects a custom maxiter and reports failure without throwing', () => {
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([10, 10]), {
      maxIter: 2
    });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/maximum number of iterations/i);
  });

  it('allows zero simplex updates when maxIter is zero', () => {
    const iterations: number[] = [];
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([10, 10]), {
      maxIter: 0,
      callback: (nIter) => {
        iterations.push(nIter);
        return { stop: false, success: false };
      }
    });

    expect(res.success).toBe(false);
    expect(res.message).toMatch(/maximum number of iterations/i);
    expect(res.evaluations).toBe(3);
    expect(iterations).toEqual([1]);
  });

  it('allows one simplex update when maxIter is one', () => {
    const iterations: number[] = [];
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([10, 10]), {
      maxIter: 1,
      callback: (nIter) => {
        iterations.push(nIter);
        return { stop: false, success: false };
      }
    });

    expect(res.success).toBe(false);
    expect(res.message).toMatch(/maximum number of iterations/i);
    expect(res.evaluations).toBeGreaterThan(3);
    expect(iterations).toEqual([1, 2]);
  });

  it.each([-1, 1.5])('rejects invalid maxIter values (%s)', (maxIter) => {
    expect(() => nelderMead(() => 0, Vector.from([1]), { maxIter })).toThrow(RangeError);
  });

  it('respects a custom maxfeval and reports failure without throwing', () => {
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([10, 10]), {
      maxFunEvals: 5
    });
    expect(res.success).toBe(false);
    expect(res.evaluations).toBeGreaterThanOrEqual(5);
    expect(res.message).toMatch(/maximum number of function evaluations/i);
  });

  it('honors a user-supplied sclx', () => {
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([1, 1]), {
      sclx: Vector.from([1, 1])
    });
    expect(res.success).toBe(true);
    expect(res.fx).toBeCloseTo(0, 5);
  });

  it('accepts a Vector for user-supplied sclx', () => {
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([1, 1]), {
      sclx: Vector.from([1, 1])
    });
    expect(res.success).toBe(true);
    expect(res.fx).toBeCloseTo(0, 5);
  });

  it('terminates when the scaled simplex distance is below tolx', () => {
    const res = nelderMead((x) => x.get(0) ** 2, Vector.from([0]), { tolx: 0.1, tolf: 0 });

    expect(res.success).toBe(true);
    expect(res.message).toMatch(/maximum distance between simplex vertices/i);
  });

  it('supports the non-adaptive parameter scheme', () => {
    const res = nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([5, -3]), {
      adaptive: false
    });
    expect(res.success).toBe(true);
    expect(res.fx).toBeCloseTo(0, 5);
  });

  it('passes the callback an (N+1) x N Matrix and an (N+1) Vector', () => {
    let seenRows = -1;
    let seenCols = -1;
    let seenFxDim = -1;

    nelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, Vector.from([5, -3]), {
      callback: (nIter, x: Matrix, fx: Vector) => {
        if (nIter === 1) {
          seenRows = x.rows;
          seenCols = x.cols;
          seenFxDim = fx.size;
        }
        return { stop: nIter >= 1, success: true };
      }
    });

    expect(seenRows).toBe(3); // N+1 vertices for N=2
    expect(seenCols).toBe(2);
    expect(seenFxDim).toBe(3);
  });
});

describe('nelderMead with VectorLike inputs', () => {
  // Badly scaled quadratic, so that `sclx` changes the initial simplex and therefore the path.
  const f = (x: Vector): number => (x.get(0) - 3) ** 2 + (x.get(1) - 2e4) ** 2 * 1e-6;
  const x0Raw = [1, 1e4];
  const scaleRaw = [1, 1e-4];
  const ref = nelderMead(f, Vector.from(x0Raw), { sclx: Vector.from(scaleRaw) });

  for (const [name, make] of vectorLikeForms) {
    it(`accepts x0 as ${name}`, () => {
      const plain = nelderMead(f, Vector.from(x0Raw));
      const res = nelderMead(f, make(x0Raw));
      expect(res.success, res.message).toBe(true);
      expect(res.x.toArray()).toEqual(plain.x.toArray());
      expect(res.evaluations).toBe(plain.evaluations);
    });

    it(`accepts sclx as ${name}`, () => {
      const res = nelderMead(f, x0Raw, { sclx: make(scaleRaw) });
      expect(res.success, res.message).toBe(true);
      expect(res.x.toArray()).toEqual(ref.x.toArray());
      expect(res.evaluations).toBe(ref.evaluations);
    });
  }

  it('uses the absolute value of sclx, whatever its form', () => {
    for (const [, make] of vectorLikeForms) {
      const res = nelderMead(f, x0Raw, { sclx: make([-1, -1e-4]) });
      expect(res.x.toArray()).toEqual(ref.x.toArray());
    }
  });

  for (const [name, make] of mutableVectorLikeForms) {
    it(`does not mutate x0 or sclx given as ${name}`, () => {
      const x0 = make(x0Raw);
      const sclx = make(scaleRaw);
      nelderMead(f, x0, { sclx });
      expect(Array.from(x0)).toEqual(x0Raw);
      expect(Array.from(sclx)).toEqual(scaleRaw);
    });
  }

  it('hands the objective function a Vector, whatever the form of x0', () => {
    for (const [, make] of vectorLikeForms) {
      const seen: unknown[] = [];
      nelderMead(
        (x) => {
          seen.push(x);
          return f(x);
        },
        make(x0Raw),
        { maxIter: 3 }
      );
      expect(seen.length).toBeGreaterThan(0);
      for (const x of seen) expect(x).toBeInstanceOf(Vector);
    }
  });

  it('throws a TypeError for unsupported inputs', () => {
    expect(() => nelderMead(f, 'abc' as never)).toThrowError(TypeError);
    expect(() => nelderMead(f, null as never)).toThrowError(/Vector\.asVector/);
    expect(() => nelderMead(f, x0Raw, { sclx: {} as never })).toThrowError(TypeError);
  });

  it('throws a RangeError for an empty x0, whatever the form', () => {
    for (const [, make] of vectorLikeForms) {
      expect(() => nelderMead(f, make([]))).toThrowError(RangeError);
    }
  });
});

describe('nelderMead sclx length', () => {
  const f = (x: Vector): number => (x.get(0) - 1) ** 2 + (x.get(1) - 2) ** 2;

  for (const [form, make] of vectorLikeForms) {
    for (const [label, values] of [
      ['shorter', [1]],
      ['longer', [1, 1, 1]]
    ] as const) {
      it(`rejects sclx given as ${form} that is ${label} than x0`, () => {
        let evaluated = false;
        const g = (x: Vector) => {
          evaluated = true;
          return f(x);
        };
        expect(() => nelderMead(g, [0, 0], { sclx: make([...values]) })).toThrowError(
          new RangeError(`nelderMead: sclx must have 2 components, got ${values.length}`)
        );
        expect(evaluated).toBe(false);
      });
    }
  }

  it('accepts sclx with exactly as many components as x0', () => {
    const res = nelderMead(f, [0, 0], { sclx: [1, 1] });
    expect(res.success, res.message).toBe(true);
  });
});
