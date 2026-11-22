import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vec3 } from '../../src/linalg/Vec3.js';
import { Vector } from '../../src/linalg/Vector.js';

/**
 * Input policy of `Vector` (and the norm range guarantees it shares with
 * `Matrix` and `Vec3`): index checks that behave like `Matrix.get`/`set`,
 * `TypeError`s (not native errors or misleading `RangeError`s) for
 * wrong-typed arguments, and norms that neither overflow nor underflow.
 */

describe('Vector index checks match Matrix', () => {
  const badIndices: number[] = [NaN, 1.5, -1, 3, Infinity, -Infinity, 2 ** 53];

  it.each(badIndices)('get(%s) and set(%s, x) throw a Vector-prefixed RangeError', (i) => {
    const v = Vector.from([1, 2, 3]);
    expect(() => v.get(i)).toThrowError(RangeError);
    expect(() => v.get(i)).toThrowError(`Vector.get: index ${i} out of bounds for dimension 3.`);
    expect(() => v.set(i, 9)).toThrowError(RangeError);
    expect(() => v.set(i, 9)).toThrowError(`Vector.set: index ${i} out of bounds for dimension 3.`);
    expect(v.toArray()).toEqual([1, 2, 3]);
  });

  it.each(badIndices)(
    'Matrix.get(0, %s) rejects the same indices with the same error type',
    (j) => {
      const m = Matrix.from([[1, 2, 3]]);
      expect(() => m.get(0, j)).toThrowError(RangeError);
      expect(() => m.set(0, j, 9)).toThrowError(RangeError);
    }
  );

  it('accepts every valid index, including -0', () => {
    const v = Vector.from([1, 2, 3]);
    expect(v.get(-0)).toBe(1);
    expect(v.get(2)).toBe(3);
    expect(v.set(-0, 7).get(0)).toBe(7);
  });

  it('rejects non-numeric indices in get without a native error', () => {
    const v = Vector.from([1, 2, 3]);
    for (const i of [null, undefined, '1', {}] as unknown[]) {
      expect(() => v.get(i as number)).toThrowError(/^Vector\.get: index .* out of bounds/);
    }
  });
});

describe('Vector constructor and from reject wrong-typed input', () => {
  const bad: [string, unknown][] = [
    ['null', null],
    ['undefined', undefined],
    ['a string', 'abc'],
    ['a boolean', true],
    ['an object without length', { 0: 1 }]
  ];

  it.each(bad)('new Vector(%s) throws a Vector.constructor TypeError', (_, x) => {
    expect(() => new Vector(x as never)).toThrowError(TypeError);
    expect(() => new Vector(x as never)).toThrowError(
      /^Vector\.constructor: input must be a dimension or an array-like of numbers, got /
    );
  });

  it.each(bad)('Vector.from(%s) throws a Vector.from TypeError', (_, x) => {
    expect(() => Vector.from(x as never)).toThrowError(TypeError);
    expect(() => Vector.from(x as never)).toThrowError(/^Vector\.from: arr must be /);
  });

  it('Vector.from(number) no longer silently builds a vector of zeros', () => {
    expect(() => Vector.from(5 as never)).toThrowError(
      /^Vector\.from: arr must be .*, got number\.$/
    );
    expect(new Vector(5).toArray()).toEqual([0, 0, 0, 0, 0]);
  });

  it.each([1.5, -1, NaN, Infinity])('an array-like of length %s is a RangeError', (length) => {
    expect(() => new Vector({ length })).toThrowError(RangeError);
    expect(() => new Vector({ length })).toThrowError(
      /^Vector\.constructor: input length must be a non-negative integer, got /
    );
  });

  it('oversized dimensions give a Vector.constructor message, not a native error', () => {
    for (const n of [2 ** 32, 2 ** 40, 1e10, 2 ** 53]) {
      expect(() => new Vector(n)).toThrowError(
        `Vector.constructor: a vector of size ${n} is too large to allocate.`
      );
      expect(() => Vector.zero(n)).toThrowError(/^Vector\.constructor: /);
    }
    expect(() => new Vector({ length: 2 ** 40 })).toThrowError(/^Vector\.constructor: .*too large/);
  });

  it('keeps accepting iterables without a length', () => {
    expect(new Vector(new Set([3, 4]) as never).toArray()).toEqual([3, 4]);
  });
});

describe('Vector.set(values)', () => {
  it('accepts a Vector, like every other VectorLike input', () => {
    const v = Vector.zero(2);
    expect(v.set(Vector.from([7, 8])).toArray()).toEqual([7, 8]);
  });

  it('rejects a non-array-like with a Vector.set TypeError', () => {
    const v = Vector.zero(2);
    for (const x of [null, undefined, 'ab', true] as unknown[]) {
      expect(() => v.set(x as never)).toThrowError(TypeError);
      expect(() => v.set(x as never)).toThrowError(/^Vector\.set: values must be /);
    }
  });

  it('still rejects a length mismatch with a RangeError', () => {
    expect(() => Vector.zero(2).set([1, 2, 3])).toThrowError(
      'Vector.set: expected 2 values, got 3.'
    );
  });
});

describe('Vector operands of the wrong type', () => {
  const v = (): Vector => Vector.from([1, 2]);
  const wrong: [string, unknown][] = [
    ['a number[]', [1, 2]],
    ['a Float64Array', new Float64Array([1, 2])],
    ['null', null],
    ['a Matrix', Matrix.identity(2)]
  ];
  const ops: [string, (a: Vector, x: never) => unknown][] = [
    ['add', (a, x) => a.add(x)],
    ['sub', (a, x) => a.sub(x)],
    ['mult', (a, x) => a.mult(x)],
    ['div', (a, x) => a.div(x)],
    ['addSelf', (a, x) => a.addSelf(x)],
    ['subSelf', (a, x) => a.subSelf(x)],
    ['dot', (a, x) => a.dot(x)],
    ['dist', (a, x) => a.dist(x)],
    ['isClose', (a, x) => a.isClose(x)],
    ['addScaledSelf', (a, x) => a.addScaledSelf(x, 1)],
    ['setSub', (a, x) => a.setSub(x, x)],
    ['outer', (a, x) => a.outer(x)]
  ];

  for (const [name, call] of ops) {
    it.each(wrong)(`${name}(%s) is a TypeError naming the method`, (_, x) => {
      expect(() => call(v(), x as never)).toThrowError(TypeError);
      expect(() => call(v(), x as never)).toThrowError(
        new RegExp(`^Vector\\.${name}: expected a Vector, got `)
      );
    });
  }

  it('documents and throws RangeError for a size mismatch in addScaledSelf, setSub and outer', () => {
    expect(() => v().addScaledSelf(Vector.zero(3), 1)).toThrowError(
      'Vector.addScaledSelf: dimension mismatch: 2 vs 3.'
    );
    expect(() => v().setSub(v(), Vector.zero(3))).toThrowError(
      'Vector.setSub: dimension mismatch: 2 vs 3.'
    );
    expect(() => new Vector(0).outer(v())).toThrowError(RangeError);
  });

  it('keeps scalar broadcasting and the RangeError for a size mismatch', () => {
    expect(v().add(1).toArray()).toEqual([2, 3]);
    expect(() => v().add(Vector.zero(3))).toThrowError('Vector.add: dimension mismatch: 2 vs 3.');
  });
});

describe('wrapUnchecked builds objects of the same shape as the constructors', () => {
  it('Vector: same own property names, in the same order', () => {
    const wrapped = Vector.wrapUnchecked(new Float64Array(3));
    expect(Object.getOwnPropertyNames(wrapped)).toEqual(Object.getOwnPropertyNames(new Vector(3)));
    expect(Object.getPrototypeOf(wrapped)).toBe(Vector.prototype);
    expect(wrapped.size).toBe(3);
  });

  it('Matrix: same own property names, in the same order', () => {
    const wrapped = Matrix.wrapUnchecked(2, 3, new Float64Array(6));
    expect(Object.getOwnPropertyNames(wrapped)).toEqual(
      Object.getOwnPropertyNames(new Matrix(2, 3))
    );
    expect(Object.getPrototypeOf(wrapped)).toBe(Matrix.prototype);
    expect([wrapped.rows, wrapped.cols]).toEqual([2, 3]);
  });

  it('arithmetic results are ordinary, fully functional vectors', () => {
    const r = Vector.from([1, 2]).add(Vector.from([3, 4]));
    expect(r).toBeInstanceOf(Vector);
    expect(r.toArray()).toEqual([4, 6]);
    expect(
      Vector.from([1, 2])
        .outer(Vector.from([3, 4, 5]))
        .toArray()
    ).toEqual([
      [3, 4, 5],
      [6, 8, 10]
    ]);
  });
});

describe('Euclidean norm range', () => {
  it('does not overflow for large components', () => {
    expect(Vector.from([3e200, 4e200]).norm()).toBeCloseTo(5e200, -190);
    expect(Vector.from([3e200, 4e200]).norm() / 5e200).toBeCloseTo(1, 14);
    expect(Matrix.from([[3e200, 4e200]]).norm() / 5e200).toBeCloseTo(1, 14);
    expect(Vector.from([3e200]).dist(Vector.from([-4e200])) / 7e200).toBeCloseTo(1, 14);
    expect(new Vec3(3e200, 4e200, 0).norm() / 5e200).toBeCloseTo(1, 14);
  });

  it('does not underflow for small components, so normalize() is not the zero vector', () => {
    expect(Vector.from([3e-200, 4e-200]).norm() / 5e-200).toBeCloseTo(1, 14);
    expect(Vector.from([3e-200, 4e-200]).normalize().toArray()).toEqual([
      expect.closeTo(0.6, 14),
      expect.closeTo(0.8, 14)
    ]);
    expect(new Vec3(3e-200, 4e-200, 0).norm() / 5e-200).toBeCloseTo(1, 14);
    expect(Vector.from([5e-324]).norm()).toBe(5e-324);
  });

  it('keeps the ordinary range bit-identical to sqrt(normSq)', () => {
    const v = Vector.from([0.1, 0.2, 0.3, 1e5, -7.25]);
    expect(v.norm()).toBe(Math.sqrt(v.normSq()));
    const w = new Vec3(0.1, -0.2, 0.3);
    expect(w.norm()).toBe(Math.sqrt(w.normSq()));
  });

  it('keeps the special values: zero, NaN and infinity', () => {
    expect(Vector.zero(3).norm()).toBe(0);
    expect(new Vector(0).norm()).toBe(0);
    expect(Vector.from([NaN, 1e200]).norm()).toBeNaN();
    expect(Vector.from([Infinity, NaN]).norm()).toBeNaN();
    expect(Vector.from([-Infinity, 1]).norm()).toBe(Infinity);
    expect(Vector.from([Infinity]).dist(Vector.from([Infinity]))).toBeNaN();
    expect(new Vec3(NaN, Infinity, 0).norm()).toBeNaN();
    expect(new Vec3(Infinity, 0, 0).norm()).toBe(Infinity);
  });

  it('reports a genuinely unrepresentable norm as Infinity', () => {
    expect(Vector.from([1.7e308, 1.7e308]).norm()).toBe(Infinity);
  });

  it('normalize() works when the norm is subnormal, where 1 / norm overflows', () => {
    // Subnormals carry fewer than 14 significant digits (granularity 5e-324), which
    // bounds the accuracy of the unit vector: hence 12 digits here, not 14.
    expect(Vector.from([5e-324, 0]).normalize().toArray()).toEqual([1, 0]);
    expect(Vector.from([1e-310, 1e-310]).normalize().norm()).toBeCloseTo(1, 12);
    expect(new Vec3(5e-324, 0, 0).normalize().toArray()).toEqual([1, 0, 0]);
    expect(new Vec3(1e-310, 1e-310, 0).normalize().norm()).toBeCloseTo(1, 12);
  });

  it('limit() and normalize() work on huge Vec3 values', () => {
    const u = new Vec3(3e200, 4e200, 0).normalize();
    expect(u.x).toBeCloseTo(0.6, 14);
    expect(u.y).toBeCloseTo(0.8, 14);
    expect(new Vec3(3e200, 4e200, 0).limit(1).norm()).toBeCloseTo(1, 14);
  });
});

describe('Vec3.from rejects wrong-typed input', () => {
  it('throws a Vec3.from TypeError instead of a native error or garbage', () => {
    for (const x of [null, undefined, 'abc', 5, true] as unknown[]) {
      expect(() => Vec3.from(x as never)).toThrowError(TypeError);
      expect(() => Vec3.from(x as never)).toThrowError(/^Vec3\.from: arr must be /);
    }
  });

  it('still pads short input and ignores extra entries', () => {
    expect(Vec3.from([1, 2]).toArray()).toEqual([1, 2, 0]);
    expect(Vec3.from([1, 2, 3, 4]).toArray()).toEqual([1, 2, 3]);
    expect(Vec3.from(new Float64Array([4, 5, 6])).toArray()).toEqual([4, 5, 6]);
  });
});
