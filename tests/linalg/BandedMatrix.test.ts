import { describe, expect, it } from 'vitest';
import { BandedMatrix, type BandedLUDecomposition } from '../../src/linalg/BandedMatrix.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { countFloat64ArrayAllocations } from './allocation.js';

/** Deterministic uniform generator on (-0.5, 0.5). */
function rng(seed: number): () => number {
  let s = seed;
  return () => (s = (s * 16807) % 2147483647) / 2147483647 - 0.5;
}

/** Dense matrix with random entries inside the band `kl`, `ku` and a small diagonal (forces pivoting). */
function randomBanded(n: number, kl: number, ku: number, rnd: () => number, diag = 0.05): Matrix {
  const A = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - kl); j <= Math.min(n - 1, i + ku); j++) {
      A.set(i, j, i === j ? diag * rnd() : rnd());
    }
  }
  return A;
}

function relDiff(x: Vector, y: Vector): number {
  return x.sub(y).normInf() / Math.max(y.normInf(), 1e-300);
}

describe('BandedMatrix construction and access', () => {
  it('stores LAPACK band storage and converts back to dense', () => {
    const A = Matrix.from([
      [4, 1, 0, 0],
      [2, 5, 1, 0],
      [7, 3, 6, 1],
      [0, 8, 2, 9]
    ]);
    const B = BandedMatrix.fromMatrix(A);
    expect([B.n, B.kl, B.ku, B.ldab]).toEqual([4, 2, 1, 4]);
    // column j holds rows ku + i - j: here the diagonal is row 1 of each column.
    expect(B.data[1]).toBe(4);
    expect(B.data[4]).toBe(1);
    expect(B.data[2]).toBe(2);
    expect(B.data[3]).toBe(7);
    expect(B.toMatrix().allClose(A, { atol: 0, rtol: 0 })).toBe(true);
    expect(B.get(2, 0)).toBe(7);
    expect(B.get(0, 3)).toBe(0);
    expect(B.get(3, 0)).toBe(0);
  });

  it('builds from band data like SciPy ab, ignoring the unused corners', () => {
    const ab = Matrix.from([
      [99, 1, 1],
      [4, 5, 6],
      [2, 3, 99]
    ]);
    const B = BandedMatrix.fromBands(ab, 1, 1);
    expect(B.toMatrix().toString()).toBe('Matrix[[4, 1, 0], [2, 5, 1], [0, 3, 6]]');
    expect(B.data[0]).toBe(0); // unused corner stays 0
    expect(B.data[B.data.length - 1]).toBe(0);
  });

  it('infers the band, or checks the entries against the one given', () => {
    const A = Matrix.from([
      [1, 2, 0],
      [0, 3, 4],
      [0, 0, 5]
    ]);
    const B = BandedMatrix.fromMatrix(A);
    expect([B.kl, B.ku]).toEqual([0, 1]);
    expect(BandedMatrix.fromMatrix(new Matrix(3, 3)).kl).toBe(0);
    expect(() => BandedMatrix.fromMatrix(A, 0, 0)).toThrow(RangeError);
    expect(() => BandedMatrix.fromMatrix(A, 0, 0)).toThrow(
      'BandedMatrix.fromMatrix: entry (0, 1) is 2 but lies outside the band (kl = 0, ku = 0).'
    );
    const wide = BandedMatrix.fromMatrix(A, 2, 2);
    expect([wide.kl, wide.ku]).toEqual([2, 2]);
    expect(wide.toMatrix().allClose(A)).toBe(true);
    // a NaN outside the band is a nonzero entry
    const bad = Matrix.from([
      [1, NaN],
      [0, 1]
    ]);
    expect(() => BandedMatrix.fromMatrix(bad, 0, 0)).toThrow(RangeError);
  });

  it('clamps bandwidths of n or more to n - 1', () => {
    const B = new BandedMatrix(3, 10, 7);
    expect([B.kl, B.ku, B.ldab, B.data.length]).toEqual([2, 2, 5, 15]);
    const ab = new Matrix(4, 2); // kl = 2, ku = 1, n = 2: the layout is not clamped
    ab.set(1, 0, 3).set(1, 1, 4).set(0, 1, 1).set(2, 0, 2);
    const C = BandedMatrix.fromBands(ab, 2, 1);
    expect([C.kl, C.ku]).toEqual([1, 1]);
    expect(C.toMatrix().toString()).toBe('Matrix[[3, 1], [2, 4]]');
  });

  it('gets and sets elements, rejecting what cannot be stored', () => {
    const B = new BandedMatrix(3, 1, 0);
    expect(B.set(1, 0, 5).set(2, 2, 7)).toBe(B);
    expect(B.get(1, 0)).toBe(5);
    expect(B.get(0, 2)).toBe(0);
    expect(() => B.set(0, 1, 1)).toThrow('BandedMatrix.set: index (0, 1) is outside the band');
    expect(() => B.set(2, 0, 1)).toThrow(RangeError);
    expect(() => B.get(3, 0)).toThrow(RangeError);
    expect(() => B.get(0, -1)).toThrow(RangeError);
    expect(() => B.get(0.5, 0)).toThrow(RangeError);
    expect(() => B.set(0, 3, 1)).toThrow(RangeError);
  });

  it('validates constructor and factory arguments', () => {
    expect(() => new BandedMatrix(0, 0, 0)).toThrow(RangeError);
    expect(() => new BandedMatrix(2.5, 0, 0)).toThrow(RangeError);
    expect(() => new BandedMatrix(3, -1, 0)).toThrow(RangeError);
    expect(() => new BandedMatrix(3, 0, 1.5)).toThrow(RangeError);
    expect(() => new BandedMatrix('3' as unknown as number, 0, 0)).toThrow(TypeError);
    expect(() => new BandedMatrix(3, 'a' as unknown as number, 0)).toThrow(TypeError);
    expect(() => new BandedMatrix(1e12, 1, 1)).toThrow(/too large to allocate/);
    expect(() => BandedMatrix.fromMatrix([[1]] as unknown as Matrix)).toThrow(TypeError);
    expect(() => BandedMatrix.fromMatrix(new Matrix(2, 3))).toThrow(
      'BandedMatrix.fromMatrix: expected a square matrix, got 2x3.'
    );
    expect(() => BandedMatrix.fromMatrix(new Matrix(2, 2), 1)).toThrow(RangeError);
    expect(() => BandedMatrix.fromBands([[1]] as unknown as Matrix, 0, 0)).toThrow(TypeError);
    expect(() => BandedMatrix.fromBands(new Matrix(2, 3), 1, 1)).toThrow(
      'BandedMatrix.fromBands: expected kl + ku + 1 = 3 rows, got 2.'
    );
  });
});

describe('BandedMatrix.mulVec', () => {
  it('matches the dense product', () => {
    const rnd = rng(3);
    for (const [n, kl, ku] of [
      [1, 0, 0],
      [5, 1, 1],
      [8, 3, 0],
      [8, 0, 3],
      [12, 2, 4],
      [4, 9, 9]
    ]) {
      const A = randomBanded(n, kl, ku, rnd, 1);
      const v = Vector.from(Array.from({ length: n }, rnd));
      const B = BandedMatrix.fromMatrix(A);
      expect(relDiff(B.mulVec(v), A.mulVec(v))).toBeLessThan(1e-14);
    }
  });

  it('accepts arrays, reuses out and validates arguments', () => {
    const B = BandedMatrix.fromMatrix(
      Matrix.from([
        [1, 2],
        [3, 4]
      ])
    );
    expect(B.mulVec([1, 1]).toString()).toBe('Vector(3, 7)');
    const out = new Vector(2);
    expect(B.mulVec(Vector.from([1, 1]), out)).toBe(out);
    expect(out.toString()).toBe('Vector(3, 7)');
    const v = Vector.from([1, 1]);
    expect(() => B.mulVec(v, v)).toThrow(RangeError);
    expect(() => B.mulVec([1, 2, 3])).toThrow('BandedMatrix.mulVec: shape mismatch');
    expect(() => B.mulVec(v, new Vector(3))).toThrow(RangeError);
    expect(() => B.mulVec(v, [0, 0] as unknown as Vector)).toThrow(TypeError);
    expect(() => B.mulVec('ab' as unknown as number[])).toThrow(TypeError);
  });

  it('does not read structural zeros', () => {
    const B = BandedMatrix.fromMatrix(
      Matrix.from([
        [1, 0],
        [0, 1]
      ])
    );
    expect(B.mulVec([Infinity, 2]).toString()).toBe('Vector(Infinity, 2)');
  });
});

describe('BandedMatrix.lu and solve', () => {
  it('agrees with dense Matrix.solve on random systems that need pivoting', () => {
    const rnd = rng(42);
    let pivoted = 0;
    let checked = 0;
    const shapes = [
      [2, 1, 1],
      [3, 1, 1],
      [6, 2, 1],
      [6, 1, 2],
      [9, 3, 2],
      [15, 4, 4],
      [20, 1, 3],
      [20, 3, 1],
      [33, 5, 2],
      [40, 2, 6],
      [50, 7, 7]
    ];
    for (const [n, kl, ku] of shapes) {
      for (let rep = 0; rep < 8; rep++) {
        const A = randomBanded(n, kl, ku, rnd);
        if (A.cond1() > 1e7) continue;
        const b = Vector.from(Array.from({ length: n }, rnd));
        const B = BandedMatrix.fromMatrix(A, kl, ku);
        const dec = B.lu();
        for (let j = 0; j < n; j++) if (dec.ipiv[j] !== j) pivoted++;
        const x = BandedMatrix.solveLU(dec, b);
        expect(relDiff(x, A.solve(b))).toBeLessThan(1e-8);
        expect(relDiff(B.solve(b), x)).toBe(0);
        // residual of the banded product
        expect(relDiff(B.mulVec(x), b)).toBeLessThan(1e-9);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(50);
    expect(pivoted).toBeGreaterThan(100);
  });

  it('reuses one factorization for several right-hand sides', () => {
    const rnd = rng(5);
    const A = randomBanded(25, 3, 2, rnd);
    const B = BandedMatrix.fromMatrix(A, 3, 2);
    const dec = B.lu();
    const snapshot = dec.ab.slice();
    for (let k = 0; k < 4; k++) {
      const b = Vector.from(Array.from({ length: 25 }, rnd));
      expect(relDiff(BandedMatrix.solveLU(dec, b), A.solve(b))).toBeLessThan(1e-8);
    }
    expect(Array.from(dec.ab)).toEqual(Array.from(snapshot));
  });

  it('reproduces P*A = L*U from the stored factors', () => {
    const rnd = rng(8);
    const n = 12;
    const kl = 2;
    const ku = 3;
    const A = randomBanded(n, kl, ku, rnd);
    const dec = BandedMatrix.fromMatrix(A).lu();
    const kv = kl + ku;
    const ld = 2 * kl + ku + 1;
    const L = Matrix.identity(n);
    const U = new Matrix(n, n);
    for (let j = 0; j < n; j++) {
      for (let i = Math.max(0, j - kv); i <= j; i++) U.set(i, j, dec.ab[kv + i - j + j * ld]);
    }
    // dgbtrf semantics: L = L_{n-1} P_{n-1} ... L_1 P_1 applied step by step.
    const PA = A.copy();
    for (let j = 0; j < n - 1; j++) {
      const l = dec.ipiv[j];
      if (l !== j) PA.swapRowsSelf(j, l);
      for (let k = 1; k <= Math.min(kl, n - 1 - j); k++) {
        const m = dec.ab[kv + k + j * ld];
        for (let c = 0; c < n; c++) PA.set(j + k, c, PA.get(j + k, c) - m * PA.get(j, c));
      }
    }
    expect(PA.allClose(U, { atol: 1e-12, rtol: 0 })).toBe(true);
    expect(L.rows).toBe(n);
  });

  it('handles kl = 0, ku = 0, n = 1 and bandwidth >= n', () => {
    const rnd = rng(9);
    for (const [n, kl, ku] of [
      [1, 0, 0],
      [7, 0, 3],
      [7, 3, 0],
      [7, 0, 0],
      [4, 20, 20],
      [5, 4, 4],
      [2, 1, 0],
      [2, 0, 1]
    ]) {
      const A = randomBanded(n, kl, ku, rnd, 1);
      const b = Vector.from(Array.from({ length: n }, rnd));
      const x = BandedMatrix.fromMatrix(A, kl, ku).solve(b);
      expect(relDiff(x, A.solve(b))).toBeLessThan(1e-10);
    }
    expect(
      BandedMatrix.fromMatrix(Matrix.from([[4]]))
        .solve([2])
        .toString()
    ).toBe('Vector(0.5)');
  });

  it('pivots on a zero diagonal', () => {
    const A = Matrix.from([
      [0, 2, 0],
      [1, 1, 3],
      [0, 4, 5]
    ]);
    const B = BandedMatrix.fromMatrix(A);
    expect(Array.from(B.lu().ipiv)).toEqual([1, 2, 2]);
    expect(relDiff(B.solve([2, 5, 9]), A.solve(Vector.from([2, 5, 9])))).toBeLessThan(1e-14);
  });

  it('throws a plain Error for a singular matrix', () => {
    const zeroColumn = BandedMatrix.fromMatrix(
      Matrix.from([
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1]
      ])
    );
    expect(() => zeroColumn.lu()).toThrow('BandedMatrix.lu: matrix is singular.');
    const err = (() => {
      try {
        zeroColumn.solve([1, 1, 1]);
      } catch (e) {
        return e as Error;
      }
    })();
    expect(err).toBeInstanceOf(Error);
    expect(err).not.toBeInstanceOf(RangeError);
    // rank deficient only after elimination
    const dependent = BandedMatrix.fromMatrix(
      Matrix.from([
        [1, 2, 0],
        [2, 4, 0],
        [0, 0, 1]
      ])
    );
    expect(() => dependent.lu()).toThrow(/singular/);
    expect(() => new BandedMatrix(3, 1, 1).lu()).toThrow(/singular/);
    expect(() => new BandedMatrix(1, 0, 0).lu()).toThrow(/singular/);
    // a nearly singular matrix does not throw
    expect(() =>
      BandedMatrix.fromMatrix(
        Matrix.from([
          [1, 1],
          [1, 1 + 1e-15]
        ])
      ).lu()
    ).not.toThrow();
  });

  it('rejects a factorization with a zero on the diagonal of U in solveLU', () => {
    const dec = BandedMatrix.fromMatrix(
      Matrix.from([
        [2, 1],
        [1, 3]
      ])
    ).lu();
    dec.ab[dec.kl + dec.ku + 0] = 0;
    expect(() => BandedMatrix.solveLU(dec, [1, 1])).toThrow(Error);
    expect(() => BandedMatrix.solveLU(dec, [1, 1])).toThrow(/zero diagonal entry/);
  });

  it('validates the arguments of solve and solveLU', () => {
    const B = BandedMatrix.fromMatrix(
      Matrix.from([
        [2, 1],
        [1, 3]
      ])
    );
    const dec = B.lu();
    expect(() => B.solve([1, 2, 3])).toThrow('BandedMatrix.solve: shape mismatch');
    expect(() => BandedMatrix.solveLU(dec, [1, 2, 3])).toThrow(RangeError);
    expect(() => BandedMatrix.solveLU(dec, [1, 2], new Vector(3))).toThrow(RangeError);
    const b = Vector.from([1, 2]);
    expect(() => BandedMatrix.solveLU(dec, b, b)).toThrow('out must not be b');
    expect(() => BandedMatrix.solveLU(dec, b, [0, 0] as unknown as Vector)).toThrow(TypeError);
    expect(() => BandedMatrix.solveLU({} as BandedLUDecomposition, b)).toThrow(TypeError);
    expect(() => BandedMatrix.solveLU(null as unknown as BandedLUDecomposition, b)).toThrow(
      TypeError
    );
    expect(() => BandedMatrix.solveLU({ ...dec, ab: new Float64Array(3) }, b)).toThrow(RangeError);
    expect(() => B.lu({ ...dec, kl: 0 })).toThrow(RangeError);
    expect(() => B.lu(randomFor(3).lu())).toThrow(RangeError);
    expect(() => B.lu({} as BandedLUDecomposition)).toThrow(TypeError);
  });
});

/** A diagonally dominant 3x3 banded matrix, to make a factorization of another shape. */
function randomFor(n: number): BandedMatrix {
  const B = new BandedMatrix(n, 1, 1);
  for (let i = 0; i < n; i++) B.set(i, i, 4);
  return B;
}

describe('BandedMatrix non-finite and scaling policy', () => {
  it('propagates NaN: no throw, everything NaN', () => {
    const B = BandedMatrix.fromMatrix(
      Matrix.from([
        [2, 1, 0],
        [1, 3, 1],
        [0, 1, 4]
      ])
    );
    B.set(1, 1, NaN);
    const dec = B.lu();
    expect(dec.ab.every(Number.isNaN)).toBe(true);
    expect(Array.from(dec.ipiv)).toEqual([0, 1, 2]);
    expect(dec.scale).toBe(1);
    expect(BandedMatrix.solveLU(dec, [1, 2, 3]).data.every(Number.isNaN)).toBe(true);
    expect(B.solve([1, 2, 3]).data.every(Number.isNaN)).toBe(true);
  });

  it('throws RangeError for an infinite entry, even next to a NaN', () => {
    const B = BandedMatrix.fromMatrix(
      Matrix.from([
        [2, 1],
        [1, 3]
      ])
    );
    B.set(1, 0, Infinity);
    expect(() => B.lu()).toThrow(RangeError);
    expect(() => B.lu()).toThrow(
      'BandedMatrix.lu: matrix contains an infinite entry at (1, 0), got Infinity.'
    );
    expect(() => B.solve([1, 1])).toThrow(RangeError);
    B.set(0, 0, NaN);
    B.set(1, 0, -Infinity);
    expect(() => B.lu()).toThrow(/got -Infinity/);
  });

  it('lets NaN and infinity in the right-hand side follow IEEE rules', () => {
    const dec = BandedMatrix.fromMatrix(
      Matrix.from([
        [2, 1],
        [1, 3]
      ])
    ).lu();
    expect(BandedMatrix.solveLU(dec, [NaN, 1]).data.every(Number.isNaN)).toBe(true);
    expect(() => BandedMatrix.solveLU(dec, [Infinity, 1])).not.toThrow();
  });

  it('is not fooled by entries near the limits of the double range', () => {
    const rnd = rng(77);
    const n = 10;
    const A0 = randomBanded(n, 2, 2, rnd);
    const b0 = Vector.from(Array.from({ length: n }, rnd));
    const x0 = A0.solve(b0);
    for (const s of [1e300, 1e-300, 2 ** 1000, 2 ** -1000]) {
      const B = BandedMatrix.fromMatrix(A0.mult(s), 2, 2);
      const dec = B.lu();
      expect(dec.scale).not.toBe(1);
      // A*x = b with A scaled by s: x = x0 / s, in the scaled units (b0 unscaled)
      const x = BandedMatrix.solveLU(dec, b0);
      expect(relDiff(x.mult(s), x0)).toBeLessThan(1e-8);
      const y = BandedMatrix.solveLU(dec, b0.mult(s));
      expect(relDiff(y, x0)).toBeLessThan(1e-8);
    }
    // an out-of-range solution is +-Infinity, never NaN
    const tiny = BandedMatrix.fromMatrix(
      Matrix.from([
        [1e-300, 0],
        [0, 1e-300]
      ])
    );
    const big = BandedMatrix.solveLU(tiny.lu(), [1e300, -1e300]);
    expect(Array.from(big.data)).toEqual([Infinity, -Infinity]);
  });

  it('leaves ordinary matrices unscaled', () => {
    expect(randomFor(4).lu().scale).toBe(1);
  });
});

describe('BandedMatrix allocation behavior', () => {
  const rnd = rng(123);
  const A = randomBanded(40, 3, 2, rnd);
  const B = BandedMatrix.fromMatrix(A, 3, 2);
  const b = Vector.from(Array.from({ length: 40 }, rnd));

  it('solveLU with out allocates no numeric storage', () => {
    const dec = B.lu();
    const out = new Vector(40);
    expect(countFloat64ArrayAllocations(() => void BandedMatrix.solveLU(dec, b, out))).toBe(0);
    expect(relDiff(out, A.solve(b))).toBeLessThan(1e-8);
    // without out: exactly the result vector
    expect(countFloat64ArrayAllocations(() => void BandedMatrix.solveLU(dec, b))).toBe(1);
  });

  it('lu with out reuses the factorization storage', () => {
    const dec = B.lu();
    expect(countFloat64ArrayAllocations(() => void B.lu(dec))).toBe(0);
    expect(relDiff(BandedMatrix.solveLU(dec, b), A.solve(b))).toBeLessThan(1e-8);
    expect(B.lu(dec)).toBe(dec);
  });

  it('mulVec with out allocates nothing', () => {
    const out = new Vector(40);
    expect(countFloat64ArrayAllocations(() => void B.mulVec(b, out))).toBe(0);
  });

  it('reusing out after a different matrix gives that matrix factors', () => {
    const dec = B.lu();
    const other = BandedMatrix.fromMatrix(randomBanded(40, 3, 2, rnd), 3, 2);
    other.lu(dec);
    expect(relDiff(BandedMatrix.solveLU(dec, b), other.toMatrix().solve(b))).toBeLessThan(1e-8);
  });
});
