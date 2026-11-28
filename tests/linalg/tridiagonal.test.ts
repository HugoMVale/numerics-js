import { describe, it, expect } from 'vitest';
import {
  factorTridiagonalInPlace as factorTridiagonal,
  solveTridiagonalInPlace as solveTridiagonal,
  solveTridiagonal as solveTridiagonalPublic,
  solveTridiagonalLU,
  tridiagonalLU,
  type TridiagonalLU
} from '../../src/linalg/tridiagonal.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { countFloat64ArrayAllocations } from './allocation.js';

/** Dense matrix with the given sub-, main and superdiagonal. */
function dense(dl: number[], d: number[], du: number[]): Matrix {
  const n = d.length;
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n).fill(0);
    row[i] = d[i];
    if (i > 0) row[i - 1] = dl[i - 1];
    if (i < n - 1) row[i + 1] = du[i];
    rows.push(row);
  }
  return Matrix.from(rows);
}

function solveTri(dl: number[], d: number[], du: number[], b: number[]): Float64Array {
  const lu = factorTridiagonal(Float64Array.from(dl), Float64Array.from(d), Float64Array.from(du));
  const x = Float64Array.from(b);
  solveTridiagonal(lu, x);
  return x;
}

describe('tridiagonal LU solver', () => {
  it('solves a 1x1 and a 2x2 system', () => {
    expect(Array.from(solveTri([], [4], [], [2]))).toEqual([0.5]);
    // [[2, 1], [1, 3]] x = [3, 5] -> x = [0.8, 1.4]
    const x = solveTri([1], [2, 3], [1], [3, 5]);
    expect(x[0]).toBeCloseTo(0.8, 14);
    expect(x[1]).toBeCloseTo(1.4, 14);
  });

  it('matches a dense solve on random diagonally dominant systems', () => {
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
    for (const n of [3, 4, 10, 57]) {
      const dl = Array.from({ length: n - 1 }, rnd);
      const du = Array.from({ length: n - 1 }, rnd);
      const d = Array.from({ length: n }, () => 3 + rnd());
      const b = Array.from({ length: n }, rnd);
      const ref = dense(dl, d, du).solve(Vector.from(b));
      const x = solveTri(dl, d, du, b);
      for (let i = 0; i < n; i++) expect(x[i]).toBeCloseTo(ref.get(i), 12);
    }
  });

  it('pivots when the diagonal is small or zero (not diagonally dominant)', () => {
    let seed = 11;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
    for (const n of [3, 5, 12, 40]) {
      const dl = Array.from({ length: n - 1 }, () => 1 + rnd());
      const du = Array.from({ length: n - 1 }, () => 1 + rnd());
      const d = Array.from({ length: n }, () => 0.01 * rnd());
      d[0] = 0; // would break Thomas elimination
      const b = Array.from({ length: n }, rnd);
      const x = solveTri(dl, d, du, b);
      for (let i = 0; i < n; i++) {
        let ax = d[i] * x[i];
        if (i > 0) ax += dl[i - 1] * x[i - 1];
        if (i < n - 1) ax += du[i] * x[i + 1];
        expect(Math.abs(ax - b[i])).toBeLessThan(1e-9);
      }
    }
  });

  it('reuses one factorization for several right-hand sides', () => {
    const dl = [1, 2, 1];
    const d = [4, 5, 6, 7];
    const du = [1, 1, 2];
    const lu = factorTridiagonal(
      Float64Array.from(dl),
      Float64Array.from(d),
      Float64Array.from(du)
    );
    const A = dense(dl, d, du);
    for (const b of [
      [1, 0, 0, 0],
      [0, 1, -1, 2],
      [3, 3, 3, 3]
    ]) {
      const x = Float64Array.from(b);
      solveTridiagonal(lu, x);
      const ref = A.solve(Vector.from(b));
      for (let i = 0; i < 4; i++) expect(x[i]).toBeCloseTo(ref.get(i), 13);
    }
  });

  it('throws on a singular matrix', () => {
    // second row is a multiple of the first
    expect(() =>
      factorTridiagonal(Float64Array.of(2), Float64Array.of(1, 2), Float64Array.of(1))
    ).toThrow(/singular/);
    expect(() =>
      factorTridiagonal(new Float64Array(0), Float64Array.of(0), new Float64Array(0))
    ).toThrow(Error);
  });
});

describe('public tridiagonal solver', () => {
  const rnd = (() => {
    let s = 31;
    return () => (s = (s * 16807) % 2147483647) / 2147483647 - 0.5;
  })();

  it('matches the dense solve, with pivoting, for arrays and vectors', () => {
    for (const n of [1, 2, 3, 9, 40]) {
      const dl = Array.from({ length: n - 1 }, () => 1 + rnd());
      const du = Array.from({ length: n - 1 }, () => 1 + rnd());
      const d = Array.from({ length: n }, () => 0.02 * rnd());
      const b = Array.from({ length: n }, rnd);
      if (n === 1) d[0] = 2;
      const ref = dense(dl, d, du).solve(b);
      const x = solveTridiagonalLU(tridiagonalLU(dl, Vector.from(d), Float64Array.from(du)), b);
      expect(x.sub(ref).normInf()).toBeLessThan(1e-8 * Math.max(1, ref.normInf()));
      expect(solveTridiagonalPublic(dl, d, du, b).sub(x).normInf()).toBe(0);
    }
  });

  it('reuses the factorization and does not modify its inputs', () => {
    const dl = [1, 2];
    const d = [4, 5, 6];
    const du = [1, 1];
    const lu = tridiagonalLU(dl, d, du);
    expect([dl, d, du]).toEqual([
      [1, 2],
      [4, 5, 6],
      [1, 1]
    ]);
    const A = dense(dl, d, du);
    for (const b of [
      [1, 0, 0],
      [0, 1, -1]
    ]) {
      expect(solveTridiagonalLU(lu, b).sub(A.solve(b)).normInf()).toBeLessThan(1e-14);
    }
    const out = new Vector(3);
    expect(solveTridiagonalLU(lu, [1, 2, 3], out)).toBe(out);
    expect(
      countFloat64ArrayAllocations(() => void solveTridiagonalLU(lu, Vector.from([1, 2, 3]), out))
    ).toBe(1);
    const b = Vector.from([1, 2, 3]);
    expect(countFloat64ArrayAllocations(() => void solveTridiagonalLU(lu, b, out))).toBe(0);
    expect(() => solveTridiagonalLU(lu, b, b)).toThrow('out must not be b');
  });

  it('follows the linear-algebra policy', () => {
    expect(() => tridiagonalLU([2], [1, 2], [1])).toThrow('tridiagonalLU: matrix is singular.');
    expect(() => tridiagonalLU([], [0], [])).toThrow(Error);
    expect(() => tridiagonalLU([1, 1], [1, 2], [1])).toThrow(RangeError);
    expect(() => tridiagonalLU([], [], [])).toThrow(RangeError);
    expect(() => tridiagonalLU([Infinity], [1, 2], [1])).toThrow(
      'tridiagonalLU: matrix contains an infinite entry in dl[0], got Infinity.'
    );
    expect(() => tridiagonalLU([NaN], [1, Infinity], [1])).toThrow(RangeError);
    expect(() => tridiagonalLU('x' as unknown as number[], [1], [])).toThrow(TypeError);
    const nan = tridiagonalLU([1], [NaN, 2], [1]);
    expect(nan.d.every(Number.isNaN)).toBe(true);
    expect(Array.from(nan.ipiv)).toEqual([0, 1]);
    expect(solveTridiagonalLU(nan, [1, 1]).data.every(Number.isNaN)).toBe(true);
    const lu = tridiagonalLU([1], [2, 3], [1]);
    expect(() => solveTridiagonalLU(lu, [1, 2, 3])).toThrow(RangeError);
    expect(() => solveTridiagonalLU({} as TridiagonalLU, [1])).toThrow(TypeError);
    expect(() => solveTridiagonalLU(lu, [1, 2], [0, 0] as unknown as Vector)).toThrow(TypeError);
  });

  it('scales entries near the limits of the double range', () => {
    const dl = [1, 2, 1];
    const d = [4, 5, 6, 7];
    const du = [1, 1, 2];
    const b = [1, -2, 3, 0.5];
    const x0 = dense(dl, d, du).solve(b);
    for (const s of [1e300, 1e-300]) {
      const sc = (v: number[]) => v.map((e) => e * s);
      const lu = tridiagonalLU(sc(dl), sc(d), sc(du));
      expect(lu.scale).not.toBe(1);
      expect(solveTridiagonalLU(lu, b).mult(s).sub(x0).normInf()).toBeLessThan(1e-13);
    }
    expect(tridiagonalLU(dl, d, du).scale).toBe(1);
  });
});
