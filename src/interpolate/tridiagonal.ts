/**
 * LU factorization of a general tridiagonal matrix with partial pivoting
 * (the scheme of LAPACK `dgttrf`), reusable for several right-hand sides.
 *
 * Plain Thomas elimination is not enough for cubic-spline systems: the
 * not-a-knot boundary rows are not diagonally dominant, so pivoting is needed
 * for backward stability.
 */
export interface TridiagonalLU {
  /** Order of the system. */
  n: number;
  /** Multipliers of `L` (length `n - 1`). */
  dl: Float64Array;
  /** Diagonal of `U` (length `n`). */
  d: Float64Array;
  /** First superdiagonal of `U` (length `n - 1`). */
  du: Float64Array;
  /** Second superdiagonal of `U`, created by row interchanges (length `n - 2`, or `0`). */
  du2: Float64Array;
  /** Row interchanges: row `i` was swapped with row `ipiv[i]` (`i` or `i + 1`). */
  ipiv: Int32Array;
}

/**
 * Factors the tridiagonal matrix with subdiagonal `dl`, diagonal `d` and
 * superdiagonal `du` (lengths `n - 1`, `n`, `n - 1`). The inputs are
 * overwritten and become part of the returned factorization.
 * @throws {RangeError} If the matrix is singular (a zero pivot is found).
 */
export function factorTridiagonal(
  dl: Float64Array,
  d: Float64Array,
  du: Float64Array
): TridiagonalLU {
  const n = d.length;
  const du2 = new Float64Array(Math.max(n - 2, 0));
  const ipiv = new Int32Array(n);

  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(d[i]) >= Math.abs(dl[i])) {
      // No row interchange
      ipiv[i] = i;
      if (d[i] !== 0) {
        const fact = dl[i] / d[i];
        dl[i] = fact;
        d[i + 1] -= fact * du[i];
      }
    } else {
      // Interchange rows i and i + 1
      ipiv[i] = i + 1;
      const fact = d[i] / dl[i];
      d[i] = dl[i];
      dl[i] = fact;
      const temp = du[i];
      du[i] = d[i + 1];
      d[i + 1] = temp - fact * d[i + 1];
      if (i < n - 2) {
        du2[i] = du[i + 1];
        du[i + 1] = -fact * du2[i];
      }
    }
  }
  ipiv[n - 1] = n - 1;

  for (let i = 0; i < n; i++) {
    if (d[i] === 0) {
      throw new RangeError('factorTridiagonal: singular matrix');
    }
  }

  return { n, dl, d, du, du2, ipiv };
}

/**
 * Solves `A x = b` in place using the factorization of `A`: on entry `b`
 * holds the right-hand side, on exit the solution.
 */
export function solveTridiagonal(lu: TridiagonalLU, b: Float64Array): void {
  const { n, dl, d, du, du2, ipiv } = lu;

  // Forward substitution: apply the row interchanges and L^-1
  for (let i = 0; i < n - 1; i++) {
    if (ipiv[i] === i) {
      b[i + 1] -= dl[i] * b[i];
    } else {
      const temp = b[i];
      b[i] = b[i + 1];
      b[i + 1] = temp - dl[i] * b[i];
    }
  }

  // Back substitution with U
  b[n - 1] /= d[n - 1];
  if (n > 1) {
    b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2];
  }
  for (let i = n - 3; i >= 0; i--) {
    b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / d[i];
  }
}
