/** Largest relative spacing of double precision numbers (QUADPACK's `epmach`). */
const EPSILON = Number.EPSILON;

/** Largest finite double (QUADPACK's `oflow`), used as "no error estimate available yet". */
const OVERFLOW = Number.MAX_VALUE;

/**
 * Maximum number of elements the epsilon table keeps (QUADPACK's `limexp`). When the table
 * reaches this length its oldest diagonal is dropped, which bounds the work and the memory
 * of the extrapolation.
 */
const MAX_ELEMENTS = 50;

/** Estimate of the limit of a sequence, with the estimated absolute error of that limit. */
export interface EpsilonEstimate {
  /** Extrapolated limit of the sequence. */
  result: number;
  /** Estimated absolute error of `result`. `Number.MAX_VALUE` while no estimate is available. */
  error: number;
}

/**
 * Wynn's epsilon algorithm: accelerates the convergence of a sequence of partial results and
 * estimates its limit and the error of that limit. This is QUADPACK's `dqelg`.
 *
 * Given the sequence `s_0, s_1, ...`, the epsilon table is the array
 *
 * ```text
 * eps(-1, k) = 0,    eps(0, k) = s_k,
 * eps(j+1, k) = eps(j-1, k+1) + 1 / (eps(j, k+1) - eps(j, k)),
 * ```
 *
 * whose even columns `eps(2m, k)` are the Shanks transforms of the sequence (the limit that a
 * sequence with `m` exponentially decaying terms converges to) and whose odd columns are
 * intermediate quantities. Only the most recent anti-diagonal of the table is stored, so each new
 * element costs one sweep over it: the table is updated in place.
 *
 * In the adaptive quadrature the sequence is that of the integral estimates obtained after each
 * round of subdividing the subintervals with the largest errors, whose errors behave like a
 * (possibly algebraic) singular series; the table converges much faster than the sequence does.
 *
 * The routine contains the QUADPACK safeguards against the instabilities of the algorithm:
 * - If three consecutive elements agree to machine precision, convergence is declared and the
 *   latest element is returned.
 * - If two neighbouring elements of a column are closer than rounding, or if the new element
 *   would be built from a reciprocal of an almost zero difference or a very small `1 / ss`
 *   (irregular behaviour), the part of the table that depends on them is dropped by shortening
 *   the table instead of letting a division by a tiny number pollute the result.
 * - The error estimate is the spread of the last three extrapolated results (it needs at least
 *   four calls), and it is never smaller than `5 * eps * |result|`.
 *
 * @internal
 */
export class EpsilonTable {
  /**
   * The stored anti-diagonal of the table (`epstab` in QUADPACK). `MAX_ELEMENTS + 2` entries are
   * needed by the algorithm; a few more are allocated so that a table that did not get shortened
   * (see the convergence exit of {@link extrapolate}) can never be written out of bounds.
   */
  private readonly table = new Float64Array(MAX_ELEMENTS + 10);

  /** Number of elements currently in the table (`numrl2`). */
  private n = 0;

  /** Number of times {@link extrapolate} has been called (`nres`). */
  private calls = 0;

  /** The last three extrapolated results, oldest first (`res3la`), used for the error estimate. */
  private readonly lastResults = new Float64Array(3);

  /** Number of elements currently in the table. */
  get length(): number {
    return this.n;
  }

  /** Appends the next element `s_k` of the sequence to the table. */
  push(value: number): void {
    this.table[this.n++] = value;
  }

  /**
   * Applies the epsilon algorithm to the elements pushed so far and returns the best extrapolated
   * limit and its error estimate. May shorten the table (see the class description).
   *
   * With fewer than three elements there is nothing to extrapolate: the last element is returned
   * with `error = Number.MAX_VALUE`. The same error is returned for the first three calls (it
   * needs the three previous results), unless convergence is detected at once.
   */
  extrapolate(): EpsilonEstimate {
    const table = this.table;
    this.calls++;
    let n = this.n;
    let error = OVERFLOW;
    let result = table[n - 1];

    if (n >= 3) {
      table[n + 1] = table[n - 1];
      const newElements = (n - 1) >> 1;
      table[n - 1] = OVERFLOW;
      const originalLength = n;
      // Index of the element of the diagonal that is currently the last one.
      let k1 = n - 1;

      for (let i = 1; i <= newElements; i++) {
        const k2 = k1 - 1;
        const k3 = k1 - 2;
        let res = table[k1 + 2];
        const e0 = table[k3];
        const e1 = table[k2];
        const e2 = res;
        const e1abs = Math.abs(e1);
        const delta2 = e2 - e1;
        const err2 = Math.abs(delta2);
        const tol2 = Math.max(Math.abs(e2), e1abs) * EPSILON;
        const delta3 = e1 - e0;
        const err3 = Math.abs(delta3);
        const tol3 = Math.max(e1abs, Math.abs(e0)) * EPSILON;

        if (!(err2 > tol2 || err3 > tol3)) {
          // e0, e1 and e2 are equal to machine accuracy: convergence is assumed. The table is
          // left as it is (QUADPACK does the same), and so is its length.
          return { result: res, error: Math.max(err2 + err3, 5 * EPSILON * Math.abs(res)) };
        }

        const e3 = table[k1];
        table[k1] = e1;
        const delta1 = e1 - e3;
        const err1 = Math.abs(delta1);
        const tol1 = Math.max(e1abs, Math.abs(e3)) * EPSILON;

        // If two elements are very close to each other, omit a part of the table by
        // shortening it.
        if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) {
          n = i + i - 1;
          break;
        }
        const ss = 1 / delta1 + 1 / delta2 - 1 / delta3;
        const epsinf = Math.abs(ss * e1);

        // Detect irregular behaviour in the table and, if so, omit a part of it by shortening.
        if (!(epsinf > 1e-4)) {
          n = i + i - 1;
          break;
        }

        // Compute a new element and adjust the result if its error estimate is the best so far.
        res = e1 + 1 / ss;
        table[k1] = res;
        k1 -= 2;
        const candidateError = err2 + Math.abs(res - e2) + err3;
        if (candidateError <= error) {
          error = candidateError;
          result = res;
        }
      }

      // Shift the table: the diagonal moved one step, so drop its oldest element.
      if (n === MAX_ELEMENTS) n = 2 * Math.floor(MAX_ELEMENTS / 2) - 1;
      let ib = originalLength % 2 === 0 ? 1 : 0;
      for (let i = 1; i <= newElements + 1; i++) {
        const ib2 = ib + 2;
        table[ib] = table[ib2];
        ib = ib2;
      }
      if (originalLength !== n) {
        let indx = originalLength - n;
        for (let i = 0; i < n; i++) {
          table[i] = table[indx];
          indx++;
        }
      }

      // Error estimate: the spread of the last three results.
      const last = this.lastResults;
      if (this.calls >= 4) {
        error =
          Math.abs(result - last[2]) + Math.abs(result - last[1]) + Math.abs(result - last[0]);
        last[0] = last[1];
        last[1] = last[2];
        last[2] = result;
      } else {
        last[this.calls - 1] = result;
        error = OVERFLOW;
      }
      this.n = n;
    }

    return { result, error: Math.max(error, 5 * EPSILON * Math.abs(result)) };
  }
}
