// Internal building blocks of the Levenberg-Marquardt solver: a robust
// Euclidean norm and the trust-region parameter search (`lmpar`) together
// with the triangular least-squares solve it relies on (`qrsolv`). Both are
// translations of the MINPACK-1 routines of the same names (Moré, Garbow &
// Hillstrom, Argonne National Laboratory, 1980), adapted to row-major
// storage. None of this is part of the public API.

/** Smallest positive normal double (MINPACK's `dwarf`). */
const DWARF = 2.2250738585072014e-308;

/**
 * Euclidean norm of the first `n` entries of `a`, protected against
 * intermediate overflow and underflow: the plain sum of squares is used when
 * it is safely representable and a rescaled accumulation otherwise.
 *
 * @internal
 * @param a Array of values.
 * @param n Number of leading entries to use.
 * @returns `sqrt(a[0]^2 + ... + a[n-1]^2)`.
 */
export function enorm(a: ArrayLike<number>, n: number = a.length): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * a[i];
  if (Number.isFinite(s) && s > 1e-280 && s < 1e280) return Math.sqrt(s);
  if (Number.isNaN(s)) return NaN;

  let amax = 0;
  for (let i = 0; i < n; i++) amax = Math.max(amax, Math.abs(a[i]));
  if (amax === 0) return 0;
  if (!Number.isFinite(amax)) return Infinity;
  let t = 0;
  for (let i = 0; i < n; i++) {
    const q = a[i] / amax;
    t += q * q;
  }
  return amax * Math.sqrt(t);
}

/**
 * Solves the least-squares problem `[A; sqrt(par)*D] x = [b; 0]` given the
 * pivoted QR factorization `A P = Q R` of `A`, for the diagonal matrix `D`
 * stored in `diag`. On output, `r` holds in its strict lower triangle the
 * transposed strict upper triangle of the upper-triangular matrix `S` with
 * `P^T (A^T A + par D^2) P = S^T S`; its upper triangle is unchanged.
 * Rank-deficient systems get a least-squares solution.
 *
 * @internal
 * @param n Order of `r`.
 * @param r `n x n` row-major array; the upper triangle holds `R`.
 * @param perm Column permutation `P` (`perm[j]` is the original column index
 * in position `j`).
 * @param diag Diagonal of `sqrt(par)*D` (already multiplied by `sqrt(par)`).
 * @param qtb First `n` entries of `Q^T b`.
 * @param x Receives the solution, in original variable order.
 * @param sdiag Receives the diagonal of `S`.
 * @param wa Work array of length `n`.
 */
function qrsolv(
  n: number,
  r: Float64Array,
  perm: number[],
  diag: Float64Array,
  qtb: Float64Array,
  x: Float64Array,
  sdiag: Float64Array,
  wa: Float64Array
): void {
  // Copy R^T into the lower triangle, keeping the diagonal of R in x, and
  // initialize the right-hand side.
  for (let j = 0; j < n; j++) {
    for (let i = j; i < n; i++) r[i * n + j] = r[j * n + i];
    x[j] = r[j * n + j];
    wa[j] = qtb[j];
  }

  // Eliminate the diagonal matrix D with Givens rotations, locating its
  // diagonal elements through the permutation.
  for (let j = 0; j < n; j++) {
    const l = perm[j];
    if (diag[l] !== 0) {
      for (let k = j; k < n; k++) sdiag[k] = 0;
      sdiag[j] = diag[l];

      // The rotations modify only one element of Q^T b beyond the
      // first n, which is initially zero.
      let qtbpj = 0;
      for (let k = j; k < n; k++) {
        if (sdiag[k] === 0) continue;
        const rkk = r[k * n + k];
        let cos: number;
        let sin: number;
        if (Math.abs(rkk) < Math.abs(sdiag[k])) {
          const cotan = rkk / sdiag[k];
          sin = 0.5 / Math.sqrt(0.25 + 0.25 * cotan * cotan);
          cos = sin * cotan;
        } else {
          const tan = sdiag[k] / rkk;
          cos = 0.5 / Math.sqrt(0.25 + 0.25 * tan * tan);
          sin = cos * tan;
        }

        // Modified diagonal element of R and modified element of (Q^T b, 0).
        const temp = cos * wa[k] + sin * qtbpj;
        qtbpj = -sin * wa[k] + cos * qtbpj;
        wa[k] = temp;

        // Accumulate the transformation in the row of S.
        r[k * n + k] = cos * rkk + sin * sdiag[k];
        for (let i = k + 1; i < n; i++) {
          const t = cos * r[i * n + k] + sin * sdiag[i];
          sdiag[i] = -sin * r[i * n + k] + cos * sdiag[i];
          r[i * n + k] = t;
        }
      }
    }

    // Store the diagonal element of S and restore that of R.
    sdiag[j] = r[j * n + j];
    r[j * n + j] = x[j];
  }

  // Solve the triangular system for z; if it is singular, obtain a
  // least-squares solution.
  let nsing = n;
  for (let j = 0; j < n; j++) {
    if (sdiag[j] === 0 && nsing === n) nsing = j;
    if (nsing < n) wa[j] = 0;
  }
  for (let k = 1; k <= nsing; k++) {
    const j = nsing - k;
    let sum = 0;
    for (let i = j + 1; i < nsing; i++) sum += r[i * n + j] * wa[i];
    wa[j] = (wa[j] - sum) / sdiag[j];
  }

  // Permute the components of z back to the components of x.
  for (let j = 0; j < n; j++) x[perm[j]] = wa[j];
}

/**
 * Determines the Levenberg-Marquardt parameter. Given the pivoted QR
 * factorization `A P = Q R`, a diagonal matrix `D`, a vector `b` and a trust
 * radius `delta`, finds `par >= 0` such that the solution `x` of
 * `(A^T A + par D^2) x = A^T b` satisfies `||D x|| ~ delta`, or `par = 0`
 * with `||D x|| <= 1.1*delta` if the Gauss-Newton step is already inside the
 * trust region. This is MINPACK's safeguarded Newton iteration on
 * `phi(par) = ||D x(par)|| - delta`, limited to 10 iterations; the best
 * value found is returned if it does not converge.
 *
 * Following cminpack, an exactly zero `||D x||` is accepted as converged
 * instead of dividing zero by zero.
 *
 * @internal
 * @param n Order of `r`.
 * @param r `n x n` row-major array; the upper triangle holds `R`. The strict
 * lower triangle is overwritten with the transposed strict upper triangle of
 * the factor `S` of `P^T (A^T A + par D^2) P = S^T S`.
 * @param perm Column permutation `P`.
 * @param diag Diagonal of `D`, strictly positive.
 * @param qtb First `n` entries of `Q^T b`.
 * @param delta Positive upper bound on `||D x||`.
 * @param par Initial estimate of the parameter (nonnegative).
 * @param x Receives the least-squares solution of `[A; sqrt(par) D] x = [b; 0]`.
 * @param sdiag Receives the diagonal of `S`.
 * @param wa1 Work array of length `n`.
 * @param wa2 Work array of length `n`.
 * @param wa3 Work array of length `n`.
 * @returns The final estimate of `par`.
 */
export function lmpar(
  n: number,
  r: Float64Array,
  perm: number[],
  diag: Float64Array,
  qtb: Float64Array,
  delta: number,
  par: number,
  x: Float64Array,
  sdiag: Float64Array,
  wa1: Float64Array,
  wa2: Float64Array,
  wa3: Float64Array
): number {
  // Gauss-Newton direction; for a rank-deficient R, a least-squares solution.
  let nsing = n;
  for (let j = 0; j < n; j++) {
    wa1[j] = qtb[j];
    if (r[j * n + j] === 0 && nsing === n) nsing = j;
    if (nsing < n) wa1[j] = 0;
  }
  for (let k = 1; k <= nsing; k++) {
    const j = nsing - k;
    wa1[j] /= r[j * n + j];
    const temp = wa1[j];
    for (let i = 0; i < j; i++) wa1[i] -= r[i * n + j] * temp;
  }
  for (let j = 0; j < n; j++) x[perm[j]] = wa1[j];

  // Accept the Gauss-Newton direction if it lies (nearly) within the trust region.
  let iter = 0;
  for (let j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
  let dxnorm = enorm(wa2, n);
  let fp = dxnorm - delta;
  if (fp <= 0.1 * delta) return 0;

  // If R is nonsingular, the Newton step gives a lower bound for the zero of phi.
  let parl = 0;
  if (nsing >= n) {
    for (let j = 0; j < n; j++) {
      const l = perm[j];
      wa1[j] = diag[l] * (wa2[l] / dxnorm);
    }
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < j; i++) sum += r[i * n + j] * wa1[i];
      wa1[j] = (wa1[j] - sum) / r[j * n + j];
    }
    const temp = enorm(wa1, n);
    parl = fp / delta / temp / temp;
  }

  // Upper bound for the zero of phi.
  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let i = 0; i <= j; i++) sum += r[i * n + j] * qtb[i];
    wa1[j] = sum / diag[perm[j]];
  }
  const gnorm = enorm(wa1, n);
  let paru = gnorm / delta;
  if (paru === 0) paru = DWARF / Math.min(delta, 0.1);

  // Clamp the input par to (parl, paru).
  par = Math.max(par, parl);
  par = Math.min(par, paru);
  if (par === 0) par = gnorm / dxnorm;

  for (;;) {
    iter++;
    if (par === 0) par = Math.max(DWARF, 0.001 * paru);
    const sqrtPar = Math.sqrt(par);
    for (let j = 0; j < n; j++) wa3[j] = sqrtPar * diag[j];
    qrsolv(n, r, perm, wa3, qtb, x, sdiag, wa2);
    for (let j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
    dxnorm = enorm(wa2, n);
    const fpPrev = fp;
    fp = dxnorm - delta;

    // Converged, exceptional case with parl = 0, or iteration limit.
    if (Math.abs(fp) <= 0.1 * delta || (parl === 0 && fp <= fpPrev && fpPrev < 0) || iter === 10)
      break;
    // ||D x|| = 0 trivially satisfies the constraint (and would divide 0/0 below).
    if (dxnorm === 0) break;

    // Newton correction.
    for (let j = 0; j < n; j++) {
      const l = perm[j];
      wa1[j] = diag[l] * (wa2[l] / dxnorm);
    }
    for (let j = 0; j < n; j++) {
      wa1[j] /= sdiag[j];
      const temp = wa1[j];
      for (let i = j + 1; i < n; i++) wa1[i] -= r[i * n + j] * temp;
    }
    const temp = enorm(wa1, n);
    const parc = fp / delta / temp / temp;

    // Update the bracket and the estimate.
    if (fp > 0) parl = Math.max(parl, par);
    if (fp < 0) paru = Math.min(paru, par);
    par = Math.max(parl, par + parc);
  }

  return par;
}
