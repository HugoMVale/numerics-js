import { copysign } from '../math.js';
import { Matrix } from './Matrix.js';
import { Vector } from './Vector.js';

// =======================================================================
// Matrix.svd.ts
//
// Singular value decomposition and singular values. Matrix's instance
// methods (`svd()`, `singularValues()`) are thin forwarders to the exported
// functions in this module; the remaining helpers are private
// implementation details.
//
// Algorithm (Golub-Kahan-Reinsch, as in LAPACK `dgesvd`):
//
//   1. Householder bidiagonalization `A = Q * B * P^T` (LAPACK `dgebd2`),
//      with `B` upper bidiagonal. A wide matrix is transposed first, so the
//      working matrix always has `rows >= cols`.
//   2. Implicit-shift QR iteration on `B` (LAPACK `dbdsqr`), with Demmel &
//      Kahan's zero-shift sweeps whenever a shift would endanger the
//      relative accuracy of the small singular values, and with LAPACK's
//      `dlasv2` for 2x2 blocks.
//   3. Singular values made nonnegative and sorted in decreasing order.
//
// The singular vectors are kept as the *rows* of two working arrays (`UT`
// and `VT`), so that every Givens rotation sweep of step 2 and every
// reflector of the accumulation runs along contiguous row-major memory.
// The iteration can also rotate arbitrary extra row-targets (it only sees
// "rows to transform"), so it does not care which singular vectors it is
// given.
//
// All index arithmetic below is 0-based; the comments name the LAPACK
// routine each block follows.
// =======================================================================

/**
 * Result of `Matrix.svd()`: a singular value decomposition such that
 * `A = U * diag(s) * V^T`.
 *
 * With `k = min(rows, cols)`, the default (thin) decomposition has `U` of
 * size `rows x k`, `V` of size `cols x k` and `k` singular values. With
 * `{ full: true }`, `U` is `rows x rows` and `V` is `cols x cols`; the
 * extra columns span the orthogonal complement and do not enter the product.
 */
export interface SVDDecomposition {
  /** Left singular vectors, as orthonormal columns. */
  U: Matrix;
  /** Singular values, nonnegative and in nonincreasing order. */
  s: Vector;
  /** Right singular vectors, as orthonormal columns (not transposed). */
  V: Matrix;
}

/** Options of `Matrix.svd()`. */
export interface SVDOptions {
  /**
   * Return the full square `U` (`rows x rows`) and `V` (`cols x cols`)
   * instead of the thin `rows x k` and `cols x k` factors, `k = min(rows,
   * cols)`. Costs `O(rows^2 * cols)` instead of `O(rows * cols^2)` for a tall
   * matrix, so leave it off unless the orthogonal complement is needed.
   * @default false
   */
  full?: boolean;
}

// LAPACK's `eps` (relative machine precision, `dlamch('E')`: half of the
// distance from 1 to the next double) and `safmin` (smallest normal number).
const EPS = Number.EPSILON / 2;
const SAFMIN = 2.2250738585072014e-308;
// `dgesvd`'s safe range for the matrix norm: outside of it the matrix is
// scaled, so that neither the Householder norms nor the QR iteration can
// overflow or lose relative accuracy to underflow.
const SMLNUM = Math.sqrt(SAFMIN) / EPS;
const BIGNUM = 1 / SMLNUM;
// `dbdsqr`: maximum number of QR sweeps per singular value, and the
// relative tolerance of its deflation tests.
const MAX_SWEEPS = 6;
const TOL = Math.max(10, Math.min(100, Math.pow(EPS, -0.125))) * EPS;

// -----------------------------------------------------------------
// Small LAPACK kernels. They return through module-level scratch buffers
// to avoid allocating in the QR iteration; JS is single-threaded and none
// of them is reentrant, so this is safe.
// -----------------------------------------------------------------

const ROT = new Float64Array(3);

/**
 * Plane rotation (LAPACK 3.10 `dlartg`): sets `ROT = [c, s, r]` with
 * `c >= 0`, `c*c + s*s = 1` and `[c s; -s c] * [f; g] = [r; 0]`.
 */
function rotg(f: number, g: number): void {
  if (g === 0) {
    ROT[0] = 1;
    ROT[1] = 0;
    ROT[2] = f;
  } else if (f === 0) {
    ROT[0] = 0;
    ROT[1] = Math.sign(g);
    ROT[2] = Math.abs(g);
  } else {
    const d = Math.hypot(f, g);
    const r = f >= 0 ? d : -d;
    ROT[0] = Math.abs(f) / d;
    ROT[1] = g / r;
    ROT[2] = r;
  }
}

const SV2 = new Float64Array(2);

/**
 * Singular values of the 2x2 triangular matrix `[f g; 0 h]` (LAPACK
 * `dlas2`), computed without overflow or cancellation: sets `SV2 = [ssmin,
 * ssmax]`.
 */
function las2(f: number, g: number, h: number): void {
  const fa = Math.abs(f);
  const ga = Math.abs(g);
  const ha = Math.abs(h);
  const fhmn = Math.min(fa, ha);
  const fhmx = Math.max(fa, ha);
  let ssmin: number;
  let ssmax: number;
  if (fhmn === 0) {
    ssmin = 0;
    if (fhmx === 0) {
      ssmax = ga;
    } else {
      const mx = Math.max(fhmx, ga);
      const q = Math.min(fhmx, ga) / mx;
      ssmax = mx * Math.sqrt(1 + q * q);
    }
  } else if (ga < fhmx) {
    const as = 1 + fhmn / fhmx;
    const at = (fhmx - fhmn) / fhmx;
    const au = (ga / fhmx) * (ga / fhmx);
    const c = 2 / (Math.sqrt(as * as + au) + Math.sqrt(at * at + au));
    ssmin = fhmn * c;
    ssmax = fhmx / c;
  } else {
    const au = fhmx / ga;
    if (au === 0) {
      ssmin = (fhmn * fhmx) / ga;
      ssmax = ga;
    } else {
      const as = 1 + fhmn / fhmx;
      const at = (fhmx - fhmn) / fhmx;
      const c = 1 / (Math.sqrt(1 + as * au * (as * au)) + Math.sqrt(1 + at * au * (at * au)));
      ssmin = 2 * (fhmn * c) * au;
      ssmax = ga / (c + c);
    }
  }
  SV2[0] = ssmin;
  SV2[1] = ssmax;
}

const SV2X2 = new Float64Array(6);

/**
 * SVD of the 2x2 triangular matrix `[f g; 0 h]` (LAPACK `dlasv2`). Sets
 * `SV2X2 = [ssmin, ssmax, snr, csr, snl, csl]` such that
 * `[csl snl; -snl csl] * [f g; 0 h] * [csr -snr; snr csr] = diag(ssmax, ssmin)`.
 * `ssmax` and `ssmin` carry signs (`|ssmax| >= |ssmin|`).
 */
function lasv2(f: number, g: number, h: number): void {
  let ft = f;
  let fa = Math.abs(ft);
  let ht = h;
  let ha = Math.abs(h);
  // pmax points at the entry of largest magnitude: 1 = f, 2 = g, 3 = h.
  let pmax = 1;
  const swap = ha > fa;
  if (swap) {
    pmax = 3;
    [ft, ht] = [ht, ft];
    [fa, ha] = [ha, fa];
  }
  const gt = g;
  const ga = Math.abs(gt);
  let clt = 1;
  let crt = 1;
  let slt = 0;
  let srt = 0;
  let ssmin: number;
  let ssmax: number;
  if (ga === 0) {
    ssmin = ha;
    ssmax = fa;
  } else {
    let gasmal = true;
    ssmin = 0;
    ssmax = 0;
    if (ga > fa) {
      pmax = 2;
      if (fa / ga < EPS) {
        // g is huge relative to f.
        gasmal = false;
        ssmax = ga;
        ssmin = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha;
        clt = 1;
        slt = ht / gt;
        srt = 1;
        crt = ft / gt;
      }
    }
    if (gasmal) {
      const d = fa - ha;
      let l = d === fa ? 1 : d / fa;
      const m = gt / ft;
      let t = 2 - l;
      const mm = m * m;
      const tt = t * t;
      const s = Math.sqrt(tt + mm);
      const r = l === 0 ? Math.abs(m) : Math.sqrt(l * l + mm);
      const a = 0.5 * (s + r);
      ssmin = ha / a;
      ssmax = fa * a;
      if (mm === 0) {
        // m is tiny.
        t = l === 0 ? copysign(2, ft) * copysign(1, gt) : gt / copysign(d, ft) + m / t;
      } else {
        t = (m / (s + t) + m / (r + l)) * (1 + a);
      }
      l = Math.sqrt(t * t + 4);
      crt = 2 / l;
      srt = t / l;
      clt = (crt + srt * m) / a;
      slt = ((ht / ft) * srt) / a;
    }
  }
  let csl: number;
  let snl: number;
  let csr: number;
  let snr: number;
  if (swap) {
    csl = srt;
    snl = crt;
    csr = slt;
    snr = clt;
  } else {
    csl = clt;
    snl = slt;
    csr = crt;
    snr = srt;
  }
  // Correct the signs of ssmax and ssmin.
  let tsign: number;
  if (pmax === 1) tsign = copysign(1, csr) * copysign(1, csl) * copysign(1, f);
  else if (pmax === 2) tsign = copysign(1, snr) * copysign(1, csl) * copysign(1, g);
  else tsign = copysign(1, snr) * copysign(1, snl) * copysign(1, h);
  SV2X2[0] = copysign(ssmin, tsign * copysign(1, f) * copysign(1, h));
  SV2X2[1] = copysign(ssmax, tsign);
  SV2X2[2] = snr;
  SV2X2[3] = csr;
  SV2X2[4] = snl;
  SV2X2[5] = csl;
}

// -----------------------------------------------------------------
// Householder bidiagonalization.
// -----------------------------------------------------------------

/**
 * The matrix to decompose, as a working array: `rows >= cols`, finite, and
 * scaled into the safe range (see `prepare()`).
 */
interface Prepared {
  /** Row-major `mw x nw` working copy, overwritten by `bidiagonalize()`. */
  a: Float64Array;
  mw: number;
  nw: number;
  /** `true` if the working matrix is the transpose of the user's matrix. */
  transposed: boolean;
  /** Factor the matrix was multiplied by; singular values are divided by it. */
  scale: number;
}

/**
 * Copies `A` (transposed if it is wide, so that `mw >= nw`) into a fresh
 * working array, rejecting non-finite entries and scaling it into the safe
 * range `[SMLNUM, BIGNUM]` of its largest entry, as `dgesvd` does.
 * @param who Name of the public routine, for the error message.
 * @throws {RangeError} If `A` has a `NaN` or infinite entry.
 */
function prepare(A: Matrix, who: string): Prepared {
  const m = A.rows;
  const n = A.cols;
  const transposed = m < n;
  const mw = transposed ? n : m;
  const nw = transposed ? m : n;
  const src = A.data;
  let a: Float64Array;
  if (transposed) {
    a = new Float64Array(mw * nw);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) a[j * nw + i] = src[i * n + j];
    }
  } else {
    a = src.slice();
  }
  let amax = 0;
  for (let p = 0; p < a.length; p++) {
    const v = a[p];
    if (!Number.isFinite(v)) {
      throw new RangeError(`Matrix ${who}: matrix contains non-finite entries`);
    }
    const av = Math.abs(v);
    if (av > amax) amax = av;
  }
  let scale = 1;
  if (amax > 0 && amax < SMLNUM) scale = SMLNUM / amax;
  else if (amax > BIGNUM) scale = BIGNUM / amax;
  if (scale !== 1) for (let p = 0; p < a.length; p++) a[p] *= scale;
  return { a, mw, nw, transposed, scale };
}

/**
 * Householder reflector `H = I - tau * v * v^T`, `v = [1; x]`, that maps
 * `[alpha; x]` to `[beta; 0]` (LAPACK `dlarfg`). `x` is overwritten by the
 * essential part of `v`, `alpha` by `beta`. Returns `tau` (`0` if `x` is
 * already zero, i.e. `H = I`).
 * @param a Array holding `alpha` and `x`.
 * @param alphaIdx Position of `alpha`.
 * @param xStart Position of the first entry of `x`.
 * @param xCount Length of `x`.
 * @param xStride Distance between consecutive entries of `x`.
 */
function larfg(
  a: Float64Array,
  alphaIdx: number,
  xStart: number,
  xCount: number,
  xStride: number
): number {
  let xs = 0;
  for (let t = 0, p = xStart; t < xCount; t++, p += xStride) xs += a[p] * a[p];
  if (xs === 0) return 0;
  const alpha = a[alphaIdx];
  const xnorm = Math.sqrt(xs);
  const beta = alpha >= 0 ? -Math.hypot(alpha, xnorm) : Math.hypot(alpha, xnorm);
  const scale = 1 / (alpha - beta);
  for (let t = 0, p = xStart; t < xCount; t++, p += xStride) a[p] *= scale;
  a[alphaIdx] = beta;
  return (beta - alpha) / beta;
}

/**
 * Upper bidiagonal reduction `A = Q * B * P^T` in LAPACK `dgebrd` storage:
 * `B` has diagonal `d` and superdiagonal `e`; `Q = H_0 * ... * H_{n-1}` and
 * `P = G_0 * ... * G_{n-2}` are products of the Householder reflectors whose
 * essential parts are stored in `a` (`H_j` in column `j` below the
 * diagonal, `G_j` in row `j` right of the superdiagonal).
 */
interface Bidiagonal {
  a: Float64Array;
  m: number;
  n: number;
  d: Float64Array;
  e: Float64Array;
  tauq: Float64Array;
  taup: Float64Array;
}

/**
 * Reduces the `m x n` (`m >= n`) row-major array `a` to upper bidiagonal
 * form (LAPACK `dgebd2`), overwriting it. Reflectors are applied row by
 * row, the cache-friendly order for the row-major layout.
 */
function bidiagonalize(a: Float64Array, m: number, n: number): Bidiagonal {
  const d = new Float64Array(n);
  const e = new Float64Array(Math.max(n - 1, 0));
  const tauq = new Float64Array(n);
  const taup = new Float64Array(n);
  const w = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    // H_i annihilates a[i+1:m, i].
    const tq = larfg(a, i * n + i, (i + 1) * n + i, m - i - 1, n);
    tauq[i] = tq;
    d[i] = a[i * n + i];
    if (tq !== 0 && i < n - 1) {
      // Apply H_i to a[i:m, i+1:n] from the left: w^T = v^T C, C -= tau * v * w^T.
      for (let c = i + 1; c < n; c++) w[c] = a[i * n + c];
      for (let r = i + 1; r < m; r++) {
        const vr = a[r * n + i];
        if (vr !== 0) {
          const off = r * n;
          for (let c = i + 1; c < n; c++) w[c] += vr * a[off + c];
        }
      }
      for (let c = i + 1; c < n; c++) {
        w[c] *= tq;
        a[i * n + c] -= w[c];
      }
      for (let r = i + 1; r < m; r++) {
        const vr = a[r * n + i];
        if (vr !== 0) {
          const off = r * n;
          for (let c = i + 1; c < n; c++) a[off + c] -= vr * w[c];
        }
      }
    }
    if (i < n - 1) {
      // G_i annihilates a[i, i+2:n].
      const tp = larfg(a, i * n + i + 1, i * n + i + 2, n - i - 2, 1);
      taup[i] = tp;
      e[i] = a[i * n + i + 1];
      if (tp !== 0) {
        // Apply G_i to a[i+1:m, i+1:n] from the right, one row at a time.
        const ro = i * n;
        for (let r = i + 1; r < m; r++) {
          const off = r * n;
          let s = a[off + i + 1];
          for (let c = i + 2; c < n; c++) s += a[off + c] * a[ro + c];
          s *= tp;
          if (s === 0) continue;
          a[off + i + 1] -= s;
          for (let c = i + 2; c < n; c++) a[off + c] -= s * a[ro + c];
        }
      }
    }
  }
  return { a, m, n, d, e, tauq, taup };
}

/**
 * Forms the leading `rowsOut` rows of `Q^T` (`rowsOut = n` for the thin
 * factor, `m` for the full one) as a row-major `rowsOut x m` array. Right-
 * multiplies `[I | 0]` by the reflectors `H_{n-1}, ..., H_0`, so each step
 * touches only the rows that can be nonzero in its columns.
 */
function formQt(bd: Bidiagonal, rowsOut: number): Float64Array {
  const { a, m, n, tauq } = bd;
  const x = new Float64Array(rowsOut * m);
  for (let i = 0; i < rowsOut; i++) x[i * m + i] = 1;
  const v = new Float64Array(m);
  for (let j = n - 1; j >= 0; j--) {
    const tj = tauq[j];
    if (tj === 0) continue;
    for (let r = j + 1; r < m; r++) v[r] = a[r * n + j];
    for (let i = j; i < rowsOut; i++) {
      const off = i * m;
      let s = x[off + j];
      for (let r = j + 1; r < m; r++) s += x[off + r] * v[r];
      s *= tj;
      if (s === 0) continue;
      x[off + j] -= s;
      for (let r = j + 1; r < m; r++) x[off + r] -= s * v[r];
    }
  }
  return x;
}

/**
 * Forms `P^T` (`n x n`, row-major) by right-multiplying the identity by
 * `G_{n-2}, ..., G_0`.
 */
function formPt(bd: Bidiagonal): Float64Array {
  const { a, n, taup } = bd;
  const x = new Float64Array(n * n);
  for (let i = 0; i < n; i++) x[i * n + i] = 1;
  for (let k = n - 2; k >= 0; k--) {
    const tk = taup[k];
    if (tk === 0) continue;
    const ro = k * n;
    for (let i = k + 1; i < n; i++) {
      const off = i * n;
      let s = x[off + k + 1];
      for (let c = k + 2; c < n; c++) s += x[off + c] * a[ro + c];
      s *= tk;
      if (s === 0) continue;
      x[off + k + 1] -= s;
      for (let c = k + 2; c < n; c++) x[off + c] -= s * a[ro + c];
    }
  }
  return x;
}

// -----------------------------------------------------------------
// SVD of the bidiagonal matrix: implicit-shift QR iteration.
// -----------------------------------------------------------------

/**
 * A row-major array with (at least) `n` rows that the QR iteration
 * transforms along with the bidiagonal matrix: the rows are rotated,
 * negated and permuted exactly like the corresponding singular vectors.
 */
interface RowTarget {
  x: Float64Array;
  /** Row stride. */
  ncols: number;
}

/**
 * Applies the plane rotations `(cs[k], sn[k])`, `k = 0..count-1`, to the
 * row pairs `(lo+k, lo+k+1)` of every target, in ascending (`forward`) or
 * descending order (LAPACK `dlasr`, pivot 'V'). Each pair is rotated as
 * `(x, y) -> (c*x + s*y, c*y - s*x)`, sweeping contiguous row segments.
 */
function applyRotations(
  targets: RowTarget[],
  lo: number,
  cs: Float64Array,
  sn: Float64Array,
  count: number,
  forward: boolean
): void {
  for (const { x, ncols } of targets) {
    for (let t = 0; t < count; t++) {
      const k = forward ? t : count - 1 - t;
      const c = cs[k];
      const s = sn[k];
      if (c === 1 && s === 0) continue;
      const r1 = (lo + k) * ncols;
      const r2 = r1 + ncols;
      for (let q = 0; q < ncols; q++) {
        const p = x[r1 + q];
        const o = x[r2 + q];
        x[r1 + q] = c * p + s * o;
        x[r2 + q] = c * o - s * p;
      }
    }
  }
}

/** Rotates the single row pair `(i, i + 1)` of every target by `(c, s)`. */
function rotatePair(targets: RowTarget[], i: number, c: number, s: number): void {
  const cs = new Float64Array([c]);
  const sn = new Float64Array([s]);
  applyRotations(targets, i, cs, sn, 1, true);
}

/**
 * SVD of the `n x n` upper bidiagonal matrix with diagonal `d` and
 * superdiagonal `e` (LAPACK `dbdsqr`, upper case): on return `d` holds the
 * singular values in nonincreasing order and `e` is destroyed.
 *
 * With `B = U_B * diag(d) * V_B^T`, the `right` targets are transformed as
 * `X -> V_B^T * X` (start them from `P^T`, or from a vector `P^T b`) and the
 * `left` targets as `X -> U_B^T * X` (start them from `Q^T`, or from `Q^T b`),
 * so that their rows follow the sorted singular values. Targets may have
 * more than `n` rows; only the first `n` are touched. With no targets only
 * the singular values are computed.
 *
 * Convergence follows `dbdsqr`: relative tolerance `TOL`, the shift is the
 * smaller singular value of the trailing (or leading, depending on the
 * direction of the sweep) 2x2 block, and zero-shift sweeps replace it
 * whenever it could spoil the relative accuracy of the smallest singular
 * values.
 * @throws {Error} If the iteration does not converge in `6 * n^2` rotation sweeps.
 */
function bdsqr(
  d: Float64Array,
  e: Float64Array,
  n: number,
  right: RowTarget[],
  left: RowTarget[]
): void {
  if (n > 1) {
    // Deflation threshold, from a lower bound on the smallest singular value.
    let sminoa = Math.abs(d[0]);
    if (sminoa !== 0) {
      let mu = sminoa;
      for (let i = 1; i < n; i++) {
        mu = Math.abs(d[i]) * (mu / (mu + Math.abs(e[i - 1])));
        sminoa = Math.min(sminoa, mu);
        if (sminoa === 0) break;
      }
    }
    sminoa /= Math.sqrt(n);
    const thresh = Math.max(TOL * sminoa, MAX_SWEEPS * (n * (n * SAFMIN)));
    const maxit = MAX_SWEEPS * n * n;

    // Rotations of one sweep: (cosr, sinr) act on `right`, (cosl, sinl) on `left`.
    const w1 = new Float64Array(n);
    const w2 = new Float64Array(n);
    const w3 = new Float64Array(n);
    const w4 = new Float64Array(n);

    // The active (unreduced) block is d[lo..hi].
    let hi = n - 1;
    let oldLo = -1;
    let oldHi = -1;
    let idir = 0; // 1: chase the bulge top to bottom; 2: bottom to top.
    let iter = 0;

    main: while (hi > 0) {
      if (iter > maxit) throw new Error('Matrix svd: QR iteration failed to converge');

      // Find the bottom unreduced block d[lo..hi], deflating negligible e's.
      let smax = Math.abs(d[hi]);
      let lo = 0;
      for (let l = hi - 1; l >= 0; l--) {
        const abss = Math.abs(d[l]);
        const abse = Math.abs(e[l]);
        if (abse <= thresh) {
          e[l] = 0;
          if (l === hi - 1) {
            hi--;
            continue main;
          }
          lo = l + 1;
          break;
        }
        smax = Math.max(smax, abss, abse);
      }

      if (lo === hi - 1) {
        // 2x2 block: solve directly.
        lasv2(d[hi - 1], e[hi - 1], d[hi]);
        d[hi - 1] = SV2X2[1];
        e[hi - 1] = 0;
        d[hi] = SV2X2[0];
        rotatePair(right, hi - 1, SV2X2[3], SV2X2[2]);
        rotatePair(left, hi - 1, SV2X2[5], SV2X2[4]);
        hi -= 2;
        continue;
      }

      // For a new block, chase the bulge from the larger end towards the smaller.
      if (lo > oldHi || hi < oldLo) idir = Math.abs(d[lo]) >= Math.abs(d[hi]) ? 1 : 2;

      // Convergence tests, which also yield sminl, an estimate of the
      // smallest singular value of the block.
      let sminl: number;
      if (idir === 1) {
        if (Math.abs(e[hi - 1]) <= TOL * Math.abs(d[hi])) {
          e[hi - 1] = 0;
          continue;
        }
        let mu = Math.abs(d[lo]);
        sminl = mu;
        for (let l = lo; l < hi; l++) {
          if (Math.abs(e[l]) <= TOL * mu) {
            e[l] = 0;
            continue main;
          }
          mu = Math.abs(d[l + 1]) * (mu / (mu + Math.abs(e[l])));
          sminl = Math.min(sminl, mu);
        }
      } else {
        if (Math.abs(e[lo]) <= TOL * Math.abs(d[lo])) {
          e[lo] = 0;
          continue;
        }
        let mu = Math.abs(d[hi]);
        sminl = mu;
        for (let l = hi - 1; l >= lo; l--) {
          if (Math.abs(e[l]) <= TOL * mu) {
            e[l] = 0;
            continue main;
          }
          mu = Math.abs(d[l]) * (mu / (mu + Math.abs(e[l])));
          sminl = Math.min(sminl, mu);
        }
      }
      oldLo = lo;
      oldHi = hi;

      // Shift: zero if a shift would ruin the relative accuracy, else the
      // smaller singular value of the 2x2 block at the end being chased to.
      let shift: number;
      if (n * TOL * (sminl / smax) <= Math.max(EPS, 0.01 * TOL)) {
        shift = 0;
      } else {
        let sll: number;
        if (idir === 1) {
          sll = Math.abs(d[lo]);
          las2(d[hi - 1], e[hi - 1], d[hi]);
        } else {
          sll = Math.abs(d[hi]);
          las2(d[lo], e[lo], d[lo + 1]);
        }
        shift = SV2[0];
        if (sll > 0 && (shift / sll) * (shift / sll) < EPS) shift = 0;
      }
      iter += hi - lo;
      const count = hi - lo;

      if (shift === 0) {
        // Zero-shift QR sweep (Demmel & Kahan).
        let cs = 1;
        let oldcs = 1;
        let sn = 0;
        let oldsn = 0;
        if (idir === 1) {
          for (let i = lo; i < hi; i++) {
            rotg(d[i] * cs, e[i]);
            cs = ROT[0];
            sn = ROT[1];
            if (i > lo) e[i - 1] = oldsn * ROT[2];
            rotg(oldcs * ROT[2], d[i + 1] * sn);
            oldcs = ROT[0];
            oldsn = ROT[1];
            d[i] = ROT[2];
            w1[i - lo] = cs;
            w2[i - lo] = sn;
            w3[i - lo] = oldcs;
            w4[i - lo] = oldsn;
          }
          const h = d[hi] * cs;
          d[hi] = h * oldcs;
          e[hi - 1] = h * oldsn;
          applyRotations(right, lo, w1, w2, count, true);
          applyRotations(left, lo, w3, w4, count, true);
          if (Math.abs(e[hi - 1]) <= thresh) e[hi - 1] = 0;
        } else {
          for (let i = hi; i > lo; i--) {
            rotg(d[i] * cs, e[i - 1]);
            cs = ROT[0];
            sn = ROT[1];
            if (i < hi) e[i] = oldsn * ROT[2];
            rotg(oldcs * ROT[2], d[i - 1] * sn);
            oldcs = ROT[0];
            oldsn = ROT[1];
            d[i] = ROT[2];
            w1[i - lo - 1] = cs;
            w2[i - lo - 1] = -sn;
            w3[i - lo - 1] = oldcs;
            w4[i - lo - 1] = -oldsn;
          }
          const h = d[lo] * cs;
          d[lo] = h * oldcs;
          e[lo] = h * oldsn;
          applyRotations(right, lo, w3, w4, count, false);
          applyRotations(left, lo, w1, w2, count, false);
          if (Math.abs(e[lo]) <= thresh) e[lo] = 0;
        }
      } else if (idir === 1) {
        // Shifted QR sweep, top to bottom.
        let f = (Math.abs(d[lo]) - shift) * (copysign(1, d[lo]) + shift / d[lo]);
        let g = e[lo];
        for (let i = lo; i < hi; i++) {
          rotg(f, g);
          const cosr = ROT[0];
          const sinr = ROT[1];
          if (i > lo) e[i - 1] = ROT[2];
          f = cosr * d[i] + sinr * e[i];
          e[i] = cosr * e[i] - sinr * d[i];
          g = sinr * d[i + 1];
          d[i + 1] = cosr * d[i + 1];
          rotg(f, g);
          const cosl = ROT[0];
          const sinl = ROT[1];
          d[i] = ROT[2];
          f = cosl * e[i] + sinl * d[i + 1];
          d[i + 1] = cosl * d[i + 1] - sinl * e[i];
          if (i < hi - 1) {
            g = sinl * e[i + 1];
            e[i + 1] = cosl * e[i + 1];
          }
          w1[i - lo] = cosr;
          w2[i - lo] = sinr;
          w3[i - lo] = cosl;
          w4[i - lo] = sinl;
        }
        e[hi - 1] = f;
        applyRotations(right, lo, w1, w2, count, true);
        applyRotations(left, lo, w3, w4, count, true);
        if (Math.abs(e[hi - 1]) <= thresh) e[hi - 1] = 0;
      } else {
        // Shifted QR sweep, bottom to top.
        let f = (Math.abs(d[hi]) - shift) * (copysign(1, d[hi]) + shift / d[hi]);
        let g = e[hi - 1];
        for (let i = hi; i > lo; i--) {
          rotg(f, g);
          const cosr = ROT[0];
          const sinr = ROT[1];
          if (i < hi) e[i] = ROT[2];
          f = cosr * d[i] + sinr * e[i - 1];
          e[i - 1] = cosr * e[i - 1] - sinr * d[i];
          g = sinr * d[i - 1];
          d[i - 1] = cosr * d[i - 1];
          rotg(f, g);
          const cosl = ROT[0];
          const sinl = ROT[1];
          d[i] = ROT[2];
          f = cosl * e[i - 1] + sinl * d[i - 1];
          d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
          if (i > lo + 1) {
            g = sinl * e[i - 2];
            e[i - 2] = cosl * e[i - 2];
          }
          w1[i - lo - 1] = cosr;
          w2[i - lo - 1] = -sinr;
          w3[i - lo - 1] = cosl;
          w4[i - lo - 1] = -sinl;
        }
        e[lo] = f;
        if (Math.abs(e[lo]) <= thresh) e[lo] = 0;
        applyRotations(right, lo, w3, w4, count, false);
        applyRotations(left, lo, w1, w2, count, false);
      }
    }
  }

  // Make the singular values nonnegative (flipping the right vectors) ...
  for (let i = 0; i < n; i++) {
    if (d[i] < 0) {
      d[i] = -d[i];
      for (const { x, ncols } of right) {
        for (let q = 0; q < ncols; q++) x[i * ncols + q] = -x[i * ncols + q];
      }
    } else if (d[i] === 0) {
      d[i] = 0; // turn -0 into +0
    }
  }
  // ... and sort them into nonincreasing order: selection sort, which moves
  // each vector at most once.
  const both = [...right, ...left];
  for (let i = 0; i < n - 1; i++) {
    const last = n - 1 - i;
    let isub = 0;
    let smin = d[0];
    for (let j = 1; j <= last; j++) {
      if (d[j] <= smin) {
        isub = j;
        smin = d[j];
      }
    }
    if (isub !== last) {
      d[isub] = d[last];
      d[last] = smin;
      for (const { x, ncols } of both) {
        for (let q = 0; q < ncols; q++) {
          const p = x[isub * ncols + q];
          x[isub * ncols + q] = x[last * ncols + q];
          x[last * ncols + q] = p;
        }
      }
    }
  }
}

// -----------------------------------------------------------------
// Public algorithms.
// -----------------------------------------------------------------

/** Returns the transpose of the row-major `rows x cols` array `x`, as a new `cols x rows` array. */
function transposeArray(x: Float64Array, rows: number, cols: number): Float64Array {
  const t = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) t[j * rows + i] = x[i * cols + j];
  }
  return t;
}

/**
 * Divides the singular values by the matrix scale factor of `prepare()`,
 * in place, and returns them.
 */
function unscale(s: Float64Array, scale: number): Float64Array {
  if (scale !== 1) for (let i = 0; i < s.length; i++) s[i] /= scale;
  return s;
}

/**
 * Computes the singular value decomposition `A = U * diag(s) * V^T` of `A`
 * (any shape): Householder bidiagonalization followed by the implicit-shift
 * QR iteration of LAPACK `dgesvd`/`dbdsqr` (Golub-Kahan-Reinsch), in
 * `O(rows * cols^2)` operations for a tall matrix (a wide one is
 * transposed first). The singular values are returned in nonincreasing
 * order, and the singular vectors are orthonormal to working precision.
 *
 * As for any backward-stable SVD, the computed singular values have absolute
 * error of order `Number.EPSILON * s[0]`; the QR iteration additionally
 * preserves the relative accuracy of the singular values of the bidiagonal
 * factor. The sign of each singular-vector pair is arbitrary (but
 * deterministic). The matrix is rescaled internally if its entries are close
 * to overflow or underflow, so extreme magnitudes are handled correctly.
 * @param A The matrix to decompose. Not modified.
 * @param options See {@link SVDOptions}.
 * @returns The `{ U, s, V }` decomposition.
 * @throws {RangeError} If `A` contains a non-finite entry.
 * @throws {Error} If the QR iteration fails to converge, which is not expected for finite input.
 */
export function svd(A: Matrix, options?: SVDOptions): SVDDecomposition {
  const m = A.rows;
  const n = A.cols;
  const full = options?.full ?? false;
  const { a, mw, nw, transposed, scale } = prepare(A, 'svd');
  const bd = bidiagonalize(a, mw, nw);
  const rowsU = full ? mw : nw;
  const ut = formQt(bd, rowsU);
  const vt = formPt(bd);
  bdsqr(bd.d, bd.e, nw, [{ x: vt, ncols: nw }], [{ x: ut, ncols: mw }]);
  const s = unscale(bd.d, scale);

  // The working matrix is `Uw * S * Vw^T`, `Uw` (mw x rowsU) being the
  // transpose of `ut` and `Vw` (nw x nw) that of `vt`. For a wide matrix
  // the roles of U and V are exchanged.
  const big = transposeArray(ut, rowsU, mw);
  const small = transposeArray(vt, nw, nw);
  const U = transposed ? Matrix.wrapUnchecked(m, m, small) : Matrix.wrapUnchecked(m, rowsU, big);
  const V = transposed ? Matrix.wrapUnchecked(n, rowsU, big) : Matrix.wrapUnchecked(n, n, small);
  return { U, s: Vector.wrapUnchecked(s), V };
}

/**
 * Computes only the singular values of `A` (any shape), in nonincreasing
 * order. Same algorithm and accuracy as `svd()` — the values are identical —
 * but the singular vectors are never formed, which makes it roughly 1.5 to 3
 * times faster than `svd()` (the bidiagonalization, `O(rows * cols^2)`,
 * dominates) and saves the `O(rows * cols)` vector storage.
 * @param A The matrix. Not modified.
 * @returns The `min(rows, cols)` singular values.
 * @throws {RangeError} If `A` contains a non-finite entry.
 * @throws {Error} If the QR iteration fails to converge, which is not expected for finite input.
 */
export function singularValues(A: Matrix): Vector {
  const { a, mw, nw, scale } = prepare(A, 'singularValues');
  const bd = bidiagonalize(a, mw, nw);
  bdsqr(bd.d, bd.e, nw, [], []);
  return Vector.wrapUnchecked(unscale(bd.d, scale));
}
