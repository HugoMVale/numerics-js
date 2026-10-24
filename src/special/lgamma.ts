/*
 * The algorithm and coefficients below are those of the FreeBSD / fdlibm
 * implementation of the log-gamma function (`e_lgamma_r.c`).
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 */
import { sinpi } from './sinpi.js';

/** Below this magnitude `lgamma(x)` is `-ln|x|` to working precision. */
const TINY = 1.3877787807814457e-17; // 2^-56

const TWO52 = 4503599627370496; // 2^52: beyond this every double is an integer
const TWO56 = 72057594037927936; // 2^56: beyond this `lgamma(x) = x (ln x - 1)`

/**
 * Boundaries of the sub-intervals of `(0, 2)`: the doubles whose high words
 * (`0x3FCDA661`, `0x3FE76944`, `0x3FECCCCD`, `0x3FF3B4C4`, `0x3FFBB4C3`, low
 * word zero) fdlibm compares against. They are close to, but not equal to,
 * `ymin - 1 - 0.23`, `ymin - 1 + 0.27`, `0.9`, `ymin - 0.23` and
 * `ymin + 0.27`, where `ymin = 1.4616...` is the location of the minimum of
 * `Gamma`. The polynomials for the region around `ymin` are fitted on
 * `[ymin - 0.23, ymin + 0.27]`: using `ymin - 0.27` as the lower end, as some
 * ports do, evaluates them up to 0.04 outside it and loses about 5e-15.
 */
const X_0_2316 = 0.23163998126983643;
const X_0_7316 = 0.7315998077392578;
const X_0_9 = 0.9000000953674316;
const X_1_2316 = 1.2316322326660156;
const X_1_7316 = 1.7316312789916992;

const TC = 1.4616321449683622; // 0x3FF762D86356BE3F, the expansion point near the minimum of Gamma
const TF = -0.12148629053584961; // 0xBFBF19B9BCC38A42, lgamma(TC)
const TT = -3.638676997039505e-18; // 0xBC50C7CAA48A971F, -(tail of TF)

// 0 < x < 2 near the ends of the interval: lgamma(x) = -ln(x) + p(x) for x < 0.9,
// with y = 1 - x (A), y = x - (TC - 1) (T) or y = x (U/V); the same forms are used on [0.9, 2).
const A1C = 0.07721566490153287;
const A1 = [
  0.06735230105312927, 0.007385550860814029, 0.0011927076318336207, 0.00022086279071390839,
  2.5214456545125733e-5
];
const A2C = 0.3224670334241136;
const A2 = [
  0.020580808432516733, 0.0028905138367341563, 0.0005100697921535113, 0.00010801156724758394,
  4.4864094961891516e-5
];
const T1C = 0.48383612272381005;
const T1 = [
  -0.032788541075985965, 0.006100538702462913, -0.0014034646998923284, 0.00031563207090362595
];
const T2C = -0.1475877229945939;
const T2 = [
  0.01797067508118204, -0.0036845201678113826, 0.000881081882437654, -0.00031275416837512086
];
const T3C = 0.06462494023913339;
const T3 = [
  -0.010314224129834144, 0.0022596478090061247, -0.0005385953053567405, 0.0003355291926355191
];
const UC = -0.07721566490153287;
const U = [
  0.6328270640250934, 1.4549225013723477, 0.9777175279633727, 0.22896372806469245,
  0.013381091853678766
];
const VC = 1;
const V = [
  2.4559779371304113, 2.128489763798934, 0.7692851504566728, 0.10422264559336913,
  0.003217092422824239
];

// 2 <= x < 8: lgamma(2 + y) = y / 2 + y * P(y) / Q(y), y in [0, 1).
const SC = -0.07721566490153287;
const S = [
  0.21498241596060885, 0.325778796408931, 0.14635047265246445, 0.02664227030336386,
  0.0018402845140733772, 3.194753265841009e-5
];
const RC = 1;
const R = [
  1.3920053346762105, 0.7219355475671381, 0.17193386563280308, 0.01864591917156529,
  0.0007779424963818936, 7.326684307446256e-6
];

// 8 <= x < 2^56: lgamma(x) = (x - 1/2) (ln x - 1) + W(1 / x^2) / x + WC.
const WC = 0.4189385332046727;
const W = [
  0.08333333333333297, -0.0027777777772877554, 0.0007936505586430196, -0.00059518755745034,
  0.0008363399189962821, -0.0016309293409657527
];

/** Horner evaluation of the polynomial with ascending coefficients `c`. */
function polyval(c: readonly number[], x: number): number {
  let p = c[c.length - 1];
  for (let i = c.length - 2; i >= 0; i--) p = c[i] + x * p;
  return p;
}

/**
 * Computes the natural logarithm of the absolute value of the gamma
 * function, `ln|Gamma(x)|`.
 *
 * Unlike `Math.log(Math.abs(gamma(x)))`, which overflows for `x > 171.62`,
 * the result stays finite up to `x ≈ 2.5e305`. The implementation follows
 * the classic FreeBSD / fdlibm algorithm:
 *
 * - `0 < x < 2`: `lgamma(x) = lgamma(x + 1) - ln(x)`, with three polynomial
 *   or rational approximations that are accurate near the zeros at `x = 1` and
 *   `x = 2` and around the minimum of `Gamma` at `x = 1.4616...`.
 * - `2 <= x < 8`: a rational approximation on `[2, 3)` and upward recurrence.
 * - `8 <= x < 2^56`: `(x - 1/2) (ln x - 1)` plus a polynomial correction in
 *   `1 / x^2`; for `x >= 2^56`, `x (ln x - 1)`.
 * - `x < 0`: the reflection formula
 *   `ln|Gamma(x)| = ln(pi / |x sin(pi x)|) - lgamma(-x)`, with `sin(pi x)`
 *   reduced exactly.
 *
 * Accuracy: for `x > 0` the error is at most 2 ulp of the result (typically
 * well below 1), including close to the zeros at `x = 1` and `x = 2` where
 * the relative accuracy is preserved. For `x < 0` the result is the difference
 * `ln(pi / |x sin(pi x)|) - lgamma(-x)` of two terms, and its error is at most
 * about 5 ulp of the larger term (about 2 ulp of the result when the terms do
 * not cancel). The relative error therefore grows where the result is much
 * smaller than the terms: near the zeros of `ln|Gamma|` on the negative axis
 * (about `-2.4570`, `-2.7477`, `-3.1436`, ...) and, mildly, next to the poles.
 * It relies on `Math.log` being accurate to about 1 ulp, as it is in V8.
 *
 * Special cases (the same conventions as C `lgamma` and SciPy's `gammaln`):
 *
 * - `lgamma(NaN)` is `NaN`.
 * - `lgamma(±0)` and `lgamma(n)` for negative integers `n` are `Infinity`.
 * - `lgamma(±Infinity)` is `Infinity`; so is `lgamma(x)` for `x > 2.56e305`, where the
 *   result overflows.
 * - For `|x| < 2^-56` the result is `-ln|x|`, also for subnormal `x`.
 *
 * The sign of `Gamma(x)` is not returned; for `x < 0` it is positive on
 * `(-2, -1)`, `(-4, -3)`, ... and negative on `(-1, 0)`, `(-3, -2)`, ...
 *
 * @param x Argument; any real number.
 * @returns The value of `ln|Gamma(x)|`.
 *
 * @example
 * ```ts
 * import { lgamma } from 'numerics-js/special';
 *
 * console.log(lgamma(0.5));
 * console.log(lgamma(1000));
 * console.log(lgamma(-2.5));
 * ```
 *
 * Output:
 * ```text
 * 0.5723649429247001
 * 5905.220423209181
 * -0.05624371649767412
 * ```
 */
export function lgamma(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity || x === -Infinity || x === 0) return Infinity;

  const negative = x < 0;
  if (negative) x = -x;

  if (x < TINY) return -Math.log(x);

  let nadj = 0;
  if (negative) {
    if (x >= TWO52) return Infinity; // an integer
    const t = sinpi(x);
    if (t === 0) return Infinity; // a pole
    nadj = Math.log(Math.PI / Math.abs(t * x));
  }

  let r: number;
  if (x === 1 || x === 2) {
    r = 0;
  } else if (x < 2) {
    // lgamma(x) = lgamma(x + 1) - ln(x), for x <= 0.9
    let y: number;
    let region: number;
    if (x < X_0_9) {
      r = -Math.log(x);
      if (x >= X_0_7316) {
        y = 1 - x;
        region = 0;
      } else if (x >= X_0_2316) {
        y = x - (TC - 1);
        region = 1;
      } else {
        y = x;
        region = 2;
      }
    } else {
      r = 0;
      if (x >= X_1_7316) {
        y = 2 - x;
        region = 0;
      } else if (x >= X_1_2316) {
        y = x - TC;
        region = 1;
      } else {
        y = x - 1;
        region = 2;
      }
    }

    if (region === 0) {
      const z = y * y;
      const p1 = A1C + z * polyval(A1, z);
      const p2 = z * (A2C + z * polyval(A2, z));
      r += y * p1 + p2 - 0.5 * y;
    } else if (region === 1) {
      const z = y * y;
      const w = z * y;
      const p1 = T1C + w * polyval(T1, w);
      const p2 = T2C + w * polyval(T2, w);
      const p3 = T3C + w * polyval(T3, w);
      r += TF + (z * p1 - (TT - w * (p2 + y * p3)));
    } else {
      const p1 = y * (UC + y * polyval(U, y));
      const p2 = VC + y * polyval(V, y);
      r += -0.5 * y + p1 / p2;
    }
  } else if (x < 8) {
    // lgamma(2 + y) = y / 2 + y P(y) / Q(y); then lgamma(1 + s) = ln(s) + lgamma(s) up to x.
    const i = Math.trunc(x);
    const y = x - i;
    const p = y * (SC + y * polyval(S, y));
    const q = RC + y * polyval(R, y);
    r = 0.5 * y + p / q;
    let z = 1;
    for (let k = i - 1; k >= 2; k--) z *= y + k;
    if (i > 2) r += Math.log(z);
  } else if (x < TWO56) {
    const t = Math.log(x);
    const z = 1 / x;
    const w = WC + z * polyval(W, z * z);
    r = (x - 0.5) * (t - 1) + w;
  } else {
    r = x * (Math.log(x) - 1);
  }

  return negative ? nadj - r : r;
}
