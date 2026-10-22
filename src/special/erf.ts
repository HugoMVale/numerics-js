/*
 * The algorithm and coefficients below are those of the FreeBSD / fdlibm
 * implementation of the error function (`s_erf.c`).
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 */

/** Below this magnitude `erf(x)` is computed from its leading Taylor term. */
const SMALL = 3.725290298461914e-9; // 2^-28

/** Below this magnitude `8 * x` is used to avoid underflow of `efx * x`. */
const VERY_TINY = 2.848094538889218e-306;

/** `erf(1)` rounded to 24 bits: the expansion point of the `0.84375 <= |x| < 1.25` branch. */
const ERX = 8.45062911510467529297e-1;

/** `2/sqrt(pi) - 1`. */
const EFX = 1.28379167095512586316e-1;

/** `8 * (2/sqrt(pi) - 1)`. */
const EFX8 = 1.02703333676410069053;

// |x| < 0.84375: erf(x) = x + x * R(x^2), R = P/Q (odd rational minimax).
const PP0 = 1.28379167095512558561e-1;
const PP1 = -3.250421072470015e-1;
const PP2 = -2.848174957559851e-2;
const PP3 = -5.770270296489442e-3;
const PP4 = -2.3763016656650163e-5;
const QQ1 = 3.9791722395915535e-1;
const QQ2 = 6.50222499887673e-2;
const QQ3 = 5.081306281875766e-3;
const QQ4 = 1.3249473800432164e-4;
const QQ5 = -3.960228278775368e-6;

// 0.84375 <= |x| < 1.25: erf(x) = ERX + P(s)/Q(s), s = |x| - 1.
const PA0 = -2.3621185607526594e-3;
const PA1 = 4.1485611868374833e-1;
const PA2 = -3.722078760357013e-1;
const PA3 = 3.1834661990116175e-1;
const PA4 = -1.1089469428239668e-1;
const PA5 = 3.5478304325618236e-2;
const PA6 = -2.166375594868791e-3;
const QA1 = 1.0642088040084423e-1;
const QA2 = 5.40397917702171e-1;
const QA3 = 7.182865441419627e-2;
const QA4 = 1.2617121980876164e-1;
const QA5 = 1.363708391202905e-2;
const QA6 = 1.1984499846799107e-2;

// 1.25 <= |x| < 1/0.35: erfc(x) = exp(-x^2 - 0.5625 + RA(z)/SA(z)) / x, z = 1/x^2.
const RA0 = -9.864944034847148e-3;
const RA1 = -6.938585727071818e-1;
const RA2 = -1.0558626225323291e1;
const RA3 = -6.2375332450326006e1;
const RA4 = -1.6239666946257347e2;
const RA5 = -1.8460509290671104e2;
const RA6 = -8.12874355063066e1;
const RA7 = -9.814329344169145;
const SA1 = 1.9651271667439257e1;
const SA2 = 1.3765775414351904e2;
const SA3 = 4.3456587747522923e2;
const SA4 = 6.453872717332679e2;
const SA5 = 4.2900814002756783e2;
const SA6 = 1.0863500554177944e2;
const SA7 = 6.570249770319282;
const SA8 = -6.04244152148581e-2;

// 1/0.35 <= |x| < 6 (erf) or < 28 (erfc): same form with RB/SB.
const RB0 = -9.86494292470009928597e-3;
const RB1 = -7.99283237680523e-1;
const RB2 = -1.7757954917754752e1;
const RB3 = -1.6063638485582192e2;
const RB4 = -6.375664433683896e2;
const RB5 = -1.0250951316110772e3;
const RB6 = -4.835191916086514e2;
const SB1 = 3.033806074348246e1;
const SB2 = 3.257925129965739e2;
const SB3 = 1.536729586084437e3;
const SB4 = 3.1998582195085955e3;
const SB5 = 2.5530504064331644e3;
const SB6 = 4.7452854120695537e2;
const SB7 = -2.244095244658582e1;

/** Scratch view used to clear the low 32 bits of a double (endianness independent). */
const scratch = new DataView(new ArrayBuffer(8));

/**
 * `exp(-x^2 - 0.5625 + r/s) / x` for `x >= 1.25`, the asymptotic form shared
 * by `erf` and `erfc`.
 *
 * To keep `x^2` exact in the exponent, `x` is split as `z + (x - z)` with `z`
 * having its low 32 bits cleared, so that
 * `exp(-x^2 - 0.5625 + r/s) = exp(-z^2 - 0.5625) * exp((z - x)(z + x) + r/s)`
 * and `z^2` is exactly representable.
 */
function expTail(x: number, r: number, s: number): number {
  scratch.setFloat64(0, x);
  scratch.setUint32(4, 0);
  const z = scratch.getFloat64(0);
  return (Math.exp(-z * z - 0.5625) * Math.exp((z - x) * (z + x) + r / s)) / x;
}

/**
 * Computes the error function,
 * `erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.
 *
 * The implementation follows the classic FreeBSD / fdlibm algorithm. The
 * argument is split into regions, each with a dedicated rational minimax
 * approximation:
 *
 * - `|x| < 0.84375`: `x + x * R(x^2)`; for `|x| < 2^-28` the leading Taylor
 *   term.
 * - `0.84375 <= |x| < 1.25`: expansion around `|x| = 1`.
 * - `1.25 <= |x| < 6`: `1 - erfc(|x|)` from the asymptotic form of `erfc`.
 * - `|x| >= 6`: exactly `±1`, since `1 - erf(6)` is below half an ulp of 1.
 *
 * Accuracy: the error is below 1 ulp over the whole real line. `erf` is odd,
 * `erf(±0) = ±0`, `erf(±Infinity) = ±1`, and `NaN` returns `NaN`.
 *
 * @param x Argument; any real number.
 * @returns The value of `erf(x)`, in `[-1, 1]`.
 *
 * @example
 * ```ts
 * import { erf } from 'numerics-js/special';
 *
 * console.log(erf(0.5));
 * console.log(erf(-2));
 * console.log(erf(10));
 * ```
 *
 * Output:
 * ```text
 * 0.5204998778130465
 * -0.9953222650189527
 * 1
 * ```
 */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === 0) return x;

  const ax = Math.abs(x);

  if (ax < 0.84375) {
    if (ax < SMALL) {
      if (ax < VERY_TINY) return 0.125 * (8 * x + EFX8 * x);
      return x + EFX * x;
    }
    const z = x * x;
    const r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
    const s = 1 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
    return x + x * (r / s);
  }

  if (ax < 1.25) {
    const s = ax - 1;
    const p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    const q = 1 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
    return x < 0 ? -ERX - p / q : ERX + p / q;
  }

  if (ax >= 6) return x < 0 ? -1 : 1;

  const z = 1 / (ax * ax);
  let r: number;
  let s: number;
  if (ax < 2.857142857142857) {
    r = RA0 + z * (RA1 + z * (RA2 + z * (RA3 + z * (RA4 + z * (RA5 + z * (RA6 + z * RA7))))));
    s =
      1 +
      z * (SA1 + z * (SA2 + z * (SA3 + z * (SA4 + z * (SA5 + z * (SA6 + z * (SA7 + z * SA8)))))));
  } else {
    r = RB0 + z * (RB1 + z * (RB2 + z * (RB3 + z * (RB4 + z * (RB5 + z * RB6)))));
    s = 1 + z * (SB1 + z * (SB2 + z * (SB3 + z * (SB4 + z * (SB5 + z * (SB6 + z * SB7))))));
  }
  const tail = expTail(ax, r, s);
  return x < 0 ? tail - 1 : 1 - tail;
}
