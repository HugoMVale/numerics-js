import { sinpi } from './sinpi.js';

/**
 * Largest argument with a finite result: `gamma(GAMMA_OVERFLOW)` is just below
 * the largest double, while the next double up already overflows.
 */
const GAMMA_OVERFLOW = 171.6243769563027;

/**
 * Below this argument (more negative) the result rounds to `±0`: next to the
 * pole at `-184` the magnitude is only `1.6e-325`, under half the smallest
 * subnormal.
 */
const GAMMA_UNDERFLOW = -184;

/** Stirling is used from here on; below, the polynomial core and recurrence. */
const STIRLING_MIN = 10;

/** Above this argument `x^(x - 1/2)` times the series factor would overflow, so `x^(x/2 - 1/4)` is squared in two steps. */
const SPLIT_POW_MIN = 143;

/** Beyond this `q = -x`, `Gamma(1 + q)` overflows and the reflection formula is scaled by hand. */
const REFLECTION_SCALED_MIN = 170;

/**
 * Coefficients (ascending powers) of `P` in `1 / Gamma(1 + t) = 1 + t * P(t)`
 * for `t` in `[0, 1]`: a degree-14 Chebyshev-node fit with maximum absolute
 * error `1.6e-18` (below 0.02 ulp of the result, which lies in `[1, 1.13]`).
 * `P(0) = Euler-Mascheroni constant`. Derived with
 * `scripts/derive_gamma_coefficients.py`.
 */
const RECIPROCAL_GAMMA_P = [
  0.5772156649015329, -0.6558780715202535, -0.042002635034127836, 0.16653861138323703,
  -0.042197734569778066, -0.009621971400288845, 0.007218942511576732, -0.0011651647525430678,
  -0.00021524914822683067, 0.000128063513304448, -0.000020149457390763776,
  -0.0000012436534652553382, 0.0000011389962411508655, -2.1896988906648608e-7, 1.7200070946431934e-8
];

/**
 * Coefficients `sqrt(2 pi) * a_n` of the Stirling series
 * `Gamma(x) = sqrt(2 pi) x^(x - 1/2) e^(-x) * sum_n a_n x^(-n)`, where
 * `a_n` are the coefficients of the exponentiated Bernoulli series
 * (`1, 1/12, 1/288, -139/51840, ...`), each correctly rounded from exact
 * rational arithmetic. Folding `sqrt(2 pi)` in saves one rounding error. Fourteen
 * terms truncate with relative error `2.3e-17` at `x = 10`. Derived with
 * `scripts/derive_gamma_coefficients.py`.
 */
const STIRLING = [
  2.5066282746310007, 0.20888568955258338, 0.008703570398024307, -0.006721090474029882,
  -0.0005752012381101712, 0.001965294881583203, 0.00017478252120455912, -0.0014843411351582762,
  -0.00012963757321125544, 0.0021043112297532062, 0.00018059994565555043, -0.004798785670546346,
  -0.0004073678593815252, 0.016050850331944595
];

/** `FACTORIAL[n] = n!` for `n = 0..170`, each correctly rounded (`BigInt` to `Number` rounds to nearest). */
const FACTORIAL: readonly number[] = (() => {
  const table: number[] = [1];
  let f = 1n;
  for (let n = 1; n <= 170; n++) {
    f *= BigInt(n);
    table.push(Number(f));
  }
  return table;
})();

/** `1 / Gamma(1 + t)` for `t` in `[0, 1]`. */
function reciprocalGamma1p(t: number): number {
  let p = RECIPROCAL_GAMMA_P[14];
  for (let i = 13; i >= 0; i--) p = RECIPROCAL_GAMMA_P[i] + t * p;
  return 1 + t * p;
}

/** `sqrt(2 pi) * sum_n a_n z^n`, the scaled Stirling series in `z = 1/x`. */
function stirlingSeries(z: number): number {
  let s = STIRLING[13];
  for (let i = 12; i >= 0; i--) s = STIRLING[i] + z * s;
  return s;
}

/** `Gamma(x)` for finite `x > 0` that is not above the overflow threshold. */
function gammaPositive(x: number): number {
  if (x < 1) return 1 / (x * reciprocalGamma1p(x));

  if (x < STIRLING_MIN) {
    if (Number.isInteger(x)) return FACTORIAL[x - 1];
    // Gamma(x) = (x-1)(x-2)...(y) * Gamma(y) with y in [1, 2); every y is an exact subtraction.
    let z = 1;
    let y = x;
    while (y >= 2) {
      y -= 1;
      z *= y;
    }
    return z / reciprocalGamma1p(y - 1);
  }

  if (x <= 171 && Number.isInteger(x)) return FACTORIAL[x - 1];
  if (x <= SPLIT_POW_MIN) return (stirlingSeries(1 / x) * Math.pow(x, x - 0.5)) / Math.exp(x);
  const v = Math.pow(x, x / 2 - 0.25);
  return stirlingSeries(1 / x) * v * (v / Math.exp(x));
}

/**
 * `Gamma(x)` for finite, non-integer `x < 0`, through the reflection formula
 * `Gamma(x) = pi / (sin(pi x) * Gamma(1 - x))` with `Gamma(1 - x) = q * Gamma(q)`,
 * `q = -x`.
 */
function gammaNegative(x: number): number {
  if (x > -1) {
    // Gamma(x) = Gamma(1 + x) / x = 1 / (x * (1 + x) * R(1 + x)). Rounding 1 + x (for -0.5 < x < 0) is harmless:
    // Gamma is smooth around 1, so a relative error of 1e-16 in the argument stays far below an ulp.
    const y = 1 + x;
    return 1 / (x * y * reciprocalGamma1p(y));
  }

  const s = sinpi(x);
  if (x <= GAMMA_UNDERFLOW) return s < 0 ? -0 : 0;

  const q = -x;
  if (q <= REFLECTION_SCALED_MIN) return Math.PI / (s * (q * gammaPositive(q)));

  // 170 < q < 184: Gamma(1 + q) overflows, but the quotient is a (sub)normal number.
  // Divide by the two factors of the split Stirling form one at a time, keeping each
  // intermediate representable, so that only the last division rounds into the subnormal range.
  const v = Math.pow(q, q / 2 - 0.25);
  return Math.PI / s / v / (q * stirlingSeries(1 / q) * (v * Math.exp(-q)));
}

/**
 * Computes the gamma function,
 * `Gamma(x) = integral_0^Infinity t^(x-1) exp(-t) dt`, extended to negative
 * non-integer arguments by analytic continuation. For positive integers,
 * `Gamma(n) = (n - 1)!`.
 *
 * The algorithm is chosen by region:
 *
 * - Integers `1 <= x <= 171`: a table of correctly rounded factorials, so
 *   `gamma(n)` is exact whenever `(n - 1)!` is representable and otherwise the
 *   nearest double.
 * - `0 < x < 10`: the reciprocal `1 / Gamma(1 + t)` is a polynomial in
 *   `t = x - 1` on `[0, 1]` (reached by exact steps of the recurrence
 *   `Gamma(x) = (x - 1) Gamma(x - 1)`, or by `Gamma(x) = Gamma(1 + x) / x` below 1).
 * - `x >= 10`: Stirling's asymptotic series with 14 terms. Above `x = 143` the
 *   power is split in two factors so that nothing overflows early.
 * - `x < 0`: the reflection formula `pi / (sin(pi x) * Gamma(1 - x))`, with an
 *   exactly reduced `sin(pi x)`. For `-184 < x < -170.57`, where the result is
 *   subnormal or very small but not zero, the quotient is scaled so that no
 *   intermediate overflows.
 *
 * Accuracy: against 60-digit reference values the error is at most 6 ulp over
 * the whole real line (observed maximum 5.1 on about 45,000 test points; about
 * 4 ulp for `x > 0`, 2 ulp for `0 < x < 3`), and the mean error is below 1 ulp.
 * It relies on `Math.pow` and `Math.exp` being accurate to about 1 ulp, as they
 * are in V8.
 *
 * Special cases (the same conventions as C `tgamma` and SciPy):
 *
 * - `gamma(NaN)` is `NaN`.
 * - `gamma(+0)` is `Infinity` and `gamma(-0)` is `-Infinity`.
 * - `gamma(x)` is `NaN` for negative integers and for `-Infinity`, where the
 *   function has poles with no single signed limit.
 * - `gamma(Infinity)` is `Infinity`; so is `gamma(x)` for `x > 171.6243769563027`,
 *   where the result overflows.
 * - For `x <= -184` the magnitude is below half the smallest subnormal and the
 *   result is `±0`, with the sign of `Gamma(x)` (positive on `(-2, -1)`,
 *   `(-4, -3)`, ..., negative on `(-1, 0)`, `(-3, -2)`, ...).
 *
 * @param x Argument; any real number.
 * @returns The value of `Gamma(x)`.
 *
 * @example
 * ```ts
 * import { gamma } from 'numerics-js/special';
 *
 * console.log(gamma(5));
 * console.log(gamma(0.5));
 * console.log(gamma(-1.5));
 * ```
 *
 * Output:
 * ```text
 * 24
 * 1.772453850905516
 * 2.363271801207355
 * ```
 */
export function gamma(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x > 0) return x > GAMMA_OVERFLOW ? Infinity : gammaPositive(x);
  if (x === 0) return 1 / x;
  if (x === -Infinity || Number.isInteger(x)) return NaN;
  return gammaNegative(x);
}
