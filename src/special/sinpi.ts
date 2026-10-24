/**
 * Computes `sin(pi * x)` without the argument error of `Math.sin(Math.PI * x)`.
 *
 * The argument is reduced exactly (`x mod 2` is exact in floating point) to
 * `r` in `[0, 1/2]` using the symmetries `sin(pi (x + 1)) = -sin(pi x)` and
 * `sin(pi (1 - r)) = sin(pi r)`, and then `sin(pi r)` or `cos(pi (1/2 - r))` is
 * evaluated, whichever has the smaller argument (`<= pi/4`). Hence the result
 * is exactly `±0` at every integer, exactly `±1` at half-integers, and keeps
 * a relative error of about 1 ulp near the zeros, however large `|x|` is.
 *
 * `NaN` is returned for `NaN` and `±Infinity`; `sinpi(-0)` is `-0`.
 *
 * @param x Argument.
 * @returns `sin(pi * x)`.
 *
 * @internal
 */
export function sinpi(x: number): number {
  if (x < 0) return -sinpi(-x);

  let r = x % 2;
  let sign = 1;
  if (r > 1) {
    r -= 1;
    sign = -1;
  }
  if (r > 0.5) r = 1 - r;

  return sign * (r <= 0.25 ? Math.sin(Math.PI * r) : Math.cos(Math.PI * (0.5 - r)));
}
