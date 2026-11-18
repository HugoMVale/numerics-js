/** Result of a {@link gaussKronrod} quadrature. */
export interface GaussKronrodResult {
  /**
   * Whether the requested error tolerance was reached, i.e. `error <= atol + rtol * |value|`;
   * see `message`. If `false` (subinterval limit reached, or bisection reached floating-point
   * resolution at an endpoint singularity), `value` and `error` are the best estimates found.
   */
  success: boolean;
  /** Message describing the reason for termination. */
  message: string;
  /** Total number of integrand evaluations performed. */
  evaluations: number;
  /** Total number of final subintervals used by the adaptive algorithm. */
  subintervals: number;
  /** Estimated value of the definite integral. */
  value: number;
  /**
   * Estimated absolute integration error: the sum of the local estimates `|kronrod - gauss|`.
   * It is not a guaranteed bound. For smooth integrands it usually exceeds the actual error
   * by orders of magnitude; for integrands with mild singularities or kinks it is close to
   * it; for strong singularities (e.g. `x^-0.9` at `x = 0`) it can be several times smaller,
   * so the requested tolerance may be missed. A narrow feature that falls between all the
   * nodes of a subinterval can go unnoticed and make the estimate far too small;
   * split the interval to resolve it.
   */
  error: number;
}

/**
 * Options for {@link gaussKronrod} and, through {@link QuadOptions}, for {@link quad}.
 *
 * The adaptive algorithm stops as soon as the estimated error of the whole
 * integral satisfies `error <= atol + rtol * |value|`, the same convention used
 * by the ODE solvers.
 */
export interface GaussKronrodOptions {
  /**
   * Absolute error tolerance. It dominates when the integral is close to zero.
   * @default 1e-8
   */
  atol?: number;
  /**
   * Relative error tolerance, with respect to the magnitude of the estimated
   * integral. With `atol = 0` the stopping criterion is purely relative, hence
   * invariant under a rescaling of the integrand.
   * @default 1e-8
   */
  rtol?: number;
  /**
   * Safety limit on the total number of subintervals of the whole
   * integral, including the initial ones.
   * @default 200
   */
  maxSubintervals?: number;
}
