# Matrix SVD, least squares, pseudoinverse and rank vs SciPy Benchmark

Reference values are generated with `scipy.linalg.svd` (LAPACK `gesdd`) for the
singular values, `scipy.linalg.lstsq(..., lapack_driver="gelsd")` for the
minimum-norm least-squares solution and its effective rank, `scipy.linalg.pinv`
for the pseudoinverse, and `numpy.linalg.matrix_rank` / `numpy.linalg.cond` for
the numerical rank and the 2-norm condition number. SciPy's `lstsq` defaults to
the cutoff `eps * s[0]`, while `numpy.linalg.lstsq`, `scipy.linalg.pinv`,
`numpy.linalg.matrix_rank` and `Matrix.lstsq()` use `max(rows, cols) * eps *
s[0]`; the generator passes that cutoff to `lstsq` explicitly, so all
quantities share one definition of the numerical rank.

The comparison suite checks, per case:

- the singular values, to `50 * eps * s[0]` (the absolute accuracy both
  implementations guarantee);
- the reconstruction residual `||U diag(s) V^T - A|| / ||A||` and the
  orthogonality of `U` and `V` (the factors themselves are not compared, since
  their signs and, for repeated singular values, their directions are not
  unique);
- the numerical rank and the 2-norm condition number;
- the least-squares solution and its rank, and the pseudoinverse, to
  `1e-13 + 20 * eps * kappa`, where `kappa = s[0] / s[rank - 1]` is the
  condition number of the part of `A` kept by the rank cutoff.

The suite covers square, tall, wide, rank-deficient (tall and wide),
ill-conditioned (Hilbert), dynamic-range and unequal-column-norm matrices, the
Läuchli matrix (whose small singular values are lost by forming `A^T A`) and a
50 x 50 unit upper triangular matrix with `-1` above the diagonal, which is
nonsingular but has numerical rank 49, the case where elimination-based
`rank()` fails and `rankSvd()` agrees with NumPy.

Measured agreement (largest singular-value difference relative to `s[0]`, in
units of machine epsilon, and relative differences of the solution and the
pseudoinverse):

| Case                 | Shape | Rank | max &#124;Δs&#124; / s[0] (eps) | Δx (relative) | Δpinv (relative) |
| -------------------- | ----- | ---- | ------------------------------- | ------------- | ---------------- |
| well_conditioned     | 3x3   | 3    | 2.0                             | 3.5e-16       | 3.3e-16          |
| tall                 | 3x2   | 2    | 0.1                             | 4.8e-16       | 1.5e-16          |
| wide                 | 2x3   | 2    | 0.8                             | 8.1e-16       | 9.5e-16          |
| tall_unequal_columns | 8x4   | 4    | 1.0                             | 8.8e-16       | 1.6e-15          |
| hilbert_8            | 8x8   | 8    | 0.3                             | 2.3e-9        | 1.4e-9           |
| hilbert_10x6         | 10x6  | 6    | 2.4                             | 1.9e-11       | 2.3e-11          |
| rank_deficient_tall  | 6x4   | 2    | 0.2                             | 3.6e-16       | 3.6e-16          |
| rank_deficient_wide  | 3x6   | 2    | 0.6                             | 1.7e-15       | 1.2e-15          |
| wide_dynamic_range   | 4x4   | 4    | 0.0                             | 7.8e-13       | 7.7e-13          |
| lauchli              | 4x3   | 3    | 1.2                             | 3.6e-16       | 2.9e-16          |
| kahan_like_50        | 50x50 | 49   | 2.1                             | 1.6e-14       | not stored       |

The Hilbert and dynamic-range differences in `x` and `pinv` are not errors of
either implementation: they reflect the condition number (`eps * kappa` is
`3e-6` for `hilbert_8`), and both sides are backward stable.
