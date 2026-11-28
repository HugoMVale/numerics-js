# BandedMatrix and tridiagonal LU vs SciPy/LAPACK Benchmark

Reference values are generated with `scipy.linalg.solve_banded` (the solution), and with the
LAPACK routines `dgbtrf` (general band LU with partial pivoting) and `dgttrf` (tridiagonal LU)
through `scipy.linalg.lapack`. Regenerate them with
`uv run benchmarks/linalg/generate_banded_fixtures.py`.

The comparison suite (`BandedMatrix.scipy.test.ts`) checks, for each of the 13 cases:

- the solution of `BandedMatrix.solve` against `solve_banded` (bound `max(1e-13, 20 * cond1 * eps)`),
  and the normwise backward error of a second right-hand side solved with the reused factorization;
- for well-conditioned cases (`cond1 < 1e4`), the row interchanges `ipiv` exactly and the factors in
  `BandedLUDecomposition.ab` against the output of `dgbtrf` entry by entry;
- for the tridiagonal cases, `tridiagonalLU` (`dl`, `d`, `du`, `du2`, `ipiv`) against `dgttrf`.

The cases cover `n = 1`, a diagonal matrix, lower and upper bidiagonal, tridiagonal, pentadiagonal
and asymmetric bands (`kl != ku`), a full band (`kl = ku = n - 1`), and systems of order 200 and
300. The cases named `*_pivoting` have a small random diagonal, so that rows are interchanged at
many steps.

Largest deviation from the reference over the 13 cases (Node 24):

| Quantity                                        | Largest deviation |
| ----------------------------------------------- | ----------------- |
| Relative solution difference (`max-norm`)       | 4.9e-15           |
| Entry of the LU factors (`dgbtrf`)              | 6.2e-15           |
| Row interchanges equal to LAPACK's (13 of 13)   | exact             |

## Timing

`banded.perf.mjs` is a hand-run check that the cost grows as predicted
(`node benchmarks/linalg/banded.perf.mjs dist --check`, after `npm run build`). It fits the
log-log slope of the best-of-15 time against the size:

| Sweep                                        | Predicted exponent | Measured (3 runs) |
| -------------------------------------------- | ------------------ | ----------------- |
| `lu()`, `kl = ku = 4`, `n` from 2e3 to 3.2e4 | 1                  | 1.04 - 1.06       |
| `lu()`, `n = 2000`, `kl = ku = k`, 64..512   | 2                  | 1.79 - 1.83       |
| `lu()`, `ku = 1000`, `kl` from 16 to 128     | 1                  | 0.82              |
| `solveLU()`, `kl = ku = 4`, `n` to 6.4e4     | 1                  | 1.07 - 1.09       |
| `solveLU()`, `n = 4000`, `kl = ku = k`       | 1                  | 0.92 - 0.97       |

The exponent of `k` for `lu()` stays a little below 2 because of the `O(n * k)` terms (copying the
band into the factor workspace, pivot search) and, at `k = 512`, cache effects.
