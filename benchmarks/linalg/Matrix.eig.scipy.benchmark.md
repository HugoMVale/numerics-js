# Matrix.eig() vs SciPy Benchmark

Reference values are generated with `scipy.linalg.eig`. The comparison suite
checks eigenvalues to a scale-aware tolerance of `1e-9` and checks the overlap
of each corresponding complex eigenvector to at least `1 - 1e-8`.

The suite covers:

| Case | Matrix |
| --- | --- |
| `well_conditioned` | Well-conditioned symmetric 3x3 |
| `complex_pair` | 4x4 matrix with two distinct complex-conjugate pairs |
| `nonsymmetric` | Nonsymmetric 4x4 matrix with distinct real eigenvalues |
| `hilbert_5x5` | 5x5 Hilbert matrix |

Eigenvectors are compared by overlap rather than component-wise equality
because their signs and complex phases are not unique.
