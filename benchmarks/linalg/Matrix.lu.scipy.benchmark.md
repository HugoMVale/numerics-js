# Matrix.lu() vs SciPy Benchmark

Reference values are generated with `scipy.linalg.lu`. The comparison suite
checks the row permutation and permutation sign for every case. It compares
the factor matrices directly for well-conditioned cases and checks the
relative reconstruction residual for ill-conditioned or dynamically scaled
cases.

The suite covers well-conditioned, row-pivoted, mixed-scale, Hilbert, and
twelve-orders-of-magnitude dynamic-range matrices.
