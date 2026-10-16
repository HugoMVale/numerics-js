# Matrix.qrPivot() vs SciPy Benchmark

Reference values are generated with `scipy.linalg.qr(..., pivoting=True)`,
using `mode="raw"` for the compact LAPACK (`dgeqp3`) storage and
`mode="economic"` for the thin `Q` and `R` factors. The comparison suite checks
shapes, the relative reconstruction residual `A[:, perm] = Q*R`, and the
orthogonality residual of the thin `Q`. For determined-rank cases it also
compares the column permutation, the compact `QR` matrix, the Householder
scalars `tau`, and `R` directly: both implementations follow LAPACK's
Householder sign convention, so the signs agree.

The suite covers square, tall, wide, ill-conditioned (Hilbert), dynamic-range,
and unequal-column-norm matrices. The rank-deficient case is validated through
reconstruction and orthogonality only, since its trailing columns are
determined by rounding noise.
