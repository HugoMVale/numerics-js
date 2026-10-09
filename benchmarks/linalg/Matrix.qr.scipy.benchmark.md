# Matrix.qr() vs SciPy Benchmark

Reference values are generated with `scipy.linalg.qr`. The comparison suite
checks matrix shape, relative reconstruction residual, and orthogonality
residual. For determined-rank cases it also compares the absolute values of
the `Q` and `R` entries to account for the sign ambiguity of QR factors.

The suite covers square, tall, wide, Hilbert, dynamic-range, and
rank-deficient matrices. The rank-deficient case is validated through
reconstruction and orthogonality rather than direct factor comparison.
