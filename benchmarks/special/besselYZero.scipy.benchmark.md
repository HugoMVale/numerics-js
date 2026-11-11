# besselYZero vs SciPy Benchmark Results

Reference: `scipy.special.yn_zeros`. Each order is checked at the zero indices
`m = 1, 2, 3, 5, 10, 20, 50, 100, 200`, requested from the largest index down
so that both the scan and the cache are exercised; a second test walks through
all consecutive zeros up to `m = 60` to detect missed or duplicated zeros. The
test enforces a relative tolerance of `1e-13`.

| Order `n` | Zeros compared | Max relative error |
| --------: | -------------: | -----------------: |
|         0 |              9 |            0.0e+0  |
|         1 |              9 |            1.6e-16 |
|         2 |              9 |            1.3e-16 |
|         5 |              9 |            1.7e-16 |
|        10 |              9 |            2.2e-16 |
|        30 |              9 |            1.6e-16 |
|       100 |              9 |            1.5e-16 |

The largest observed relative error is `2.2e-16` (`n = 10`), i.e. the zeros
agree with SciPy to round-off.
