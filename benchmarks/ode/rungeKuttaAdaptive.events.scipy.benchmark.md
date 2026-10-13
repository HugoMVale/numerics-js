# Adaptive Runge-Kutta Events vs SciPy Benchmark Results

Reference values come from `scipy.integrate.solve_ivp` using `RK23` and `RK45`.
Both solvers use `atol = 1e-8` and `rtol = 1e-8`; each case uses the same fixed
initial step (`first_step` in SciPy, `h0` in numerics-js).

| Test problem                                  | RK23 events |  RK23 final t | RK45 events |  RK45 final t | Terminal |
| :-------------------------------------------- | ----------: | ------------: | ----------: | ------------: | :------: |
| Harmonic oscillator zero crossings            |           4 |            12 |           4 |            12 |    No    |
| Exponential decay threshold                   |           1 | 1.3862942967  |           1 | 1.3862943617  |   Yes    |
| Logistic growth rising threshold              |           1 |             1 |           1 |             1 |    No    |
| Harmonic oscillator, stop at third crossing   |           3 | 7.8539816323  |           3 | 7.8539816251  |   Yes    |
| Backward exponential growth                   |           1 |            -1 |           1 |            -1 |    No    |
| Harmonic oscillator with two events           |  4 (2 each) | 5.7595864327  |  4 (2 each) | 5.7595865098  |   Yes    |

The comparison test checks that event counts match and that every event time,
event-state component, and final integration time differs from SciPy by less
than `1e-8`. It also checks that terminal events stop integration at the
reference final time. The cases cover crossings in both directions, a direction
filter, terminal events, stopping at the Nth occurrence, multiple event
functions, and integration backward in time.

The focused run of the SciPy comparison and event feature tests passed: **42
tests across 2 files**. These tests compare event behavior and reference values;
they do not benchmark or assert derivative-evaluation performance.
