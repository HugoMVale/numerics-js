# Adaptive TR-BDF2 vs SciPy Benchmark Results

SciPy has no TR-BDF2 solver, so accuracy is checked against a very tight
`scipy.integrate.solve_ivp` `Radau` integration (`atol = 1e-12`, `rtol = 1e-11`),
and the cost of `trBdf2Adaptive` is listed next to SciPy's `Radau`, `BDF` and
`LSODA` at the same tolerances. The cost columns are for context only; the test
asserts accuracy, not cost.

Two tolerance settings are used for every problem. The first is the default of this library
and of SciPy, MATLAB and Julia; the second is tighter. TR-BDF2 is a second-order method
and is aimed at moderate tolerances, as in Hosea & Shampine (1996), who test at `rtol = 5e-3`.

`numerics-js` uses finite-difference Jacobians here (the default), as SciPy does.
`Evaluations` counts every call to `f`, including those spent on Jacobians.

## atol = 1e-6, rtol = 1e-3 (the library defaults)

| Test problem                           | Solver      | Steps | Evaluations | Jacobians | LU decompositions | Error / tol |
| :------------------------------------- | :---------- | ----: | ----------: | --------: | ----------------: | ----------: |
| Prothero-Robinson, lambda = -1e5       | numerics-js |    42 |         172 |         1 |                17 |         0.0 |
|                                        | SciPy Radau |    24 |         176 |         1 |                26 |        0.00 |
|                                        | SciPy BDF   |    72 |         146 |         1 |                20 |        0.00 |
|                                        | SciPy LSODA |    70 |         130 |         6 |                 6 |        0.04 |
| Robertson                              | numerics-js |    43 |         253 |         7 |                47 |         4.1 |
|                                        | SciPy Radau |    33 |         323 |        18 |                78 |        0.04 |
|                                        | SciPy BDF   |    84 |         210 |         8 |                31 |        1.33 |
|                                        | SciPy LSODA |   105 |         209 |        20 |                20 |        0.74 |
| Van der Pol, mu = 1000                 | numerics-js |   160 |       1,336 |        21 |               123 |         9.1 |
|                                        | SciPy Radau |   110 |       1,129 |        38 |               186 |        0.01 |
|                                        | SciPy BDF   |   211 |         703 |        35 |                78 |       21.04 |
|                                        | SciPy LSODA |   278 |         692 |        53 |                53 |       14.92 |
| Linear chain, rates -1000, -1, -0.1    | numerics-js |    42 |         167 |         1 |                19 |         6.2 |
|                                        | SciPy Radau |    23 |         175 |         2 |                32 |        0.01 |
|                                        | SciPy BDF   |    70 |         148 |         1 |                18 |        0.96 |
|                                        | SciPy LSODA |    74 |         162 |        11 |                11 |        1.15 |

## atol = 1e-8, rtol = 1e-5

| Test problem                           | Solver      | Steps | Evaluations | Jacobians | LU decompositions | Error / tol |
| :------------------------------------- | :---------- | ----: | ----------: | --------: | ----------------: | ----------: |
| Prothero-Robinson, lambda = -1e5       | numerics-js |   177 |         709 |         1 |                26 |         0.0 |
|                                        | SciPy Radau |    61 |         435 |         1 |                36 |        0.06 |
|                                        | SciPy BDF   |   140 |         282 |         1 |                31 |        0.00 |
|                                        | SciPy LSODA |   138 |         274 |        13 |                13 |        0.00 |
| Robertson                              | numerics-js |   168 |         937 |         6 |                65 |        21.4 |
|                                        | SciPy Radau |    79 |         681 |        30 |               134 |        0.04 |
|                                        | SciPy BDF   |   184 |         523 |         8 |                48 |        4.18 |
|                                        | SciPy LSODA |   239 |         469 |        36 |                36 |        2.28 |
| Van der Pol, mu = 1000                 | numerics-js |   675 |       4,431 |        41 |               210 |       111.0 |
|                                        | SciPy Radau |   279 |       2,514 |        74 |               252 |        0.03 |
|                                        | SciPy BDF   |   462 |       1,416 |        43 |               115 |       14.88 |
|                                        | SciPy LSODA |   538 |       1,151 |        77 |                77 |       27.21 |
| Linear chain, rates -1000, -1, -0.1    | numerics-js |   182 |         730 |         1 |                26 |        26.6 |
|                                        | SciPy Radau |    68 |         490 |         2 |                44 |        0.00 |
|                                        | SciPy BDF   |   144 |         306 |         1 |                30 |        0.35 |
|                                        | SciPy LSODA |   138 |         286 |        12 |                12 |        0.94 |

"Error / tol" is the largest component-wise final-state error relative to the
reference, divided by `atol + rtol * |y|`. The tolerances control the local error
of each step, so this ratio can exceed 1; the test allows a factor of 200. The largest
ratio observed for numerics-js is 111, on Van der Pol at the tight setting, where phase errors
accumulate over many steps. SciPy's variable-order BDF and LSODA stay below about 30 on the
same cases, and Radau is far below 1 because it is of higher order.

The solver follows the implementation recommendations of Hosea & Shampine: the first stage of
a step reuses the last stage of the previous step instead of evaluating `f`, the Newton iteration
stops when the correction falls below half the tolerance, the error estimate is filtered through the
factored iteration matrix, and the Jacobian is refreshed only when the Newton iteration fails with a
stale Jacobian. Being second order with a fixed Newton strategy, it needs more evaluations than
SciPy's variable-order BDF and LSODA, with the gap largest on Van der Pol.

The Van der Pol case ends at `t = 1500`, in the middle of the second slow phase, after one fast
jump. Ending later (for example at `t = 3000`, just after a jump) would make the final-state
check depend on the exact timing of the jump, which is not a property of the solver's accuracy.
At `t = 1500` the error ratio stays between about 8 and 15 at the default setting and between 90
and 125 at the tight one for `safety` in 0.7 to 0.9 and different initial steps.
