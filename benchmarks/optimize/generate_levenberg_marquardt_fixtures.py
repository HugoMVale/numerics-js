# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate SciPy `leastsq` (MINPACK lmder/lmdif) reference values for the levenbergMarquardt fixture.

Run with: uv run benchmarks/optimize/generate_levenberg_marquardt_fixtures.py

Each problem here must mirror the function (and Jacobian) of the same name in
tests/optimize/leastSquaresFunctions.ts, which benchmarks/optimize/levenbergMarquardt.scipy.test.ts uses.
Every problem is run with a finite-difference Jacobian (`lmdif`) and, if an analytic Jacobian exists, also
with it (`lmder`, id suffix `_jac`).
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import leastsq

SQRT5 = np.sqrt(5.0)
SQRT10 = np.sqrt(10.0)


def linear_full_rank(x):
    m = 10
    s = 2.0 / m * x.sum()
    return np.array([(x[i] if i < len(x) else 0.0) - s - 1.0 for i in range(m)])


def linear_full_rank_jac(x):
    m, n = 10, len(x)
    return np.array([[(1.0 if i == j else 0.0) - 2.0 / m for j in range(n)] for i in range(m)])


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def rosenbrock_jac(x):
    return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


def powell_singular(x):
    return np.array([x[0] + 10.0 * x[1], SQRT5 * (x[2] - x[3]), (x[1] - 2.0 * x[2]) ** 2, SQRT10 * (x[0] - x[3]) ** 2])


def powell_singular_jac(x):
    a = x[1] - 2.0 * x[2]
    b = x[0] - x[3]
    return np.array(
        [
            [1.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, SQRT5, -SQRT5],
            [0.0, 2.0 * a, -4.0 * a, 0.0],
            [2.0 * SQRT10 * b, 0.0, 0.0, -2.0 * SQRT10 * b],
        ]
    )


def helical_valley(x):
    theta = np.arctan2(x[1], x[0]) / (2.0 * np.pi)
    if theta < -0.25:
        theta += 1.0
    return np.array([10.0 * (x[2] - 10.0 * theta), 10.0 * (np.hypot(x[0], x[1]) - 1.0), x[2]])


def beale(x):
    c = [1.5, 2.25, 2.625]
    return np.array([c[i - 1] - x[0] * (1.0 - x[1] ** i) for i in (1, 2, 3)])


def beale_jac(x):
    return np.array([[-(1.0 - x[1] ** i), x[0] * i * x[1] ** (i - 1)] for i in (1, 2, 3)])


def jennrich_sampson(x):
    return np.array([2.0 + 2.0 * i - (np.exp(i * x[0]) + np.exp(i * x[1])) for i in range(1, 11)])


def jennrich_sampson_jac(x):
    return np.array([[-i * np.exp(i * x[0]), -i * np.exp(i * x[1])] for i in range(1, 11)])


def brown_badly_scaled(x):
    return np.array([x[0] - 1e6, x[1] - 2e-6, x[0] * x[1] - 2.0])


def brown_badly_scaled_jac(x):
    return np.array([[1.0, 0.0], [0.0, 1.0], [x[1], x[0]]])


def freudenstein_roth(x):
    a, b = x
    return np.array([-13.0 + a + ((5.0 - b) * b - 2.0) * b, -29.0 + a + ((b + 1.0) * b - 14.0) * b])


def freudenstein_roth_jac(x):
    b = x[1]
    return np.array([[1.0, 10.0 * b - 3.0 * b * b - 2.0], [1.0, 3.0 * b * b + 2.0 * b - 14.0]])


def watson(x):
    n = len(x)
    r = np.zeros(31)
    for i in range(1, 30):
        t = i / 29.0
        s1 = sum(j * x[j] * t ** (j - 1) for j in range(1, n))
        s2 = sum(x[j] * t**j for j in range(n))
        r[i - 1] = s1 - s2 * s2 - 1.0
    r[29] = x[0]
    r[30] = x[1] - x[0] ** 2 - 1.0
    return r


def rank_deficient(x):
    s = x[0] + x[1] - 2.0
    return np.array([s, 2.0 * s, 3.0 * s])


def rank_deficient_jac(x):
    return np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


# id -> (function, jacobian or None, x0, description)
PROBLEMS = {
    "linearFullRank": (linear_full_rank, linear_full_rank_jac, [1.0] * 5, "Linear function, full rank (MGH 1), n = 5, m = 10"),
    "rosenbrock": (rosenbrock, rosenbrock_jac, [-1.2, 1.0], "Rosenbrock function (MGH 4)"),
    "powellSingular": (powell_singular, powell_singular_jac, [3.0, -1.0, 0.0, 1.0], "Powell singular function (MGH 13)"),
    "helicalValley": (helical_valley, None, [-1.0, 0.0, 0.0], "Helical valley function (MGH 7)"),
    "beale": (beale, beale_jac, [1.0, 1.0], "Beale function (MGH 5)"),
    "jennrichSampson": (jennrich_sampson, jennrich_sampson_jac, [0.3, 0.4], "Jennrich and Sampson function (MGH 6)"),
    "brownBadlyScaled": (brown_badly_scaled, brown_badly_scaled_jac, [1.0, 1.0], "Brown badly scaled function"),
    "freudensteinRoth": (freudenstein_roth, freudenstein_roth_jac, [0.5, -2.0], "Freudenstein and Roth function, local minimum"),
    "watson": (watson, None, [0.0] * 6, "Watson function (MGH 20), n = 6, m = 31"),
    "rankDeficient": (rank_deficient, rank_deficient_jac, [0.0, 0.0], "Rank-deficient linear problem"),
}

# Problems whose minimizer is not unique (the solution depends on the path taken).
NON_UNIQUE = {"rankDeficient"}

FTOL = XTOL = 1.49012e-8
GTOL = 0.0


def main() -> None:
    rng = np.random.default_rng(12345)
    results = []
    for pid, (f, jac, x0, description) in PROBLEMS.items():
        x0 = np.array(x0, dtype=float)
        mirror = []
        for _ in range(2):
            x = x0 + rng.uniform(-1.0, 1.0, size=x0.shape) * np.maximum(np.abs(x0), 1.0) * 0.5
            mirror.append({"x": x.tolist(), "fx": f(x).tolist()})

        for use_jac in (False, True):
            if use_jac and jac is None:
                continue
            n = len(x0)
            maxfev = (200 if use_jac else 100) * (n + 1)
            x, _, info, msg, ier = leastsq(
                f,
                x0,
                Dfun=jac if use_jac else None,
                full_output=True,
                ftol=FTOL,
                xtol=XTOL,
                gtol=GTOL,
                maxfev=maxfev,
                factor=100,
            )
            results.append(
                {
                    "id": pid + ("_jac" if use_jac else ""),
                    "problem": pid,
                    "description": description + (", analytic Jacobian" if use_jac else ", finite-difference Jacobian"),
                    "useJac": use_jac,
                    "x0": x0.tolist(),
                    "fx0": f(x0).tolist(),
                    "mirrorPoints": mirror,
                    "tolf": FTOL,
                    "tolx": XTOL,
                    "tolg": GTOL,
                    "maxFunEvals": maxfev,
                    "uniqueMinimizer": pid not in NON_UNIQUE,
                    "scipyX": x.tolist(),
                    "scipySumSquares": float((f(x) ** 2).sum()),
                    "scipyEvaluations": int(info["nfev"]),
                    "scipyJacobianEvaluations": int(info.get("njev", 0)) if use_jac else 0,
                    "scipyIer": int(ier),
                    "scipyMessage": msg,
                }
            )

    out_path = Path(__file__).resolve().with_name("levenbergMarquardt.scipy.json")
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")
    for r in results:
        print(f"{r['id']:24s} ier={r['scipyIer']} nfev={r['scipyEvaluations']:4d} ss={r['scipySumSquares']:.6e}")


if __name__ == "__main__":
    main()
