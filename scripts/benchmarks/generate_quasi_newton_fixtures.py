# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate SciPy hybr reference values for the roots.quasiNewton test fixture.

Run with: uv run scripts/benchmarks/generate_quasi_newton_fixtures.py

Each case here must mirror the function and starting point in
tests/roots/quasiNewton.scipy.test.ts, which in turn uses
tests/roots/testFunctions.ts.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import root

TOL_X = 1e-10
TOL_F = 0.0
MAX_FEVAL = 1000
RESIDUAL_TOLERANCE = 1e-7


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Extended Rosenbrock function (Moré, Garbow & Hillstrom 1981, problem 21).

    For each pair of variables (k = 0, 2, 4, ...)::

        f_k(x)     = 1 - x_k
        f_{k+1}(x) = 10 * (x_{k+1} - x_k^2)

    Parameters
    ----------
    x : ndarray, shape (n,)
        Point at which to evaluate the function. The number of variables `n` must be even.

    Returns
    -------
    ndarray, shape (n,)
        Residual vector f(x).

    Notes
    -----
    The two residuals of a pair are listed in the opposite order of the original definition.

    Root: ``x = (1, ..., 1)``. Standard starting point: ``x0 = (-1.2, 1, -1.2, 1, ...)``.
    """
    values = np.zeros_like(x)
    for i in range(0, len(x), 2):
        values[i] = 1.0 - x[i]
        values[i + 1] = 10.0 * (x[i + 1] - x[i] ** 2)
    return values


def powell_singular(x: np.ndarray) -> np.ndarray:
    """Extended Powell singular function (Powell 1962; Moré, Garbow & Hillstrom 1981, problem 13).

    For each block of four variables (k = 0, 4, 8, ...)::

        f_k(x)     = x_k - 10 * x_{k+1}
        f_{k+1}(x) = sqrt(5) * (x_{k+2} - x_{k+3})
        f_{k+2}(x) = (x_{k+1} - 2 * x_{k+2})^2
        f_{k+3}(x) = sqrt(10) * (x_k - x_{k+3})^2

    Parameters
    ----------
    x : ndarray, shape (n,)
        Point at which to evaluate the function. The number of variables `n` must be a
        multiple of 4.

    Returns
    -------
    ndarray, shape (n,)
        Residual vector f(x).

    Notes
    -----
    The definition of Moré, Garbow & Hillstrom has ``f_k = x_k + 10 * x_{k+1}``; the sign differs
    here.

    Root: ``x = 0``. The Jacobian is singular there (rank 2 per block), so Newton-type methods
    converge only linearly. Standard starting point: ``x0 = (3, -1, 0, 1, 3, -1, 0, 1, ...)``.
    """
    values = np.zeros_like(x)
    for i in range(0, len(x), 4):
        values[i] = x[i] - 10.0 * x[i + 1]
        values[i + 1] = np.sqrt(5.0) * (x[i + 2] - x[i + 3])
        values[i + 2] = (x[i + 1] - 2.0 * x[i + 2]) ** 2
        values[i + 3] = np.sqrt(10.0) * (x[i] - x[i + 3]) ** 2
    return values


def trigonometric(x: np.ndarray) -> np.ndarray:
    """Trigonometric function (Moré, Garbow & Hillstrom 1981, problem 26).

    For i = 1, ..., n::

        f_i(x) = n - sum_j cos(x_j) + i * (1 - cos(x_i)) - sin(x_i)

    Parameters
    ----------
    x : ndarray, shape (n,)
        Point at which to evaluate the function.

    Returns
    -------
    ndarray, shape (n,)
        Residual vector f(x).

    Notes
    -----
    Root: ``x = 0``, but the problem has other roots as well. The standard starting point
    ``x0 = (1/n, ..., 1/n)`` can lead to one of them (and hybr may fail from it for n = 10), so the
    benchmark starts closer to ``x = 0``.
    """
    n = len(x)
    sum_cos = np.cos(x).sum()
    return np.array([n - sum_cos + (i + 1) * (1.0 - np.cos(x[i])) - np.sin(x[i]) for i in range(n)])


def powell_badly_scaled(x: np.ndarray) -> np.ndarray:
    """Powell badly scaled function (Powell 1970; Moré, Garbow & Hillstrom 1981, problem 3).

    ::

        f_0(x) = 10^4 * x_0 * x_1 - 1
        f_1(x) = exp(-x_0) + exp(-x_1) - 1.0001

    Parameters
    ----------
    x : ndarray, shape (2,)
        Point at which to evaluate the function.

    Returns
    -------
    ndarray, shape (2,)
        Residual vector f(x).

    Notes
    -----
    The components of the root, ``x = (1.098159e-5, 9.106147)``, differ by roughly six orders of
    magnitude, which makes the problem a test of variable scaling.

    Standard starting point: ``x0 = (0, 1)``.
    """
    return np.array([1e4 * x[0] * x[1] - 1.0, np.exp(-x[0]) + np.exp(-x[1]) - 1.0001])


def helical_valley(x: np.ndarray) -> np.ndarray:
    """Helical valley function (Fletcher & Powell 1963; Moré, Garbow & Hillstrom 1981, problem 7).

    ::

        f_0(x) = 10 * (x_2 - 10 * theta)
        f_1(x) = 10 * (sqrt(x_0^2 + x_1^2) - 1)
        f_2(x) = x_2

    where ``theta = arctan(x_1 / x_0) / (2 pi)`` for ``x_0 > 0`` and
    ``arctan(x_1 / x_0) / (2 pi) + 1/2`` for ``x_0 < 0``, i.e. the polar angle of ``(x_0, x_1)``
    divided by ``2 pi``, taken in the range ``(-1/4, 3/4]``.

    Parameters
    ----------
    x : ndarray, shape (3,)
        Point at which to evaluate the function.

    Returns
    -------
    ndarray, shape (3,)
        Residual vector f(x).

    Notes
    -----
    The function is discontinuous across the half-line ``x_0 = 0, x_1 < 0``, where ``theta`` jumps
    by 1 (and ``f_0`` by 100).

    Root: ``x = (1, 0, 0)``. Standard starting point: ``x0 = (-1, 0, 0)``.
    """
    x1, x2, x3 = x
    theta = np.arctan2(x2, x1) / (2.0 * np.pi)
    if theta < -0.25:
        theta += 1.0
    return np.array([10.0 * (x3 - 10.0 * theta), 10.0 * (np.hypot(x1, x2) - 1.0), x3])


def example65(x: np.ndarray) -> np.ndarray:
    """Example 6.5 of Dennis & Schnabel (1996): a nonlinear system of two equations.

    ::

        f_0(x) = x_0^2 + x_1^2 - 2
        f_1(x) = exp(x_0 - 1) + x_1^3 - 2

    Parameters
    ----------
    x : ndarray, shape (2,)
        Point at which to evaluate the function.

    Returns
    -------
    ndarray, shape (2,)
        Residual vector f(x).

    Notes
    -----
    Root: ``x = (1, 1)``.
    """
    x1, x2 = x
    return np.array([x1**2 + x2**2 - 2.0, np.exp(x1 - 1.0) + x2**3 - 2.0])


# `sclx`: scaling factors for the numerics-js solver; None means the solver's default scaling.
# SciPy's hybr always runs with its own defaults (adaptive scaling).
CASES = [
    {
        "id": "rosenbrock",
        "description": "Extended Rosenbrock function (4 variables)",
        "f": rosenbrock,
        "x0": np.array([-1.2, 1.0, -1.2, 1.0]),
        "sclx": None,
    },
    {
        "id": "powell_singular",
        "description": "Extended Powell singular function (4 variables)",
        "f": powell_singular,
        "x0": np.array([3.0, -1.0, 0.0, 1.0]),
        "sclx": None,
    },
    {
        "id": "trigonometric",
        "description": "Trigonometric function (10 variables)",
        "f": trigonometric,
        "x0": np.full(10, 0.01),
        "sclx": None,
    },
    {
        "id": "trigonometric_5",
        "description": "Trigonometric function (5 variables)",
        "f": trigonometric,
        "x0": np.full(5, 0.01),
        "sclx": None,
    },
    {
        "id": "rosenbrock_100",
        "description": "Extended Rosenbrock function (100 variables)",
        "f": rosenbrock,
        "x0": np.tile([-1.2, 1.0], 50),
        "sclx": None,
    },
    {
        "id": "powell_badly_scaled",
        "description": "Powell badly scaled function (2 variables)",
        "f": powell_badly_scaled,
        "x0": np.array([0.0, 1.0]),
        "sclx": None,
    },
    {
        "id": "helical_valley",
        "description": "Helical valley function (3 variables)",
        "f": helical_valley,
        "x0": np.array([-1.0, 0.0, 0.0]),
        # Explicit scaling for numerics-js. The default heuristic gives the zero entries of x0 a
        # scale of 10, i.e. (1, 10, 10), which distorts the trust region on this problem
        # (52 / 103 evaluations without / with Broyden updates, vs 33 / 26 with unit scaling).
        "sclx": np.ones(3),
    },
    {
        "id": "example65",
        "description": "Example 6.5 of Dennis & Schnabel (1996), x0 = [1.5, 1]",
        "f": example65,
        "x0": np.array([1.5, 1.0]),
        "sclx": None,
    },
]


def mirror_points(f, x0: np.ndarray, count: int = 3) -> list[dict]:
    """Deterministic generic points (no special structure) and the values of `f` there.

    The TypeScript test uses them to verify that its implementation of `f` mirrors this one;
    `x0` and the solution alone can be special (e.g. lie on a surface where a term vanishes).

    Parameters
    ----------
    f : callable
        Test function, ``f(x) -> ndarray``.
    x0 : ndarray, shape (n,)
        Starting point of the problem; the generic points are perturbations of it.
    count : int, optional
        Number of points to generate. Default is 3.

    Returns
    -------
    list of dict
        One entry ``{"x": [...], "fx": [...]}`` per point.
    """
    points = []
    for k in range(count):
        x = x0 + 0.5 * np.sin((k + 1) * np.arange(1, len(x0) + 1))
        points.append({"x": x.tolist(), "fx": f(x).tolist()})
    return points


def main() -> None:
    results = []
    for case in CASES:
        result = root(
            case["f"],
            case["x0"],
            method="hybr",
            options={"xtol": TOL_X, "maxfev": MAX_FEVAL},
        )
        residual_norm = float(np.linalg.norm(result.fun, ord=np.inf))
        if residual_norm > RESIDUAL_TOLERANCE:
            if not result.success:
                raise RuntimeError(f"SciPy hybr failed for {case['id']}: {result.message}")
            raise RuntimeError(
                f"SciPy hybr residual for {case['id']} was {residual_norm}, "
                f"above {RESIDUAL_TOLERANCE}"
            )
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "x0": case["x0"].tolist(),
                # f(x0) lets the TypeScript test verify that its test function mirrors this one.
                "fx0": case["f"](case["x0"]).tolist(),
                "mirrorPoints": mirror_points(case["f"], case["x0"]),
                # `tolx` and `tolf` are settings for the numerics-js solver (SciPy's hybr has no
                # `tolf`); `tolx` is also the `xtol` passed to SciPy.
                "tolx": TOL_X,
                "tolf": TOL_F,
                "sclx": None if case["sclx"] is None else case["sclx"].tolist(),
                "scipyX": result.x.tolist(),
                "scipyFx": result.fun.tolist(),
                "scipyEvaluations": result.nfev,
                "scipySuccess": bool(result.success),
                "scipyMessage": str(result.message),
            }
        )

    out_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "roots"
        / "fixtures"
        / "quasiNewton.scipy.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()