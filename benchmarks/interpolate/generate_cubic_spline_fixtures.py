# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate SciPy reference values for the interpolate.CubicSpline* test fixture.

Run with: uv run benchmarks/interpolate/generate_cubic_spline_fixtures.py

The reference is scipy.interpolate.CubicSpline with bc_type 'not-a-knot',
'natural' or 'clamped' (first derivatives prescribed at both ends). Unlike the
other benchmark generators, every case stores its own input data (knots,
values, end derivatives, query points) in the fixture, so
benchmarks/interpolate/cubicSpline.scipy.test.ts needs no duplicated case
definitions and only has to feed the stored data to the library.

Queries and integration bounds stay inside [x[0], x[-1]]: outside it SciPy
extrapolates the end polynomials while numerics-js clamps to constants.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

SEED = 20260101
N_QUERIES = 40


def datasets() -> list[dict]:
    rng = np.random.default_rng(SEED)
    cases = []

    x = np.linspace(0.0, 2.0 * np.pi, 12)
    cases.append(("uniform_sine_12", "12 uniform knots, y = sin(x) + 0.1 x", x, np.sin(x) + 0.1 * x))

    # Spacing grows geometrically: the largest step is ~50x the smallest.
    h = 0.05 * 1.17 ** np.arange(24)
    x = np.concatenate([[0.0], np.cumsum(h)])
    cases.append(("graded_25", "25 knots, spacing growing by 17 % per step", x, np.exp(-0.3 * x) * np.cos(2.0 * x)))

    x = np.sort(rng.uniform(0.0, 10.0, 40))
    x = x[np.concatenate([[True], np.diff(x) > 0.02])]  # keep knots well separated
    cases.append(("random_noisy", "random non-uniform knots, y ~ N(0, 1)", x, rng.normal(size=x.size)))

    for n in (2, 3, 4, 5):
        x = np.cumsum(rng.uniform(0.4, 1.6, n))
        cases.append((f"tiny_{n}", f"{n} knots (smallest systems and closed-form branches)", x, rng.normal(size=n)))

    x = np.cumsum(rng.uniform(0.3, 1.7, 16))
    y = np.column_stack([np.sin(x), x**2 - 3 * x, np.exp(-x), rng.normal(size=x.size)])
    cases.append(("vector_valued_16x4", "16 knots shared by 4 output components", x, y))

    return cases


def main() -> None:
    rng = np.random.default_rng(SEED + 1)
    results = []

    for case_id, description, x, y in datasets():
        width = x[-1] - x[0]
        queries = np.sort(np.concatenate([x, x[:-1] + 0.5 * np.diff(x), rng.uniform(x[0], x[-1], N_QUERIES)]))
        bounds = [
            (x[0], x[-1]),
            (x[0] + 0.17 * width, x[0] + 0.63 * width),
            (x[0] + 0.5 * width, x[0] + 0.51 * width),
        ]
        dim = 1 if y.ndim == 1 else y.shape[1]
        for bc in ("not-a-knot", "natural", "clamped"):
            if bc == "clamped":
                left = rng.uniform(-1.5, 1.5, size=None if dim == 1 else dim)
                right = rng.uniform(-1.5, 1.5, size=None if dim == 1 else dim)
                spline = CubicSpline(x, y, bc_type=((1, left), (1, right)))
            else:
                left = right = None
                spline = CubicSpline(x, y, bc_type=bc)

            def as_json(value):
                return None if value is None else np.asarray(value, dtype=float).tolist()

            results.append(
                {
                    "id": f"{case_id}__{bc}",
                    "description": f"{description}, bc = {bc}",
                    "bc": bc,
                    "dim": None if y.ndim == 1 else dim,
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "leftDerivative": as_json(left),
                    "rightDerivative": as_json(right),
                    "queries": queries.tolist(),
                    "eval": spline(queries).tolist(),
                    "derivative": spline(queries, 1).tolist(),
                    "integrals": [
                        {"a": float(a), "b": float(b), "value": as_json(spline.integrate(a, b))}
                        for a, b in bounds
                    ],
                }
            )

    out_path = Path(__file__).resolve().with_name("cubicSpline.scipy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
