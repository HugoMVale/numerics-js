# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference fixtures for events in the adaptive Runge-Kutta solvers.

Run with: uv run benchmarks/ode/generate_runge_kutta_adaptive_events_fixtures.py

Each case must have a matching derivative and event definition in
benchmarks/ode/rungeKuttaAdaptive.events.scipy.test.ts (same id, math, span, state,
initial step, and events). The initial step is fixed so that SciPy and numerics-js take
the same steps; the event times then differ only by root-finding and rounding.
"""

import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp

ATOL = 1e-8
RTOL = 1e-8


def event(fn, direction=0, terminal=False):
    """SciPy event: direction 0 = both, +1 = rising, -1 = falling."""

    def wrapped(t, y):
        return fn(t, y)

    wrapped.direction = direction
    wrapped.terminal = terminal
    return wrapped


OSC = lambda _t, y: [y[1], -y[0]]

CASES = [
    {
        "id": "oscillator_zero_crossings",
        "description": "x' = v, v' = -x on [0, 12]; event x = 0 (both directions)",
        "f": OSC,
        "t0": 0.0,
        "tEnd": 12.0,
        "y0": [1.0, 0.0],
        "h0": 0.1,
        "events": [event(lambda _t, y: y[0])],
    },
    {
        "id": "decay_threshold_terminal",
        "description": "y' = -y on [0, 5]; terminal falling event y = 0.25",
        "f": lambda _t, y: [-y[0]],
        "t0": 0.0,
        "tEnd": 5.0,
        "y0": [1.0],
        "h0": 0.1,
        "events": [event(lambda _t, y: y[0] - 0.25, direction=-1, terminal=True)],
    },
    {
        "id": "logistic_rising",
        "description": "y' = 10 y (1 - y) on [0, 1]; rising event y = 0.5",
        "f": lambda _t, y: [10.0 * y[0] * (1.0 - y[0])],
        "t0": 0.0,
        "tEnd": 1.0,
        "y0": [0.01],
        "h0": 0.01,
        "events": [event(lambda _t, y: y[0] - 0.5, direction=1)],
    },
    {
        "id": "oscillator_terminal_count",
        "description": "x' = v, v' = -x on [0, 20]; event x = 0 terminal at the 3rd occurrence",
        "f": OSC,
        "t0": 0.0,
        "tEnd": 20.0,
        "y0": [1.0, 0.0],
        "h0": 0.1,
        "events": [event(lambda _t, y: y[0], terminal=3)],
    },
    {
        "id": "backward_growth",
        "description": "y' = -2 y from t = 2 to t = -1; event y = 1 (crossed at t = 0)",
        "f": lambda _t, y: [-2.0 * y[0]],
        "t0": 2.0,
        "tEnd": -1.0,
        "y0": [math.exp(-4.0)],
        "h0": 0.1,
        "events": [event(lambda _t, y: y[0] - 1.0)],
    },
    {
        "id": "oscillator_two_events",
        "description": "x' = v, v' = -x on [0, 7]; events x = 0 and v = 0.5, terminal on the second",
        "f": OSC,
        "t0": 0.0,
        "tEnd": 7.0,
        "y0": [1.0, 0.0],
        "h0": 0.1,
        "events": [event(lambda _t, y: y[0]), event(lambda _t, y: y[1] - 0.5, terminal=2)],
    },
]

METHODS = {"rk23": "RK23", "rk45": "RK45"}


def main() -> None:
    results = []
    for case in CASES:
        scipy = {}
        for method_id, scipy_method in METHODS.items():
            result = solve_ivp(
                case["f"],
                (case["t0"], case["tEnd"]),
                case["y0"],
                method=scipy_method,
                atol=ATOL,
                rtol=RTOL,
                first_step=case["h0"],
                events=case["events"],
            )
            if result.status < 0:
                raise RuntimeError(f"SciPy {scipy_method} failed for {case['id']}: {result.message}")
            scipy[method_id] = {
                "terminated": result.status == 1,
                "finalT": float(result.t[-1]),
                "tEvents": [[float(t) for t in ts] for ts in result.t_events],
                "yEvents": [[[float(v) for v in y] for y in ys] for ys in result.y_events],
            }

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "t0": case["t0"],
                "tEnd": case["tEnd"],
                "y0": case["y0"],
                "h0": case["h0"],
                "atol": ATOL,
                "rtol": RTOL,
                "scipy": scipy,
            }
        )

    out_path = Path(__file__).resolve().with_name("rungeKuttaAdaptive.events.scipy.json")
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
