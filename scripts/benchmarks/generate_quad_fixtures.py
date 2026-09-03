# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the 'quad' integrator test fixture.

Run with: uv run scripts/benchmarks/generate_quad_fixtures.py

Each case here must have a matching hand-written integrand in
tests/integrate/quad.scipy.test.ts (same id, same math, same bounds).

Cases with a non-smooth endpoint (marked "singular": True) are excluded from the
evaluation-count comparison there: 'quad' has no singularity-aware subdivision
like QUADPACK's QAGS, so panel counts aren't meaningfully comparable for those.
"""

import json
import math
from pathlib import Path

from scipy.integrate import quad

# Requested tolerance shared with the quad(fn, a, b, tol) call in
# tests/integrate/quad.scipy.test.ts, so evaluation counts are comparable.
# Matches quad's own default tol, since that's the representative use case.
TOL = 1e-8

CASES = [
    {
        "id": "sine",
        "description": "sin(x) on [0, pi]",
        "f": math.sin,
        "a": 0.0,
        "b": math.pi,
    },
    {
        "id": "rational_pi",
        "description": "4 / (1 + x^2) on [0, 1]",
        "f": lambda x: 4.0 / (1.0 + x * x),
        "a": 0.0,
        "b": 1.0,
    },
    {
        "id": "oscillatory",
        "description": "sin(100 x) on [0, pi]",
        "f": lambda x: math.sin(100.0 * x),
        "a": 0.0,
        "b": math.pi,
    },
    {
        "id": "sqrt_singularity",
        "description": "1 / sqrt(x) on [0, 1] (integrable endpoint singularity)",
        "f": lambda x: 1.0 / math.sqrt(x),
        "a": 0.0,
        "b": 1.0,
        "singular": True,
    },
    {
        "id": "log_singularity",
        "description": "-log(x) on [0, 1] (integrable endpoint singularity)",
        "f": lambda x: -math.log(x),
        "a": 0.0,
        "b": 1.0,
        "singular": True,
    },
    {
        "id": "gaussian",
        "description": "exp(-x^2) on [-2, 2]",
        "f": lambda x: math.exp(-x * x),
        "a": -2.0,
        "b": 2.0,
    },
    {
        "id": "runge",
        "description": "1 / (1 + 25 x^2) on [-1, 1]",
        "f": lambda x: 1.0 / (1.0 + 25.0 * x * x),
        "a": -1.0,
        "b": 1.0,
    },
    {
        "id": "high_freq_cos",
        "description": "cos(50 x) on [0, 2 pi]",
        "f": lambda x: math.cos(50.0 * x),
        "a": 0.0,
        "b": 2.0 * math.pi,
    },
    {
        "id": "lorentzian_peak",
        "description": "1 / ((x - 0.5)^2 + 1e-4) on [0, 1]",
        "f": lambda x: 1.0 / ((x - 0.5) ** 2 + 1e-4),
        "a": 0.0,
        "b": 1.0,
    },
    {
        "id": "exp_growth",
        "description": "exp(x) on [0, 1]",
        "f": math.exp,
        "a": 0.0,
        "b": 1.0,
    },
]


def main() -> None:
    results = []
    for case in CASES:
        value, abserr, infodict = quad(
            case["f"], case["a"], case["b"], epsabs=TOL, epsrel=TOL, full_output=1
        )
        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "a": case["a"],
                "b": case["b"],
                "scipyValue": value,
                "scipyAbsError": abserr,
                "scipyEvaluations": infodict["neval"],
                "singular": case.get("singular", False),
            }
        )

    out_path = Path(__file__).resolve().parents[2] / "tests" / "integrate" / "fixtures" / "quad.scipy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    main()
