# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate SciPy `curve_fit` (method="lm") reference values for the curveFit fixture.

Run with: uv run benchmarks/optimize/generate_curve_fit_fixtures.py

The data (`xdata`, `ydata`, `sigma`) are stored in the generated JSON fixture, which
benchmarks/optimize/curveFit.scipy.test.ts reads, so no noise generator has to be mirrored. Each model here
must mirror the model of the same id in that test.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

FTOL = XTOL = 1.49012e-8
GTOL = 0.0


def wobble(i: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-noise in about [-0.08, 0.08]."""
    return 0.05 * np.sin(12.9898 * (i + 1)) + 0.03 * np.cos(4.1414 * (i + 1))


def exp_decay(x, a, b, c):
    """a * exp(-b x) + c"""
    return a * np.exp(-b * x) + c


def gauss(x, a, mu, s):
    """a * exp(-((x - mu)/s)^2 / 2)"""
    return a * np.exp(-(((x - mu) / s) ** 2) / 2.0)


def logistic(x, k, r, x0):
    """k / (1 + exp(-r (x - x0)))"""
    return k / (1.0 + np.exp(-r * (x - x0)))


def line(x, a, b):
    """a + b x"""
    return a + b * x


def michaelis_menten(x, vmax, km):
    """vmax x / (km + x)"""
    return vmax * x / (km + x)


def power_law(x, a, b):
    """a x^b"""
    return a * x**b


def case(pid, model, x, truth, noise_scale, p0, description, sigma=None, absolute_sigma=False):
    i = np.arange(len(x), dtype=float)
    y = model(x, *truth) + noise_scale * wobble(i) / 0.08
    return {
        "id": pid,
        "model": model,
        "description": description,
        "x": x,
        "y": y,
        "p0": np.array(p0, dtype=float),
        "sigma": sigma,
        "absolute_sigma": absolute_sigma,
    }


def build_cases():
    x1 = np.linspace(0.0, 5.0, 25)
    x2 = np.linspace(-3.0, 5.0, 40)
    x3 = np.linspace(0.0, 20.0, 30)
    x4 = np.linspace(0.0, 10.0, 15)
    x5 = np.linspace(0.2, 6.0, 20)
    x6 = np.linspace(1.0, 8.0, 18)
    return [
        case("exp_decay", exp_decay, x1, (2.5, 1.3, 0.5), 0.05, [1.0, 1.0, 0.0], "a*exp(-b*x) + c, unweighted"),
        case("gauss", gauss, x2, (4.0, 1.0, 0.8), 0.1, [3.0, 0.5, 1.0], "Gaussian peak, unweighted"),
        case(
            "logistic_sigma",
            logistic,
            x3,
            (1000.0, 0.6, 9.0),
            20.0,
            [800.0, 0.3, 7.0],
            "logistic curve, relative sigma growing with y",
            sigma=np.linspace(10.0, 40.0, 30),
        ),
        case(
            "line_absolute_sigma",
            line,
            x4,
            (1.0, 2.0),
            0.2,
            [0.0, 0.0],
            "straight line, absolute sigma",
            sigma=np.full(15, 0.15),
            absolute_sigma=True,
        ),
        case(
            "michaelis_menten_sigma",
            michaelis_menten,
            x5,
            (3.0, 1.2),
            0.05,
            [1.0, 1.0],
            "Michaelis-Menten, relative sigma",
            sigma=0.02 + 0.03 * x5,
        ),
        case("power_law", power_law, x6, (0.8, 1.7), 0.3, [1.0, 1.0], "power law, unweighted"),
    ]


def main() -> None:
    rng = np.random.default_rng(2024)
    results = []
    for c in build_cases():
        popt, pcov, info, msg, ier = curve_fit(
            c["model"],
            c["x"],
            c["y"],
            p0=c["p0"],
            sigma=c["sigma"],
            absolute_sigma=c["absolute_sigma"],
            method="lm",
            full_output=True,
            ftol=FTOL,
            xtol=XTOL,
            gtol=GTOL,
        )
        sigma = c["sigma"] if c["sigma"] is not None else np.ones_like(c["x"])
        residuals = c["y"] - c["model"](c["x"], *popt)
        mirror = []
        for _ in range(2):
            p = c["p0"] * (1.0 + 0.3 * rng.uniform(-1.0, 1.0, size=c["p0"].shape)) + 0.1 * rng.uniform(0.0, 1.0, size=c["p0"].shape)
            mirror.append({"p": p.tolist(), "fx": c["model"](c["x"], *p).tolist()})
        results.append(
            {
                "id": c["id"],
                "description": c["description"],
                "xdata": c["x"].tolist(),
                "ydata": c["y"].tolist(),
                "sigma": None if c["sigma"] is None else sigma.tolist(),
                "absoluteSigma": c["absolute_sigma"],
                "p0": c["p0"].tolist(),
                "tolf": FTOL,
                "tolx": XTOL,
                "tolg": GTOL,
                "mirrorPoints": mirror,
                "scipyP": popt.tolist(),
                "scipyCov": pcov.tolist(),
                "scipyStderr": np.sqrt(np.diag(pcov)).tolist(),
                "scipyRss": float(np.sum((residuals / sigma) ** 2)),
                "scipyResiduals": residuals.tolist(),
                "scipyIer": int(ier),
                "scipyEvaluations": int(info["nfev"]),
            }
        )

    out_path = Path(__file__).resolve().with_name("curveFit.scipy.json")
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    print(f"Wrote {len(results)} cases to {out_path}")
    for r in results:
        print(f"{r['id']:26s} ier={r['scipyIer']} nfev={r['scipyEvaluations']:3d} p={np.round(r['scipyP'], 5)} rss={r['scipyRss']:.5e}")


if __name__ == "__main__":
    main()
