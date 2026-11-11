# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Generate SciPy reference values for the special.besselYZero test fixture.

Run with: uv run benchmarks/special/generate_besselYZero_fixtures.py

The reference is scipy.special.yn_zeros. Each order here is checked for the
listed zero indices by benchmarks/special/besselYZero.scipy.test.ts, which
iterates over the fixture (the ids only label the orders).
"""

import json
from pathlib import Path

from scipy.special import yn_zeros

ORDERS = [0, 1, 2, 5, 10, 30, 100]
INDICES = [1, 2, 3, 5, 10, 20, 50, 100, 200]


def main() -> None:
    results = []
    for n in ORDERS:
        zeros = yn_zeros(n, max(INDICES))
        results.append(
            {
                "id": f"order_{n}",
                "description": f"zeros of Y_{n}",
                "n": n,
                "zeros": [{"m": m, "scipy": float(zeros[m - 1])} for m in INDICES],
            }
        )

    out_path = Path(__file__).resolve().with_name("besselYZero.scipy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=4) + "\n")
    total = sum(len(r["zeros"]) for r in results)
    print(f"Wrote {len(results)} orders ({total} zeros) to {out_path}")


if __name__ == "__main__":
    main()
