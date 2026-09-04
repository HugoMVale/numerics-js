import { Vector } from '../../src/array/Vector.js';

/**
 * Multivariate test functions commonly used in optimize benchmarks.
 * Ported from PolyKin's `_tester.py` (`TEST_FUNCTIONS_MULTIVAR`).
 *
 * Only `fn` (the objective), `initialPoint`, and `globalMinimum` are
 * included, since these are the pieces needed by derivative-free methods
 * like `fminNelderMead`. Gradients/Hessians and the `properties` tags from
 * the Python source are omitted; add them back here if a gradient-based
 * method needs them later, so all optimizer tests keep sharing one table.
 */
export interface TestFunctionData {
    /** The objective function to minimize. */
    fn: (x: Vector) => number;
    /** Returns a standard N-dimensional starting point for this function. */
    initialPoint: (n: number) => Vector;
    /** The known global minimum value. */
    globalMinimum: number;
}

/** `linspace(start, end, n)`, matching `numpy.linspace`. */
function linspace(start: number, end: number, n: number): Vector {
    if (n === 1) return Vector.from([start]);
    const step = (end - start) / (n - 1);
    return Vector.from(Array.from({ length: n }, (_, i) => start + step * i));
}

/** Per-component weights for the `ellipsoid` test function. */
function ellipsoidCoeffs(n: number): number[] {
    const denom = n > 1 ? n - 1 : 1;
    return Array.from({ length: n }, (_, i) => 1e6 ** (i / denom));
}

/** `u(x) = 0.5 * sum((i+1) * x_i)`, shared by the `zakharov` function. */
function zakharovU(x: Vector): number {
    let u = 0;
    for (let i = 0; i < x.size; i++) u += 0.5 * (i + 1) * x.get(i);
    return u;
}

export const TEST_FUNCTIONS_MULTIVAR: Record<string, TestFunctionData> = {
    sphere: {
        fn: (x) => {
            let s = 0;
            for (let i = 0; i < x.size; i++) s += x.get(i) ** 2;
            return s;
        },
        // Simple start to verify basic algorithm correctness
        initialPoint: (n) => Vector.from(Array(n).fill(5.0)),
        globalMinimum: 0,
    },
    ellipsoid: {
        fn: (x) => {
            const coeffs = ellipsoidCoeffs(x.size);
            let s = 0;
            for (let i = 0; i < x.size; i++) s += coeffs[i] * x.get(i) ** 2;
            return s;
        },
        // Asymmetric start -> exposes conditioning issues
        initialPoint: (n) => linspace(1.0, 2.0, n),
        globalMinimum: 0,
    },
    rosenbrock: {
        fn: (x) => {
            let s = 0;
            for (let i = 0; i < x.size - 1; i++) {
                s += 100 * (x.get(i + 1) - x.get(i) ** 2) ** 2 + (1 - x.get(i)) ** 2;
            }
            return s;
        },
        // Classic challenging but not extreme start
        initialPoint: (n) => Vector.from(Array(n).fill(-1.2)),
        globalMinimum: 0,
    },
    zakharov: {
        fn: (x) => {
            let s = 0;
            for (let i = 0; i < x.size; i++) s += x.get(i) ** 2;
            const u = zakharovU(x);
            return s + u ** 2 + u ** 4;
        },
        // Initial point recommended by standard benchmark literature
        initialPoint: (n) => Vector.from(Array(n).fill(1.5)),
        globalMinimum: 0,
    },
};