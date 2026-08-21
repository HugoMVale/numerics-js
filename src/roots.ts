export type MathFunction = (x: number) => number;

/**
 * Finds a root of fn in the interval [a, b] using the bisection method.
 *
 * @param fn - Continuous function to find a root of.
 * @param a - Left endpoint of the bracketing interval.
 * @param b - Right endpoint of the bracketing interval.
 * @param tolerance - Stop when the interval half-width is below this.
 * @param maxIterations - Maximum number of iterations.
 * @returns Approximate root.
 * @throws {Error} If fn(a) and fn(b) don't bracket a root, or if convergence fails.
 */
export function bisection(
    fn: MathFunction,
    a: number,
    b: number,
    tolerance: number = 1e-8,
    maxIterations: number = 100
): number {
    if (a === b) {
        throw new Error('bisection: a and b must be different');
    }
    if (a > b) {
        [a, b] = [b, a];
    }

    let fa = fn(a);
    let fb = fn(b);

    if (fa === 0) return a;
    if (fb === 0) return b;

    if (Math.sign(fa) === Math.sign(fb)) {
        throw new Error(
            `bisection: fn(a) and fn(b) must have opposite signs (got fn(a)=${fa}, fn(b)=${fb})`
        );
    }

    let mid = (a + b) / 2;

    for (let k = 0; k < maxIterations; k++) {
        mid = (a + b) / 2;
        const fmid = fn(mid);

        if (fmid === 0 || (b - a) / 2 < tolerance) {
            return mid;
        }

        if (Math.sign(fmid) === Math.sign(fa)) {
            a = mid;
            fa = fmid;
        } else {
            b = mid;
        }
    }

    console.warn(
        `bisection: reached maxIterations (${maxIterations}) without converging to tolerance ${tolerance}`
    );
    return mid;
}

/**
 * Finds a root of fn using the secant method, starting from two initial guesses.
 *
 * Unlike bisection, the secant method does not require a bracketing interval
 * (fn(x0) and fn(x1) need not have opposite signs), and typically converges
 * faster (superlinear, order ~1.618) when it converges. It is not guaranteed
 * to converge for all inputs.
 *
 * @param fn - Function to find a root of.
 * @param x0 - First initial guess.
 * @param x1 - Second initial guess (should differ from x0).
 * @param tolerance - Stop when |x1 - x0| (the step size) is below this.
 * @param maxIterations - Maximum number of iterations.
 * @returns Approximate root.
 * @throws {Error} If x0 and x1 are equal, or if a zero derivative estimate is encountered.
 */
export function secant(
    fn: MathFunction,
    x0: number,
    x1: number,
    tolerance: number = 1e-8,
    maxIterations: number = 100
): number {
    if (x0 === x1) {
        throw new Error('secant: x0 and x1 must be different');
    }

    let f0 = fn(x0);
    let f1 = fn(x1);

    if (f0 === 0) return x0;
    if (f1 === 0) return x1;

    for (let k = 0; k < maxIterations; k++) {
        const denom = f1 - f0;

        if (denom === 0) {
            throw new Error(
                `secant: zero denominator encountered (fn(x0)=${f0}, fn(x1)=${f1}); cannot continue`
            );
        }

        const x2 = x1 - (f1 * (x1 - x0)) / denom;

        if (Math.abs(x2 - x1) < tolerance) {
            return x2;
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = fn(x1);

        if (f1 === 0) {
            return x1;
        }
    }

    console.warn(
        `secant: reached maxIterations (${maxIterations}) without converging to tolerance ${tolerance}`
    );
    return x1;
}