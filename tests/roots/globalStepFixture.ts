import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import type { GlobalStepContext } from '../../src/roots/types.js';

// F(x) = x - [1, 2] (Jacobian = I), f = 0.5 * ||F||^2, root at [1, 2].
const target = Vector.from([1, 2]);
export const fN = (x: Vector): [number, Vector] => {
    const F = x.sub(target);
    return [0.5 * F.dot(F), F];
};

/** Context at xc = [5, 5] for the fixture above, with Newton step `p`. */
export function makeCtx(p: number[], overrides: Partial<GlobalStepContext> = {}): GlobalStepContext {
    const xc = Vector.from([5, 5]);
    const [fc, Fc] = fN(xc);
    return {
        fN,
        p: Vector.from(p),
        xc,
        fc,
        gc: Fc, // J^T F with J = I
        R: Matrix.identity(2),
        tolx: 1e-8,
        sclx: Vector.from([1, 1]),
        maxLen: 100,
        ...overrides,
    };
}
