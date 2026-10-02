import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { quasiNewton } from '../../src/roots/newton.js';
import {
    fExample65,
    fPowellSingular,
    fRosenbrock,
    fTrigonometric,
} from './testFunctions.js';

describe('rootVecQNewton', () => {
    it('rejects an initial guess with no components before evaluating f', () => {
        let evaluated = false;

        expect(() => quasiNewton(() => {
            evaluated = true;
            return Vector.from([]);
        }, Vector.from([]))).toThrow(
            'quasiNewton: x0 must have at least one component',
        );
        expect(evaluated).toBe(false);
    });

    it('rejects an initial function value with a different dimension before evaluating the Jacobian', () => {
        let jacobianEvaluated = false;

        expect(() => quasiNewton(() => Vector.from([1]), Vector.from([1, 2]), {
            jac: () => {
                jacobianEvaluated = true;
                return Matrix.from([[1, 0], [0, 1]]);
            },
        })).toThrow('quasiNewton: f(x0) must have dimension 2, got 1');
        expect(jacobianEvaluated).toBe(false);
    });

    it.each(['sclx', 'sclf'] as const)('requires strictly positive %s values before evaluating f', (name) => {
        let evaluated = false;

        expect(() => quasiNewton(() => {
            evaluated = true;
            return Vector.from([0]);
        }, Vector.from([1]), { [name]: Vector.from([1, 0]) })).toThrow(
            `quasiNewton: ${name} must contain strictly positive values`,
        );
        expect(evaluated).toBe(false);
    });

    it.each([
        { trustLen: 1, maxLenFactor: 1e3, expectedStepLen: 1 },
        { trustLen: 100, maxLenFactor: 0.2, expectedStepLen: Math.sqrt(50) * 0.2 },
    ])('uses the initial dogleg radius and caps it at maxLen', ({ trustLen, maxLenFactor, expectedStepLen }) => {
        const x0 = Vector.from([5, 5]);
        const target = Vector.from([1, 2]);
        const evaluatedXs: Vector[] = [];
        const f = (x: Vector) => {
            evaluatedXs.push(x.copy());
            return x.sub(target);
        };

        quasiNewton(f, x0, {
            globalMethod: 'dogleg',
            jac: () => Matrix.from([[1, 0], [0, 1]]),
            jacCheck: false,
            maxIter: 1,
            maxLenFactor,
            sclx: Vector.ones(2),
            trustLen,
        });

        expect(evaluatedXs.length).toBeGreaterThan(1);
        expect(evaluatedXs[1].sub(x0).norm()).toBeCloseTo(expectedStepLen, 12);
    });

    it('solves benchmark functions with line search and dogleg', () => {
        for (const globalMethod of ['line-search', 'dogleg'] as const) {
            for (const broydenUpdate of [false, true]) {
                for (const testFunction of [fRosenbrock, fPowellSingular, fTrigonometric, fExample65]) {
                    const result = quasiNewton(testFunction.f, testFunction.x0, {
                        tolf: 1e-8,
                        globalMethod,
                        broydenUpdate,
                    });

                    expect(result.success, `${globalMethod}, ${testFunction.f.name}, ${broydenUpdate}`).toBe(true);
                    if (testFunction !== fPowellSingular) {
                        expect(result.x.allClose(testFunction.xs, 1e-7, 1e-7)).toBe(true);
                    }
                }
            }
        }
    });

    it('solves the supported benchmark functions without a global method', () => {
        for (const testFunction of [fRosenbrock, fPowellSingular, fTrigonometric]) {
            const result = quasiNewton(testFunction.f, testFunction.x0, {
                tolf: 1e-8,
                globalMethod: null,
            });

            expect(result.success, testFunction.f.name).toBe(true);
            if (testFunction !== fPowellSingular) {
                expect(result.x.allClose(testFunction.xs, 1e-7, 1e-7)).toBe(true);
            }
        }
    });

    it('solves the Broyden case from an approximate identity Jacobian', () => {
        const result = quasiNewton(fExample65.f, fExample65.x0, {
            broydenUpdate: true,
            J0: Matrix.from([
                [1, 0],
                [0, 1],
            ]),
        });

        expect(result.success).toBe(true);
    });

    it('solves with an analytic Jacobian and optional Broyden updates', () => {
        for (const broydenUpdate of [false, true]) {
            const result = quasiNewton(fExample65.f, fExample65.x0, {
                jac: fExample65.jac!,
                jacCheck: true,
                broydenUpdate,
            });

            expect(result.success).toBe(true);
            expect(result.x.allClose(fExample65.xs)).toBe(true);
        }
    });

    it('rejects an analytic Jacobian with the wrong dimensions', () => {
        expect(() => quasiNewton(fExample65.f, fExample65.x0, {
            jac: () => Matrix.from([[1]]),
        })).toThrow('quasiNewton: jac(x0) must have dimensions 2x2, got 1x1');
    });

    it('returns a failure result when the analytic Jacobian fails the finite-difference check', () => {
        const J0 = Matrix.from([
            [1, 0],
            [0, 1],
        ]);
        const result = quasiNewton(fExample65.f, fExample65.x0, {
            jac: () => J0,
        });

        expect(result.success).toBe(false);
        expect(result.iterations).toBe(0);
        expect(result.message).toBe('User-provided Jacobian `jac` does not match finite-difference approximation.');
        expect(result.x.allClose(fExample65.x0)).toBe(true);
        expect(result.Jx?.allClose(J0)).toBe(true);
    });

    it('returns immediately when the initial guess is a solution', () => {
        const result = quasiNewton(fExample65.f, fExample65.xs);

        expect(result.success).toBe(true);
        expect(result.iterations).toBe(0);
    });

});