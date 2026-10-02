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

    it('solves benchmark functions with line search', () => {
        for (const testFunction of [fRosenbrock, fPowellSingular, fTrigonometric, fExample65]) {
            for (const broydenUpdate of [false, true]) {
                const result = quasiNewton(testFunction.f, testFunction.x0, {
                    tolf: 1e-8,
                    globalMethod: 'line-search',
                    broydenUpdate,
                });

                expect(result.success, `${testFunction.f.name}, ${broydenUpdate}`).toBe(true);
                if (testFunction !== fPowellSingular) {
                    expect(result.x.allClose(testFunction.xs, 1e-7, 1e-7)).toBe(true);
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

    it('returns immediately when the initial guess is a solution', () => {
        const result = quasiNewton(fExample65.f, fExample65.xs);

        expect(result.success).toBe(true);
        expect(result.iterations).toBe(0);
    });

});