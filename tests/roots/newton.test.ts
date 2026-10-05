import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { lineSearch } from '../../src/roots/lineSearch.js';
import { quasiNewton } from '../../src/roots/newton.js';
import {
    fExample65,
    fPowellSingular,
    fRosenbrock,
    fTrigonometric,
} from './testFunctions.js';

// `lineSearch` is wrapped so that individual calls can be made to fail (see the Broyden
// restart tests). Unless a test overrides a call, the real implementation is used.
vi.mock('../../src/roots/lineSearch.js', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../../src/roots/lineSearch.js')>();
    return { ...actual, lineSearch: vi.fn(actual.lineSearch) };
});
const { lineSearch: realLineSearch } = await vi.importActual<typeof import('../../src/roots/lineSearch.js')>(
    '../../src/roots/lineSearch.js',
);

describe('quasiNewton', () => {
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

    it.each([0, -1, 2.5, NaN, Infinity])('requires maxIter to be a finite strictly positive integer before evaluating f (%s)', (maxIter) => {
        let evaluated = false;

        for (const broydenUpdate of [false, true]) {
            expect(() => quasiNewton(() => {
                evaluated = true;
                return Vector.from([1]);
            }, Vector.from([0]), { maxIter, broydenUpdate })).toThrow(
                `quasiNewton: maxIter must be a finite strictly positive integer, got ${maxIter}`,
            );
        }
        expect(evaluated).toBe(false);
    });

    it('accepts maxIter = 1 and reports when the iteration limit is reached', () => {
        const result = quasiNewton(fRosenbrock.f, fRosenbrock.x0, { maxIter: 1 });

        expect(result.success).toBe(false);
        expect(result.iterations).toBe(1);
        expect(result.message).toBe('Maximum number of iterations (1) reached.');
    });

    it('rejects an unknown globalMethod before evaluating f', () => {
        let evaluated = false;

        expect(() => quasiNewton(() => {
            evaluated = true;
            return Vector.from([0]);
        }, Vector.from([1]), { globalMethod: 'bogus' as never })).toThrow(
            "quasiNewton: unknown globalMethod 'bogus'",
        );
        expect(evaluated).toBe(false);
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

    it('uses the provided function scaling for the initial convergence check', () => {
        const x0 = Vector.from([0.005]);
        const result = quasiNewton((x) => x, x0, {
            jac: () => Matrix.from([[1]]),
            jacCheck: false,
            sclf: Vector.from([1e-5]),
            tolf: 1e-5,
        });

        expect(result.success).toBe(true);
        expect(result.iterations).toBe(0);
        expect(result.x.allClose(x0)).toBe(true);
    });

    it('handles a zero Jacobian row when deriving default function scaling', () => {
        const f = (x: Vector) => Vector.from([
            x.get(0) - 1,
            x.get(0) ** 2 + x.get(1) ** 2 - 1,
        ]);
        const jac = (x: Vector) => Matrix.from([
            [1, 0],
            [2 * x.get(0), 2 * x.get(1)],
        ]);

        const result = quasiNewton(f, Vector.from([0, 0]), {
            globalMethod: null,
            jac,
            jacCheck: false,
        });

        expect(result.success).toBe(true);
        expect(result.x.allClose(Vector.from([1, 0]), 1e-6, 1e-6)).toBe(true);
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

    describe('consecutive steps of length maxLen', () => {
        // With x0 = 0, sclx = 1, so maxLen = maxLenFactor = 1. For the linear function
        // f(x) = x - root, the Newton step is exact and only the maxLen cap shortens it.
        const run = (root: number, globalMethod: 'line-search' | 'dogleg' | null) => quasiNewton(
            (x) => Vector.from([x.get(0) - root]),
            Vector.from([0]),
            {
                globalMethod,
                jac: () => Matrix.from([[1]]),
                jacCheck: false,
                maxLenFactor: 1,
            },
        );

        it.each(['line-search', 'dogleg'] as const)('stops after 5 consecutive maximum-length steps (%s)', (globalMethod) => {
            const result = run(1000, globalMethod);

            expect(result.success).toBe(false);
            expect(result.iterations).toBe(5);
            expect(result.x.get(0)).toBeCloseTo(5, 10);
            expect(result.message).toContain('Maximum number (5) of consecutive steps of length `maxLen` reached.');
        });

        it.each(['line-search', 'dogleg'] as const)('stops at 5 consecutive maximum-length steps even if the root is just beyond (%s)', (globalMethod) => {
            const result = run(5.5, globalMethod);

            expect(result.success).toBe(false);
            expect(result.iterations).toBe(5);
            expect(result.message).toContain('Maximum number (5) of consecutive steps');
        });

        it.each(['line-search', 'dogleg'] as const)('does not stop if only 4 maximum-length steps are needed (%s)', (globalMethod) => {
            const result = run(4.5, globalMethod);

            expect(result.success).toBe(true);
            expect(result.iterations).toBe(5);
            expect(result.x.get(0)).toBeCloseTo(4.5, 10);
        });

        it('does not limit the step length counter without a global method', () => {
            const result = run(1000, null);

            expect(result.success).toBe(true);
            expect(result.iterations).toBe(1);
            expect(result.x.get(0)).toBeCloseTo(1000, 10);
        });
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

    it.each([
        { rows: 1, cols: 1 },
        { rows: 2, cols: 3 },
    ])('rejects an initial Jacobian J0 with the wrong dimensions ($rows x $cols)', ({ rows, cols }) => {
        let evaluatedJacobian = false;

        expect(() => quasiNewton(fExample65.f, fExample65.x0, {
            J0: new Matrix(rows, cols),
            jac: () => {
                evaluatedJacobian = true;
                return fExample65.jac!(fExample65.x0);
            },
        })).toThrow(`quasiNewton: J0 must have dimensions 2x2, got ${rows}x${cols}`);
        expect(evaluatedJacobian).toBe(false);
    });

    describe('Broyden restart after a failed global step', () => {
        /**
         * The first `lineSearch` call runs normally; the second one fails (and runs `onFailure`
         * first); later calls run normally again.
         */
        const failSecondLineSearch = (onFailure: () => void = () => { }) => {
            vi.mocked(lineSearch)
                .mockImplementationOnce((ctx) => realLineSearch(ctx))
                .mockImplementationOnce((ctx) => {
                    onFailure();
                    return { success: false, wasMaxStep: false, xp: ctx.xc, fp: ctx.fc, Fp: new Vector(0), trustLen: 0 };
                });
        };

        beforeEach(() => {
            vi.mocked(lineSearch).mockClear();
        });

        it.each([{ useJac: true }, { useJac: false }])(
            'recomputes the Jacobian and continues (user jac: $useJac)',
            ({ useJac }) => {
                let fCalls = 0;
                let jacCalls = 0;
                failSecondLineSearch();

                const result = quasiNewton((x) => {
                    fCalls += 1;
                    return fExample65.f(x);
                }, fExample65.x0, {
                    globalMethod: 'line-search',
                    broydenUpdate: true,
                    tolf: 1e-8,
                    ...(useJac ? {
                        jac: (x: Vector) => {
                            jacCalls += 1;
                            return fExample65.jac!(x);
                        },
                        jacCheck: false,
                    } : {}),
                });

                expect(vi.mocked(lineSearch).mock.calls.length).toBeGreaterThan(2);
                expect(result.success).toBe(true);
                expect(result.x.allClose(fExample65.xs, 1e-6, 1e-6)).toBe(true);
                // Broyden: one Jacobian at x0 plus one for the restart, and no others.
                expect(result.evaluationsJacobian).toBe(useJac ? 2 : 0);
                expect(jacCalls).toBe(useJac ? 2 : 0);
                expect(result.evaluationsFunction).toBe(fCalls);
            },
        );

        it.each([{ useJac: true }, { useJac: false }])(
            'returns a failure result if the restart Jacobian is not finite (user jac: $useJac)',
            ({ useJac }) => {
                let poisoned = false;
                failSecondLineSearch(() => { poisoned = true; });

                // Everything evaluated after the failed step, including the restart Jacobian
                // (user jac or finite differences), returns NaN.
                const f = (x: Vector) => (poisoned ? Vector.from([NaN, NaN]) : fExample65.f(x));

                const result = quasiNewton(f, fExample65.x0, {
                    globalMethod: 'line-search',
                    broydenUpdate: true,
                    ...(useJac ? {
                        jac: (x: Vector) => (poisoned ? Matrix.from([[NaN, NaN], [NaN, NaN]]) : fExample65.jac!(x)),
                        jacCheck: false,
                    } : {}),
                });

                expect(result.success).toBe(false);
                expect(result.message).toBe('Jacobian at the current iterate contains non-finite values.');
                expect(result.iterations).toBe(2);
                expect(vi.mocked(lineSearch).mock.calls.length).toBe(2);
                // The iterate and function value are those after the first, successful step.
                expect(result.x.allClose(fExample65.x0)).toBe(false);
                expect(result.fx.allClose(fExample65.f(result.x))).toBe(true);
                expect(result.Jx!.data.every(Number.isFinite)).toBe(false);
            },
        );
    });

    describe('Broyden update', () => {
        it('discards round-off noise in the secant residual (exact Jacobian, linear function)', () => {
            // f(x) = 5x - 15. With an exact J0 the secant equation holds up to the round-off in f,
            // so the update must leave the Jacobian unchanged. maxLen = 0.7 forces four capped steps
            // with inexact arithmetic before the last, uncapped step reaches the root x = 3.
            // Without the thresholding of the residual, Jx drifts to 5.000000000000001.
            const result = quasiNewton((x) => Vector.from([5 * x.get(0) - 15]), Vector.from([0]), {
                broydenUpdate: true,
                J0: Matrix.from([[5]]),
                sclx: Vector.from([1]),
                sclf: Vector.from([0.2]),
                maxLenFactor: 0.7,
                tolf: 1e-12,
            });

            expect(result.success).toBe(true);
            expect(result.iterations).toBe(5);
            expect(result.x.get(0)).toBeCloseTo(3, 12);
            expect(result.Jx!.get(0, 0)).toBe(5);
        });

        it('returns the unscaled Jacobian approximation for non-uniform sclx and sclf', () => {
            // Linear function with an exact J0: the Broyden updates must keep Jx equal to A.
            const A = Matrix.from([[4, 1], [2, 3]]);
            const b = Vector.from([6, 8]);
            const result = quasiNewton((x) => A.mulVec(x).sub(b), Vector.from([0, 0]), {
                broydenUpdate: true,
                J0: A,
                sclx: Vector.from([2, 0.5]),
                sclf: Vector.from([0.5, 2]),
                maxLenFactor: 0.05,
                maxIter: 3,
            });

            expect(result.iterations).toBe(3);
            expect(result.message).toBe('Maximum number of iterations (3) reached.');
            expect(result.Jx!.allClose(A, 1e-10, 1e-10)).toBe(true);
        });
    });

    describe('epsf', () => {
        const mismatch = 'User-provided Jacobian `jac` does not match finite-difference approximation.';
        const scaledJac = (factor: number) => (x: Vector) => fExample65.jac!(x).mult(factor);

        it('loosens the jacCheck tolerance when the function values are less precise', () => {
            // jac is off by 0.1%: rejected at machine precision, accepted for epsf = 1e-8.
            const strict = quasiNewton(fExample65.f, fExample65.x0, { jac: scaledJac(1.001) });
            const loose = quasiNewton(fExample65.f, fExample65.x0, { jac: scaledJac(1.001), epsf: 1e-8 });

            expect(strict.success).toBe(false);
            expect(strict.message).toBe(mismatch);
            expect(loose.message).not.toBe(mismatch);
            expect(loose.iterations).toBeGreaterThan(0);
        });

        it('never uses an epsf below machine precision', () => {
            // Without the clamp, the finite-difference step would be ~1e-15 and the exact jac would fail the check.
            const exact = quasiNewton(fExample65.f, fExample65.x0, { jac: fExample65.jac!, epsf: 1e-30 });
            const wrong = quasiNewton(fExample65.f, fExample65.x0, { jac: scaledJac(1.001), epsf: 1e-30 });

            expect(exact.success).toBe(true);
            expect(wrong.success).toBe(false);
            expect(wrong.message).toBe(mismatch);
        });

        it('is passed on to the finite-difference Jacobian', () => {
            const jacobianAfterOneIteration = (epsf?: number) => quasiNewton(fExample65.f, fExample65.x0, {
                maxIter: 1,
                ...(epsf === undefined ? {} : { epsf }),
            }).Jx!;

            // A much coarser epsf means a much larger finite-difference step.
            expect(jacobianAfterOneIteration(1e-2).allClose(jacobianAfterOneIteration(), 1e-4, 1e-4)).toBe(false);
        });

        it('still solves the problem with a coarser epsf and finite-difference Jacobians', () => {
            const result = quasiNewton(fExample65.f, fExample65.x0, { epsf: 1e-8 });

            expect(result.success).toBe(true);
            expect(result.x.allClose(fExample65.xs, 1e-4, 1e-4)).toBe(true);
        });
    });

    describe('jacCheck together with J0', () => {
        const identity = () => Matrix.from([[1, 0], [0, 1]]);

        it('checks jac(x0), not J0, against the finite-difference approximation', () => {
            const wrongJac = identity();
            const result = quasiNewton(fExample65.f, fExample65.x0, {
                J0: fExample65.jac!(fExample65.x0), // exact, but must not hide the wrong jac
                jac: () => wrongJac,
            });

            expect(result.success).toBe(false);
            expect(result.iterations).toBe(0);
            expect(result.message).toBe('User-provided Jacobian `jac` does not match finite-difference approximation.');
            expect(result.Jx?.allClose(wrongJac)).toBe(true);
        });

        it('does not reject a crude J0 when jac is correct', () => {
            for (const broydenUpdate of [false, true]) {
                const result = quasiNewton(fExample65.f, fExample65.x0, {
                    J0: identity(),
                    jac: fExample65.jac!,
                    jacCheck: true,
                    broydenUpdate,
                });

                // The solver must start (J0 is not compared with finite differences).
                // Convergence from a crude J0 is not guaranteed without Broyden updates.
                expect(result.message, `broydenUpdate=${broydenUpdate}`).not.toBe(
                    'User-provided Jacobian `jac` does not match finite-difference approximation.',
                );
                expect(result.iterations, `broydenUpdate=${broydenUpdate}`).toBeGreaterThan(0);
                if (broydenUpdate) {
                    expect(result.success).toBe(true);
                    expect(result.x.allClose(fExample65.xs)).toBe(true);
                }
            }
        });

        it('rejects a non-finite jac(x0) even if J0 is provided', () => {
            const result = quasiNewton(fExample65.f, fExample65.x0, {
                J0: fExample65.jac!(fExample65.x0),
                jac: () => Matrix.from([[NaN, 0], [0, 1]]),
            });

            expect(result.success).toBe(false);
            expect(result.iterations).toBe(0);
            expect(result.message).toBe('Jacobian at x0 contains non-finite values.');
            expect(result.Jx!.data.every(Number.isFinite)).toBe(false);
        });

        it.each([
            { jacCheck: true, expectedJacCalls: 1, expectedFCalls: 3 }, // f(x0) + 2 finite-difference columns
            { jacCheck: false, expectedJacCalls: 0, expectedFCalls: 1 },
        ])('evaluates jac(x0) only if requested by jacCheck ($jacCheck)', ({ jacCheck, expectedJacCalls, expectedFCalls }) => {
            let jacCalls = 0;
            let fCalls = 0;
            const result = quasiNewton((x) => {
                fCalls += 1;
                return fExample65.f(x);
            }, fExample65.xs, {
                J0: identity(),
                jac: (x) => {
                    jacCalls += 1;
                    return fExample65.jac!(x);
                },
                jacCheck,
            });

            expect(result.success).toBe(true);
            expect(result.iterations).toBe(0);
            expect(jacCalls).toBe(expectedJacCalls);
            expect(fCalls).toBe(expectedFCalls);
            expect(result.evaluationsJacobian).toBe(expectedJacCalls);
            expect(result.evaluationsFunction).toBe(expectedFCalls);
        });

        it('does not check jac at all when jacCheck is false', () => {
            const result = quasiNewton(fExample65.f, fExample65.xs, {
                J0: identity(),
                jac: () => Matrix.from([[NaN, NaN], [NaN, NaN]]),
                jacCheck: false,
            });

            expect(result.success).toBe(true);
            expect(result.iterations).toBe(0);
        });
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

    describe('non-finite values', () => {
        const globalMethods = ['line-search', 'dogleg', null] as const;

        /** Wraps `f` so that a runaway loop fails the test instead of hanging it. */
        const guarded = (f: (x: Vector) => Vector, limit = 5000) => {
            let calls = 0;
            return (x: Vector) => {
                if (++calls > limit) throw new Error('f evaluated too often (runaway loop?)');
                return f(x);
            };
        };

        it.each([NaN, Infinity, -Infinity])('returns a failure result when f(x0) is not finite (%s)', (value) => {
            for (const globalMethod of globalMethods) {
                let jacobianEvaluated = false;
                const x0 = Vector.from([1]);

                const result = quasiNewton(() => Vector.from([value]), x0, {
                    globalMethod,
                    jac: () => {
                        jacobianEvaluated = true;
                        return Matrix.from([[1]]);
                    },
                });

                expect(result.success).toBe(false);
                expect(result.message).toBe('f(x0) contains non-finite values.');
                expect(result.iterations).toBe(0);
                expect(result.evaluationsFunction).toBe(1);
                expect(result.evaluationsJacobian).toBe(0);
                expect(result.x.allClose(x0)).toBe(true);
                expect(result.fx.get(0)).toBe(value);
                expect(result.Jx).toBeNull();
                expect(jacobianEvaluated).toBe(false);
            }
        });

        it.each([
            {
                source: 'user-supplied jac',
                f: (x: Vector) => x.sub(Vector.from([3])),
                options: { jac: () => Matrix.from([[NaN]]) },
            },
            {
                source: 'initial Jacobian J0',
                f: (x: Vector) => x.sub(Vector.from([3])),
                options: { J0: Matrix.from([[Infinity]]) },
            },
            {
                // The forward-difference step leaves the domain where f is defined.
                source: 'finite-difference approximation',
                f: (x: Vector) => (x.get(0) > 0 ? Vector.from([NaN]) : Vector.from([-1])),
                options: {},
            },
        ])('returns a failure result when the Jacobian at x0 is not finite ($source)', ({ f, options }) => {
            const result = quasiNewton(f, Vector.from([0]), options);

            expect(result.success).toBe(false);
            expect(result.message).toBe('Jacobian at x0 contains non-finite values.');
            expect(result.iterations).toBe(0);
            expect(result.x.allClose(Vector.from([0]))).toBe(true);
            expect(result.Jx).not.toBeNull();
            expect(result.Jx!.data.every(Number.isFinite)).toBe(false);
        });

        it.each(globalMethods)('returns a failure result when the Jacobian becomes non-finite during the iteration (%s)', (globalMethod) => {
            const f = (x: Vector) => Vector.from([x.get(0) ** 2 - 4]);
            const result = quasiNewton(f, Vector.from([1]), {
                globalMethod,
                broydenUpdate: false,
                jac: (x) => Matrix.from([[x.get(0) === 1 ? 2 : NaN]]),
                jacCheck: false,
            });

            expect(result.success).toBe(false);
            expect(result.message).toBe('Jacobian at the current iterate contains non-finite values.');
            expect(result.iterations).toBe(1);
            expect(result.x.allClose(Vector.from([1]))).toBe(false);
            expect(result.fx.allClose(f(result.x))).toBe(true);
            expect(result.Jx!.data.every(Number.isFinite)).toBe(false);
        });

        it.each(['line-search', 'dogleg'] as const)('backtracks from trial points where f is not finite (%s)', (globalMethod) => {
            // The first Newton step from x0 = 3 lands at x < 0, where log(x) is NaN.
            let nonFiniteSeen = false;
            const f = guarded((x) => {
                const value = Math.log(x.get(0));
                if (!Number.isFinite(value)) nonFiniteSeen = true;
                return Vector.from([value]);
            });

            const result = quasiNewton(f, Vector.from([3]), {
                globalMethod,
                jac: (x) => Matrix.from([[1 / x.get(0)]]),
                jacCheck: false,
                tolf: 1e-10,
            });

            expect(nonFiniteSeen).toBe(true);
            expect(result.success).toBe(true);
            expect(result.x.allClose(Vector.from([1]), 1e-8, 1e-8)).toBe(true);
        });

        it('returns a failure result when f is not finite at the full step and no global method is used', () => {
            const x0 = Vector.from([3]);
            const result = quasiNewton(guarded((x) => Vector.from([Math.log(x.get(0))])), x0, {
                globalMethod: null,
                jac: (x) => Matrix.from([[1 / x.get(0)]]),
                jacCheck: false,
            });

            expect(result.success).toBe(false);
            expect(result.message).toContain('f returned non-finite values at the proposed step.');
            expect(result.iterations).toBe(1);
            expect(result.x.allClose(x0)).toBe(true);
            expect(result.fx.allClose(Vector.from([Math.log(3)]))).toBe(true);
        });

        it.each(globalMethods)('terminates with finite results when f is undefined beyond a boundary (%s)', (globalMethod) => {
            // The root x = 3 lies outside the domain (f is NaN for x > 1).
            const raw = (x: Vector) => (x.get(0) > 1 ? Vector.from([NaN]) : Vector.from([x.get(0) - 3]));
            const result = quasiNewton(guarded(raw), Vector.from([0]), {
                globalMethod,
                jac: () => Matrix.from([[1]]),
                jacCheck: false,
                maxIter: 50,
            });

            // `success` is not asserted: stopping on `tolx` is reported as success even though
            // the iterate is stuck at the boundary, far from a root.
            expect(Math.abs(result.fx.get(0))).toBeGreaterThan(1);
            expect(result.x.data.every(Number.isFinite)).toBe(true);
            expect(result.fx.data.every(Number.isFinite)).toBe(true);
            expect(result.fx.allClose(raw(result.x))).toBe(true);
        });
    });

    describe('termination on tolx', () => {
        it.each(['line-search', 'dogleg'] as const)('is reported as success even if the residual is not small (%s)', (globalMethod) => {
            // f(x) = x^2 has a double root: Newton halves x each iteration, so the step
            // falls below tolx while the residual is still far above tolf.
            const result = quasiNewton((x) => Vector.from([x.get(0) ** 2]), Vector.from([1]), {
                globalMethod,
                jac: (x) => Matrix.from([[2 * x.get(0)]]),
                jacCheck: false,
                tolf: 1e-30,
                tolx: 1e-6,
            });

            expect(result.success).toBe(true);
            expect(result.message).toContain('||Δx/max(x, 1/sclx)||∞ ≤ tolx');
            expect(result.fx.get(0)).toBeGreaterThan(1e-30);
        });
    });

    it.each(['line-search', 'dogleg'] as const)('returns fx = f(x) at the unchanged iterate when the global step fails (%s)', (globalMethod) => {
        // The Jacobian has the wrong sign, so no step decreases the residual.
        const x0 = Vector.from([2]);
        const f = (x: Vector) => Vector.from([1 + x.get(0) ** 2]);

        const result = quasiNewton(f, x0, {
            globalMethod,
            jac: () => Matrix.from([[-1]]),
            jacCheck: false,
            maxIter: 20,
        });

        expect(result.success).toBe(false);
        expect(result.x.allClose(x0)).toBe(true);
        expect(result.fx.get(0)).toBe(5);
    });

    describe('evaluation counters', () => {
        const counting = <A extends unknown[], R>(fn: (...args: A) => R) => {
            const wrapped = Object.assign((...args: A): R => {
                wrapped.calls += 1;
                return fn(...args);
            }, { calls: 0 });
            return wrapped;
        };

        const cases = [
            { name: 'no jac', jacCheck: false, useJac: false, functions: [fRosenbrock, fPowellSingular, fTrigonometric, fExample65] },
            { name: 'jac, jacCheck = false', jacCheck: false, useJac: true, functions: [fExample65] },
            { name: 'jac, jacCheck = true', jacCheck: true, useJac: true, functions: [fExample65] },
        ];

        it.each(cases)('matches the actual number of evaluations of f and jac ($name)', ({ jacCheck, useJac, functions }) => {
            for (const globalMethod of ['line-search', 'dogleg', null] as const) {
                for (const broydenUpdate of [false, true]) {
                    for (const testFunction of functions) {
                        const f = counting(testFunction.f);
                        const jac = counting(testFunction.jac ?? (() => Matrix.from([[1]])));

                        const result = quasiNewton(f, testFunction.x0, {
                            tolf: 1e-8,
                            globalMethod,
                            broydenUpdate,
                            ...(useJac ? { jac, jacCheck } : {}),
                        });

                        const label = `${globalMethod}, ${broydenUpdate}, ${testFunction.f.name}`;
                        expect(result.evaluationsFunction, label).toBe(f.calls);
                        expect(result.evaluationsJacobian, label).toBe(jac.calls);
                    }
                }
            }
        });

        it('does not count the evaluation of a provided initial Jacobian J0', () => {
            const f = counting(fExample65.f);
            const jac = counting(fExample65.jac!);

            const result = quasiNewton(f, fExample65.x0, {
                J0: fExample65.jac!(fExample65.x0),
                broydenUpdate: true,
            });

            expect(result.evaluationsFunction).toBe(f.calls);
            expect(result.evaluationsJacobian).toBe(0);
            expect(jac.calls).toBe(0);
        });
    });

});