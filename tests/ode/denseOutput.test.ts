import { describe, expect, it, vi } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { OdeDenseOutput, StepPolynomial } from '../../src/ode/denseOutput.js';

// Segment of y(t) = t^2 starting at `a` with signed size `h`: y(a + h x) = a^2 + 2 a h x + h^2 x^2.
function squareStep(a: number, h: number): StepPolynomial {
    const step = new StepPolynomial(1, 2, (c) => c.set([a * a, 2 * a * h, h * h]));
    step.reset(a, h);
    return step;
}

// Dense output of y(t) = t^2 from `breaks[0]` through the given step points, in integration order.
function squareOutput(breaks: number[]): OdeDenseOutput {
    const output = new OdeDenseOutput(breaks[0], new Vector([breaks[0] ** 2]));
    for (let i = 0; i + 1 < breaks.length; i++) {
        output.append(breaks[i + 1], squareStep(breaks[i], breaks[i + 1] - breaks[i]));
    }
    return output;
}

describe('StepPolynomial', () => {
    it('builds the coefficients lazily, once per step', () => {
        const build = vi.fn((c: Float64Array) => c.set([1, 2, 3]));
        const step = new StepPolynomial(1, 2, build);

        step.reset(0, 1);
        expect(build).not.toHaveBeenCalled();

        const out = new Vector(1);
        step.evaluate(0.5, out);
        step.evaluate(0.25, out);
        expect(build).toHaveBeenCalledTimes(1);
        expect(Array.from(step.ensure())).toEqual([1, 2, 3]);

        step.reset(1, 1);
        expect(build).toHaveBeenCalledTimes(1);
        step.ensure();
        expect(build).toHaveBeenCalledTimes(2);
    });

    it('evaluates every component at the relative position (t - t_n) / h', () => {
        // Component 0: 1 + 2x + 3x^2 + 4x^3; component 1: 5 - x.
        const step = new StepPolynomial(2, 3, (c) => c.set([1, 5, 2, -1, 3, 0, 4, 0]));
        step.reset(10, 0.5);
        const out = new Vector(2);

        expect(step.evaluate(10, out)).toBe(out);
        expect(out.data[0]).toBe(1);
        expect(out.data[1]).toBe(5);

        step.evaluate(10.25, out); // x = 0.5
        expect(out.data[0]).toBeCloseTo(1 + 1 + 0.75 + 0.5, 15);
        expect(out.data[1]).toBeCloseTo(4.5, 15);

        step.reset(10, -0.5); // a backward step: the same position is reached with t decreasing
        step.evaluate(9.75, out);
        expect(out.data[0]).toBeCloseTo(1 + 1 + 0.75 + 0.5, 15);
    });
});

describe('OdeDenseOutput', () => {
    describe('forward integration', () => {
        const output = squareOutput([0, 0.5, 2, 3]);

        it('reports its dimension and solved interval', () => {
            expect(output.dim).toBe(1);
            expect(output.tStart).toBe(0);
            expect(output.tEnd).toBe(3);
        });

        it('reproduces the solution inside every segment', () => {
            for (const t of [0.01, 0.25, 0.49, 0.5, 0.51, 1, 1.99, 2, 2.01, 2.5, 2.99]) {
                expect(output.eval(t).get(0)).toBeCloseTo(t * t, 14);
            }
        });

        it('returns the stored state exactly at the start of every segment', () => {
            expect(output.eval(0).get(0)).toBe(0);
            expect(output.eval(0.5).get(0)).toBe(0.25);
            expect(output.eval(2).get(0)).toBe(4);
        });

        it('includes the final point', () => {
            expect(output.eval(3).get(0)).toBeCloseTo(9, 14);
        });

        it('returns a Vector for a number and a Matrix, one row per time, for an array or Vector', () => {
            const ts = [2.5, 0, 1, 3, 0.25];

            const single = output.eval(1);
            expect(single).toBeInstanceOf(Vector);
            expect(single.size).toBe(1);

            for (const arg of [ts, new Vector(ts)]) {
                const many = output.eval(arg);
                expect(many).toBeInstanceOf(Matrix);
                expect(many.rows).toBe(5);
                expect(many.cols).toBe(1);
                ts.forEach((t, i) => expect(many.get(i, 0)).toBeCloseTo(t * t, 14));
            }
        });
    });

    describe('backward integration', () => {
        const output = squareOutput([3, 2, 0.5, 0]);

        it('reports the solved interval in the integration direction', () => {
            expect(output.tStart).toBe(3);
            expect(output.tEnd).toBe(0);
        });

        it('reproduces the solution inside every segment and at the end points', () => {
            for (const t of [3, 2.99, 2.5, 2.01, 2, 1.99, 1, 0.51, 0.5, 0.49, 0.25, 0.01, 0]) {
                expect(output.eval(t).get(0)).toBeCloseTo(t * t, 14);
            }
            expect(output.eval(3).get(0)).toBe(9);
            expect(output.eval(2).get(0)).toBe(4);
        });

        it('rejects times outside the solved interval', () => {
            expect(() => output.eval(3.01)).toThrow(RangeError);
            expect(() => output.eval(-0.01)).toThrow(RangeError);
        });
    });

    describe('at a step point', () => {
        it('uses the segment that starts there, and the last segment at the final point', () => {
            // Deliberately discontinuous pieces, so the segment that is used can be told apart:
            // y = 1 + x on [0, 1] (ends at 2), y = 10 + x on [1, 2] (starts at 10, ends at 11).
            const first = new StepPolynomial(1, 1, (c) => c.set([1, 1]));
            first.reset(0, 1);
            const second = new StepPolynomial(1, 1, (c) => c.set([10, 1]));
            second.reset(1, 1);
            const output = new OdeDenseOutput(0, new Vector([1]));
            output.append(1, first);
            output.append(2, second);

            expect(output.eval(1).get(0)).toBe(10);
            expect(output.eval(2).get(0)).toBe(11);
            expect(output.eval(1 - 1e-12).get(0)).toBeCloseTo(2, 10);
        });
    });

    describe('components', () => {
        it('are evaluated independently', () => {
            // y0 = 1 + 2x, y1 = -3x^2 on [0, 2] (h = 2): in t, y0 = 1 + t and y1 = -0.75 t^2.
            const step = new StepPolynomial(2, 2, (c) => c.set([1, 0, 2, 0, 0, -3]));
            step.reset(0, 2);
            const output = new OdeDenseOutput(0, new Vector([1, 0]));
            output.append(2, step);

            expect(output.dim).toBe(2);
            const y = output.eval(1);
            expect(y.get(0)).toBeCloseTo(2, 15);
            expect(y.get(1)).toBeCloseTo(-0.75, 15);

            const m = output.eval([0, 1, 2]);
            expect(m.rows).toBe(3);
            expect(m.cols).toBe(2);
            expect(m.get(2, 0)).toBeCloseTo(3, 15);
            expect(m.get(2, 1)).toBeCloseTo(-3, 15);
        });
    });

    describe('a step cut short by a terminal event', () => {
        it('ends the solved interval at the event time and keeps the full-step polynomial', () => {
            const output = new OdeDenseOutput(0, new Vector([0]));
            output.append(1, squareStep(0, 1));
            output.append(1.5, squareStep(1, 1)); // the polynomial spans [1, 2], the segment ends at 1.5

            expect(output.tEnd).toBe(1.5);
            expect(output.eval(1.5).get(0)).toBeCloseTo(2.25, 14);
            expect(output.eval(1.25).get(0)).toBeCloseTo(1.5625, 14);
            expect(() => output.eval(1.5 + 1e-9)).toThrow(RangeError);
        });
    });

    describe('without any step', () => {
        it('evaluates only at the initial time, where it returns the initial state', () => {
            const y0 = new Vector([1, 2]);
            const output = new OdeDenseOutput(5, y0);

            expect(output.tStart).toBe(5);
            expect(output.tEnd).toBe(5);
            expect(output.eval(5).toString()).toBe('Vector(1, 2)');
            expect(output.eval([5, 5]).rows).toBe(2);
            expect(() => output.eval(5.1)).toThrow(RangeError);
            expect(() => output.eval(4.9)).toThrow(RangeError);
        });
    });

    describe('ownership of the data', () => {
        it('copies the initial state', () => {
            const y0 = new Vector([1]);
            const output = new OdeDenseOutput(0, y0);
            y0.data[0] = 99;
            expect(output.eval(0).get(0)).toBe(1);
        });

        it('copies the coefficients of an appended step', () => {
            const step = squareStep(0, 1);
            const output = new OdeDenseOutput(0, new Vector([0]));
            output.append(1, step);
            step.coefficients.fill(1000);
            expect(output.eval(0.5).get(0)).toBeCloseTo(0.25, 15);
        });
    });

    describe('invalid input', () => {
        const output = squareOutput([0, 1, 2]);

        it('rejects times outside the solved interval, reporting the interval', () => {
            expect(() => output.eval(-1e-12)).toThrow(RangeError);
            expect(() => output.eval(2 + 1e-12)).toThrow(RangeError);
            expect(() => output.eval(-1)).toThrow(/outside the solved interval \[0, 2\]/);
            expect(() => output.eval(Infinity)).toThrow(RangeError);
        });

        it('rejects NaN', () => {
            expect(() => output.eval(NaN)).toThrow(RangeError);
            expect(() => output.eval([0.5, NaN])).toThrow(RangeError);
        });

        it('rejects an array in which any time is out of range', () => {
            expect(() => output.eval([0.5, 1, 2.5])).toThrow(RangeError);
            expect(() => output.eval(new Vector([-0.5, 1]))).toThrow(RangeError);
        });

        it('rejects an empty array or Vector', () => {
            expect(() => output.eval([])).toThrow(RangeError);
            expect(() => output.eval(new Vector(0))).toThrow(RangeError);
        });
    });
});
