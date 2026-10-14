import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { StepPolynomial } from '../../src/ode/denseOutput.js';
import { trBdf2Adaptive, trBdf2AdaptiveDenseCoefficients } from '../../src/ode/trBdf2Adaptive.js';
import type { DerivativeFunction, JacobianFunction, OdeEvent } from '../../src/ode/types.js';

const tight = { atol: 1e-10, rtol: 1e-10 };
const GAMMA = 2 - Math.SQRT2;

const decay: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);
const oscillator: DerivativeFunction = (_t, y, out) => {
    out.data[0] = y.data[1];
    out.data[1] = -y.data[0];
    return out;
};

// Stiff relaxation y' = K (y - cos t), y(0) = 0, with K = -1000; exact solution below.
const K = -1000;
const stiff: DerivativeFunction = (t, y, out) => {
    out.data[0] = K * (y.data[0] - Math.cos(t));
    return out;
};
const stiffExact = (t: number): number => {
    const a = (K * K) / (1 + K * K);
    const b = -K / (1 + K * K);
    return a * Math.cos(t) + b * Math.sin(t) - a * Math.exp(K * t);
};
function bisect(g: (t: number) => number, lo: number, hi: number): number {
    for (let i = 0; i < 200; i++) {
        const mid = 0.5 * (lo + hi);
        if (g(lo) * g(mid) <= 0) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}

describe('trBdf2Adaptive events', () => {
    it('locates a threshold crossing of an exponential decay', () => {
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] - 0.5 }];
        const result = trBdf2Adaptive(decay, 0, 3, new Vector([1]), { atol: 1e-8, rtol: 1e-8, events });

        expect(result.success).toBe(true);
        expect(result.events).toHaveLength(1);
        const [e] = result.events!;
        expect(e.direction).toBe('falling');
        expect(Math.abs(e.t - Math.LN2)).toBeLessThan(1e-5);
        expect(Math.abs(e.y.data[0] - 0.5)).toBeLessThan(1e-5);
        expect(result.t.get(result.t.size - 1)).toBe(3);
    });

    it('finds all zero crossings of an oscillator with alternating directions', () => {
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] }];
        const result = trBdf2Adaptive(oscillator, 0, 4 * Math.PI, new Vector([1, 0]), { ...tight, events });

        expect(result.events).toHaveLength(4);
        result.events!.forEach((e, k) => {
            expect(Math.abs(e.t - (k + 0.5) * Math.PI)).toBeLessThan(1e-5);
            expect(e.direction).toBe(k % 2 === 0 ? 'falling' : 'rising');
        });
    });

    it('locates an event on a stiff problem, with finite-difference and analytic Jacobians', () => {
        const root = bisect(stiffExact, 1.5, 1.7);
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], direction: 'falling' }];
        const jac: JacobianFunction = (_t, _y, J) => J.set(0, 0, K);

        for (const extra of [{}, { jac }]) {
            const result = trBdf2Adaptive(stiff, 0, 3, new Vector([0]), { atol: 1e-8, rtol: 1e-8, events, ...extra });
            expect(result.events).toHaveLength(1);
            expect(Math.abs(result.events![0].t - root)).toBeLessThan(1e-7);
            // far fewer steps than an explicit method would need for this stiff problem
            expect(result.t.size).toBeLessThan(1500);
        }
    });

    it('respects the direction filter', () => {
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], direction: 'rising' }];
        const result = trBdf2Adaptive(oscillator, 0, 4 * Math.PI, new Vector([1, 0]), { ...tight, events });
        expect(result.events!.map((e) => e.direction)).toEqual(['rising', 'rising']);
        expect(Math.abs(result.events![0].t - 1.5 * Math.PI)).toBeLessThan(1e-5);
    });

    it('leaves the integration bit-identical when events are not terminal', () => {
        const run = (extra: object) => trBdf2Adaptive(stiff, 0, 3, new Vector([0]), extra);
        const plain = run({});
        const withEvents = run({ events: [{ fn: (_t: number, y: Vector) => y.data[0] }, { fn: (_t: number, y: Vector) => y.data[0] - 0.2 }] });

        expect(Array.from(withEvents.t.data)).toEqual(Array.from(plain.t.data));
        expect(Array.from(withEvents.y.data)).toEqual(Array.from(plain.y.data));
        expect(withEvents.evaluations).toBe(plain.evaluations);
        expect(withEvents.evaluationsJacobian).toBe(plain.evaluationsJacobian);
        expect(withEvents.decompositions).toBe(plain.decompositions);
        expect(withEvents.events!.length).toBeGreaterThan(0);
        expect('events' in plain).toBe(false);
    });

    it('does not count event function calls as derivative evaluations', () => {
        let fCalls = 0;
        const f: DerivativeFunction = (t, y, out) => (fCalls++, decay(t, y, out));
        const result = trBdf2Adaptive(f, 0, 2, new Vector([1]), { events: [{ fn: (_t, y) => y.data[0] - 0.5 }] });
        expect(result.evaluations).toBe(fCalls);
    });

    describe('terminal events', () => {
        it('stops at the event, which becomes the last recorded point', () => {
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] - 0.5, terminal: true }];
            const result = trBdf2Adaptive(decay, 0, 3, new Vector([1]), { atol: 1e-8, rtol: 1e-8, events });

            expect(result.success).toBe(true);
            expect(result.message).toBe('Integration stopped by a terminal event.');
            const e = result.events!.at(-1)!;
            expect(result.t.get(result.t.size - 1)).toBe(e.t);
            expect(Math.abs(e.t - Math.LN2)).toBeLessThan(1e-5);
            const last = result.y.row(result.y.rows - 1).data;
            expect(Array.from(last)).toEqual(Array.from(e.y.data));
            expect(result.t.size).toBe(result.y.rows);
            for (let i = 1; i < result.t.size; i++) expect(result.t.get(i)).toBeGreaterThan(result.t.get(i - 1));
        });

        it('stops at the N-th occurrence', () => {
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], terminal: 3 }];
            const result = trBdf2Adaptive(oscillator, 0, 10 * Math.PI, new Vector([1, 0]), { ...tight, events });
            expect(result.events).toHaveLength(3);
            expect(Math.abs(result.events!.at(-1)!.t - 2.5 * Math.PI)).toBeLessThan(1e-5);
        });

        it('supports restarting a bouncing ball from its impact point', () => {
            const g = 9.81;
            const ball: DerivativeFunction = (_t, y, out) => {
                out.data[0] = y.data[1];
                out.data[1] = -g;
                return out;
            };
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], direction: 'falling', terminal: true }];
            const first = trBdf2Adaptive(ball, 0, 5, new Vector([1, 0]), { ...tight, events });
            const hit1 = first.events!.at(-1)!;
            expect(Math.abs(hit1.t - Math.sqrt(2 / g))).toBeLessThan(1e-6);

            const e = 0.5;
            const v1 = -hit1.y.data[1];
            const second = trBdf2Adaptive(ball, hit1.t, 5, new Vector([0, e * v1]), { ...tight, events });
            const hit2 = second.events!.at(-1)!;
            expect(Math.abs(hit2.t - (hit1.t + (2 * e * v1) / g))).toBeLessThan(1e-6);
            expect(second.success).toBe(true);
        });
    });

    describe('boundaries', () => {
        it('integrates backward in time', () => {
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] - 2 }];
            const result = trBdf2Adaptive(decay, 0, -2, new Vector([1]), { atol: 1e-8, rtol: 1e-8, events });
            expect(result.events).toHaveLength(1);
            expect(Math.abs(result.events![0].t + Math.LN2)).toBeLessThan(1e-5);
            expect(result.events![0].direction).toBe('rising');
        });

        it('reports an event at tEnd', () => {
            const f: DerivativeFunction = (_t, _y, out) => {
                out.data[0] = 1;
                return out;
            };
            const result = trBdf2Adaptive(f, 0, 1, new Vector([0]), { events: [{ fn: (_t, y) => y.data[0] - 1 }] });
            expect(result.events).toHaveLength(1);
            expect(result.events![0].t).toBeCloseTo(1, 10);
        });

        it('does not report an event function that is zero at t0', () => {
            const f: DerivativeFunction = (_t, _y, out) => {
                out.data[0] = 1;
                return out;
            };
            const result = trBdf2Adaptive(f, 0, 1, new Vector([0]), { events: [{ fn: (_t, y) => y.data[0] }] });
            expect(result.events).toEqual([]);
        });

        it('returns an empty event list when t0 === tEnd', () => {
            expect(trBdf2Adaptive(decay, 1, 1, new Vector([1]), { events: [{ fn: () => 0 }] }).events).toEqual([]);
            expect(trBdf2Adaptive(decay, 1, 1, new Vector([1])).events).toBeUndefined();
        });

        it('returns an empty event list when no crossing occurs', () => {
            const result = trBdf2Adaptive(decay, 0, 1, new Vector([1]), { events: [{ fn: (_t, y) => y.data[0] + 1 }] });
            expect(result.events).toEqual([]);
        });

        it('keeps events found before an unsuccessful termination', () => {
            const result = trBdf2Adaptive(oscillator, 0, 100, new Vector([1, 0]), {
                maxSteps: 40,
                events: [{ fn: (_t, y) => y.data[0] }],
            });
            expect(result.success).toBe(false);
            expect(result.events!.length).toBeGreaterThan(0);
        });
    });

    describe('input validation', () => {
        const run = (events: OdeEvent[]) => () => trBdf2Adaptive(decay, 0, 1, new Vector([1]), { events });

        it('rejects invalid events, also when t0 === tEnd', () => {
            expect(run([{} as OdeEvent])).toThrow(TypeError);
            expect(run([{ fn: () => 0, direction: 'up' as 'rising' }])).toThrow(RangeError);
            expect(run([{ fn: () => 0, terminal: 0 }])).toThrow(RangeError);
            expect(() => trBdf2Adaptive(decay, 1, 1, new Vector([1]), { events: [{} as OdeEvent] })).toThrow(TypeError);
        });
    });
});

describe('trBdf2AdaptiveDenseCoefficients', () => {
    // Data from an exact solution y(t) at the step's nodes: start, GAMMA, end, plus the start slope.
    function interpolate(y: (t: number) => number, dy: (t: number) => number, t0: number, h: number, x: number): number {
        const yStart = new Vector([y(t0)]);
        const yEnd = new Vector([y(t0 + h)]);
        const yG = new Vector([y(t0 + GAMMA * h)]);
        const zN = new Vector([h * dy(t0)]);
        const step = new StepPolynomial(1, 3, (c) => trBdf2AdaptiveDenseCoefficients(yStart, yEnd, yG, zN, c));
        step.reset(t0, h);
        return step.evaluate(t0 + x * h, new Vector(1)).data[0];
    }

    it('reproduces cubic polynomials exactly', () => {
        const y = (t: number): number => 1 + 2 * t - 3 * t ** 2 + 0.5 * t ** 3;
        const dy = (t: number): number => 2 - 6 * t + 1.5 * t ** 2;
        for (const x of [0, 0.1, GAMMA, 0.5, 0.9, 1]) {
            expect(interpolate(y, dy, 0.7, 0.3, x)).toBeCloseTo(y(0.7 + 0.3 * x), 13);
        }
    });

    it('joins the step end points and the intermediate stage state', () => {
        const y = Math.exp;
        const h = 0.25;
        expect(interpolate(y, y, 1, h, 0)).toBe(y(1));
        expect(Math.abs(interpolate(y, y, 1, h, 1) - y(1 + h))).toBeLessThan(1e-15);
        expect(Math.abs(interpolate(y, y, 1, h, GAMMA) - y(1 + GAMMA * h))).toBeLessThan(1e-15);
    });

    it('converges with fourth order on smooth exact data', () => {
        const err = (h: number): number => {
            let e = 0;
            for (let i = 0; i <= 50; i++) e = Math.max(e, Math.abs(interpolate(Math.exp, Math.exp, 0, h, i / 50) - Math.exp(h * (i / 50))));
            return e;
        };
        expect(Math.log2(err(0.2) / err(0.1))).toBeGreaterThan(3.7);
    });

    it('interpolates every component independently', () => {
        const yStart = new Vector([0, 1]);
        const yEnd = new Vector([1, 1]);
        const yG = new Vector([GAMMA, 1]);
        const zN = new Vector([1, 0]);
        const step = new StepPolynomial(2, 3, (c) => trBdf2AdaptiveDenseCoefficients(yStart, yEnd, yG, zN, c));
        step.reset(0, 1);
        const out = step.evaluate(0.5, new Vector(2));
        expect(out.data[0]).toBeCloseTo(0.5, 14); // linear component y = x
        expect(out.data[1]).toBeCloseTo(1, 14); // constant component
    });

    it('writes y_n, z_n, c_2 and c_3 as consecutive blocks', () => {
        const out = new Float64Array(8).fill(NaN);
        // Per component, the cubic p(x) = 1 + 2x + 3x^2 + 4x^3 (and 5 - x) fixes all four blocks.
        const p0 = (x: number): number => 1 + 2 * x + 3 * x ** 2 + 4 * x ** 3;
        const p1 = (x: number): number => 5 - x;
        trBdf2AdaptiveDenseCoefficients(
            new Vector([p0(0), p1(0)]),
            new Vector([p0(1), p1(1)]),
            new Vector([p0(GAMMA), p1(GAMMA)]),
            new Vector([2, -1]),
            out
        );
        const expected = [1, 5, 2, -1, 3, 0, 4, 0];
        expected.forEach((v, i) => expect(out[i]).toBeCloseTo(v, 13));
    });
});
