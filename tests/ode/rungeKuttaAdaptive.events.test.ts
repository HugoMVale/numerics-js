import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import {
    rungeKuttaAdaptive,
    rungeKuttaAdaptiveDense,
    rungeKuttaAdaptiveStep,
} from '../../src/ode/rungeKuttaAdaptive.js';
import type { AdaptiveScratch } from '../../src/ode/rungeKuttaAdaptive.js';
import type { DerivativeFunction, OdeEvent } from '../../src/ode/types.js';

const methods = ['rk23', 'rk45'] as const;
const tight = { atol: 1e-10, rtol: 1e-10 };

const decay: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);
const oscillator: DerivativeFunction = (_t, y, out) => {
    out.data[0] = y.data[1];
    out.data[1] = -y.data[0];
    return out;
};
const lastRow = (r: { y: { rows: number; row(i: number): Vector } }): Float64Array => r.y.row(r.y.rows - 1).data;

describe('rungeKuttaAdaptive events', () => {
    it.each(methods)('(%s) locates a threshold crossing of an exponential decay', (method) => {
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] - 0.5 }];
        const result = rungeKuttaAdaptive(method, decay, 0, 3, new Vector([1]), { ...tight, events });

        expect(result.success).toBe(true);
        expect(result.events).toHaveLength(1);
        const [e] = result.events!;
        expect(e.direction).toBe('falling');
        expect(e.t).toBeCloseTo(Math.LN2, 7);
        expect(e.y.data[0]).toBeCloseTo(0.5, 7);
        // the solution grid is unaffected by a non-terminal event
        expect(result.t.get(result.t.size - 1)).toBe(3);
    });

    it.each(methods)('(%s) finds all zero crossings of an oscillator with alternating directions', (method) => {
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] }];
        const result = rungeKuttaAdaptive(method, oscillator, 0, 4 * Math.PI, new Vector([1, 0]), { ...tight, events });

        const found = result.events!;
        expect(found).toHaveLength(4);
        found.forEach((e, k) => {
            expect(e.t).toBeCloseTo((k + 0.5) * Math.PI, 6);
            expect(e.direction).toBe(k % 2 === 0 ? 'falling' : 'rising');
            expect(e.y.data[1]).toBeCloseTo(k % 2 === 0 ? -1 : 1, 6);
        });
    });

    it.each(methods)('(%s) respects the direction filter', (method) => {
        const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], direction: 'rising' }];
        const result = rungeKuttaAdaptive(method, oscillator, 0, 4 * Math.PI, new Vector([1, 0]), { ...tight, events });

        expect(result.events!.map((e) => e.t)).toEqual([
            expect.closeTo(1.5 * Math.PI, 6),
            expect.closeTo(3.5 * Math.PI, 6),
        ]);
    });

    it.each(methods)('(%s) leaves the integration bit-identical when events are not terminal', (method) => {
        const plain = rungeKuttaAdaptive(method, oscillator, 0, 10, new Vector([1, 0]));
        const withEvents = rungeKuttaAdaptive(method, oscillator, 0, 10, new Vector([1, 0]), {
            events: [{ fn: (_t, y) => y.data[0] }, { fn: (_t, y) => y.data[1] - 0.3 }],
        });

        expect(Array.from(withEvents.t.data)).toEqual(Array.from(plain.t.data));
        expect(Array.from(withEvents.y.data)).toEqual(Array.from(plain.y.data));
        expect(withEvents.evaluations).toBe(plain.evaluations);
        expect(withEvents.events!.length).toBeGreaterThan(0);
        expect(plain.events).toBeUndefined();
        expect('events' in plain).toBe(false);
    });

    it.each(methods)('(%s) does not count event function calls as derivative evaluations', (method) => {
        let fCalls = 0;
        const f: DerivativeFunction = (t, y, out) => (fCalls++, decay(t, y, out));
        const result = rungeKuttaAdaptive(method, f, 0, 2, new Vector([1]), { events: [{ fn: (_t, y) => y.data[0] - 0.5 }] });
        expect(result.evaluations).toBe(fCalls);
    });

    describe('terminal events', () => {
        it.each(methods)('(%s) stops at the event, which becomes the last recorded point', (method) => {
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] - 0.5, terminal: true }];
            const result = rungeKuttaAdaptive(method, decay, 0, 3, new Vector([1]), { ...tight, events });

            expect(result.success).toBe(true);
            expect(result.message).toBe('Integration stopped by a terminal event.');
            const e = result.events!.at(-1)!;
            expect(result.t.get(result.t.size - 1)).toBe(e.t);
            expect(e.t).toBeCloseTo(Math.LN2, 7);
            expect(Array.from(lastRow(result))).toEqual(Array.from(e.y.data));
            expect(result.t.size).toBe(result.y.rows);
            // times strictly increase up to and including the event
            for (let i = 1; i < result.t.size; i++) expect(result.t.get(i)).toBeGreaterThan(result.t.get(i - 1));
        });

        it.each(methods)('(%s) stops at the N-th occurrence', (method) => {
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], terminal: 3 }];
            const result = rungeKuttaAdaptive(method, oscillator, 0, 10 * Math.PI, new Vector([1, 0]), { ...tight, events });

            expect(result.events).toHaveLength(3);
            expect(result.events!.at(-1)!.t).toBeCloseTo(2.5 * Math.PI, 6);
            expect(result.t.get(result.t.size - 1)).toBe(result.events!.at(-1)!.t);
        });

        it.each(methods)('(%s) keeps earlier non-terminal events and drops later ones', (method) => {
            const events: OdeEvent[] = [
                { fn: (_t, y) => y.data[0] - 0.4 },
                { fn: (_t, y) => y.data[0] - 0.2, terminal: true },
                { fn: (_t, y) => y.data[0] - 0.1 },
            ];
            const result = rungeKuttaAdaptive(method, decay, 0, 5, new Vector([1]), { ...tight, events });
            expect(result.events!.map((e) => e.index)).toEqual([0, 1]);
        });

        it.each(methods)('(%s) supports restarting a bouncing ball from its impact point', (method) => {
            // Height y and velocity v under gravity; the impact is a falling zero crossing of y. The state
            // at the located root has y of the order of rounding error, of either sign: because the restarted
            // ball moves upward, the 'falling' filter keeps that residual from re-triggering the event.
            const g = 9.81;
            const ball: DerivativeFunction = (_t, y, out) => {
                out.data[0] = y.data[1];
                out.data[1] = -g;
                return out;
            };
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0], direction: 'falling', terminal: true }];
            const first = rungeKuttaAdaptive(method, ball, 0, 5, new Vector([1, 0]), { ...tight, events });
            const hit1 = first.events!.at(-1)!;
            expect(hit1.t).toBeCloseTo(Math.sqrt(2 / g), 6);

            const e = 0.5; // restitution
            const v1 = -hit1.y.data[1];
            const second = rungeKuttaAdaptive(method, ball, hit1.t, 5, new Vector([0, e * v1]), { ...tight, events });
            const hit2 = second.events!.at(-1)!;

            expect(hit2.t).toBeCloseTo(hit1.t + (2 * e * v1) / g, 6);
            expect(second.success).toBe(true);
        });
    });

    describe('boundaries', () => {
        it.each(methods)('(%s) integrates backward in time', (method) => {
            const events: OdeEvent[] = [{ fn: (_t, y) => y.data[0] - 2 }];
            // y = e^{-t}; from t = 0 back to t = -2 it grows through 2 at t = -ln 2
            const result = rungeKuttaAdaptive(method, decay, 0, -2, new Vector([1]), { ...tight, events });

            expect(result.events).toHaveLength(1);
            expect(result.events![0].t).toBeCloseTo(-Math.LN2, 7);
            expect(result.events![0].direction).toBe('rising');
        });

        it.each(methods)('(%s) reports an event at tEnd', (method) => {
            const f: DerivativeFunction = (_t, _y, out) => {
                out.data[0] = 1;
                return out;
            };
            const result = rungeKuttaAdaptive(method, f, 0, 1, new Vector([0]), { events: [{ fn: (_t, y) => y.data[0] - 1 }] });
            expect(result.events).toHaveLength(1);
            expect(result.events![0].t).toBeCloseTo(1, 12);
        });

        it.each(methods)('(%s) does not report an event function that is zero at t0', (method) => {
            const f: DerivativeFunction = (_t, _y, out) => {
                out.data[0] = 1;
                return out;
            };
            const result = rungeKuttaAdaptive(method, f, 0, 1, new Vector([0]), { events: [{ fn: (_t, y) => y.data[0] }] });
            expect(result.events).toEqual([]);
        });

        it('returns an empty event list when t0 === tEnd', () => {
            const result = rungeKuttaAdaptive('rk45', decay, 1, 1, new Vector([1]), { events: [{ fn: () => 0 }] });
            expect(result.events).toEqual([]);
            expect(rungeKuttaAdaptive('rk45', decay, 1, 1, new Vector([1])).events).toBeUndefined();
        });

        it('returns an empty event list when no crossing occurs', () => {
            const result = rungeKuttaAdaptive('rk45', decay, 0, 1, new Vector([1]), { events: [{ fn: (_t, y) => y.data[0] + 1 }] });
            expect(result.events).toEqual([]);
        });

        it('keeps events found before an unsuccessful termination', () => {
            const result = rungeKuttaAdaptive('rk45', oscillator, 0, 100, new Vector([1, 0]), {
                maxSteps: 12,
                events: [{ fn: (_t, y) => y.data[0] }],
            });
            expect(result.success).toBe(false);
            expect(result.events!.length).toBeGreaterThan(0);
        });
    });

    describe('input validation', () => {
        const run = (events: OdeEvent[]) => () => rungeKuttaAdaptive('rk45', decay, 0, 1, new Vector([1]), { events });

        it('rejects invalid events, also when t0 === tEnd', () => {
            expect(run([{} as OdeEvent])).toThrow(TypeError);
            expect(run([{ fn: () => 0, direction: 'up' as 'rising' }])).toThrow(RangeError);
            expect(run([{ fn: () => 0, terminal: 0 }])).toThrow(RangeError);
            expect(() => rungeKuttaAdaptive('rk45', decay, 1, 1, new Vector([1]), { events: [{} as OdeEvent] })).toThrow(TypeError);
        });
    });
});

describe('rungeKuttaAdaptiveDense', () => {
    // Logistic y' = 3 y (1 - y): exact y(t) = 1 / (1 + (1/y0 - 1) e^{-3t}) with y0 = 0.1.
    const f: DerivativeFunction = (_t, y, out) => {
        out.data[0] = 3 * y.data[0] * (1 - y.data[0]);
        return out;
    };
    const exact = (t: number): number => 1 / (1 + (1 / 0.1 - 1) * Math.exp(-3 * t));
    const makeScratch = (): AdaptiveScratch => ({
        k1: new Vector(1), k2: new Vector(1), k3: new Vector(1), k4: new Vector(1),
        k5: new Vector(1), k6: new Vector(1), k7: new Vector(1), yTemp: new Vector(1),
    });

    function maxError(method: 'rk23' | 'rk45', h: number): { error: number; endMismatch: number; startMismatch: number } {
        const t0 = 0.5;
        const y0 = new Vector([exact(t0)]);
        const scratch = makeScratch();
        const next = new Vector(1);
        const tmp = new Vector(1);
        rungeKuttaAdaptiveStep(method, f, t0, y0, h, next, scratch);
        let error = 0;
        for (let i = 0; i <= 50; i++) {
            const x = i / 50;
            rungeKuttaAdaptiveDense(method, y0, h, scratch, x, tmp);
            error = Math.max(error, Math.abs(tmp.data[0] - exact(t0 + x * h)));
        }
        rungeKuttaAdaptiveDense(method, y0, h, scratch, 0, tmp);
        const startMismatch = Math.abs(tmp.data[0] - y0.data[0]);
        rungeKuttaAdaptiveDense(method, y0, h, scratch, 1, tmp);
        return { error, endMismatch: Math.abs(tmp.data[0] - next.data[0]), startMismatch };
    }

    it.each([
        ['rk23', 3.5],
        ['rk45', 4.8],
    ] as const)('(%s) converges with the expected order and joins the step end points', (method, minOrder) => {
        const coarse = maxError(method, 0.2);
        const fine = maxError(method, 0.1);

        expect(Math.log2(coarse.error / fine.error)).toBeGreaterThan(minOrder);
        for (const r of [coarse, fine]) {
            expect(r.startMismatch).toBe(0);
            expect(r.endMismatch).toBeLessThan(1e-15);
        }
    });
});
