import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { EventDetector, validateEvents } from '../../src/ode/events.js';
import type { StepInterpolant } from '../../src/ode/events.js';
import type { OdeEvent } from '../../src/ode/types.js';

// Synthetic solution y(t) = t, whose dense output is exact: the detector can be tested in isolation.
const dense: StepInterpolant = (t, out) => {
  out.data[0] = t;
  return out;
};
const at = (t: number): Vector => new Vector([t]);

/** Feeds the detector the accepted points `ts[1..]`, starting from `ts[0]`; stops at a terminal event. */
function run(events: OdeEvent[], ts: number[], interpolant: StepInterpolant = dense) {
  const detector = new EventDetector(events, ts[0], at(ts[0]));
  let terminal = null;
  for (let i = 1; i < ts.length && terminal === null; i++) {
    terminal = detector.check(ts[i], at(ts[i]), interpolant);
  }
  return { detector, terminal };
}

describe('EventDetector', () => {
  it('locates a rising crossing inside a step', () => {
    const { detector } = run([{ fn: (_t, y) => y.data[0] - 0.5 }], [0, 0.4, 0.8, 1.2]);
    expect(detector.occurrences).toHaveLength(1);
    const [e] = detector.occurrences;
    expect(e.index).toBe(0);
    expect(e.direction).toBe('rising');
    expect(e.t).toBeCloseTo(0.5, 14);
    expect(e.y.data[0]).toBeCloseTo(0.5, 14);
  });

  it('honours the direction filter', () => {
    const rising = { fn: (_t: number, y: Vector) => y.data[0] - 0.5 };
    const falling = { fn: (_t: number, y: Vector) => 0.5 - y.data[0] };
    const ts = [0, 0.4, 0.8];

    expect(run([{ ...rising, direction: 'rising' }], ts).detector.occurrences).toHaveLength(1);
    expect(run([{ ...rising, direction: 'falling' }], ts).detector.occurrences).toHaveLength(0);
    expect(run([{ ...falling, direction: 'rising' }], ts).detector.occurrences).toHaveLength(0);
    const both = run([{ ...falling, direction: 'both' }], ts).detector.occurrences;
    expect(both).toHaveLength(1);
    expect(both[0].direction).toBe('falling');
  });

  it('reports a root that falls exactly on a step boundary once', () => {
    const { detector } = run([{ fn: (_t, y) => y.data[0] - 0.5 }], [0, 0.25, 0.5, 0.75, 1]);
    expect(detector.occurrences).toHaveLength(1);
    expect(detector.occurrences[0].t).toBe(0.5);
    expect(detector.occurrences[0].y.data[0]).toBe(0.5);
  });

  it('does not report an event function that is exactly zero at the start', () => {
    const { detector } = run([{ fn: (_t, y) => y.data[0] }], [0, 0.5, 1]);
    expect(detector.occurrences).toHaveLength(0);
  });

  it('does not call the dense output when no event function changes sign', () => {
    let calls = 0;
    const counting: StepInterpolant = (t, out) => {
      calls++;
      return dense(t, out);
    };
    run([{ fn: (_t, y) => y.data[0] + 1 }], [0, 0.5, 1], counting);
    expect(calls).toBe(0);
  });

  it('evaluates each event function once per accepted step when nothing is found', () => {
    let calls = 0;
    run([{ fn: (_t, y) => (calls++, y.data[0] + 1) }], [0, 0.5, 1, 1.5]);
    expect(calls).toBe(1 + 3); // t0 plus one per step
  });

  it('orders several events of one step by time and keeps the event index', () => {
    const events: OdeEvent[] = [
      { fn: (_t, y) => y.data[0] - 0.8 },
      { fn: (_t, y) => y.data[0] - 0.2 },
      { fn: (_t, y) => y.data[0] - 0.5 }
    ];
    const { detector } = run(events, [0, 1]);
    expect(detector.occurrences.map((e) => e.index)).toEqual([1, 2, 0]);
    expect(detector.occurrences.map((e) => e.t)).toEqual([
      expect.closeTo(0.2, 14),
      expect.closeTo(0.5, 14),
      expect.closeTo(0.8, 14)
    ]);
  });

  it('orders events backward in time when the integration runs backward', () => {
    const events: OdeEvent[] = [
      { fn: (_t, y) => y.data[0] - 0.8 },
      { fn: (_t, y) => y.data[0] - 0.2 }
    ];
    const { detector } = run(events, [1, 0]);
    expect(detector.occurrences.map((e) => e.index)).toEqual([0, 1]);
    // y = t decreases along the integration, so g = y - c goes from positive to negative.
    expect(detector.occurrences.every((e) => e.direction === 'falling')).toBe(true);
  });

  it('applies direction relative to the integration direction', () => {
    const fn = (_t: number, y: Vector): number => y.data[0] - 0.5;
    expect(run([{ fn, direction: 'falling' }], [1, 0.6, 0.2]).detector.occurrences).toHaveLength(1);
    expect(run([{ fn, direction: 'rising' }], [1, 0.6, 0.2]).detector.occurrences).toHaveLength(0);
  });

  it('stops at the first terminal event and drops later occurrences of the step', () => {
    const events: OdeEvent[] = [
      { fn: (_t, y) => y.data[0] - 0.3 },
      { fn: (_t, y) => y.data[0] - 0.6, terminal: true },
      { fn: (_t, y) => y.data[0] - 0.9 }
    ];
    const { detector, terminal } = run(events, [0, 1, 2]);
    expect(terminal).not.toBeNull();
    expect(terminal?.index).toBe(1);
    expect(detector.occurrences.map((e) => e.index)).toEqual([0, 1]);
    expect(detector.occurrences[1]).toBe(terminal);
  });

  it('counts occurrences for a terminal event given as an integer', () => {
    // g crosses zero at t = 0.5, 1.5, 2.5, ... (rising and falling)
    const events: OdeEvent[] = [{ fn: (t) => Math.sin(Math.PI * (t - 0.5)), terminal: 3 }];
    const { detector, terminal } = run(events, [0, 0.8, 1.6, 2.4, 3.2, 4]);
    expect(terminal?.t).toBeCloseTo(2.5, 12);
    expect(detector.occurrences.map((e) => e.t)).toEqual([
      expect.closeTo(0.5, 12),
      expect.closeTo(1.5, 12),
      expect.closeTo(2.5, 12)
    ]);
  });

  it('counts only crossings accepted by the direction filter towards `terminal`', () => {
    const events: OdeEvent[] = [
      { fn: (t) => Math.sin(Math.PI * (t - 0.5)), direction: 'rising', terminal: 2 }
    ];
    const { terminal } = run(events, [0, 0.8, 1.6, 2.4, 3.2, 4]);
    expect(terminal?.t).toBeCloseTo(2.5, 12); // rising at 0.5, 2.5
  });

  it('returns an event on a flat root without throwing', () => {
    const { detector } = run([{ fn: (t) => (t - 0.3) ** 3 }], [0, 1]);
    expect(detector.occurrences).toHaveLength(1);
    expect(detector.occurrences[0].t).toBeCloseTo(0.3, 8);
  });

  it('gives each occurrence its own state buffer', () => {
    const events: OdeEvent[] = [
      { fn: (_t, y) => y.data[0] - 0.2 },
      { fn: (_t, y) => y.data[0] - 0.7 }
    ];
    const { detector } = run(events, [0, 1]);
    const [first, second] = detector.occurrences;
    expect(first.y.data).not.toBe(second.y.data);
    expect(first.y.data[0]).toBeCloseTo(0.2, 14);
    expect(second.y.data[0]).toBeCloseTo(0.7, 14);
  });

  it('ignores non-finite event values (no crossing is detected)', () => {
    const { detector } = run([{ fn: (t) => (t > 0.4 && t < 0.6 ? NaN : t - 0.8) }], [0, 0.5, 1]);
    expect(detector.occurrences).toHaveLength(0);
  });

  it('accepts an empty event list', () => {
    const { detector, terminal } = run([], [0, 1]);
    expect(detector.occurrences).toEqual([]);
    expect(terminal).toBeNull();
  });
});

describe('validateEvents', () => {
  const fn = (): number => 0;

  it('accepts valid events', () => {
    expect(() =>
      validateEvents(
        [
          { fn },
          { fn, direction: 'rising', terminal: true },
          { fn, direction: 'falling', terminal: 3 }
        ],
        'x'
      )
    ).not.toThrow();
  });

  it('rejects a non-array', () => {
    expect(() => validateEvents({} as unknown as OdeEvent[], 'x')).toThrow(TypeError);
  });

  it('rejects an event without a function', () => {
    expect(() => validateEvents([{} as OdeEvent], 'x')).toThrow(
      /x: events\[0\]\.fn must be a function/
    );
    expect(() => validateEvents([null as unknown as OdeEvent], 'x')).toThrow(TypeError);
  });

  it('rejects an invalid direction', () => {
    expect(() => validateEvents([{ fn, direction: 'up' as 'rising' }], 'x')).toThrow(RangeError);
  });

  it.each([0, -1, 1.5, NaN, Infinity, '2' as unknown as number])(
    'rejects terminal = %s',
    (terminal) => {
      expect(() => validateEvents([{ fn, terminal }], 'x')).toThrow(RangeError);
    }
  );
});
