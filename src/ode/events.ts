import { Vector } from '../linalg/Vector.js';
import { brent } from '../roots/brent.js';
import type { OdeEvent, OdeEventOccurrence } from './types.js';

/**
 * Dense output of an accepted step: writes the interpolated state at time `t`, inside the step,
 * into `out` and returns it. It must reproduce the step's starting state exactly at the starting
 * time and agree with the step's end state at the end time up to rounding.
 */
export type StepInterpolant = (t: number, out: Vector) => Vector;

/**
 * Validates the `events` option shared by the adaptive ODE solvers.
 *
 * @param events Events to validate.
 * @param who Name of the calling solver, used as prefix of error messages.
 * @throws {TypeError} If `events` is not an array or an event function is not a function.
 * @throws {RangeError} If `direction` or `terminal` of an event is invalid.
 */
export function validateEvents(events: readonly OdeEvent[], who: string): void {
  if (!Array.isArray(events)) throw new TypeError(`${who}: events must be an array.`);
  events.forEach((event, i) => {
    if (typeof event?.fn !== 'function')
      throw new TypeError(`${who}: events[${i}].fn must be a function.`);
    const { direction = 'both', terminal = false } = event;
    if (direction !== 'rising' && direction !== 'falling' && direction !== 'both') {
      throw new RangeError(
        `${who}: events[${i}].direction must be 'rising', 'falling' or 'both', got ${String(direction)}.`
      );
    }
    if (typeof terminal !== 'boolean' && !(Number.isInteger(terminal) && terminal > 0)) {
      throw new RangeError(
        `${who}: events[${i}].terminal must be a boolean or a positive integer, got ${String(terminal)}.`
      );
    }
  });
}

/**
 * Detects and locates zero crossings of event functions along an adaptive ODE integration.
 *
 * After every accepted step, {@link EventDetector.check} evaluates each event function at the new
 * point. A sign change relative to the previous accepted point triggers a root search of
 * `g(t, y(t))` inside the step, using the step's dense output, so no further derivative
 * evaluations are needed. Crossings are detected with a half-open test (`g` goes from `< 0` to
 * `>= 0`, or from `> 0` to `<= 0`): an event is never reported twice for a root that falls exactly
 * on a step boundary, and a function that is exactly zero at the start is not reported. A located
 * event point generally has a residual `g` of the order of rounding error, of either sign, so a
 * restart from it can re-trigger a crossing in the same direction; restarts that reverse the
 * motion (e.g. a bouncing ball) are protected by the `direction` filter.
 *
 * Only sign changes between accepted points are detected; two crossings inside one step cancel
 * and go unnoticed, so `hMax` should be used to bound the step if that matters.
 */
export class EventDetector {
  /** Located occurrences so far, in chronological order. */
  readonly occurrences: OdeEventOccurrence[] = [];

  private readonly events: readonly OdeEvent[];
  private readonly dim: number;
  private readonly rising: boolean[];
  private readonly falling: boolean[];
  private readonly limit: number[];
  private readonly count: number[];
  private readonly gOld: Float64Array;
  private readonly gNew: Float64Array;
  private readonly work: Vector;
  private tOld: number;

  /**
   * @param events Validated events; see {@link validateEvents}.
   * @param t0 Initial time.
   * @param y0 Initial state, at which the event functions are first evaluated. Not mutated.
   */
  constructor(events: readonly OdeEvent[], t0: number, y0: Vector) {
    const n = events.length;
    this.events = events;
    this.dim = y0.size;
    this.rising = events.map((e) => (e.direction ?? 'both') !== 'falling');
    this.falling = events.map((e) => (e.direction ?? 'both') !== 'rising');
    this.limit = events.map((e) =>
      e.terminal === undefined || e.terminal === false
        ? Infinity
        : e.terminal === true
          ? 1
          : e.terminal
    );
    this.count = new Array<number>(n).fill(0);
    this.gOld = new Float64Array(n);
    this.gNew = new Float64Array(n);
    this.work = new Vector(this.dim);
    this.tOld = t0;
    for (let i = 0; i < n; i++) this.gOld[i] = events[i].fn(t0, y0);
  }

  /**
   * Processes an accepted step from the previous accepted time to `tNew`.
   *
   * Must be called once per accepted step, in order, and before the solver discards the data
   * that `dense` reads. Newly located occurrences are appended to {@link occurrences}.
   *
   * @param tNew End time of the accepted step.
   * @param yNew State at `tNew`. Not mutated.
   * @param dense Dense output of the step; see {@link StepInterpolant}. Called only if an event
   * function changes sign.
   * @returns The occurrence that terminates the integration (also the last entry of
   * {@link occurrences}), or `null` if the integration continues.
   */
  check(tNew: number, yNew: Vector, dense: StepInterpolant): OdeEventOccurrence | null {
    const { events, gOld, gNew, tOld, work } = this;
    const n = events.length;
    const forward = tNew > tOld;
    const found: OdeEventOccurrence[] = [];
    const tolx = 4 * Number.EPSILON * Math.max(Math.abs(tOld), Math.abs(tNew));

    for (let i = 0; i < n; i++) {
      const g1 = events[i].fn(tNew, yNew);
      gNew[i] = g1;
      const g0 = gOld[i];
      const up = g0 < 0 && g1 >= 0;
      const down = g0 > 0 && g1 <= 0;
      if (!((up && this.rising[i]) || (down && this.falling[i]))) continue;

      const fn = events[i].fn;
      const root = brent((t) => fn(t, dense(t, work)), tOld, tNew, {
        tolx,
        tolf: 0,
        maxIter: 100,
        fa: g0,
        fb: g1
      });
      // `success` is deliberately ignored: on a flat root the iteration cap is reached,
      //  but the best estimate is still a point of the bracket very close to the root.
      const t = root.x;
      const y = new Vector(this.dim);
      y.set((t === tNew ? yNew : dense(t, work)).data);
      this.count[i]++;
      found.push({ index: i, t, y, direction: up ? 'rising' : 'falling' });
    }

    if (found.length > 1) {
      const sign = forward ? 1 : -1;
      found.sort((a, b) => sign * (a.t - b.t) || a.index - b.index);
    }

    let terminal: OdeEventOccurrence | null = null;
    let keep = found.length;
    for (let k = 0; k < found.length; k++) {
      if (this.count[found[k].index] >= this.limit[found[k].index]) {
        terminal = found[k];
        keep = k + 1;
        break;
      }
    }
    for (let k = 0; k < keep; k++) this.occurrences.push(found[k]);

    gOld.set(gNew);
    this.tOld = tNew;
    return terminal;
  }
}
