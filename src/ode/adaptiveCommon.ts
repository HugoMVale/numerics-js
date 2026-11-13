import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { OdeDenseOutput } from './denseOutput.js';
import type { OdeEvent, OdeMethod, OdeResult } from './types.js';

/**
 * Step-size control options shared by the adaptive ODE solvers, with defaults already applied
 * (except for `h0`, which is estimated when omitted).
 *
 * @internal
 */
export interface AdaptiveControlOptions {
  atol: number;
  rtol: number;
  h0: number | undefined;
  hMax: number;
  hMin: number;
  maxSteps: number;
  safety: number;
  minScale: number;
  maxScale: number;
}

/**
 * Validates the state dimension and the step-size control options shared by the adaptive ODE
 * solvers.
 *
 * @internal
 *
 * @param who Name of the calling solver, used as prefix of error messages.
 * @param dim Dimension of the initial state.
 * @param options Options to validate, with defaults applied.
 * @throws {RangeError} If the state has no components, the tolerances are invalid, or a
 * step-size or controller limit is invalid.
 */
export function validateAdaptiveOptions(
  who: string,
  dim: number,
  options: AdaptiveControlOptions
): void {
  const { atol, rtol, h0, hMax, hMin, maxSteps, safety, minScale, maxScale } = options;
  if (dim <= 0) {
    throw new RangeError(`${who}: y0 must have at least one component, got dimension ${dim}.`);
  }
  if (!Number.isFinite(atol) || !Number.isFinite(rtol) || atol < 0 || rtol < 0) {
    throw new RangeError(
      `${who}: atol (${atol}) and rtol (${rtol}) must be finite and non-negative.`
    );
  }
  if (atol === 0 && rtol === 0) {
    throw new RangeError(
      `${who}: atol and rtol cannot both be 0; no step could ever satisfy the tolerance.`
    );
  }
  if (h0 !== undefined && h0 === 0) throw new RangeError(`${who}: h0 must be nonzero.`);
  if (!(hMin > 0)) throw new RangeError(`${who}: hMin (${hMin}) must be positive.`);
  if (!(hMax > 0)) throw new RangeError(`${who}: hMax (${hMax}) must be positive.`);
  if (hMin > hMax) throw new RangeError(`${who}: hMin (${hMin}) must not exceed hMax (${hMax}).`);
  if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
    throw new RangeError(`${who}: maxSteps (${maxSteps}) must be a positive integer.`);
  }
  if (!(safety > 0 && safety <= 1)) {
    throw new RangeError(`${who}: safety (${safety}) must be in (0, 1].`);
  }
  if (!(minScale > 0) || minScale > maxScale) {
    throw new RangeError(
      `${who}: minScale (${minScale}) must be positive and not exceed maxScale (${maxScale}).`
    );
  }
}

/**
 * Accumulates the accepted time points and states of an adaptive integration, and assembles the
 * {@link OdeResult}.
 *
 * @internal
 */
export class SolutionRecorder {
  private readonly dim: number;
  private readonly tBuf: number[];
  private readonly yBuf: Float64Array[];

  /**
   * @param t0 Initial time.
   * @param y0 Initial state, recorded as the first point. It is copied.
   */
  constructor(t0: number, y0: Vector) {
    this.dim = y0.size;
    this.tBuf = [t0];
    this.yBuf = [y0.data.slice()];
  }

  /**
   * Records a point.
   *
   * @param t Time of the point.
   * @param y State at `t`. It is copied.
   */
  push(t: number, y: Vector): void {
    this.tBuf.push(t);
    this.yBuf.push(y.data.slice());
  }

  /**
   * Builds the result from the recorded points.
   *
   * @param info Result fields other than `t` and `y`.
   */
  result(info: Omit<OdeResult, 't' | 'y'>): OdeResult {
    const n = this.tBuf.length;
    const t = new Vector(n);
    t.set(this.tBuf);
    const y = new Matrix(n, this.dim);
    for (let i = 0; i < n; i++) y.setRow(i, this.yBuf[i]);
    return { ...info, t, y };
  }
}

/**
 * Result of an adaptive integration over an empty interval (`tEnd === t0`): a single point, the
 * initial state, with no derivative evaluations.
 *
 * @internal
 *
 * @param method Solver method.
 * @param t0 Initial (and final) time.
 * @param y0 Initial state. It is copied into the result.
 * @param events The `events` option; an empty `events` array is attached if it is given.
 * @param denseOutput Whether the `denseOutput` option is enabled; an empty dense output is
 * attached if it is.
 */
export function emptyIntervalResult(
  method: OdeMethod,
  t0: number,
  y0: Vector,
  events: readonly OdeEvent[] | undefined,
  denseOutput: boolean
): OdeResult {
  const result = new SolutionRecorder(t0, y0).result({
    method,
    success: true,
    message: 'Integration successful.',
    evaluations: 0,
    evaluationsJacobian: 0,
    decompositions: 0
  });
  if (events !== undefined) result.events = [];
  if (denseOutput) result.denseOutput = new OdeDenseOutput(t0, y0);
  return result;
}
