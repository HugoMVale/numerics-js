import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import {
  emptyIntervalResult,
  SolutionRecorder,
  validateAdaptiveOptions,
  type AdaptiveControlOptions
} from '../../src/ode/adaptiveCommon.js';
import { OdeDenseOutput } from '../../src/ode/denseOutput.js';

const valid: AdaptiveControlOptions = {
  atol: 1e-6,
  rtol: 1e-3,
  h0: undefined,
  hMax: Infinity,
  hMin: 1e-12,
  maxSteps: 1e5,
  safety: 0.9,
  minScale: 0.2,
  maxScale: 10
};

describe('validateAdaptiveOptions', () => {
  it('accepts valid options, with or without h0', () => {
    expect(() => validateAdaptiveOptions('solver', 2, valid)).not.toThrow();
    expect(() => validateAdaptiveOptions('solver', 2, { ...valid, h0: -0.1 })).not.toThrow();
    expect(() => validateAdaptiveOptions('solver', 1, { ...valid, atol: 0 })).not.toThrow();
  });

  it.each<[string, number, Partial<AdaptiveControlOptions>, RegExp]>([
    ['empty state', 0, {}, /^solver: y0 must have at least one component, got dimension 0\.$/],
    ['negative atol', 1, { atol: -1 }, /^solver: atol \(-1\) and rtol \(0.001\) must be finite/],
    ['infinite rtol', 1, { rtol: Infinity }, /must be finite and non-negative/],
    ['NaN atol', 1, { atol: NaN }, /must be finite and non-negative/],
    ['both tolerances zero', 1, { atol: 0, rtol: 0 }, /cannot both be 0/],
    ['zero h0', 1, { h0: 0 }, /^solver: h0 must be nonzero\.$/],
    ['zero hMin', 1, { hMin: 0 }, /^solver: hMin \(0\) must be positive\.$/],
    ['NaN hMax', 1, { hMax: NaN }, /^solver: hMax \(NaN\) must be positive\.$/],
    ['hMin above hMax', 1, { hMin: 1, hMax: 0.5 }, /hMin \(1\) must not exceed hMax \(0.5\)/],
    ['fractional maxSteps', 1, { maxSteps: 1.5 }, /maxSteps \(1.5\) must be a positive integer/],
    ['zero maxSteps', 1, { maxSteps: 0 }, /must be a positive integer/],
    ['safety above 1', 1, { safety: 1.1 }, /safety \(1.1\) must be in \(0, 1\]/],
    ['zero safety', 1, { safety: 0 }, /must be in \(0, 1\]/],
    ['zero minScale', 1, { minScale: 0 }, /minScale \(0\) must be positive/],
    ['minScale above maxScale', 1, { minScale: 2, maxScale: 1 }, /not exceed maxScale \(1\)/]
  ])('rejects %s', (_label, dim, override, message) => {
    expect(() => validateAdaptiveOptions('solver', dim, { ...valid, ...override })).toThrow(
      RangeError
    );
    expect(() => validateAdaptiveOptions('solver', dim, { ...valid, ...override })).toThrow(
      message
    );
  });
});

describe('SolutionRecorder', () => {
  it('records copies of the points, starting with the initial state', () => {
    const y = new Vector([1, 2]);
    const recorder = new SolutionRecorder(0, y);
    y.set([3, 4]);
    recorder.push(0.5, y);
    y.set([5, 6]);
    recorder.push(1, y);

    const result = recorder.result({
      method: 'rk45',
      success: true,
      message: 'ok',
      evaluations: 7,
      evaluationsJacobian: 0,
      decompositions: 0
    });

    expect(Array.from(result.t.data)).toEqual([0, 0.5, 1]);
    expect(result.y.rows).toBe(3);
    expect(result.y.cols).toBe(2);
    expect(Array.from(result.y.data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(result.evaluations).toBe(7);
    expect(Object.keys(result)).toEqual([
      'method',
      'success',
      'message',
      'evaluations',
      'evaluationsJacobian',
      'decompositions',
      't',
      'y'
    ]);
  });
});

describe('emptyIntervalResult', () => {
  it('returns the initial state as the only point, without evaluations', () => {
    const y0 = new Vector([1, 2]);
    const result = emptyIntervalResult('trbdf2', 3, y0, undefined, false);

    expect(result.method).toBe('trbdf2');
    expect(result.success).toBe(true);
    expect(result.evaluations).toBe(0);
    expect(Array.from(result.t.data)).toEqual([3]);
    expect(Array.from(result.y.data)).toEqual([1, 2]);
    expect(result.y.data).not.toBe(y0.data);
    expect(result.events).toBeUndefined();
    expect(result.denseOutput).toBeUndefined();
  });

  it('attaches empty events and a dense output only when requested', () => {
    const result = emptyIntervalResult('rk23', 0, new Vector([1]), [], true);

    expect(result.events).toEqual([]);
    expect(result.denseOutput).toBeInstanceOf(OdeDenseOutput);
    expect(Array.from(result.denseOutput!.eval(0).data)).toEqual([1]);
  });
});
