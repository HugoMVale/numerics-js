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

const validate = (options: AdaptiveControlOptions, y0: number[] = [1], t0 = 0, tEnd = 1): void =>
  validateAdaptiveOptions('solver', t0, tEnd, new Vector(y0), options);

describe('validateAdaptiveOptions', () => {
  it('accepts valid options, with or without h0, and either integration direction', () => {
    expect(() => validate(valid, [1, 2])).not.toThrow();
    expect(() => validate({ ...valid, h0: -0.1 }, [1, 2])).not.toThrow();
    expect(() => validate({ ...valid, atol: 0 })).not.toThrow();
    expect(() => validate(valid, [1], 1, 0)).not.toThrow();
    expect(() => validate(valid, [1], 2, 2)).not.toThrow();
  });

  it.each<[string, number[], number, number, RegExp]>([
    ['NaN t0', [1], NaN, 1, /^solver: t0 \(NaN\) and tEnd \(1\) must be finite\.$/],
    ['NaN tEnd', [1], 0, NaN, /^solver: t0 \(0\) and tEnd \(NaN\) must be finite\.$/],
    ['infinite tEnd', [1], 0, Infinity, /tEnd \(Infinity\) must be finite/],
    ['infinite t0', [1], -Infinity, 0, /t0 \(-Infinity\) and tEnd \(0\) must be finite/],
    ['NaN t0 and tEnd', [1], NaN, NaN, /must be finite/],
    ['NaN in y0', [1, NaN], 0, 1, /^solver: y0 must be finite, got y0\[1\] = NaN\.$/],
    ['infinite y0', [-Infinity], 0, 1, /y0\[0\] = -Infinity/]
  ])('rejects %s', (_label, y0, t0, tEnd, message) => {
    expect(() => validate(valid, y0, t0, tEnd)).toThrow(RangeError);
    expect(() => validate(valid, y0, t0, tEnd)).toThrow(message);
  });

  it.each<[string, number, Partial<AdaptiveControlOptions>, RegExp]>([
    ['empty state', 0, {}, /^solver: y0 must have at least one component, got dimension 0\.$/],
    ['negative atol', 1, { atol: -1 }, /^solver: atol \(-1\) and rtol \(0.001\) must be finite/],
    ['infinite rtol', 1, { rtol: Infinity }, /must be finite and non-negative/],
    ['NaN atol', 1, { atol: NaN }, /must be finite and non-negative/],
    ['both tolerances zero', 1, { atol: 0, rtol: 0 }, /cannot both be 0/],
    ['zero h0', 1, { h0: 0 }, /^solver: h0 \(0\) must be finite and nonzero\.$/],
    ['NaN h0', 1, { h0: NaN }, /^solver: h0 \(NaN\) must be finite and nonzero\.$/],
    ['infinite h0', 1, { h0: -Infinity }, /h0 \(-Infinity\) must be finite/],
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
    const y0 = new Array<number>(dim).fill(1);
    expect(() => validate({ ...valid, ...override }, y0)).toThrow(RangeError);
    expect(() => validate({ ...valid, ...override }, y0)).toThrow(message);
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
