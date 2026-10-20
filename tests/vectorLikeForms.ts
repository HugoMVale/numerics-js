import { Vector } from '../src/linalg/Vector.js';
import type { VectorLike } from '../src/linalg/Vector.js';

/** Equivalent ways of passing the same numbers, as accepted by `VectorLike`. */
export const vectorLikeForms: [string, (a: number[]) => VectorLike][] = [
  ['number[]', (a) => a.slice()],
  ['readonly number[]', (a) => Object.freeze(a.slice())],
  ['Float64Array', (a) => new Float64Array(a)],
  ['Vector', (a) => Vector.from(a)]
];

/** The forms of {@link vectorLikeForms} whose contents can be changed (a frozen array cannot). */
export const mutableVectorLikeForms = vectorLikeForms.filter(
  ([name]) => name !== 'readonly number[]'
);

/** Overwrites the contents of a mutable `VectorLike` with zeros. */
export function clobber(a: VectorLike): void {
  if (a instanceof Vector) a.fill(0);
  else (a as number[] | Float64Array).fill(0);
}
