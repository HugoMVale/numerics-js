import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import fixtures from './fixtures/Matrix.eig.scipy.json' with { type: 'json' };

function sortIndices(values: Array<{ re: number; im: number }>): number[] {
    return values.map((_, index) => index).sort((a, b) => values[a].re - values[b].re || values[a].im - values[b].im);
}

function complexOverlap(
    actualRe: number[],
    actualIm: number[],
    referenceRe: number[],
    referenceIm: number[]
): number {
    let innerRe = 0;
    let innerIm = 0;
    for (let i = 0; i < actualRe.length; i++) {
        innerRe += referenceRe[i] * actualRe[i] + referenceIm[i] * actualIm[i];
        innerIm += referenceRe[i] * actualIm[i] - referenceIm[i] * actualRe[i];
    }
    return Math.hypot(innerRe, innerIm);
}

describe('Matrix.eig() vs scipy.linalg.eig reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
            const matrix = Matrix.from(fixture.matrix);
            const actual = matrix.eig();
            const actualValues = actual.map((pair) => pair.value);
            const actualOrder = sortIndices(actualValues);
            const referenceOrder = sortIndices(fixture.scipyValues);

            expect(actual).toHaveLength(fixture.scipyValues.length);
            for (let position = 0; position < referenceOrder.length; position++) {
                const actualPair = actual[actualOrder[position]];
                const referenceIndex = referenceOrder[position];
                const referenceValue = fixture.scipyValues[referenceIndex];
                const scale = Math.max(1, Math.abs(referenceValue.re), Math.abs(referenceValue.im));

                expect(Math.abs(actualPair.value.re - referenceValue.re)).toBeLessThanOrEqual(1e-9 * scale);
                expect(Math.abs(actualPair.value.im - referenceValue.im)).toBeLessThanOrEqual(1e-9 * scale);

                const actualRe = actualPair.vectorRe.toArray();
                const actualIm = actualPair.vectorIm.toArray();
                const overlap = complexOverlap(
                    actualRe,
                    actualIm,
                    fixture.scipyVectorsRe[referenceIndex],
                    fixture.scipyVectorsIm[referenceIndex]
                );
                expect(overlap).toBeGreaterThanOrEqual(1 - 1e-8);
            }
        });
    }
});