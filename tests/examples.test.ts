import { describe, expect, it } from 'vitest';
import { collectExamples } from '../scripts/lib/examples.mjs';

/**
 * Every `@example` in `src/` must type-check as a self-contained program
 * against the library sources. The same extraction feeds the playground
 * (`npm run docs:playground`), so an example that fails here has no working
 * "Run" button either. Declarations tagged `@internal` are not extracted.
 */
const examples = await collectExamples();

const label = ({ file, code }: { file: string; code: string }) => {
    const first = code.split('\n').find((line) => line.trim() !== '') ?? '';
    return `${file}: ${first.trim().slice(0, 70)}`;
};

describe('@example blocks', () => {
    it('are found in the sources', () => {
        expect(examples.length).toBeGreaterThan(0);
    });

    for (const example of examples) {
        it(label(example), () => {
            expect(example.diagnostics).toEqual([]);
        });
    }
});
