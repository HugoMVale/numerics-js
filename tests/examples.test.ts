import { beforeAll, describe, expect, it } from 'vitest';
import { createChecker, extractExamples } from '../scripts/lib/examples.mjs';

/**
 * Every `@example` in `src/` must type-check as a self-contained program
 * against the library sources. The same extraction feeds the playground
 * (`npm run docs:playground`), so an example that fails here has no working
 * "Run" button either. Declarations tagged `@internal` are not extracted.
 *
 * Extraction is fast and happens up front; the type-check runs inside each test
 * (one shared checker caches the parsed library files), so progress is reported
 * per example and a failure points at the offending example.
 */
const examples = await extractExamples();
const check = createChecker();

const label = ({ file, code }: { file: string; code: string }) => {
  const first = code.split('\n').find((line) => line.trim() !== '') ?? '';
  return `${file}: ${first.trim().slice(0, 70)}`;
};

describe('@example blocks', () => {
  // Pay the one-off cost of parsing `lib.*.d.ts` and `src/` here, not inside the first
  // example's own (5 s default) timeout.
  beforeAll(() => {
    check("import 'numerics-js';");
  }, 120_000);

  it('are found in the sources', () => {
    expect(examples.length).toBeGreaterThan(0);
  });

  for (const example of examples) {
    it(label(example), () => {
      expect(check(example.program).diagnostics).toEqual([]);
    });
  }
});
