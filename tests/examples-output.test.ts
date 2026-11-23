import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { format } from 'node:util';
import ts from 'typescript';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { extractExamples } from '../scripts/lib/examples.mjs';

/**
 * Every `@example` with an "Output:" block is executed against the library sources and the
 * text it prints with `console.log` is compared with that block. `tests/examples.test.ts`
 * only type-checks the examples; this test is what keeps the documented numbers honest.
 *
 * The comparison is exact except for numbers: a number may differ from the documented one by
 * at most `1e-9 * max(1, |documented|)`. V8's `Math.*` and `**` differ in the last bits
 * between Node majors (22 vs 24+), and an example that prints a cancellation residual or an
 * error estimate would otherwise fail depending on the runtime. Everything else (integer
 * counts, strings, object layout, array and matrix structure) must match character for
 * character, so a changed iteration count or a stale message is still caught. The documented
 * outputs are the ones produced by Node >= 24.
 */
const NUMBER = /[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/gi;
const TOLERANCE = 1e-9;

/** Why `actual` does not match `expected`, or `null` if it does (see the file comment). */
function difference(expected: string, actual: string): string | null {
  if (expected === actual) return null;
  const skeleton = (text: string) => text.replace(NUMBER, '#');
  if (skeleton(expected) !== skeleton(actual)) return 'the text around the numbers differs';
  const e = expected.match(NUMBER) ?? [];
  const a = actual.match(NUMBER) ?? [];
  for (let i = 0; i < e.length; i++) {
    const want = Number(e[i]);
    const got = Number(a[i]);
    if (!(Math.abs(got - want) <= TOLERANCE * Math.max(1, Math.abs(want)))) {
      return `number #${i + 1} is ${a[i]}, documented ${e[i]}`;
    }
  }
  return null;
}

const examples = (await extractExamples()).filter(
  (example): example is typeof example & { output: string } => example.output !== null
);

const here = dirname(fileURLToPath(import.meta.url));
const work = join(here, '.examples-run');

/** The example as a plain module whose `numerics-js` imports point at the sources. */
function toModule(program: string): string {
  const js = ts.transpileModule(program, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext }
  }).outputText;
  return js.replace(/from\s+(['"])numerics-js(?:\/([^'"]+))?\1/g, (_, quote, sub) => {
    return `from ${quote}../../src/${sub ?? 'index'}.ts${quote}`;
  });
}

const label = ({ file, code }: { file: string; code: string }) => {
  const first = code.split('\n').find((line) => line.trim() !== '' && !line.startsWith('import'));
  return `${file}: ${(first ?? '').trim().slice(0, 70)}`;
};

describe('@example outputs', () => {
  beforeAll(() => {
    rmSync(work, { recursive: true, force: true });
    mkdirSync(work, { recursive: true });
  });
  afterAll(() => {
    rmSync(work, { recursive: true, force: true });
  });

  it('are found in the sources', () => {
    expect(examples.length).toBeGreaterThan(100);
  });

  it('compare numbers with a tolerance and everything else exactly', () => {
    expect(difference('{ x: 1 }', '{ x: 1 }')).toBeNull();
    expect(difference('{ x: 0.1, n: 12 }', '{ x: 0.1000000000001, n: 12 }')).toBeNull();
    expect(difference('{ n: 12 }', '{ n: 13 }')).not.toBeNull();
    expect(difference('{ x: 1 }', '{ x: 1.001 }')).not.toBeNull();
    expect(difference("{ message: 'a' }", "{ message: 'b' }")).not.toBeNull();
    expect(difference('{ x: 1 }', '{\n  x: 1\n}')).not.toBeNull();
    expect(difference('[1e-20]', '[3e-20]')).toBeNull();
  });

  examples.forEach((example, index) => {
    it(label(example), async () => {
      const path = join(work, `example${index}.ts`);
      writeFileSync(path, toModule(example.program));
      const lines: string[] = [];
      const spy = vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
        lines.push(format(...args));
      });
      try {
        await import(/* @vite-ignore */ path);
      } finally {
        spy.mockRestore();
      }
      const actual = lines.join('\n');
      const why = difference(example.output, actual);
      expect(
        why === null
          ? 'matches'
          : `${why}\n--- documented\n${example.output}\n--- actual\n${actual}`
      ).toBe('matches');
    });
  });
});
