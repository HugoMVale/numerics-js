import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { version } from '../src/index.js';

describe('version', () => {
  it('matches package.json', () => {
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as {
      version: string;
    };
    expect(version).toBe(pkg.version);
  });
});
