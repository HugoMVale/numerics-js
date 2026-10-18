// Copies the version from package.json into src/index.ts.
// npm runs this automatically through the `version` script during `npm version`.
import { readFileSync, writeFileSync } from 'node:fs';

const packageUrl = new URL('../package.json', import.meta.url);
const indexUrl = new URL('../src/index.ts', import.meta.url);

const { version } = JSON.parse(readFileSync(packageUrl, 'utf8'));
const source = readFileSync(indexUrl, 'utf8');

const pattern = /^export const version = '[^']*';$/gm;
const matches = source.match(pattern) ?? [];

if (matches.length !== 1) {
  console.error(
    `sync-version: expected exactly one "export const version" line in src/index.ts, found ${matches.length}`
  );
  process.exit(1);
}

const updated = source.replace(pattern, `export const version = '${version}';`);

if (updated !== source) {
  writeFileSync(indexUrl, updated);
}

console.log(`sync-version: src/index.ts -> ${version}`);
