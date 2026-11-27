// Prints the CHANGELOG.md section of one version, for use as GitHub release notes.
// Usage: node scripts/changelog-notes.mjs [version]   (default: version in package.json;
// a leading "v" is accepted). Exits with 1 if the section is missing or empty.
import { readFileSync } from 'node:fs';

const packageUrl = new URL('../package.json', import.meta.url);
const changelogUrl = new URL('../CHANGELOG.md', import.meta.url);

const requested = process.argv[2] ?? JSON.parse(readFileSync(packageUrl, 'utf8')).version;
const version = requested.replace(/^v/, '');

const lines = readFileSync(changelogUrl, 'utf8').split(/\r?\n/);
const heading = (line) => /^## \[/.test(line);
const linkDefinition = (line) => /^\[[^\]]+\]:\s/.test(line);

const start = lines.findIndex((line) => heading(line) && line.startsWith(`## [${version}]`));
if (start === -1) {
  console.error(`changelog-notes: no "## [${version}]" section found in CHANGELOG.md.`);
  process.exit(1);
}

const body = [];
for (const line of lines.slice(start + 1)) {
  if (heading(line) || linkDefinition(line)) break;
  body.push(line);
}

const notes = body.join('\n').trim();
if (notes === '') {
  console.error(`changelog-notes: the "## [${version}]" section of CHANGELOG.md is empty.`);
  process.exit(1);
}

console.log(notes);
