import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

async function updateDirectory(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);

        if (entry.isDirectory()) {
            await updateDirectory(path);
        } else if (entry.name.endsWith('.js') || entry.name.endsWith('.d.ts')) {
            const source = await readFile(path, 'utf8');
            const updated = source.replace(
                /((?:from\s+|import\s*\(\s*)['"])(\.\.?\/[^'"]+)(['"])/g,
                (match, prefix, specifier, quote) => /\.[a-z]+$/i.test(specifier) ? match : `${prefix}${specifier}.js${quote}`
            );

            if (updated !== source) {
                await writeFile(path, updated);
            }
        }
    }
}

await updateDirectory('dist');