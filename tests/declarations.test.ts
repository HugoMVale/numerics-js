import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

/**
 * What `npm run build` publishes as `.d.ts`: the declarations are emitted in memory with the
 * real `tsconfig.build.json` (so with `stripInternal`) and checked as a consumer would see them.
 *
 * `@internal` members are used across modules and by the solvers, so they must stay in the
 * sources but not in the declarations. And a declaration that survives must not refer to one
 * that was stripped: such a reference is an error in the emitted file, which `skipLibCheck`
 * hides from consumers by silently turning the type into `any`.
 */
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function emitDeclarations(): { files: Map<string, string>; options: ts.CompilerOptions } {
  const configPath = join(root, 'tsconfig.build.json');
  const config = ts.readConfigFile(configPath, ts.sys.readFile);
  const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root);
  const outDir = join(root, '__declarations__');
  const options: ts.CompilerOptions = {
    ...parsed.options,
    emitDeclarationOnly: true,
    declarationMap: false,
    sourceMap: false,
    noEmit: false,
    outDir
  };
  const files = new Map<string, string>();
  const host = ts.createCompilerHost(options);
  host.writeFile = (name, text) => {
    files.set(resolve(name), text);
  };
  const program = ts.createProgram(parsed.fileNames, options, host);
  const result = program.emit();
  expect(result.emitSkipped).toBe(false);
  return { files, options };
}

describe('emitted declarations', { timeout: 60_000 }, () => {
  const { files, options } = emitDeclarations();
  const texts = [...files.entries()];

  it('are emitted for the whole library', () => {
    expect(files.size).toBeGreaterThan(50);
    expect([...files.keys()].some((f) => f.endsWith(join('linalg', 'Matrix.d.ts')))).toBe(true);
  });

  it('contain no @internal declaration', () => {
    const leaks = texts.filter(([, text]) => text.includes('@internal')).map(([name]) => name);
    expect(leaks).toEqual([]);
  });

  it('hide the unchecked accessors and constructor overloads of Vector and Matrix', () => {
    const find = (suffix: string) => texts.find(([name]) => name.endsWith(suffix))?.[1] ?? '';
    for (const text of [find(join('linalg', 'Matrix.d.ts')), find(join('linalg', 'Vector.d.ts'))]) {
      expect(text).not.toMatch(/wrapUnchecked|flatIndex|getUnchecked|setUnchecked/);
      expect(text).not.toMatch(/adopt: Float64Array/);
    }
  });

  it('type-check without skipLibCheck: no declaration refers to a stripped one', () => {
    const host = ts.createCompilerHost(options);
    const realFileExists = host.fileExists.bind(host);
    const realReadFile = host.readFile.bind(host);
    const realGetSourceFile = host.getSourceFile.bind(host);
    // Module resolution also asks about directories and real paths of the in-memory files.
    const directories = new Set<string>();
    for (const name of files.keys()) {
      for (
        let dir = dirname(name);
        !directories.has(dir) && dir !== dirname(dir);
        dir = dirname(dir)
      ) {
        directories.add(dir);
      }
    }
    const realDirectoryExists = host.directoryExists?.bind(host);
    const realRealpath = host.realpath?.bind(host);
    host.directoryExists = (name) =>
      directories.has(resolve(name)) || !!realDirectoryExists?.(name);
    host.realpath = (name) =>
      files.has(resolve(name)) || directories.has(resolve(name))
        ? name
        : (realRealpath?.(name) ?? name);
    host.fileExists = (name) => files.has(resolve(name)) || realFileExists(name);
    host.readFile = (name) => files.get(resolve(name)) ?? realReadFile(name);
    host.getSourceFile = (name, languageVersion, onError) => {
      const text = files.get(resolve(name));
      return text === undefined
        ? realGetSourceFile(name, languageVersion, onError)
        : ts.createSourceFile(name, text, languageVersion);
    };
    const program = ts.createProgram(
      [...files.keys()],
      { ...options, skipLibCheck: false, noEmit: true },
      host
    );
    const errors = ts
      .getPreEmitDiagnostics(program)
      .filter((d) => d.file !== undefined && files.has(resolve(d.file.fileName)))
      .map((d) => `${d.file?.fileName}: ${ts.flattenDiagnosticMessageText(d.messageText, '\n')}`);
    expect(errors).toEqual([]);
  });
});
