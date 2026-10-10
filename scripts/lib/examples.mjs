/**
 * Extracts every `@example` from `src/` and turns it into a self-contained
 * program (imports added, variables from earlier examples of the same
 * declaration pulled in). Each program is type-checked against the library
 * sources, so broken examples are detected instead of shipped.
 *
 * Shared by `scripts/build-playground.mjs` (playground) and
 * `tests/examples.test.ts` (CI check).
 */
import { readdir, readFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const src = join(root, 'src');
const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));


// Public top-level modules, e.g. './roots' -> 'roots' (nested subpaths excluded).
export const MODULES = Object.keys(pkg.exports)
    .filter((key) => /^\.\/[^/]+$/.test(key))
    .map((key) => key.slice(2));

async function walk(dir) {
    const files = [];
    for (const entry of await readdir(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name);
        if (entry.isDirectory()) files.push(...(await walk(path)));
        else files.push(path);
    }
    return files;
}

const posix = (path) => path.split('\\').join('/');

/** Whitespace-insensitive key under which an example is looked up (same as `docs-assets/playground-links.js`). */
export const normalize = (code) => code.replace(/\s+/g, ' ').trim();

/** Which public module exports which name, and whether it is a value or a type. */
function loadExports() {
    const entries = MODULES.map((m) => join(src, `${m}.ts`));
    const program = ts.createProgram(entries, {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.NodeNext,
        moduleResolution: ts.ModuleResolutionKind.NodeNext,
        noEmit: true,
        skipLibCheck: true,
        types: [],
    });
    const checker = program.getTypeChecker();
    const byName = new Map(); // name -> Map(module -> isValue)
    for (const module of MODULES) {
        const file = program.getSourceFile(join(src, `${module}.ts`));
        const symbol = file && checker.getSymbolAtLocation(file);
        if (!symbol) throw new Error(`Cannot read exports of src/${module}.ts`);
        for (let exported of checker.getExportsOfModule(symbol)) {
            const name = exported.getName();
            if (exported.flags & ts.SymbolFlags.Alias) exported = checker.getAliasedSymbol(exported);
            const isValue = (exported.flags & ts.SymbolFlags.Value) !== 0;
            if (!byName.has(name)) byName.set(name, new Map());
            byName.get(name).set(module, isValue);
        }
    }
    return byName;
}

function bindingNames(name) {
    if (!name) return [];
    if (ts.isIdentifier(name)) return [name.text];
    return name.elements.flatMap((element) =>
        ts.isOmittedExpression(element) ? [] : bindingNames(element.name),
    );
}

/** Names a top-level statement introduces into the example's scope. */
function topLevelNames(statement) {
    if (ts.isVariableStatement(statement)) {
        return statement.declarationList.declarations.flatMap((d) => bindingNames(d.name));
    }
    if (
        ts.isFunctionDeclaration(statement) ||
        ts.isClassDeclaration(statement) ||
        ts.isInterfaceDeclaration(statement) ||
        ts.isTypeAliasDeclaration(statement) ||
        ts.isEnumDeclaration(statement)
    ) {
        return statement.name ? [statement.name.text] : [];
    }
    if (ts.isImportDeclaration(statement) && statement.importClause) {
        const { name, namedBindings } = statement.importClause;
        const names = name ? [name.text] : [];
        if (namedBindings) {
            if (ts.isNamespaceImport(namedBindings)) names.push(namedBindings.name.text);
            else names.push(...namedBindings.elements.map((e) => e.name.text));
        }
        return names;
    }
    return [];
}

function isReference(id) {
    const p = id.parent;
    if (ts.isPropertyAccessExpression(p) && p.name === id) return false;
    if (ts.isQualifiedName(p) && p.right === id) return false;
    if (
        (ts.isPropertyAssignment(p) ||
            ts.isPropertyDeclaration(p) ||
            ts.isPropertySignature(p) ||
            ts.isMethodDeclaration(p) ||
            ts.isMethodSignature(p) ||
            ts.isGetAccessorDeclaration(p) ||
            ts.isSetAccessorDeclaration(p) ||
            ts.isEnumMember(p)) &&
        p.name === id
    ) {
        return false;
    }
    if (ts.isBindingElement(p) && p.propertyName === id) return false;
    if (ts.isImportSpecifier(p) || ts.isExportSpecifier(p)) return false;
    return true;
}

/** Names a statement uses but does not declare itself (at any depth). */
function freeNames(statement) {
    const used = new Set();
    const declared = new Set();
    const visit = (node) => {
        if (ts.isIdentifier(node) && isReference(node)) used.add(node.text);
        if (
            ts.isVariableDeclaration(node) ||
            ts.isParameter(node) ||
            ts.isFunctionDeclaration(node) ||
            ts.isFunctionExpression(node) ||
            ts.isClassDeclaration(node) ||
            ts.isClassExpression(node) ||
            ts.isTypeAliasDeclaration(node) ||
            ts.isInterfaceDeclaration(node) ||
            ts.isTypeParameterDeclaration(node)
        ) {
            bindingNames(node.name).forEach((n) => declared.add(n));
        }
        ts.forEachChild(node, visit);
    };
    visit(statement);
    return [...used].filter((name) => !declared.has(name));
}

function analyze(code) {
    const file = ts.createSourceFile('example.ts', code, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    return file.statements.map((statement) => ({
        text: statement.getFullText(file).trim(),
        declares: topLevelNames(statement),
        free: freeNames(statement),
    }));
}

/** The `@example` code fences of every declaration, in source order. */
function extractExampleGroups(file) {
    const groups = [];
    const seen = new Set();
    const visit = (node) => {
        // Internal declarations are not part of the public API (and not in the docs).
        if (ts.getJSDocTags(node).some((tag) => tag.tagName.text === 'internal')) return;
        const tags = ts
            .getJSDocTags(node)
            .filter((tag) => tag.tagName.text === 'example' && !seen.has(tag.pos));
        if (tags.length > 0) {
            const codes = [];
            for (const tag of tags) {
                seen.add(tag.pos);
                const text = ts.getTextOfJSDocComment(tag.comment) ?? '';
                const match = /```ts\n([\s\S]*?)\n```/.exec(text);
                if (match) codes.push(match[1]);
            }
            if (codes.length > 0) groups.push(codes);
        }
        ts.forEachChild(node, visit);
    };
    visit(file);
    return groups;
}

/** Self-contained program for example `k` of a declaration's examples. */
function buildProgram(codes, k, homeModule, exportsByName) {
    const parsed = codes.map(analyze);
    const included = new Map(); // "j:s" -> { j, s }
    const queue = [];
    const include = (j, s) => {
        const key = `${j}:${s}`;
        if (!included.has(key)) {
            included.set(key, { j, s });
            queue.push({ j, s });
        }
    };
    parsed[k].forEach((_, s) => include(k, s));

    const unresolved = new Set();
    while (queue.length > 0) {
        const { j, s } = queue.pop();
        for (const name of parsed[j][s].free) {
            let found = false;
            for (let jj = j; jj >= 0 && !found; jj--) {
                const t = parsed[jj].findIndex((st, idx) => !(jj === j && idx === s) && st.declares.includes(name));
                if (t >= 0) {
                    include(jj, t);
                    found = true;
                }
            }
            if (!found) unresolved.add(name);
        }
    }

    // Library imports for what is still unresolved.
    const importsByModule = new Map();
    for (const name of unresolved) {
        const modules = exportsByName.get(name);
        if (!modules) continue;
        const module = modules.has(homeModule) ? homeModule : [...modules.keys()].sort()[0];
        if (!importsByModule.has(module)) importsByModule.set(module, []);
        importsByModule.get(module).push({ name, isValue: modules.get(module) });
    }
    const imports = [...importsByModule.keys()].sort().map((module) => {
        const specifiers = importsByModule
            .get(module)
            .sort((a, b) => a.name.localeCompare(b.name))
            .map(({ name, isValue }) => (isValue ? name : `type ${name}`));
        return `import { ${specifiers.join(', ')} } from 'numerics-js/${module}';`;
    });

    // Statements pulled in from earlier examples, in original order.
    const dependencies = [...included.values()]
        .filter(({ j }) => j !== k)
        .sort((a, b) => a.j - b.j || a.s - b.s)
        .map(({ j, s }) => parsed[j][s].text);

    const parts = [];
    if (imports.length > 0) parts.push(imports.join('\n'));
    if (dependencies.length > 0) parts.push(dependencies.join('\n'));
    parts.push(codes[k]);
    return parts.join('\n\n');
}

/**
 * Type-checks a generated program against the compiled library (`dist/`).
 * Returns the names of unresolved identifiers, which means the program cannot run
 * (typically an example that refers to something that is not public API).

/**
 * Type-checks programs against the library sources (`src/`, as `numerics-js`).
 * Returns every diagnostic, and whether any is an unresolved name or module,
 * which means the program cannot run (typically it uses something that is not
 * public API).
 */
function createChecker() {
    const options = {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ESNext,
        moduleResolution: ts.ModuleResolutionKind.Bundler,
        lib: ['lib.esnext.d.ts', 'lib.dom.d.ts'],
        types: [],
        strict: true,
        noEmit: true,
        skipLibCheck: true,
        paths: { 'numerics-js': [join(src, 'index')], 'numerics-js/*': [join(src, '*')] },
    };
    const virtual = join(root, '__example__.ts');
    let previous;
    return (code) => {
        const host = ts.createCompilerHost(options);
        const readFile = host.readFile.bind(host);
        const fileExists = host.fileExists.bind(host);
        const getSourceFile = host.getSourceFile.bind(host);
        host.readFile = (f) => (f === virtual ? code : readFile(f));
        host.fileExists = (f) => f === virtual || fileExists(f);
        host.getSourceFile = (f, languageVersion, ...rest) =>
            f === virtual ? ts.createSourceFile(f, code, languageVersion, true) : getSourceFile(f, languageVersion, ...rest);
        const program = ts.createProgram([virtual], options, host, previous);
        previous = program;
        const diagnostics = ts.getPreEmitDiagnostics(program, program.getSourceFile(virtual));
        return {
            diagnostics: diagnostics.map((d) => {
                const message = ts.flattenDiagnosticMessageText(d.messageText, ' ');
                if (!d.file) return message;
                const { line } = ts.getLineAndCharacterOfPosition(d.file, d.start ?? 0);
                return `line ${line + 1}: ${message}`;
            }),
            // 2304/2552: unknown name; 2307: unknown module; 2305/2614/2724: module has no such export.
            unresolved: diagnostics.some((d) => [2304, 2305, 2307, 2552, 2614, 2724].includes(d.code)),
        };
    };
}

/**
 * Every `@example` in `src/` as `{ file, code, program, diagnostics, unresolved }`:
 * `code` is the example as written, `program` its self-contained version.
 */
export async function collectExamples() {
    const exportsByName = loadExports();
    const check = createChecker();
    const examples = [];
    for (const path of (await walk(src)).filter((p) => p.endsWith('.ts')).sort()) {
        const rel = posix(relative(src, path));
        const homeModule = rel.includes('/') ? rel.split('/')[0] : rel.replace(/\.ts$/, '');
        const file = ts.createSourceFile(path, await readFile(path, 'utf8'), ts.ScriptTarget.ES2022, true);
        for (const codes of extractExampleGroups(file)) {
            codes.forEach((code, k) => {
                const program = buildProgram(codes, k, homeModule, exportsByName);
                examples.push({ file: `src/${rel}`, code, program, ...check(program) });
            });
        }
    }
    return examples;
}
