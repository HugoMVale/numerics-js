// numerics-js playground: Monaco editor + TypeScript transpile + sandboxed worker.
import { EXAMPLES } from './examples.js';

const MONACO_VERSION = '0.52.2';
const MONACO_BASE = `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min`;
const RUN_TIMEOUT_MS = 15_000;
const LIB_URL = new URL('lib/', location.href).href;

const $ = (id) => document.getElementById(id);
const ui = {
    run: $('run'),
    stop: $('stop'),
    share: $('share'),
    clear: $('clear'),
    examples: $('examples'),
    console: $('console'),
    version: $('version'),
    editor: $('editor'),
};

// ---------------------------------------------------------------------------
// Sharing: code travels in the URL hash, deflate-compressed (#z=) or plain (#code=).
// ---------------------------------------------------------------------------

const toBase64Url = (bytes) =>
    btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

const fromBase64Url = (text) =>
    Uint8Array.from(atob(text.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0));

async function pipe(bytes, stream) {
    const piped = new Blob([bytes]).stream().pipeThrough(stream);
    return new Uint8Array(await new Response(piped).arrayBuffer());
}

async function encodeCode(code) {
    const bytes = await pipe(new TextEncoder().encode(code), new CompressionStream('deflate-raw'));
    return toBase64Url(bytes);
}

async function decodeCode(text) {
    const bytes = await pipe(fromBase64Url(text), new DecompressionStream('deflate-raw'));
    return new TextDecoder().decode(bytes);
}

async function codeFromHash() {
    const params = new URLSearchParams(location.hash.slice(1));
    try {
        if (params.has('z')) return await decodeCode(params.get('z'));
        if (params.has('code')) return params.get('code');
    } catch {
        // Fall through to the default example.
    }
    return null;
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

function loadMonaco() {
    return new Promise((resolve, reject) => {
        // Monaco's workers live on the CDN; bootstrap them through a same-origin blob.
        self.MonacoEnvironment = {
            getWorkerUrl: () =>
                URL.createObjectURL(
                    new Blob(
                        [
                            `self.MonacoEnvironment = { baseUrl: '${MONACO_BASE}/' };` +
                                `importScripts('${MONACO_BASE}/vs/base/worker/workerMain.js');`,
                        ],
                        { type: 'text/javascript' },
                    ),
                ),
        };
        const script = document.createElement('script');
        script.src = `${MONACO_BASE}/vs/loader.js`;
        script.onerror = () => reject(new Error(`Could not load the editor from ${MONACO_BASE}.`));
        script.onload = () => {
            window.require.config({ paths: { vs: `${MONACO_BASE}/vs` } });
            window.require(['vs/editor/editor.main'], () => resolve(window.monaco), reject);
        };
        document.head.append(script);
    });
}

async function setupTypeScript(monaco) {
    const ts = monaco.languages.typescript;
    ts.typescriptDefaults.setCompilerOptions({
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ESNext,
        moduleResolution: ts.ModuleResolutionKind.NodeJs,
        lib: ['esnext', 'dom'],
        strict: true,
        skipLibCheck: true,
        allowNonTsExtensions: true,
    });
    // Mirror dist/ at node_modules/numerics-js/ so that
    // `import ... from 'numerics-js/roots'` resolves to roots.d.ts.
    try {
        const typings = await (await fetch(new URL('typings.json', location.href))).json();
        for (const [path, content] of Object.entries(typings)) {
            ts.typescriptDefaults.addExtraLib(content, `file:///node_modules/numerics-js/${path}`);
        }
    } catch (error) {
        console.warn('numerics-js typings unavailable; autocomplete will be limited.', error);
    }
}

const prefersDark = matchMedia('(prefers-color-scheme: dark)');
const themeName = () => (prefersDark.matches ? 'vs-dark' : 'vs');

// ---------------------------------------------------------------------------
// Compile and run
// ---------------------------------------------------------------------------

/** Transpile with Monaco's own TypeScript worker (no extra compiler download). */
async function compile(monaco, model) {
    const getWorker = await monaco.languages.typescript.getTypeScriptWorker();
    const worker = await getWorker(model.uri);
    const result = await worker.getEmitOutput(model.uri.toString());
    const file = result.outputFiles.find((f) => f.name.endsWith('.js'));
    if (!file) throw new Error('TypeScript produced no output.');
    return file.text;
}

/** Point `numerics-js` imports at the library files served next to this page. */
function rewriteImports(js) {
    return js.replace(
        /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s*)(['"])numerics-js(?:\/([\w/.-]+))?\2/g,
        (_match, lead, quote, subpath) => `${lead}${quote}${LIB_URL}${subpath ?? 'index'}.js${quote}`,
    );
}

function print(kind, text) {
    const line = document.createElement('div');
    line.className = kind;
    line.textContent = text;
    ui.console.append(line);
    ui.console.scrollTop = ui.console.scrollHeight;
}

let worker = null;
let timer = null;

function finish(summary) {
    clearTimeout(timer);
    worker?.terminate();
    worker = null;
    if (summary) print('meta', summary);
    ui.run.hidden = false;
    ui.stop.hidden = true;
}

function stop() {
    if (worker) finish('Stopped.');
}

async function run(monaco, model) {
    if (worker) finish();
    ui.console.textContent = '';
    ui.run.hidden = true;
    ui.stop.hidden = false;

    let code;
    try {
        code = rewriteImports(await compile(monaco, model));
    } catch (error) {
        print('error', `Compilation failed: ${error.message}`);
        finish();
        return;
    }

    worker = new Worker(new URL('worker.js', location.href), { type: 'module' });
    worker.onmessage = ({ data }) => {
        if (data.kind === 'done') finish(`Finished in ${data.text}.`);
        else print(data.kind, data.text);
    };
    worker.onerror = (event) => {
        print('error', event.message || 'The program could not be started.');
        finish();
    };
    timer = setTimeout(() => {
        print('error', `Stopped: still running after ${RUN_TIMEOUT_MS / 1000} s.`);
        finish();
    }, RUN_TIMEOUT_MS);
    worker.postMessage({ code });
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function main() {
    for (const [index, example] of EXAMPLES.entries()) {
        ui.examples.add(new Option(example.title, String(index)));
    }

    fetch(new URL('manifest.json', location.href))
        .then((response) => response.json())
        .then(({ version }) => (ui.version.textContent = `v${version}`))
        .catch(() => {});

    let monaco;
    try {
        monaco = await loadMonaco();
    } catch (error) {
        $('editor-status').textContent = error.message;
        return;
    }
    await setupTypeScript(monaco);

    const initial = (await codeFromHash()) ?? EXAMPLES[0].code;
    const model = monaco.editor.createModel(initial, 'typescript', monaco.Uri.parse('file:///main.ts'));
    $('editor-status').remove();
    const editor = monaco.editor.create(ui.editor, {
        model,
        theme: themeName(),
        automaticLayout: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        fontSize: 14,
        tabSize: 4,
    });
    prefersDark.addEventListener('change', () => monaco.editor.setTheme(themeName()));

    const runNow = () => run(monaco, model);
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, runNow);

    ui.run.addEventListener('click', runNow);
    ui.stop.addEventListener('click', stop);
    ui.clear.addEventListener('click', () => (ui.console.textContent = ''));
    ui.examples.addEventListener('change', () => {
        model.setValue(EXAMPLES[Number(ui.examples.value)].code);
        ui.console.textContent = '';
    });
    ui.share.addEventListener('click', async () => {
        const url = `${location.origin}${location.pathname}#z=${await encodeCode(model.getValue())}`;
        try {
            await navigator.clipboard.writeText(url);
            ui.share.textContent = 'Link copied';
        } catch {
            history.replaceState(null, '', url);
            ui.share.textContent = 'Link in address bar';
        }
        setTimeout(() => (ui.share.textContent = 'Copy link'), 2000);
    });
    window.addEventListener('hashchange', async () => {
        const code = await codeFromHash();
        if (code !== null) model.setValue(code);
    });

    ui.run.disabled = false;
    ui.share.disabled = false;
}

main();
