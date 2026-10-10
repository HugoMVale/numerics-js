// Runs one compiled program. The main thread terminates this worker on Stop,
// on timeout, or once the program has finished.
import { format, inspect } from './vendor/inspect.mjs';

const post = (kind, text) => self.postMessage({ kind, text });

for (const kind of ['log', 'info', 'debug', 'warn', 'error']) {
    console[kind] = (...args) => post(kind, format(...args));
}
console.dir = (value, options) => post('log', inspect(value, options));

function describe(error) {
    const text = error instanceof Error ? (error.stack ?? String(error)) : `Uncaught ${format(error)}`;
    return text.replace(/blob:\S+?:(\d+):(\d+)/g, '<program>:$1:$2');
}

self.addEventListener('unhandledrejection', (event) => post('error', describe(event.reason)));

self.onmessage = async ({ data }) => {
    const url = URL.createObjectURL(new Blob([data.code], { type: 'text/javascript' }));
    const start = performance.now();
    try {
        await import(url);
    } catch (error) {
        post('error', describe(error));
    } finally {
        URL.revokeObjectURL(url);
        post('done', `${Math.round(performance.now() - start)} ms`);
    }
};
