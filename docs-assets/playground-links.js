// Adds a "Run" button next to the Copy button of every TypeDoc example
// (a ```ts block inside .tsd-tag-example). It opens the playground with the
// self-contained version of that example, looked up in playground/examples.json.
(function () {
    'use strict';

    const normalize = (code) => code.replace(/\s+/g, ' ').trim();
    const root = document.documentElement.dataset.base || './';
    const playground = new URL(root + 'playground/', location.href);

    // TypeDoc's highlighter writes line breaks as <br/>, which textContent drops.
    function sourceText(code) {
        const copy = code.cloneNode(true);
        copy.querySelectorAll('br').forEach((br) => br.replaceWith('\n'));
        return copy.textContent;
    }

    let examples = null;
    const load = () =>
        (examples ??= fetch(new URL('examples.json', playground))
            .then((response) => (response.ok ? response.json() : {}))
            .catch(() => ({})));

    async function encode(code) {
        const piped = new Blob([new TextEncoder().encode(code)])
            .stream()
            .pipeThrough(new CompressionStream('deflate-raw'));
        const bytes = new Uint8Array(await new Response(piped).arrayBuffer());
        return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    }

    async function init() {
        if (typeof CompressionStream === 'undefined') return;
        const blocks = document.querySelectorAll('.tsd-tag-example pre > code.ts');
        if (blocks.length === 0) return;
        const table = await load();
        for (const code of blocks) {
            const pre = code.parentElement;
            if (pre.querySelector('.playground-run')) continue;
            const program = table[normalize(sourceText(code))];
            if (!program) continue;
            const link = document.createElement('a');
            link.className = 'playground-run';
            link.textContent = 'Run ▶';
            link.title = 'Open this example in the playground';
            link.target = '_blank';
            link.rel = 'noopener';
            link.href = new URL(playground).href;
            encode(program).then((z) => (link.href = `${playground.href}#z=${z}`));
            pre.append(link);
        }
    }

    const style = document.createElement('style');
    style.textContent = `
        pre > a.playground-run {
            position: absolute; top: 10px; right: 80px; opacity: 0;
            transition: opacity 0.1s; padding: 2px 10px; border-radius: 6px;
            border: 1px solid var(--color-accent); background: var(--color-background);
            color: var(--color-text); font-size: 0.8rem; text-decoration: none;
        }
        pre:hover > a.playground-run, pre > a.playground-run:focus-visible { opacity: 1; }
    `;
    document.head.append(style);

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
