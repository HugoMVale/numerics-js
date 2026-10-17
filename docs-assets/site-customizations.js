// Adds a GitHub icon to the header and a "Run" button to supported examples.
// Example links open the self-contained version from playground/examples.json.
(function () {
  'use strict';

  const normalize = (code) => code.replace(/\s+/g, ' ').trim();
  const root = document.documentElement.dataset.base || './';
  const playground = new URL(root + 'playground/', location.href);

  function addGithubIcon() {
    const link = document.querySelector(
      '#tsd-toolbar-links a[href="https://github.com/HugoMVale/numerics-js"]'
    );
    if (!link) return;

    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    icon.setAttribute('viewBox', '0 0 16 16');
    icon.setAttribute('width', '20');
    icon.setAttribute('height', '20');
    icon.setAttribute('aria-hidden', 'true');
    icon.setAttribute('fill', 'currentColor');
    icon.classList.add('github-icon');

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute(
      'd',
      'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z'
    );
    icon.append(path);

    link.replaceChildren(icon);
    link.setAttribute('aria-label', 'GitHub repository');
    link.title = 'GitHub repository';
  }

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
    return btoa(String.fromCharCode(...bytes))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }

  async function init() {
    addGithubIcon();
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
        #tsd-toolbar-links .github-icon { display: block; }
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
