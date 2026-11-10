import { version } from 'numerics-js';
import { startLoop } from './util.js';
import { mountHero } from './hero.js';
import { mountDrum } from './drum.js';
import { mountOrbits } from './orbits.js';
import { mountSimplex } from './simplex.js';
import { mountCurve } from './curve.js';
import { mountPhase } from './phase.js';

// tiny syntax highlighter for the code snippets
function highlight(el) {
  const src = el.textContent;
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  el.innerHTML = esc(src).replace(
    /(\/\/.*$)|('(?:[^'\\]|\\.)*')|\b(import|from|const|new|return)\b|\b(\d+(?:\.\d+)?)\b|\b(ode|special|optimize|interpolate|integrate|numdiff|roots|linalg|math)\b(?=\.|,|\s*\})/gm,
    (_m, c, s, k, n, mod) =>
      c
        ? `<span class="c">${c}</span>`
        : s
          ? `<span class="s">${s}</span>`
          : k
            ? `<span class="k">${k}</span>`
            : n
              ? `<span class="n">${n}</span>`
              : `<span class="m">${mod}</span>`
  );
}
document.querySelectorAll('.call code').forEach(highlight);

// reveal on scroll
const io = new IntersectionObserver(
  (es) => {
    for (const e of es) {
      if (!e.isIntersecting) continue;
      e.target.classList.add('in');
      io.unobserve(e.target);
    }
  },
  { threshold: 0.12 }
);
document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

document.getElementById('ver').textContent = 'v' + version;

for (const mount of [mountHero, mountDrum, mountOrbits, mountSimplex, mountCurve, mountPhase]) {
  try {
    mount();
  } catch (err) {
    console.error('[numerics-gallery]', mount.name, err);
  }
}
startLoop();
