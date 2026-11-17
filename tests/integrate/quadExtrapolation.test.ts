import { describe, expect, it } from 'vitest';
import { extrapolatedIntegral } from '../../src/integrate/adaptiveExtrapolation.js';
import { gaussKronrod } from '../../src/integrate/gaussKronrod.js';
import { quad } from '../../src/integrate/quad.js';

type Kind = 'smooth' | 'singular' | 'kink' | 'pathological';

interface Problem {
  name: string;
  f: (x: number) => number;
  a: number;
  b: number;
  exact: number;
  kind: Kind;
}

// 16 integrands with exact answers. The "singular" ones have integrable singularities at an
// endpoint or inside the range, the "kink" ones are continuous with a kink or have a jump.
const PROBLEMS: Problem[] = [
  { name: 'sin', f: Math.sin, a: 0, b: Math.PI, exact: 2, kind: 'smooth' },
  { name: '4/(1+x^2)', f: (x) => 4 / (1 + x * x), a: 0, b: 1, exact: Math.PI, kind: 'smooth' },
  {
    name: 'sin(100x)',
    f: (x) => Math.sin(100 * x),
    a: 0,
    b: 1,
    exact: (1 - Math.cos(100)) / 100,
    kind: 'smooth'
  },
  {
    name: 'cos(50x)',
    f: (x) => Math.cos(50 * x),
    a: 0,
    b: 1,
    exact: Math.sin(50) / 50,
    kind: 'smooth'
  },
  {
    name: 'gaussian [-3,3]',
    f: (x) => Math.exp(-x * x),
    a: -3,
    b: 3,
    exact: Math.sqrt(Math.PI) * 0.9999779095030014,
    kind: 'smooth'
  },
  {
    name: 'runge',
    f: (x) => 1 / (1 + 25 * x * x),
    a: -1,
    b: 1,
    exact: 0.4 * Math.atan(5),
    kind: 'smooth'
  },
  {
    name: 'lorentzian peak',
    f: (x) => 1 / ((x - 0.5) ** 2 + 1e-4),
    a: 0,
    b: 1,
    exact: (2 * Math.atan(0.5 / 0.01)) / 0.01,
    kind: 'smooth'
  },
  {
    name: 'exp(-x) on [0,50]',
    f: (x) => Math.exp(-x),
    a: 0,
    b: 50,
    exact: 1 - Math.exp(-50),
    kind: 'smooth'
  },
  { name: 'sqrt(x)', f: Math.sqrt, a: 0, b: 1, exact: 2 / 3, kind: 'singular' },
  { name: '-log(x)', f: (x) => -Math.log(x), a: 0, b: 1, exact: 1, kind: 'singular' },
  { name: '1/sqrt(x)', f: (x) => 1 / Math.sqrt(x), a: 0, b: 1, exact: 2, kind: 'singular' },
  { name: 'x^-0.9', f: (x) => x ** -0.9, a: 0, b: 1, exact: 10, kind: 'singular' },
  {
    name: '|x-1/3|^-1/2',
    f: (x) => Math.abs(x - 1 / 3) ** -0.5,
    a: 0,
    b: 1,
    exact: 2 * (Math.sqrt(1 / 3) + Math.sqrt(2 / 3)),
    kind: 'singular'
  },
  {
    name: 'kink |x-1/3|',
    f: (x) => Math.abs(x - 1 / 3),
    a: 0,
    b: 1,
    exact: 5 / 18,
    kind: 'kink'
  },
  {
    name: 'jump at 1/3',
    f: (x) => (x < 1 / 3 ? 1 : 2),
    a: 0,
    b: 1,
    exact: 5 / 3,
    kind: 'kink'
  },
  {
    name: 'sin(1/x)',
    f: (x) => Math.sin(1 / x),
    a: 0,
    b: 1,
    exact: 0.504067061906928,
    kind: 'pathological'
  }
];

const RTOLS = [1e-4, 1e-6, 1e-8, 1e-10];
const EPS = Number.EPSILON;

/** Whether `value` meets the requested relative tolerance, up to a few ulps. */
function meetsTolerance(p: Problem, value: number, rtol: number, floor = 0): boolean {
  const slack = Math.max(rtol * Math.abs(p.exact), 4 * EPS * Math.abs(p.exact), floor);
  return Math.abs(value - p.exact) <= slack;
}

/** Runs `quad` on all the problems of a kind with all tolerances. */
function runKind(kind: Kind) {
  const runs = [];
  for (const p of PROBLEMS.filter((q) => q.kind === kind)) {
    for (const rtol of RTOLS) {
      const res = quad(p.f, p.a, p.b, { atol: 0, rtol, maxSubintervals: 200 });
      runs.push({ p, rtol, res });
    }
  }
  return runs;
}

describe('quad - extrapolation on problems with exact answers', () => {
  // Evaluation counts of the plain adaptive engine (gaussKronrod) over the same runs were
  // 7950 (smooth), 30855 (singular, 3 runs not converged) and 3330 (kinks).
  const BUDGET: Record<'smooth' | 'singular' | 'kink', number> = {
    smooth: 7950,
    singular: 3500,
    kink: 1100
  };

  for (const kind of ['smooth', 'singular', 'kink'] as const) {
    it(`converges to the requested tolerance on every ${kind} run within the evaluation budget`, () => {
      const runs = runKind(kind);
      for (const { p, rtol, res } of runs) {
        expect(res.success, `${p.name} rtol=${rtol}: ${res.message}`).toBe(true);
        expect(meetsTolerance(p, res.value, rtol), `${p.name} rtol=${rtol}`).toBe(true);
      }
      const total = runs.reduce((sum, r) => sum + r.res.evaluations, 0);
      expect(total).toBeLessThanOrEqual(BUDGET[kind]);
    });
  }

  it('needs far fewer evaluations than the plain engine on integrable singularities', () => {
    const singular = PROBLEMS.filter((p) => p.kind === 'singular');
    for (const p of singular) {
      const plain = gaussKronrod(p.f, p.a, p.b, { atol: 0, rtol: 1e-8, maxSubintervals: 200 });
      const res = quad(p.f, p.a, p.b, { atol: 0, rtol: 1e-8, maxSubintervals: 200 });
      expect(res.evaluations, p.name).toBeLessThan(plain.evaluations / 2);
    }
    // 1/sqrt(x) at 1e-8: SciPy's QAGS with the 21-point rule needs 231.
    const sqrt = PROBLEMS.find((p) => p.name === '1/sqrt(x)')!;
    expect(quad(sqrt.f, 0, 1, { atol: 0, rtol: 1e-8 }).evaluations).toBeLessThanOrEqual(231);
  });

  it('never reports success without meeting the tolerance, also for pathological sin(1/x)', () => {
    const p = PROBLEMS.find((q) => q.kind === 'pathological')!;
    for (const rtol of RTOLS) {
      const res = quad(p.f, p.a, p.b, { atol: 0, rtol, maxSubintervals: 200 });
      // The oscillations accumulate at 0: it may legitimately not converge, but then it says so.
      if (res.success) expect(meetsTolerance(p, res.value, rtol, 1e-13)).toBe(true);
      else expect(res.message.length).toBeGreaterThan(0);
    }
    expect(quad(p.f, p.a, p.b, { atol: 0, rtol: 1e-10, maxSubintervals: 200 }).success).toBe(false);
  });

  it('does not report success while missing the tolerance (cases of the plain engine)', () => {
    // The plain engine claimed success with a true error 3.5-5x its estimate on these 4 runs.
    for (const name of ['x^-0.9', '|x-1/3|^-1/2']) {
      const p = PROBLEMS.find((q) => q.name === name)!;
      for (const rtol of [1e-4, 1e-6]) {
        const res = quad(p.f, p.a, p.b, { atol: 0, rtol, maxSubintervals: 200 });
        expect(res.success, `${name} rtol=${rtol}`).toBe(true);
        expect(Math.abs(res.value - p.exact), `${name} rtol=${rtol}`).toBeLessThanOrEqual(
          rtol * p.exact
        );
        expect(Math.abs(res.value - p.exact)).toBeLessThanOrEqual(Math.max(res.error, 1e-13));
      }
    }
  });

  it('is cheap on a single smooth subinterval', () => {
    expect(quad(Math.sin, 0, Math.PI).evaluations).toBe(15);
    expect(quad((x) => x ** 3, 0, 1).evaluations).toBe(15);
  });
});

describe('quad - divergent integrals', () => {
  it('flags 1/x^2 and x^-1.01 as probably divergent', () => {
    for (const f of [(x: number) => 1 / (x * x), (x: number) => x ** -1.01]) {
      const res = quad(f, 0, 1, { atol: 0, rtol: 1e-8 });
      expect(res.success).toBe(false);
      expect(res.message).toMatch(/probably divergent/);
    }
  });

  it('flags 1/x as failed, with the limit message', () => {
    // QUADPACK stops 1/x on the subinterval limit; the value is meaningless anyway.
    const res = quad((x) => 1 / x, 0, 1);
    expect(res.success).toBe(false);
    expect(res.message).toBe('Maximum number of subintervals (200) reached.');
  });

  it('flags a divergent singularity at a breakpoint', () => {
    const res = quad((x) => 1 / (x - 0.5) ** 2, 0, 1, { breakpoints: [0.5] });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/probably divergent/);
    const res2 = quad((x) => 1 / Math.abs(x - 0.5), 0, 1, { breakpoints: [0.5] });
    expect(res2.success).toBe(false);
  });

  it('flags a divergent interior singularity as bad integrand behaviour', () => {
    // 1/(1-x): non-integrable singularity at the upper limit. With a larger budget, all the
    // bisection ends in the test on the width of the subinterval.
    const res = quad((x) => 1 / (1 - x), 0, 1, { maxSubintervals: 1000 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Bad integrand behaviour/);
  });
});

describe('quad - breakpoints, transforms and infinite limits with extrapolation', () => {
  it('handles a jump with and without a breakpoint', () => {
    const f = (x: number) => (x < 1 / 3 ? 1 : 2);
    const withBreakpoint = quad(f, 0, 1, { breakpoints: [1 / 3] });
    const without = quad(f, 0, 1);
    expect(withBreakpoint.success).toBe(true);
    expect(withBreakpoint.value).toBeCloseTo(5 / 3, 14);
    expect(withBreakpoint.subintervals).toBe(2);
    expect(withBreakpoint.evaluations).toBe(30);
    expect(without.success).toBe(true);
    expect(without.value).toBeCloseTo(5 / 3, 13);
    expect(without.evaluations).toBeGreaterThan(withBreakpoint.evaluations);
  });

  it('handles singularities at the breakpoints', () => {
    const f = (x: number) => Math.abs(x - 0.3) ** -0.5 + Math.abs(x - 0.7) ** -0.5;
    const exact = 4 * (Math.sqrt(0.3) + Math.sqrt(0.7));
    const res = quad(f, 0, 1, { atol: 0, rtol: 1e-10, breakpoints: [0.3, 0.7] });
    expect(res.success).toBe(true);
    expect(Math.abs(res.value - exact)).toBeLessThan(1e-10 * exact);
    // Reversed limits give the opposite sign.
    const rev = quad(f, 1, 0, { atol: 0, rtol: 1e-10, breakpoints: [0.7, 0.3] });
    expect(rev.value).toBe(-res.value);
  });

  it('shares one subinterval limit among the initial subintervals', () => {
    const f = (x: number) => Math.abs(x - 1 / 3) ** -0.5;
    const res = quad(f, 0, 1, { atol: 0, rtol: 1e-10, maxSubintervals: 3 });
    expect(res.success).toBe(false);
    expect(res.subintervals).toBe(3);
    expect(res.message).toBe('Maximum number of subintervals (3) reached.');
    expect(Number.isFinite(res.value)).toBe(true);
  });

  it('integrates infinite intervals', () => {
    const exp = quad((x) => Math.exp(-x), 0, Infinity);
    expect(exp.success).toBe(true);
    expect(exp.value).toBeCloseTo(1, 10);
    const lorentz = quad((x) => 1 / (1 + x * x), 0, Infinity);
    expect(lorentz.success).toBe(true);
    expect(lorentz.value).toBeCloseTo(Math.PI / 2, 9);
    const gauss = quad((x) => Math.exp(-x * x), -Infinity, Infinity);
    expect(gauss.success).toBe(true);
    expect(gauss.value).toBeCloseTo(Math.sqrt(Math.PI), 8);
    const lower = quad((x) => (-x) ** -0.5 * Math.exp(x), -Infinity, 0, { atol: 0, rtol: 1e-8 });
    expect(lower.success).toBe(true);
    expect(lower.value).toBeCloseTo(Math.sqrt(Math.PI), 8);
  });

  it('integrates endpoint singularities of infinite intervals: x^-1/2 exp(-x) and x^-0.9 exp(-x)', () => {
    // 285 evaluations for x^-1/2 exp(-x) at rtol 1e-6 (1005 with the plain engine).
    const half = quad((x) => x ** -0.5 * Math.exp(-x), 0, Infinity, { atol: 0, rtol: 1e-6 });
    expect(half.success).toBe(true);
    expect(half.evaluations).toBeLessThanOrEqual(300);
    expect(half.value).toBeCloseTo(Math.sqrt(Math.PI), 5);

    const tight = quad((x) => x ** -0.5 * Math.exp(-x), 0, Infinity, { atol: 0, rtol: 1e-10 });
    expect(tight.success).toBe(true);
    expect(tight.value).toBeCloseTo(Math.sqrt(Math.PI), 9);

    // Gamma(0.1)
    const gamma = 9.513507698668732;
    const res = quad((x) => x ** -0.9 * Math.exp(-x), 0, Infinity, { atol: 0, rtol: 1e-6 });
    expect(res.success).toBe(true);
    expect(Math.abs(res.value - gamma)).toBeLessThan(1e-6 * gamma);
  });

  it('combines a user transform with extrapolation', () => {
    // x = t^2 removes the 1/sqrt(x) singularity: the integrand becomes the constant 2.
    const res = quad((x) => 1 / Math.sqrt(x), 0, 1, {
      transform: { map: (t) => ({ x: t * t, dxDt: 2 * t }), inverse: (x) => Math.sqrt(x) }
    });
    expect(res.success).toBe(true);
    expect(res.value).toBeCloseTo(2, 13);
    expect(res.evaluations).toBe(15);
  });
});

describe('quad - the termination status is mapped to success and message', () => {
  const status = (f: (x: number) => number, a: number, b: number, o = {}) =>
    extrapolatedIntegral('quad', f, [a, b], {
      atol: 0,
      rtol: 1e-8,
      maxSubintervals: 200,
      ...o
    });

  it('0: success only when the accuracy was achieved', () => {
    const r = status(Math.sin, 0, Math.PI);
    expect(r.status).toBe(0);
    const res = quad(Math.sin, 0, Math.PI);
    expect(res.success).toBe(true);
    expect(res.message).toBe('Estimated error is at most `atol + rtol*|value|`.');
  });

  it('1: maximum number of subintervals', () => {
    const f = (x: number) => Math.abs(x - 1 / 3) ** -0.5;
    expect(status(f, 0, 1, { maxSubintervals: 3 }).status).toBe(1);
    const res = quad(f, 0, 1, { atol: 0, rtol: 1e-10, maxSubintervals: 3 });
    expect(res.success).toBe(false);
    expect(res.message).toBe('Maximum number of subintervals (3) reached.');
  });

  it('2: roundoff error prevents the requested accuracy', () => {
    expect(status(Math.sin, 0, Math.PI, { rtol: 1e-15 }).status).toBe(2);
    const res = quad(Math.sin, 0, Math.PI, { atol: 0, rtol: 1e-15 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Roundoff error detected: /);
    // The value is still the best estimate.
    expect(res.value).toBeCloseTo(2, 13);
  });

  it('3: bad integrand behaviour', () => {
    const r = status((x) => 1 / (1 - x), 0, 1, { maxSubintervals: 1000 });
    expect(r.status).toBe(3);
    expect(r.endpointSingularity).toBe(false);
    const res = quad((x) => 1 / (1 - x), 0, 1, { maxSubintervals: 1000 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Bad integrand behaviour/);
  });

  it('3: bisection reaches floating-point resolution at a non-finite endpoint', () => {
    const r = status((x) => 1 / (1.5 - x), 0, 1.5, { rtol: 1e-12 });
    expect(r.status).toBe(3);
    expect(r.endpointSingularity).toBe(true);
    const res = quad((x) => 1 / (1.5 - x), 0, 1.5, { atol: 0, rtol: 1e-12 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Bisection reached floating-point resolution/);
  });

  it('4: roundoff error in the extrapolation', () => {
    const f = (x: number) => x ** -0.9 * Math.exp(-x);
    const res = quad(f, 0, Infinity, { atol: 0, rtol: 1e-12 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Roundoff error detected in the extrapolation/);
    expect(res.value).toBeCloseTo(9.513507698668732, 8);
  });

  it('5: probably divergent', () => {
    const r = status((x) => 1 / (x * x), 0, 1);
    expect(r.status).toBe(5);
    const res = quad((x) => 1 / (x * x), 0, 1);
    expect(res.success).toBe(false);
    expect(res.message).toBe('The integral is probably divergent, or converges too slowly.');
  });

  it('handles maxSubintervals equal to the number of initial subintervals', () => {
    // No room to bisect: fine if the first integration already meets the tolerance, "limit
    // reached" otherwise (also with a single initial subinterval).
    expect(quad(Math.sin, 0, Math.PI, { maxSubintervals: 1 }).success).toBe(true);
    const hard = quad((x) => Math.sin(100 * x), 0, 3, { maxSubintervals: 1 });
    expect(hard.success).toBe(false);
    expect(hard.message).toBe('Maximum number of subintervals (1) reached.');
    expect(hard.subintervals).toBe(1);
    expect(Number.isFinite(hard.value)).toBe(true);
  });
});

describe('quad - tolerances at the roundoff level', () => {
  const integrands: [string, (x: number) => number, number, number][] = [
    ['sin', Math.sin, 0, Math.PI],
    ['x^2', (x) => x * x, 0, 1],
    ['sin(100x)', (x) => Math.sin(100 * x), 0, 1],
    ['runge', (x) => 1 / (1 + 25 * x * x), -1, 1],
    ['1/sqrt(x)', (x) => 1 / Math.sqrt(x), 0, 1],
    ['exp(-x) on [0, Infinity)', (x) => Math.exp(-x), 0, Infinity],
    ['kink', (x) => Math.abs(x - 1 / 3), 0, 1],
    ['sin(1/x)', (x) => Math.sin(1 / x), 0, 1]
  ];
  const tolerances: [number, number][] = [
    [0, 1e-12],
    [0, 1e-14],
    [0, 1e-15],
    [0, 0],
    [1e-14, 0],
    [0, Number.MIN_VALUE]
  ];

  for (const [name, f, a, b] of integrands) {
    it(`terminates with a finite result and a message for ${name}`, () => {
      for (const [atol, rtol] of tolerances) {
        const res = quad(f, a, b, { atol, rtol, maxSubintervals: 200 });
        const where = `${name} atol=${atol} rtol=${rtol}`;
        expect(Number.isFinite(res.value), where).toBe(true);
        expect(Number.isFinite(res.error), where).toBe(true);
        expect(res.subintervals, where).toBeLessThanOrEqual(200);
        expect(res.message.length, where).toBeGreaterThan(0);
        // success is only claimed if the estimate meets the requested tolerance
        if (res.success)
          expect(res.error, where).toBeLessThanOrEqual(atol + rtol * Math.abs(res.value));
        // otherwise the reason is the roundoff level or the limit, never a crash
        else expect(res.message, where).toMatch(/^(Roundoff error detected|Maximum number)/);
      }
    });
  }

  it('reports roundoff for atol = rtol = 0, unless the integral is exactly zero', () => {
    const res = quad(Math.sin, 0, Math.PI, { atol: 0, rtol: 0 });
    expect(res.success).toBe(false);
    expect(res.message).toMatch(/^Roundoff error detected/);
    expect(res.value).toBeCloseTo(2, 13);

    const zero = quad(() => 0, 0, 1, { atol: 0, rtol: 0 });
    expect(zero.success).toBe(true);
    expect(zero.value).toBe(0);
    expect(zero.error).toBe(0);
  });

  it('reaches 1e-12 on smooth and singular integrands that are far from the roundoff floor', () => {
    for (const f of [(x: number) => x * x, (x: number) => 1 / Math.sqrt(x), Math.sin]) {
      const res = quad(f, 0, 1, { atol: 0, rtol: 1e-12 });
      expect(res.success).toBe(true);
    }
  });

  it('can reach an absolute tolerance of the order of the roundoff floor', () => {
    const res = quad((x) => x * x, 0, 1, { atol: 1e-14, rtol: 0 });
    expect(res.success).toBe(true);
    expect(res.value).toBeCloseTo(1 / 3, 15);
  });
});
