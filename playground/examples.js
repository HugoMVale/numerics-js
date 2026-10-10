// Curated starter programs for the playground dropdown.
// The first entry is the default. Keep each one self-contained and runnable.
export const EXAMPLES = [
    {
        title: 'Cooling model (ODE + interpolation + quadrature)',
        code: `import { linalg, integrate, interpolate, ode } from 'numerics-js';

// Newton cooling: dT/dt = -0.4 (T - 20), T(0) = 100
const solution = ode.rungeKuttaAdaptive(
    'rk45',
    (_t, y, dydt) => dydt.set([-(y.data[0] - 20) * 0.4]),
    0,
    10,
    new linalg.Vector([100]),
);

const times = solution.t;
const temperatures = solution.y.col(0);
const temperature = new interpolate.PchipInterpolator1D(times, temperatures);
const accumulatedHeat = integrate.quad((time) => temperature.eval(time) - 20, 0, 10);

console.log(temperature.eval(2.25));
console.log(accumulatedHeat.value);
`,
    },
    {
        title: 'Find a root (Brent)',
        code: `import { brent } from 'numerics-js/roots';

// Solve x^3 - 2x - 5 = 0 on [2, 3]
const result = brent((x: number) => x ** 3 - 2 * x - 5, 2, 3);
console.log(result);
`,
    },
    {
        title: 'Integrate a function (adaptive quadrature)',
        code: `import { quad } from 'numerics-js/integrate';

// Integral of exp(-x^2) over the real line = sqrt(pi)
const result = quad((x: number) => Math.exp(-x * x), -Infinity, Infinity);
console.log(result.value, Math.sqrt(Math.PI));
`,
    },
    {
        title: 'Minimize a function (Nelder-Mead)',
        code: `import { nelderMead } from 'numerics-js/optimize';
import { Vector } from 'numerics-js/linalg';

// Rosenbrock function, minimum at (1, 1)
const rosenbrock = (x: Vector): number =>
    (1 - x.get(0)) ** 2 + 100 * (x.get(1) - x.get(0) ** 2) ** 2;

const result = nelderMead(rosenbrock, [-1.2, 1]);
console.log(result);
`,
    },
    {
        title: 'Solve a linear system',
        code: `import { Matrix, Vector } from 'numerics-js/linalg';

const A = new Matrix(3, 3, [
    4, -2, 1,
    -2, 4, -2,
    1, -2, 4,
]);
const b = new Vector([11, -16, 17]);

const x = A.solve(b);
console.log(x.toString());
`,
    },
];
