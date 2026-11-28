// Curated starter programs for the playground dropdown.
// The first entry is the default. Keep each one self-contained and runnable.
export const EXAMPLES = [
  {
    title: 'Pendulum period (ODE events + quadrature + root finding)',
    code: `import { integrate, linalg, ode, roots } from 'numerics-js';

const g = 9.81; // m/s²
const L = 1; // m

// Period from the equation of motion θ'' = -(g/L) sin θ, for a pendulum released from rest at θ0.
// The solver locates the first crossing of θ = 0, a quarter period, as a terminal event.
function periodFromOde(theta0: number): number {
    const result = ode.rungeKuttaAdaptive(
        'rk45',
        (_t, y, dydt) => dydt.set([y.data[1], -(g / L) * Math.sin(y.data[0])]),
        0,
        10,
        new linalg.Vector([theta0, 0]),
        {
            rtol: 1e-10,
            atol: 1e-12,
            events: [{ fn: (_t, y) => y.data[0], direction: 'falling', terminal: true }],
        },
    );
    return 4 * result.events![0].t;
}

// Exact period, 4 √(L/g) K(k), where K(k) = ∫₀^{π/2} dφ / √(1 − k² sin²φ) is the complete
// elliptic integral of the first kind and k = sin(θ0/2).
function periodFromIntegral(theta0: number): number {
    const k = Math.sin(theta0 / 2);
    const K = integrate.quad((phi) => 1 / Math.sqrt(1 - (k * Math.sin(phi)) ** 2), 0, Math.PI / 2);
    return 4 * Math.sqrt(L / g) * K.value;
}

const T0 = 2 * Math.PI * Math.sqrt(L / g); // small-angle period
const theta0 = Math.PI / 3; // 60°

console.log('small-angle period:', T0);
console.log('60° period, ODE:   ', periodFromOde(theta0));
console.log('60° period, quad:  ', periodFromIntegral(theta0));

// Which amplitude makes the period 10 % longer than the small-angle one?
const amplitude = roots.brent((theta) => periodFromIntegral(theta) / T0 - 1.1, 0.1, 3);
const degrees = (amplitude.x * 180) / Math.PI;
console.log('amplitude for +10 % (deg):', degrees.toFixed(2));
`
  },
  {
    title: 'Find a root (Brent)',
    code: `import { brent } from 'numerics-js/roots';

// Solve x^3 - 2x - 5 = 0 on [2, 3]
const result = brent((x: number) => x ** 3 - 2 * x - 5, 2, 3);
console.log(result);
`
  },
  {
    title: 'Integrate a function (adaptive quadrature)',
    code: `import { quad } from 'numerics-js/integrate';

// Integral of exp(-x^2) over the real line = sqrt(pi)
const result = quad((x: number) => Math.exp(-x * x), -Infinity, Infinity);
console.log(result.value, Math.sqrt(Math.PI));
`
  },
  {
    title: 'Minimize a function (Nelder-Mead)',
    code: `import { nelderMead } from 'numerics-js/optimize';
import { Vector } from 'numerics-js/linalg';

// Rosenbrock function, minimum at (1, 1)
const rosenbrock = (x: Vector): number =>
    (1 - x.get(0)) ** 2 + 100 * (x.get(1) - x.get(0) ** 2) ** 2;

const result = nelderMead(rosenbrock, new Vector([-1.2, 1]));
console.log(result);
`
  },
  {
    title: '2D heat equation (stiff ODE with a banded Jacobian)',
    code: `import { ode, linalg } from 'numerics-js';

// 2D heat equation u_t = 0.01 (u_xx + u_yy) on the unit square with u = 0 on the boundary and
// u(x, y, 0) = sin(πx) sin(πy), by the method of lines on an m × m grid of interior points.
// With the grid numbered row by row, the five-point stencil gives a banded Jacobian with
// kl = ku = m: a dense Jacobian would be 2500 × 2500 (50 MB), the band is 101 × 2500.
const m = 50;
const n = m * m;
const h = 1 / (m + 1);
const k = 0.01 / (h * h);

const f: ode.DerivativeFunction = (_t, y, out) => {
    for (let r = 0; r < m; r++) {
        for (let c = 0; c < m; c++) {
            const i = r * m + c;
            const left = c > 0 ? y.data[i - 1] : 0;
            const right = c < m - 1 ? y.data[i + 1] : 0;
            const down = r > 0 ? y.data[i - m] : 0;
            const up = r < m - 1 ? y.data[i + m] : 0;
            out.data[i] = k * (left + right + down + up - 4 * y.data[i]);
        }
    }
    return out;
};

const jac: ode.BandedJacobianFunction = (_t, _y, J) => {
    for (let r = 0; r < m; r++) {
        for (let c = 0; c < m; c++) {
            const i = r * m + c;
            J.set(i, i, -4 * k);
            if (c > 0) J.set(i, i - 1, k);
            if (c < m - 1) J.set(i, i + 1, k);
            if (r > 0) J.set(i, i - m, k);
            if (r < m - 1) J.set(i, i + m, k);
        }
    }
    return J;
};

const shape = (r: number, c: number) => Math.sin(Math.PI * (r + 1) * h) * Math.sin(Math.PI * (c + 1) * h);
const u0 = new linalg.Vector(n);
for (let i = 0; i < n; i++) u0.data[i] = shape(Math.floor(i / m), i % m);

const t0 = performance.now();
const result = ode.trBdf2Adaptive(f, 0, 1, u0, { atol: 1e-8, rtol: 1e-6, kl: m, ku: m, jac });
const ms = performance.now() - t0;

// The exact solution keeps its shape and decays like exp(-2 π² 0.01 t).
const decay = Math.exp(-2 * Math.PI ** 2 * 0.01);
let maxError = 0;
for (let i = 0; i < n; i++) {
    const exact = decay * shape(Math.floor(i / m), i % m);
    maxError = Math.max(maxError, Math.abs(result.y.get(result.y.rows - 1, i) - exact));
}
console.log('unknowns:', n, 'steps:', result.t.size - 1, 'f evaluations:', result.evaluations);
console.log('max error vs exact:', maxError);
console.log('time (ms):', Math.round(ms));
`
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
`
  }
];
