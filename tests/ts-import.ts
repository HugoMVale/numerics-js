import { array, ode, roots, special, version } from '../src/index.js';

const a = new array.Array1D([1, 2, 3]);
const b = new array.Array2D(2, 2, [1, 2, 3, 4]);
const v = new array.Vec3(1, 2, 3);
const value = special.bessel.J(0, 1.5);
const root = roots.bisection((x) => x * x - 2, 1, 2);
const sec = roots.secant((x) => Math.cos(x) - x, 0, 1);
const solution = ode.rk4Integrate((t, y, dydt) => dydt.set([-y.data[0]]), 0, 1, new array.Array1D([1]), 0.1);
const step = ode.createVelocityVerlet((u, v, aNext) => {
    for (let i = 0; i < aNext.length; i++) aNext[i] = -9.81;
});

void a;
void b;
void v;
void value;
void root;
void sec;
void solution;
void step;
void version;
