/** 
 * Common test integrands for quadrature validation.
 */
export const fPolynomial = (x: number) => x * x;
export const fSin = Math.sin;
export const fArctanDeriv = (x: number) => 4 / (1 + x * x);
export const fIdentity = (x: number) => x;
export const fOscillatory = (x: number) => Math.sin(100 * x);

// Function with an endpoint singularity (integrates to 2 over [0, 1])
export const fEndpointSingularity = (x: number) => 1 / Math.sqrt(x);

// Function designed to throw a non-finite error at exactly x = 0.5
export const fBlowupMidpoint = (x: number) => (x === 0.5 ? Infinity : 1 / x);