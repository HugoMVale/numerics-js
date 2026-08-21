# numerics-js

A lightweight JavaScript library for numerical methods and common mathematical utilities.

## Project structure

```text
numerics-js/
├── src/
│   ├── algorithms/
│   │   ├── rootFinding.js
│   │   └── statistics.js
│   ├── index.js
│   └── utils/
│       └── validation.js
├── tests/
│   ├── rootFinding.test.js
│   └── statistics.test.js
├── package.json
├── README.md
├── .gitignore
└── LICENSE
```

## Features

- Root-finding algorithms such as bisection and Newton-Raphson
- Statistical helpers like mean, variance, and standard deviation
- Small and extensible API for adding more numerical methods

## Getting started

```bash
npm install
npm test
```

## Example

```js
import { bisection, mean } from 'numerics-js';

const root = bisection((x) => x * x - 2, 1, 2);
const average = mean([1, 2, 3, 4]);

console.log(root);    // ~1.41421356
console.log(average); // 2.5
```
