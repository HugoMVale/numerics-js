# Agent Instructions

## Project at a glance

- `numerics-js` is a dependency-free TypeScript library of numerical methods.
- It publishes native ESM only. Source is in `src/`; compiled artifacts are in `dist/`.
- Tests are in `tests/`, organized to mirror the source modules.
- `docs/` is generated API documentation. Do not edit it by hand; regenerate it with `npm run docs` when documentation output must change.

## Commands

- `npm test` runs the complete Vitest suite.
- `npm run typecheck` runs TypeScript without emitting files.
- `npm run build` compiles to `dist/` and fixes emitted ESM imports.
- `npm run docs` regenerates the TypeDoc site in `docs/`.

## Independent numerical benchmarks

- Some algorithms have a companion test that checks results against an independent reference implementation (e.g. SciPy) instead of only hand-derived closed-form values, guarding against correct-looking-but-wrong numerics.
- Reference values are generated offline by a PEP 723 standalone Python script under `scripts/benchmarks/` (e.g. `generate_gauss_kronrod_fixtures.py`), run via `uv run scripts/benchmarks/<script>.py`. `uv` resolves the script's inline `dependencies` into an ephemeral, cached environment, so no `pyproject.toml`/`requirements.txt` or persistent Python env is added to the repo.
- Each script writes a JSON fixture to `tests/<module>/fixtures/`, pairing an integrand/case id with the reference value and its reported error estimate.
- A matching Vitest file (e.g. `gaussKronrod.scipy.test.ts`) loads the fixture and re-implements each case's function in TypeScript by id, then asserts the library result agrees with the reference within a tolerance derived from both sides' error estimates.
- The Python script and the TS test's integrand map must stay in sync (same id, math, and bounds). When adding or changing benchmark cases, update both, then rerun the generation script to refresh the fixture before rerunning the test.
- This is a generation-time tool only; Python/SciPy are never required to run `npm test` or build.

Use the narrowest relevant test while iterating, then run `npm run typecheck` and the relevant test suite before completing a code change.

## Source and public API

- Top-level module barrels are in `src/*.ts`; `src/index.ts` is the package-root API.
- Keep `package.json`'s `exports` map, source barrels, and `src/index.ts` aligned whenever a new public module or subpath is intended.
- Do not expose a symbol publicly unless the request or established adjacent API makes that intent clear. Add or update public-API tests for export-surface changes.
- Preserve the package's existing API style: module namespaces at the package root and direct exports from focused module subpaths.

## Numerical code expectations

- Prefer clear, allocation-conscious loops and reusable working buffers in hot solver paths.
- Prefer `Array1D`, `Matrix`, and `Vec3` for numerical operations and inputs that represent vectors, matrices, or three-dimensional vectors, rather than raw JavaScript arrays. Where an API accepts array-like input, prefer the corresponding library array type unless accepting plain arrays is intentional for ergonomics or interoperability.
- Preserve documented mutability contracts. In particular, methods ending in `Self` mutate their instance; other `Array1D` and `Vec3` operations return new values.
- `Array1D` and `Matrix` public accessors and exposed `data` storage use zero-based indexing.
- Numerical changes need targeted tests covering expected values, boundary conditions, invalid inputs, and convergence or iteration-limit behavior where applicable.
- Do not silently alter tolerance defaults, convergence criteria, error handling, callback signatures, output shapes, or indexing conventions.

## Docstring rules

- Use TypeDoc-compatible JSDoc (`/** ... */`) for exported functions, classes, interfaces, constants, and public methods or accessors.
- Start with a concise description of the operation or mathematical method and its observable behavior. For numerical algorithms, explain important guarantees, limitations, special cases, and fallback behavior that callers need to know.
- Document every parameter with `@param`, including accepted shapes, constraints, defaults, and meaningful units or conventions. Use `@returns` for non-void results and describe the result's shape or semantics.
- Add `@throws {ErrorType}` for documented validation or failure paths. Keep the documented errors aligned with the implementation.
- Include a compact, runnable `@example` for new exported user-facing algorithms or types when one clarifies normal use. Include expected output when it makes the result unambiguous.
- Document public classes at the class declaration and their constructor parameters, public methods, and public accessors. Document interfaces and non-obvious public properties, including cache/indexing conventions.
- Document private helpers when their mathematical role, performance purpose, or non-obvious behavior is not clear from the name and implementation. Do not add boilerplate comments to straightforward code.
- Keep documentation accurate when changing behavior; update README prose and generated TypeDoc only when the public contract or published documentation needs to change.

## Working rules

- Read the owning implementation and its neighboring tests before editing. Follow local naming, formatting, and assertion patterns.
- Keep changes small and scoped. Do not refactor unrelated modules or regenerate build/docs output unless needed for the requested change.
- Do not revert or overwrite existing changes that are unrelated to the task.
- Use `.js` extensions in relative ESM imports when that is the existing source convention.

## When to ask instead of guessing

Ask the user before proceeding when any of these is unclear:

- Whether a new symbol is public, which import path should expose it, or whether a breaking API change is acceptable.
- The intended mathematical definition, units, domain, sign/indexing convention, tolerance, or error behavior of a numerical routine.
- Whether ambiguous invalid input should throw, warn, clamp, return a sentinel value, or follow another policy.
- Whether generated `dist/` or `docs/` files should be updated and included.
- The scope of a request that could reasonably mean a behavior fix, a performance change, or an API redesign.

State the specific uncertainty and the options it affects. Do not invent semantics merely to make an implementation complete.
