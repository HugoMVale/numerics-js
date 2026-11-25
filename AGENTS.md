# Agent Instructions

## Project at a glance

- `numerics-js` is a TypeScript numerical-methods library with no runtime dependencies.
- It publishes native ESM only. Source is in `src/`; compiled artifacts are in `dist/`.
- Tests are in `tests/`, organized to mirror the source modules.
- `docs/` is generated API documentation. Do not edit it by hand; regenerate it with `npm run docs` when documentation output must change.

## Commands

- `npm test` runs the complete Vitest suite. Use `npm test -- --reporter=verbose` for per-example progress and failures labeled with the offending example. `tests/examples.test.ts` warms the shared checker in a `beforeAll` with a 120 s timeout so slow startup does not make the first example hit Vitest's 5 s default timeout.
- `npm run test:benchmarks` runs only the independent-reference comparisons under `benchmarks/`; these tests also run as part of `npm test`.
- `npm run lint` runs Oxlint; this also runs from the pre-commit hook.
- `npm run format` formats the project with oxfmt (configured in `.oxfmtrc.json`); `npm run format:check` verifies formatting without writing and runs in CI. Generated outputs (`dist/`, `docs/`, `coverage/`, and the benchmark JSON/Markdown reports) are excluded.
- `npm run typecheck` runs TypeScript without emitting files.
- `npm run build` compiles to `dist/` and fixes emitted ESM imports.
- `npm run docs` regenerates the TypeDoc site in `docs/`.
- `npm run docs:apps` assembles the browser playground into `docs/playground/` and the visual gallery into `docs/gallery/` (needs `dist/` and `docs/`; `npm run docs:site` does build, docs and apps in one go). Sources: `playground/`, `gallery/` and `docs-assets/playground-links.js`. The gallery is plain ES modules that import the compiled library from `docs/playground/lib/` through an import map, so serve `docs/` over HTTP to view it.
- `tests/examples.test.ts` type-checks every `@example` as a self-contained program (imports added automatically where missing, see `scripts/lib/examples.mjs`), so examples must compile against the public API. Known-broken examples are listed in that test and must be fixed, not added to.
- `tests/examples.test.ts` also runs every `@example` that has an `Output:` block against `src/` and compares what it prints with that block (about 5 s). Numbers may differ by `1e-9 * max(1, |documented|)` because V8's `Math.*`/`**` differ in the last bits between Node 22 and 24+; everything else must match character for character. Write `Output:` blocks from a real run on Node >= 24, never by hand. To refresh a block, run the example and paste what it prints.
- `npm run prepare` sets `core.hooksPath` to `.githooks`, which runs the committed pre-commit hook; package installation runs this automatically.

## Independent numerical benchmarks

- Some algorithms have companion tests that check results against independent references (e.g. SciPy or mpmath), supplementing hand-derived values and guarding against plausible-looking but incorrect numerics.
- Reference values are generated offline by PEP 723 standalone Python scripts under `benchmarks/<module>/`, run via `uv run benchmarks/<module>/<script>.py`. `uv` resolves each script's inline `dependencies` into an ephemeral, cached environment, so no `pyproject.toml`/`requirements.txt` or persistent Python env is added to the repo.
- Each benchmark directory contains its Python generator, JSON fixture, matching Vitest comparison, and optional Markdown report.
- Matching Vitest files load the fixture and compare library results with the reference using tolerances appropriate to the algorithm and reference data; some also check reported errors, documented accuracy, or performance-related counts.
- The Python script and the TS test's integrand map must stay in sync (same id, math, and bounds). When adding or changing benchmark cases, update both, then rerun the generation script to refresh the fixture before rerunning the test.
- Python and the reference packages are needed only to regenerate fixtures; they are not required to run `npm test` or build.

Use the narrowest relevant test while iterating, then run `npm run typecheck` and the relevant test suite before completing a code change.

## Source and public API

- Top-level module barrels are in `src/*.ts`; `src/index.ts` is the package-root API.
- Keep `package.json`'s `exports` map, source barrels, and `src/index.ts` aligned whenever a new public module or subpath is intended.
- Do not expose a symbol publicly unless the request or established adjacent API makes that intent clear. Add or update public-API tests for export-surface changes.
- Preserve the package's existing API style: module namespaces at the package root and direct exports from focused module subpaths.

## Numerical code expectations

- Prefer clear, allocation-conscious loops and reusable working buffers in hot solver paths.
- APIs that take vector-shaped numeric inputs use `VectorLike` and normalize inputs once with `Vector.asVector`; follow each API's documented return shape (`Vector`, `Matrix`, or scalar).
- Preserve documented mutability contracts. In particular, methods ending in `Self` mutate their instance and return it; other `Vector` and `Vec3` operations return new values. The only other mutators are the setters `set`, `setSub`, `reset` and `fill`, which also return `this`.
- `Vector` and `Matrix` public accessors and exposed `data` storage use zero-based indexing.
- Numerical changes need targeted tests covering expected values, boundary conditions, invalid inputs, and convergence or iteration-limit behavior where applicable.
- Do not silently alter tolerance defaults, convergence criteria, error handling, callback signatures, output shapes, or indexing conventions.

## Linear-algebra error and non-finite policy

Applies to `Matrix` (`Matrix.ts`, `Matrix.linalg.ts`, `Matrix.svd.ts`); `Vector` and `ArrayND` follow the error-type and message rules only.

- **Error types.** `RangeError` for bad values: non-square input where a square matrix is required, shape or size mismatches, invalid options or tolerances, a matrix too large to allocate, and any `±Infinity` entry (see below). `TypeError` for wrong types. Plain `Error` for numerical failure that depends on the data: a singular matrix, a matrix that is not positive definite, a QR iteration that does not converge. Do not add error classes without being asked.
- **Operand types.** Arguments that supply _data_ read as a vector (the vector of `mulVec`, the right-hand sides of `solve`/`solveLower`/`solveUpper`/`solveCholesky`/`lstsq`, `setRow`/`setCol` values, `Matrix.diag`, `Vector.set`) accept `VectorLike` and are normalized with `Vector.asVector`. Operands of an operation between two objects (`add`/`sub`/`mult`/`div` and their `Self` variants, `isClose`/`allClose`, `dot`, `dist`, `outer`, `addScaledSelf`, `setSub`, `matmul`) and every `out` argument must be an actual `Vector`/`Matrix` (a `Matrix` operand may also be a number or a `Vector` broadcast over the rows) and throw `TypeError` (`Vector.add: expected a Vector, got Array.`) otherwise: converting an array on every call would allocate in hot loops, and a bare array is ambiguous as a `Matrix` operand. `Vec3` is a plain value type for hot loops and does not validate its operands. Constructors and `from` take raw data (a number for a dimension, or an array-like) and reject anything else with `TypeError`.
- **Message format.** `Matrix.method: description.` (always the public method name, always a trailing period), e.g. `Matrix.solve: expected a square matrix, got 2x3.`
- **NaN propagates.** A `NaN` entry in the matrix argument is not an error and is not singularity: no throw, and every numeric result is `NaN` (`determinant`, `rank`, `cond1`, `cond2` return `NaN`; matrices, vectors and singular values are filled with `NaN`; a permutation is the identity, `sign` is `1`). This holds for `lu`, `solve`, `inverse`, `determinant`, `rank`, `cholesky`, `qr`, `qrPivot`, `svd`, `singularValues`, `pinv`, `lstsq`, `cond1`, `cond2`, `eigenvalues` and `eig`.
- **±Infinity throws.** An infinite entry in the matrix argument throws `RangeError` (`Matrix.method: matrix contains an infinite entry, ...`) from the same methods, because elimination silently turns it into plausible but wrong finite numbers. It wins over `NaN`: a matrix with both throws. The one carve-out is `cholesky`, which reads only the lower triangle (diagonal included), so it scans only that part.
- **`normalize` and infinite components.** `Vector.normalize` and `Vec3.normalize` return the limit direction (an infinite entry counts as `±1`, a finite one as `0`), which is what a huge finite entry already gives, instead of the half-`NaN` that `x / Infinity` produced. A `NaN` entry still gives an all-`NaN` result.
- **What is not scanned.** The `O(n^2)` kernels (`solveLower`, `solveUpper`, `solveLU`, `solveCholesky`, `cond1Upper`, `cond1Lower`, `qrPivot*` helpers, `qrUpdate*`) and right-hand-side vectors follow plain IEEE arithmetic; they throw only on an exactly zero diagonal entry. Callers that need a guarantee check `allFinite()` first (the solvers in `roots` and `optimize` already do).
- **Singularity.** Throw, never return a flag or `NaN`. A pivot counts as missing only if it is exactly `0` (the LAPACK `dgetrf` convention): `lu`, `solve`, `inverse`, `solveLower` and `solveUpper` throw `Error`, `determinant` returns `0`. There is no pivot tolerance in these methods: a nearly singular matrix returns finite, possibly inaccurate numbers. A finite matrix is never reported singular because of its magnitude: the elimination of `lu`, `solve`, `inverse`, `determinant`, `rank` (elimination) and `cond1` runs on the matrix divided by a power of two (`safeScale` in `Matrix.linalg.ts`) when its largest entry is outside `[2^-900, 2^900]` (ordinary matrices keep their exact bits; a matrix whose entries span the whole double range is left unscaled). `determinant` is exact to rounding whenever the true value is representable and `±Infinity`/`±0` when it is not; an entry of `inverse`, `solve` or `U` that is out of range is `±Infinity`, never `NaN`. Detecting near-singularity is the job of `rank` (with its tolerances), `cond1`/`cond2` and `svd`, which never throw for rank deficiency.
- **Scaling.** Squared norms must not saturate. `qr` and `qrPivot` give the factorization of the unscaled matrix for entries around `1e±200` (tested; `svd`, `pinv` and `lstsq` hold too, since `svd` rescales into a safe range as `dgesvd` does). Compute the 2-norm of a reflector column with `norm2`/`powerOfTwoScale` in `Matrix.linalg.ts`: a plain sum of squares while it lies in `[1e-280, 1e300]` (ordinary inputs keep their exact bits), scaled otherwise. `eigenvalues`/`eig` scale `A` by an exact power of two (`eigenScale`, only outside `[2^-100, 2^100]`, so ordinary inputs keep their bits) before the Hessenberg/QR iteration and scale the eigenvalues back (eigenvectors are unaffected); the eigenvector back-substitution uses floors relative to `T`, never absolute ones. `qrPivot` divides, instead of multiplying by `1 / (alpha - beta)`, when that reciprocal would overflow (subnormal columns). Known limit: subnormal entries carry fewer than 16 significant digits, so results for them are only as accurate as their input, and an eigenvalue far below `eps * max|A|` is subnormal or flushed to 0 when scaled back.
- **Special shapes.** Every method documents its behavior for singular, rank-deficient, non-square, 1x1 and non-finite input; matrices are never empty (`rows, cols >= 1`).

## Linear-algebra naming

- **Factorizations** are nouns named after the factorization: `lu`, `cholesky`, `qr`, `qrPivot`, `svd`.
- **Solvers** are `solve` followed by the structure being solved with: `solve` (general square `A`, factorizes internally), `solveLower` and `solveUpper` (triangular receiver), `solveLU` (a decomposition returned by `lu()`) and `solveCholesky` (the factor `L` returned by `cholesky()`). A new solver is `solve<Structure>`, never `<structure>Solve`, so the family sorts together in the API docs.
- A solver is an instance method when its receiver is an ordinary `Matrix` (`solveLower`, `solveUpper`, `solveCholesky`) and a static method when it takes a plain decomposition object (`solveLU`).

## Tolerance naming

- `atol` is an absolute tolerance on a quantity (`|x| <= atol`) and `rtol` a relative one, scaled by a reference magnitude that the docs name. An API that takes both documents how they combine (`atol + rtol * |reference|` for closeness and error control, `max(atol, rtol * reference)` for cutoffs). This covers `isClose`/`allClose`, `integrate`, `ode`, the `rank`/SVD cutoffs, the `Matrix` structural predicates (`isSymmetric`, `isLowerTriangular`, `isUpperTriangular`: `atol`, default `0` = exact) and `eigenvalues`/`eig` (`rtol`: relative deflation threshold, default `Number.EPSILON`).
- `tolx`, `tolf` and `tolg` are the stopping tests of the iterative solvers in `roots` and `optimize`, named by the quantity they test (step in `x`, residual `f`, gradient `g`); each option documents its own scaling. `epsf` is the noise level of `f` used by finite differences.
- No public API uses a bare `tol`. A local variable or internal parameter named `tol` is fine when it holds one already-resolved threshold (for example `max(atol, rtol * norm)`).
- `isClose`/`allClose` (`ArrayND`, `Vec3`, `math.isClose`) take an options object `{ atol = 1e-8, rtol = 1e-5 }` and test `|a - b| <= atol + rtol * |b|`. The `ArrayND` methods validate it (`TypeError` for a non-object such as the former positional `rtol`, or a non-number; `RangeError` for a negative or `NaN` value); `Vec3` and `math.isClose` do not validate, like their other arguments. Do not add positional tolerance pairs: take an options object with named `atol`/`rtol`.

## Internal declarations

- Tag anything that other modules need but users must not see with `@internal` (helpers, shared types, the `wrapUnchecked`/`getUnchecked`/`setUnchecked`/`flatIndex` fast paths of `Vector` and `Matrix`, the unchecked constructor overloads). TypeDoc hides it (`excludeInternal`) and the build removes it from the published `.d.ts` (`stripInternal` in `tsconfig.build.json`). The emitted JavaScript keeps it, so code in `src` uses it freely; only consumers of the declarations lose sight of it.
- A public signature must not mention an `@internal` type: the declaration that remains would import a name that no longer exists. Either make the type public or tag the whole module's exports. `tests/declarations.test.ts` emits the declarations in memory and fails on a leaked `@internal` or on such a dangling reference (TypeScript hides the latter from consumers with `skipLibCheck`, as an `any`).
- Do not mention an `@internal` member in public prose.

## Docstring rules

- Use TypeDoc-compatible JSDoc (`/** ... */`) for exported functions, classes, interfaces, constants, and public methods or accessors.
- Start with a concise description of the operation or mathematical method and its observable behavior. For numerical algorithms, explain important guarantees, limitations, special cases, and fallback behavior that callers need to know.
- For properties of exported option interfaces, state defaults with a `@default` tag (plain value or expression, no backticks) instead of "Defaults to ..." prose; keep any explanation of how the default is derived in the description.
- Document every parameter with `@param`, including accepted shapes, constraints, defaults, and meaningful units or conventions. Use `@returns` for non-void results and describe the result's shape or semantics.
- Add `@throws {ErrorType}` for documented validation or failure paths. Keep the documented errors aligned with the implementation.
- Include a compact, runnable `@example` for new exported user-facing algorithms or types when one clarifies normal use. Run the example before documenting it; show the actual observed output in an `Output:` block, preserving the presentation produced by the code (for example, a scalar, `toString()` text, a formatted template string, or the runtime's object/array representation). Never invent numeric output or rely on hand calculation. Omit the output block only when the example intentionally produces no output.
- Document public classes at the class declaration and their constructor parameters, public methods, and public accessors. Document interfaces and non-obvious public properties, including cache/indexing conventions.
- Document private helpers when their mathematical role, performance purpose, or non-obvious behavior is not clear from the name and implementation. Do not add boilerplate comments to straightforward code.
- Keep documentation accurate when changing behavior; update README prose and generated TypeDoc only when the public contract or published documentation needs to change.

## Working rules

- Read the owning implementation and its neighboring tests before editing. Follow local naming, formatting, and assertion patterns.
- Keep changes small and scoped. Do not refactor unrelated modules or regenerate build/docs output unless needed for the requested change.
- Do not revert or overwrite existing changes that are unrelated to the task.
- Use `.js` extensions in relative ESM imports when that is the existing source convention.
- Every error message created by the codebase must end with a trailing period.

## When to ask instead of guessing

Ask the user before proceeding when any of these is unclear:

- Whether a new symbol is public, which import path should expose it, or whether a breaking API change is acceptable.
- The intended mathematical definition, units, domain, sign/indexing convention, tolerance, or error behavior of a numerical routine.
- Whether ambiguous invalid input should throw, warn, clamp, return a sentinel value, or follow another policy.
- Whether generated `dist/` or `docs/` files should be updated and included.
- The scope of a request that could reasonably mean a behavior fix, a performance change, or an API redesign.

State the specific uncertainty and the options it affects. Do not invent semantics merely to make an implementation complete.
