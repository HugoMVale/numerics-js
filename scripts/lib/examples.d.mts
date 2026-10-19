export const MODULES: string[];

/** Whitespace-insensitive key under which an example is looked up. */
export function normalize(code: string): string;

export interface UncheckedExample {
  /** Source file, e.g. `src/roots/brent.ts`. */
  file: string;
  /** The example as written in the `@example` block. */
  code: string;
  /** Self-contained version: imports added, earlier statements pulled in. */
  program: string;
}

export interface CheckResult {
  /** All TypeScript diagnostics of the program, as `line N: message`. */
  diagnostics: string[];
  /** True if the program uses a name or module that does not resolve. */
  unresolved: boolean;
}

export type ExtractedExample = UncheckedExample & CheckResult;

/**
 * Creates a type-checker for self-contained example programs. Create it once and reuse it:
 * parsed library files are cached across calls, which makes later checks cheap.
 */
export function createChecker(): (program: string) => CheckResult;

/** Extracts every `@example` without type-checking it (fast). */
export function extractExamples(): Promise<UncheckedExample[]>;

/** Extracts and type-checks every `@example`. */
export function collectExamples(): Promise<ExtractedExample[]>;
