export const MODULES: string[];

/** Whitespace-insensitive key under which an example is looked up. */
export function normalize(code: string): string;

export interface ExtractedExample {
    /** Source file, e.g. `src/roots/brent.ts`. */
    file: string;
    /** The example as written in the `@example` block. */
    code: string;
    /** Self-contained version: imports added, earlier statements pulled in. */
    program: string;
    /** All TypeScript diagnostics of `program`, as `line N: message`. */
    diagnostics: string[];
    /** True if `program` uses a name or module that does not resolve. */
    unresolved: boolean;
}

export function collectExamples(): Promise<ExtractedExample[]>;
