import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
    {
        ignores: ['dist/**', 'docs/**', 'node_modules/**'],
    },
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
    {
        // Browser-side files: the playground and the TypeDoc customization.
        files: ['playground/**/*.js', 'docs-assets/**/*.js'],
        languageOptions: {
            globals: Object.fromEntries(
                [
                    'Blob', 'CompressionStream', 'DecompressionStream', 'Option', 'Response',
                    'TextDecoder', 'TextEncoder', 'URL', 'URLSearchParams', 'Worker', 'atob',
                    'btoa', 'clearTimeout', 'console', 'document', 'fetch', 'history',
                    'location', 'matchMedia', 'navigator', 'performance', 'self',
                    'setTimeout', 'window',
                ].map((name) => [name, 'readonly']),
            ),
        },
    },
    {
        files: ['scripts/**/*.mjs'],
        languageOptions: { globals: { console: 'readonly', process: 'readonly' } },
    },
    {
        rules: {
            'no-loss-of-precision': 'off',
            'no-useless-assignment': 'off',
            '@typescript-eslint/no-explicit-any': 'off',
            '@typescript-eslint/no-unused-vars': 'off',
        },
    },
);