import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
    {
        ignores: ['dist/**', 'docs/**', 'node_modules/**'],
    },
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
    {
        rules: {
            'no-loss-of-precision': 'off',
            'no-useless-assignment': 'off',
            '@typescript-eslint/no-explicit-any': 'off',
            '@typescript-eslint/no-unused-vars': 'off',
        },
    },
);