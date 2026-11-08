import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    fsModuleCache: true,
    coverage: {
      include: ['src/**/*.ts'],
      reporter: ['text', 'json', 'html']
    }
  }
});
