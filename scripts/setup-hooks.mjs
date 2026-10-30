import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

// Skip outside a git checkout (published tarballs, CI copies without .git).
if (!existsSync('.git')) process.exit(0);

try {
  execFileSync('git', ['config', 'core.hooksPath', '.githooks'], { stdio: 'ignore' });
} catch {
  // git is unavailable; hooks stay unconfigured.
}
