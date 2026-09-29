import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { lineSearch } from '../../src/roots/lineSearch.js';
import { makeCtx } from './globalStepFixture.js';

describe('lineSearch', () => {
    it('accepts the full Newton step when it decreases the objective', () => {
        const res = lineSearch(makeCtx([-4, -3]));
        expect(res.success).toBe(true);
        expect(res.nFev).toBe(1);
        expect(res.xp.get(0)).toBeCloseTo(1, 12);
        expect(res.xp.get(1)).toBeCloseTo(2, 12);
        expect(res.fp).toBeCloseTo(0, 12);
        expect(res.Fp.size).toBe(2);
    });

    it('backtracks when the full step overshoots', () => {
        const ctx = makeCtx([-12, -9]);
        const res = lineSearch(ctx);
        expect(res.success).toBe(true);
        expect(res.nFev).toBeGreaterThan(1);
        expect(res.fp).toBeLessThan(ctx.fc);
    });

    it('truncates the step to maxLen and flags it', () => {
        const res = lineSearch(makeCtx([-4, -3], { maxLen: 1 }));
        expect(res.success).toBe(true);
        expect(res.wasMaxStep).toBe(true);
        expect(res.xp.sub(Vector.from([5, 5])).norm()).toBeCloseTo(1, 12);
    });

    it('fails without evaluating for a non-descent direction', () => {
        const ctx = makeCtx([4, 3]);
        const res = lineSearch(ctx);
        expect(res.success).toBe(false);
        expect(res.nFev).toBe(0);
        expect(res.xp.get(0)).toBe(5);
        expect(res.xp.get(1)).toBe(5);
        expect(res.fp).toBe(ctx.fc);
    });

    it('echoes trustLen unchanged (0 if unset)', () => {
        expect(lineSearch(makeCtx([-4, -3])).trustLen).toBe(0);
        expect(lineSearch(makeCtx([-4, -3], { trustLen: 3 })).trustLen).toBe(3);
    });
});
