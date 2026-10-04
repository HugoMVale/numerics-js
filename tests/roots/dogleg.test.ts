import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { dogleg } from '../../src/roots/dogleg.js';
import { makeCtx } from './globalStepFixture.js';

describe('dogleg', () => {
    it('accepts the full Newton step when it fits in the trust region', () => {
        const res = dogleg(makeCtx([-4, -3], { trustLen: 10 }));
        expect(res.success).toBe(true);
        expect(res.xp.get(0)).toBeCloseTo(1, 12);
        expect(res.xp.get(1)).toBeCloseTo(2, 12);
        expect(res.fp).toBeCloseTo(0, 12);
        expect(res.Fp.size).toBe(2);
    });

    it('shrinks the trust region when the full step overshoots', () => {
        const ctx = makeCtx([-12, -9], { trustLen: 100 });
        const res = dogleg(ctx);
        expect(res.success).toBe(true);
        expect(res.fp).toBeLessThan(ctx.fc);
        expect(res.trustLen).toBeLessThan(15);
    });

    it('truncates the step to maxLen and flags it', () => {
        const res = dogleg(makeCtx([-4, -3], { maxLen: 1 }));
        expect(res.success).toBe(true);
        expect(res.wasMaxStep).toBe(true);
        expect(res.xp.sub(Vector.from([5, 5])).norm()).toBeCloseTo(1, 12);
    });

    it('converges without moving when the step is below the tolerance', () => {
        const res = dogleg(makeCtx([1e-12, 1e-12], { trustLen: 10 }));
        expect(res.success).toBe(false);
        expect(res.xp.get(0)).toBe(5);
        expect(res.xp.get(1)).toBe(5);
    });

    it('initializes the trust region when trustLen is not positive', () => {
        const res = dogleg(makeCtx([-4, -3]));
        expect(res.success).toBe(true);
        expect(res.trustLen).toBeGreaterThan(0);
    });
});
