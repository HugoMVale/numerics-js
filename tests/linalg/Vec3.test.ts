import { describe, it, expect } from 'vitest';
import { Vec3 } from '../../src/linalg/Vec3.js';

describe('Vec3', () => {
    it('initializes with default values or provided components', () => {
        const v1: Vec3 = new Vec3();
        expect(v1.toArray()).toEqual([0, 0, 0]);

        const v2: Vec3 = new Vec3(1, 2, 3);
        expect(v2.x).toBe(1);
        expect(v2.y).toBe(2);
        expect(v2.z).toBe(3);
    });

    it('performs immutable arithmetic operations', () => {
        const v1: Vec3 = new Vec3(1, 2, 3);
        const v2: Vec3 = new Vec3(4, 5, 6);

        const sum: Vec3 = v1.add(v2);
        expect(sum.toArray()).toEqual([5, 7, 9]);
        expect(v1.toArray()).toEqual([1, 2, 3]);

        const dot: number = v1.dot(v2);
        expect(dot).toBe(1 * 4 + 2 * 5 + 3 * 6);

        const cross: Vec3 = v1.cross(v2);
        expect(cross.toArray()).toEqual([-3, 6, -3]);
        expect(v1.toArray()).toEqual([1, 2, 3]);
        expect(v2.toArray()).toEqual([4, 5, 6]);

        expect(v1.sub(v2).toArray()).toEqual([-3, -3, -3]);
        expect(v1.mult(2).toArray()).toEqual([2, 4, 6]);
    });

    it('performs mutating operations correctly', () => {
        const v: Vec3 = new Vec3(1, 1, 1);
        expect(v.addSelf(new Vec3(2, 2, 2))).toBe(v);
        expect(v.toArray()).toEqual([3, 3, 3]);

        expect(v.subSelf(new Vec3(1, 2, 3))).toBe(v);
        expect(v.toArray()).toEqual([2, 1, 0]);
        expect(v.multSelf(2)).toBe(v);
        expect(v.toArray()).toEqual([4, 2, 0]);
        expect(v.addScaled(new Vec3(1, 2, 3), 3)).toBe(v);
        expect(v.toArray()).toEqual([7, 8, 9]);

        expect(v.set(4, 5)).toBe(v);
        expect(v.toArray()).toEqual([4, 5, 0]);
        expect(v.reset()).toBe(v);
        expect(v.toArray()).toEqual([0, 0, 0]);

        const scratch: Vec3 = new Vec3();
        expect(scratch.subVectors(new Vec3(5, 6, 7), new Vec3(2, 4, 8))).toBe(scratch);
        expect(scratch.toArray()).toEqual([3, 2, -1]);
    });

    it('calculates norm and normalizes', () => {
        const v: Vec3 = new Vec3(0, 3, 4);
        expect(v.normSq()).toBe(25);
        expect(v.norm()).toBe(5);

        const unit: Vec3 = v.normalize();

        expect(unit.x).toBeCloseTo(0);
        expect(unit.y).toBeCloseTo(0.6);
        expect(unit.z).toBeCloseTo(0.8);

        expect(new Vec3().normalize().toArray()).toEqual([0, 0, 0]);
    });

    it('computes distance, copies, and limits magnitude', () => {
        const v: Vec3 = new Vec3(3, 4, 0);
        expect(v.dist(new Vec3(0, 0, 0))).toBe(5);

        const unchanged: Vec3 = v.limit(5);
        expect(unchanged.toArray()).toEqual([3, 4, 0]);
        expect(unchanged).not.toBe(v);

        expect(v.limit(2).norm()).toBeCloseTo(2);
        expect(v.limit(2).toArray()).toEqual([expect.closeTo(1.2), expect.closeTo(1.6), 0]);

        const copy: Vec3 = v.copy();
        copy.x = 99;
        expect(v.toArray()).toEqual([3, 4, 0]);
    });

    it('checks componentwise closeness and supports string and iteration output', () => {
        const v: Vec3 = new Vec3(1, 2, 3);
        expect(v.isClose(new Vec3(1.00001, 2, 3))).toBe(true);
        expect(v.isClose(new Vec3(1.1, 2, 3))).toBe(false);
        expect(v.isClose(new Vec3(1.1, 2, 3), 0.1, 0)).toBe(true);
        expect(v.toString()).toBe('Vec3(1, 2, 3)');
        expect([...v]).toEqual([1, 2, 3]);
    });

    it('creates vectors through static constructors', () => {
        expect(Vec3.zero().toArray()).toEqual([0, 0, 0]);
        expect(Vec3.from([4, 5, 6]).toArray()).toEqual([4, 5, 6]);
        expect(Vec3.from([4]).toArray()).toEqual([4, 0, 0]);
    });
});
