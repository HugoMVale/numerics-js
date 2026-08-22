import { describe, it, expect } from 'vitest';
import { Vec3 } from '../../src/index';

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
    });

    it('performs mutating operations correctly', () => {
        const v: Vec3 = new Vec3(1, 1, 1);
        v.addSelf(new Vec3(2, 2, 2));
        expect(v.toArray()).toEqual([3, 3, 3]);
    });

    it('calculates norm and normalizes', () => {
        const v: Vec3 = new Vec3(0, 3, 4);
        expect(v.normSq()).toBe(25);
        expect(v.norm()).toBe(5);

        const unit: Vec3 = v.normalize();

        expect(unit.x).toBeCloseTo(0);
        expect(unit.y).toBeCloseTo(0.6);
        expect(unit.z).toBeCloseTo(0.8);
    });
});
