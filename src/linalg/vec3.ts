/**
 * A 3-component vector, used for positions, directions, velocities, etc.
 *
 * All operations that produce a new vector are non-mutating (they return
 * a new Vec3 instance rather than modifying `this`).
 */
export class Vec3 {
    x: number;
    y: number;
    z: number;

    constructor(x: number = 0, y: number = 0, z: number = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * Adds another vector to this one.
     * @param v The vector to add.
     * @returns A new vector equal to `this + v`.
     */
    add(v: Vec3): Vec3 { return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z); }

    /**
     * Subtracts another vector from this one.
     * @param v The vector to subtract.
     * @returns A new vector equal to `this - v`.
     */
    sub(v: Vec3): Vec3 { return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z); }

    /**
     * Scales this vector by a scalar.
     * @param s The scale factor.
     * @returns A new vector equal to `this * s`.
     */
    mult(s: number): Vec3 { return new Vec3(this.x * s, this.y * s, this.z * s); }

    /**
     * Computes the squared magnitude (length) of this vector.
     * Cheaper than `norm()` since it avoids a square root — useful for
     * comparisons where the exact length isn't needed.
     * @returns The squared length of the vector.
     */
    normSq(): number { return this.x * this.x + this.y * this.y + this.z * this.z; }

    /**
     * Computes the magnitude (length) of this vector.
     * @returns The length of the vector.
     */
    norm(): number { return Math.sqrt(this.normSq()); }

    /**
     * Returns a unit-length version of this vector (same direction, length 1).
     * If this vector has zero length, returns a zero vector instead of
     * dividing by zero.
     * @returns The normalized vector, or `(0, 0, 0)` if this vector is zero.
     */
    normalize(): Vec3 {
        const m = this.norm();
        return m === 0 ? new Vec3() : this.mult(1 / m);
    }

    /**
     * Computes the dot product of this vector with another.
     * @param v The other vector.
     * @returns The scalar dot product `this · v`.
     */
    dot(v: Vec3): number { return this.x * v.x + this.y * v.y + this.z * v.z; }

    /**
     * Computes the Euclidean distance between this vector and another.
     * @param v The other vector.
     * @returns The distance between `this` and `v`.
     */
    dist(v: Vec3): number { return this.sub(v).norm(); }

    /**
     * Returns a copy of this vector clamped to a maximum magnitude, preserving
     * direction. If the vector's magnitude is already at or below `max`,
     * returns an unchanged copy.
     * @param max The maximum allowed magnitude (must be >= 0).
     * @returns A new vector with magnitude at most `max`.
     */
    limit(max: number): Vec3 {
        const m = this.norm();
        return m > max ? this.mult(max / m) : this.copy();
    }

    /**
     * Creates an independent copy of this vector.
     * @returns A new Vec3 with the same x, y, z values.
     */
    copy(): Vec3 { return new Vec3(this.x, this.y, this.z); }

    /**
     * Checks whether this vector is elementwise close to another, modeled on
     * `numpy.isclose`: a component `a` is close to `b` if
     * `|a - b| <= atol + rtol * |b|`.
     * @param v The other vector.
     * @param rtol Relative tolerance.
     * @param atol Absolute tolerance.
     * @returns `true` if all components of `this` are close to `v`'s.
     */
    isClose(v: Vec3, rtol: number = 1e-5, atol: number = 1e-8): boolean {
        return Math.abs(this.x - v.x) <= atol + rtol * Math.abs(v.x) &&
            Math.abs(this.y - v.y) <= atol + rtol * Math.abs(v.y) &&
            Math.abs(this.z - v.z) <= atol + rtol * Math.abs(v.z);
    }

    /**
     * Returns this vector's components as a tuple.
     * @returns `[x, y, z]`.
     */
    toArray(): [number, number, number] { return [this.x, this.y, this.z]; }

    /**
     * Returns a human-readable string representation of this vector.
     * @returns e.g. `"Vec3(1, 2, 3)"`.
     */
    toString(): string { return `Vec3(${this.x}, ${this.y}, ${this.z})`; }

    /**
     * Makes Vec3 iterable, e.g. `const [x, y, z] = someVec3;`.
     */
    [Symbol.iterator](): IterableIterator<number> {
        return [this.x, this.y, this.z][Symbol.iterator]();
    }

    // -----------------------------------------------------------------
    // In-place (mutating) operations.
    // -----------------------------------------------------------------

    /**
     * Sets this vector's components directly, mutating it in place.
     * @returns `this`, for chaining.
     */
    set(x: number, y: number, z: number = 0): Vec3 {
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }

    /**
     * Resets this vector to `(0, 0, 0)` in place.
     * @returns `this`, for chaining.
     */
    reset(): Vec3 { return this.set(0, 0, 0); }

    /**
     * Adds another vector to this one in place: `this += v`.
     * @param v The vector to add.
     * @returns `this`, for chaining.
     */
    addSelf(v: Vec3): Vec3 { this.x += v.x; this.y += v.y; this.z += v.z; return this; }

    /**
     * Subtracts another vector from this one in place: `this -= v`.
     * @param v The vector to subtract.
     * @returns `this`, for chaining.
     */
    subSelf(v: Vec3): Vec3 { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }

    /**
     * Scales this vector in place: `this *= s`.
     * @param s The scale factor.
     * @returns `this`, for chaining.
     */
    multSelf(s: number): Vec3 { this.x *= s; this.y *= s; this.z *= s; return this; }

    /**
     * Adds a scaled vector to this one in place, in a single pass and
     * without an intermediate vector: `this += v * s`.
     * @param v The vector to scale and add.
     * @param s The scale factor applied to `v`.
     * @returns `this`, for chaining.
     */
    addScaled(v: Vec3, s: number): Vec3 { this.x += v.x * s; this.y += v.y * s; this.z += v.z * s; return this; }

    /**
     * Sets this vector to `a - b` in place, without allocating. Useful as a
     * reusable scratch vector inside a loop.
     * @returns `this`, set to `a - b`.
     */
    subVectors(a: Vec3, b: Vec3): Vec3 { this.x = a.x - b.x; this.y = a.y - b.y; this.z = a.z - b.z; return this; }

    /**
     * Creates a zero vector `(0, 0, 0)`.
     * @returns A new zero vector.
     */
    static zero(): Vec3 { return new Vec3(); }

    /**
     * Creates a Vec3 from an array or array-like of `[x, y, z]`.
     * @param arr Source values; missing entries default to 0.
     * @returns A new vector.
     */
    static from(arr: ArrayLike<number>): Vec3 {
        return new Vec3(arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0);
    }
}