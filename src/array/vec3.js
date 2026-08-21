/**
 * A 3-component vector, used for positions, directions, velocities, etc.
 * All operations that produce a new vector are non-mutating (they return
 * a new Vec3 instance rather than modifying `this`).
 */
export class Vec3 {
    /**
     * @param {number} [x=0] - X component.
     * @param {number} [y=0] - Y component.
     * @param {number} [z=0] - Z component.
     */
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * Adds another vector to this one.
     * @param {Vec3} v - The vector to add.
     * @returns {Vec3} A new vector equal to `this + v`.
     */
    add(v) { return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z); }

    /**
     * Subtracts another vector from this one.
     * @param {Vec3} v - The vector to subtract.
     * @returns {Vec3} A new vector equal to `this - v`.
     */
    sub(v) { return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z); }

    /**
     * Scales this vector by a scalar.
     * @param {number} s - The scale factor.
     * @returns {Vec3} A new vector equal to `this * s`.
     */
    mult(s) { return new Vec3(this.x * s, this.y * s, this.z * s); }

    /**
     * Computes the squared magnitude (length) of this vector.
     * Cheaper than `norm()` since it avoids a square root — useful for
     * comparisons where the exact length isn't needed.
     * @returns {number} The squared length of the vector.
     */
    normSq() { return this.x * this.x + this.y * this.y + this.z * this.z; }

    /**
     * Computes the magnitude (length) of this vector.
     * @returns {number} The length of the vector.
     */
    norm() { return Math.sqrt(this.normSq()); }

    /**
     * Returns a unit-length version of this vector (same direction, length 1).
     * If this vector has zero length, returns a zero vector instead of
     * dividing by zero.
     * @returns {Vec3} The normalized vector, or `(0, 0, 0)` if this vector is zero.
     */
    normalize() {
        const m = this.norm();
        return m === 0 ? new Vec3() : this.mult(1 / m);
    }

    /**
     * Computes the dot product of this vector with another.
     * @param {Vec3} v - The other vector.
     * @returns {number} The scalar dot product `this · v`.
     */
    dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; }

    /**
     * Computes the Euclidean distance between this vector and another.
     * @param {Vec3} v - The other vector.
     * @returns {number} The distance between `this` and `v`.
     */
    dist(v) { return this.sub(v).norm(); }

    /**
     * Returns a copy of this vector clamped to a maximum magnitude, preserving
     * direction. If the vector's magnitude is already at or below `max`,
     * returns an unchanged copy.
     * @param {number} max - The maximum allowed magnitude (must be >= 0).
     * @returns {Vec3} A new vector with magnitude at most `max`.
     */
    limit(max) {
        const m = this.norm();
        return m > max ? this.mult(max / m) : this.copy();
    }

    /**
     * Creates an independent copy of this vector.
     * @returns {Vec3} A new Vec3 with the same x, y, z values.
     */
    copy() { return new Vec3(this.x, this.y, this.z); }

    /**
     * Checks whether this vector is elementwise close to another, modeled on
     * `numpy.isclose`: a component `a` is close to `b` if
     * `|a - b| <= atol + rtol * |b|`.
     * @param {Vec3} v - The other vector.
     * @param {number} [rtol=1e-5] - Relative tolerance.
     * @param {number} [atol=1e-8] - Absolute tolerance.
     * @returns {boolean} `true` if all components of `this` are close to `v`'s.
     */
    isClose(v, rtol = 1e-5, atol = 1e-8) {
        return Math.abs(this.x - v.x) <= atol + rtol * Math.abs(v.x) &&
            Math.abs(this.y - v.y) <= atol + rtol * Math.abs(v.y) &&
            Math.abs(this.z - v.z) <= atol + rtol * Math.abs(v.z);
    }

    /**
     * Returns this vector's components as a plain array.
     * @returns {number[]} `[x, y, z]`.
     */
    toArray() { return [this.x, this.y, this.z]; }

    /**
     * Returns a human-readable string representation of this vector.
     * @returns {string} e.g. `"Vec3(1, 2, 3)"`.
     */
    toString() { return `Vec3(${this.x}, ${this.y}, ${this.z})`; }

    /**
     * Makes Vec3 iterable, e.g. `const [x, y, z] = someVec3;`.
     * @returns {Iterator<number>}
     */
    [Symbol.iterator]() { return [this.x, this.y, this.z][Symbol.iterator](); }

    // -----------------------------------------------------------------
    // In-place (mutating) operations.
    //
    // Only use these on local accumulators or scratch vectors that are
    // never aliased elsewhere (i.e. nothing else holds a reference to the
    // same object expecting it to stay constant). Mutating a shared or
    // unexpectedly-referenced vector is a classic, hard-to-spot source of
    // physics bugs — when in doubt, use the immutable methods instead.
    // -----------------------------------------------------------------

    /**
     * Sets this vector's components directly, mutating it in place.
     * @param {number} x
     * @param {number} y
     * @param {number} [z=0]
     * @returns {Vec3} `this`, for chaining.
     */
    set(x, y, z = 0) { this.x = x; this.y = y; this.z = z; return this; }

    /**
     * Resets this vector to `(0, 0, 0)` in place.
     * @returns {Vec3} `this`, for chaining.
     */
    reset() { return this.set(0, 0, 0); }

    /**
     * Adds another vector to this one in place: `this += v`.
     * @param {Vec3} v - The vector to add.
     * @returns {Vec3} `this`, for chaining.
     */
    addSelf(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }

    /**
     * Subtracts another vector from this one in place: `this -= v`.
     * @param {Vec3} v - The vector to subtract.
     * @returns {Vec3} `this`, for chaining.
     */
    subSelf(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }

    /**
     * Scales this vector in place: `this *= s`.
     * @param {number} s - The scale factor.
     * @returns {Vec3} `this`, for chaining.
     */
    multSelf(s) { this.x *= s; this.y *= s; this.z *= s; return this; }

    /**
     * Adds a scaled vector to this one in place, in a single pass and
     * without an intermediate vector: `this += v * s`.
     * @param {Vec3} v - The vector to scale and add.
     * @param {number} s - The scale factor applied to `v`.
     * @returns {Vec3} `this`, for chaining.
     */
    addScaled(v, s) { this.x += v.x * s; this.y += v.y * s; this.z += v.z * s; return this; }

    /**
     * Sets this vector to `a - b` in place, without allocating. Useful as a
     * reusable scratch vector inside a loop, e.g.
     * `scratch.subVectors(other.pos, bird.pos)` instead of
     * `other.pos.sub(bird.pos)`.
     * @param {Vec3} a
     * @param {Vec3} b
     * @returns {Vec3} `this`, set to `a - b`.
     */
    subVectors(a, b) { this.x = a.x - b.x; this.y = a.y - b.y; this.z = a.z - b.z; return this; }

    /**
     * Creates a zero vector `(0, 0, 0)`.
     * @returns {Vec3} A new zero vector.
     */
    static zero() { return new Vec3(); }

    /**
     * Creates a Vec3 from an array or array-like of `[x, y, z]`.
     * @param {number[]|Float64Array} arr - Source values; missing entries default to 0.
     * @returns {Vec3} A new vector.
     */
    static from(arr) { return new Vec3(arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0); }
}
