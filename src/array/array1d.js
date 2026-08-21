/**
 * An N-component vector utilizing Float64Array for performance.
 */
export class Array1D {
    /**
     * @param {number|number[]|Float64Array} input - The dimension length (initialized to 0s) or initial data.
     */
    constructor(input) {
        this.data = new Float64Array(input);
        this.dim = this.data.length;
    }

    /**
     * Throws if `v` is not a Array1D of the same dimension as this one. Used
     * internally to guard binary operations against silent shape mismatches.
     * @param {Array1D} v - The other vector.
     * @throws {RangeError} If `v.dim !== this.dim`.
     * @private
     */
    _checkDim(v) {
        if (v.dim !== this.dim) {
            throw new RangeError(`Array1D dimension mismatch: ${this.dim} vs ${v.dim}`);
        }
    }

    // -----------------------------------------------------------------
    // Immutable Operations (return new instances)
    // -----------------------------------------------------------------

    /**
     * Adds another vector to this one.
     * @param {Array1D} v - The vector to add. Must have the same `dim` as this one.
     * @returns {Array1D} A new vector equal to `this + v`.
     */
    add(v) {
        this._checkDim(v);
        const res = new Array1D(this.dim);
        for (let i = 0; i < this.dim; i++) res.data[i] = this.data[i] + v.data[i];
        return res;
    }

    /**
     * Subtracts another vector from this one.
     * @param {Array1D} v - The vector to subtract. Must have the same `dim` as this one.
     * @returns {Array1D} A new vector equal to `this - v`.
     */
    sub(v) {
        this._checkDim(v);
        const res = new Array1D(this.dim);
        for (let i = 0; i < this.dim; i++) res.data[i] = this.data[i] - v.data[i];
        return res;
    }

    /**
     * Scales this vector by a scalar.
     * @param {number} s - The scale factor.
     * @returns {Array1D} A new vector equal to `this * s`.
     */
    mult(s) {
        const res = new Array1D(this.dim);
        for (let i = 0; i < this.dim; i++) res.data[i] = this.data[i] * s;
        return res;
    }

    /**
     * Computes the squared magnitude (length) of this vector.
     * Cheaper than `norm()` since it avoids a square root.
     * @returns {number} The squared length of the vector.
     */
    normSq() {
        let sum = 0;
        for (let i = 0; i < this.dim; i++) sum += this.data[i] * this.data[i];
        return sum;
    }

    /**
     * Computes the magnitude (length) of this vector.
     * @returns {number} The length of the vector.
     */
    norm() { return Math.sqrt(this.normSq()); }

    /**
     * Returns a unit-length version of this vector (same direction, length 1).
     * If this vector has zero length, returns a zero vector instead of
     * dividing by zero.
     * @returns {Array1D} The normalized vector, or a zero vector if this vector is zero.
     */
    normalize() {
        const m = this.norm();
        return m === 0 ? new Array1D(this.dim) : this.mult(1 / m);
    }

    /**
     * Computes the dot product of this vector with another.
     * @param {Array1D} v - The other vector. Must have the same `dim` as this one.
     * @returns {number} The scalar dot product `this · v`.
     */
    dot(v) {
        this._checkDim(v);
        let sum = 0;
        for (let i = 0; i < this.dim; i++) sum += this.data[i] * v.data[i];
        return sum;
    }

    /**
     * Computes the Euclidean distance between this vector and another.
     * @param {Array1D} v - The other vector. Must have the same `dim` as this one.
     * @returns {number} The distance between `this` and `v`.
     */
    dist(v) { return this.sub(v).norm(); }

    /**
     * Computes the sum of this vector's components.
     * @returns {number} The sum, or `0` if `dim === 0`.
     */
    sum() {
        let s = 0;
        for (let i = 0; i < this.dim; i++) s += this.data[i];
        return s;
    }

    /**
     * Finds the smallest component of this vector.
     * @returns {number} The minimum value.
     * @throws {RangeError} If `dim === 0`.
     */
    min() {
        if (this.dim === 0) throw new RangeError('Array1D min() called on an empty vector');
        let m = this.data[0];
        for (let i = 1; i < this.dim; i++) if (this.data[i] < m) m = this.data[i];
        return m;
    }

    /**
     * Finds the largest component of this vector.
     * @returns {number} The maximum value.
     * @throws {RangeError} If `dim === 0`.
     */
    max() {
        if (this.dim === 0) throw new RangeError('Array1D max() called on an empty vector');
        let m = this.data[0];
        for (let i = 1; i < this.dim; i++) if (this.data[i] > m) m = this.data[i];
        return m;
    }

    /**
     * Returns a copy of this vector clamped to a maximum magnitude, preserving
     * direction. If the vector's magnitude is already at or below `max`,
     * returns an unchanged copy.
     * @param {number} max - The maximum allowed magnitude (must be >= 0).
     * @returns {Array1D} A new vector with magnitude at most `max`.
     */
    limit(max) {
        const m = this.norm();
        return m > max ? this.mult(max / m) : this.copy();
    }

    /**
     * Creates an independent copy of this vector.
     * @returns {Array1D} A new Array1D with the same values.
     */
    copy() { return new Array1D(this.data); }

    /**
     * Checks whether this vector is elementwise close to another, modeled on
     * `numpy.isclose`: a component `a` is close to `b` if
     * `|a - b| <= atol + rtol * |b|`.
     * @param {Array1D} v - The other vector.
     * @param {number} [rtol=1e-5] - Relative tolerance.
     * @param {number} [atol=1e-8] - Absolute tolerance.
     * @returns {boolean} `true` if `v` has the same `dim` and all components of `this` are close to `v`'s.
     */
    isClose(v, rtol = 1e-5, atol = 1e-8) {
        if (v.dim !== this.dim) return false;
        for (let i = 0; i < this.dim; i++) {
            if (Math.abs(this.data[i] - v.data[i]) > atol + rtol * Math.abs(v.data[i])) return false;
        }
        return true;
    }

    /**
     * Returns this vector's components as a plain array.
     * @returns {number[]} The components, in order.
     */
    toArray() { return Array.from(this.data); }

    /**
     * Returns a human-readable string representation of this vector.
     * @returns {string} e.g. `"Array1D(1, 2, 3)"`.
     */
    toString() { return `Array1D(${this.data.join(', ')})`; }

    /**
     * Makes Array1D iterable, e.g. `const [a, b, c] = someVector;` or `for (const x of v)`.
     * @returns {Iterator<number>}
     */
    [Symbol.iterator]() { return this.data[Symbol.iterator](); }

    // -----------------------------------------------------------------
    // In-place (mutating) operations.
    // -----------------------------------------------------------------

    /**
     * Sets this vector's components directly, mutating it in place.
     * @param {number[]|Float64Array} values - Values to copy in; must have length `dim`.
     * @returns {Array1D} `this`, for chaining.
     */
    set(values) {
        this.data.set(values);
        return this;
    }

    /**
     * Resets this vector to all zeros in place.
     * @returns {Array1D} `this`, for chaining.
     */
    reset() {
        this.data.fill(0);
        return this;
    }

    /**
     * Adds another vector to this one in place: `this += v`.
     * @param {Array1D} v - The vector to add. Must have the same `dim` as this one.
     * @returns {Array1D} `this`, for chaining.
     */
    addSelf(v) {
        this._checkDim(v);
        for (let i = 0; i < this.dim; i++) this.data[i] += v.data[i];
        return this;
    }

    /**
     * Subtracts another vector from this one in place: `this -= v`.
     * @param {Array1D} v - The vector to subtract. Must have the same `dim` as this one.
     * @returns {Array1D} `this`, for chaining.
     */
    subSelf(v) {
        this._checkDim(v);
        for (let i = 0; i < this.dim; i++) this.data[i] -= v.data[i];
        return this;
    }

    /**
     * Scales this vector in place: `this *= s`.
     * @param {number} s - The scale factor.
     * @returns {Array1D} `this`, for chaining.
     */
    multSelf(s) {
        for (let i = 0; i < this.dim; i++) this.data[i] *= s;
        return this;
    }

    /**
     * Adds a scaled vector to this one in place, in a single pass and
     * without an intermediate vector: `this += v * s`.
     * @param {Array1D} v - The vector to scale and add. Must have the same `dim` as this one.
     * @param {number} s - The scale factor applied to `v`.
     * @returns {Array1D} `this`, for chaining.
     */
    addScaled(v, s) {
        this._checkDim(v);
        for (let i = 0; i < this.dim; i++) this.data[i] += v.data[i] * s;
        return this;
    }

    /**
     * Sets this vector to `a - b` in place, without allocating. Useful as a
     * reusable scratch vector inside a loop.
     * @param {Array1D} a
     * @param {Array1D} b - Must have the same `dim` as `a` and as this vector.
     * @returns {Array1D} `this`, set to `a - b`.
     */
    subVectors(a, b) {
        this._checkDim(a);
        this._checkDim(b);
        for (let i = 0; i < this.dim; i++) this.data[i] = a.data[i] - b.data[i];
        return this;
    }

    /**
     * Creates a zero vector of the given dimension.
     * @param {number} dim - The number of components.
     * @returns {Array1D} A new zero vector.
     */
    static zero(dim) { return new Array1D(dim); }

    /**
     * Creates a Array1D from an array or typed array.
     * @param {number[]|Float64Array} arr - Source values.
     * @returns {Array1D} A new vector with dimension equal to `arr.length`.
     */
    static from(arr) { return new Array1D(arr); }
}
