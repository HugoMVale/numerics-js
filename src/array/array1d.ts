/**
 * An N-component vector utilizing Float64Array for performance.
 */
export class Array1D {
    public data: Float64Array;
    public dim: number;

    /**
     * @param input - The dimension length (initialized to 0s) or initial data.
     */
    constructor(input: number | number[] | Float64Array) {
        if (typeof input === 'number') {
            this.data = new Float64Array(input);
        } else {
            this.data = new Float64Array(input);
        }
        this.dim = this.data.length;
    }

    /**
     * Throws if `v` is not a Array1D of the same dimension as this one. Used
     * internally to guard binary operations against silent shape mismatches.
     * @param v - The other vector.
     * @throws {RangeError} If `v.dim !== this.dim`.
     */
    private _checkDim(v: Array1D): void {
        if (v.dim !== this.dim) {
            throw new RangeError(`Array1D dimension mismatch: ${this.dim} vs ${v.dim}`);
        }
    }

    // -----------------------------------------------------------------
    // Immutable Operations (return new instances)
    // -----------------------------------------------------------------

    /**
     * Adds another vector to this one.
     * @param v - The vector to add. Must have the same `dim` as this one.
     * @returns A new vector equal to `this + v`.
     */
    add(v: Array1D): Array1D {
        this._checkDim(v);
        const res = new Array1D(this.dim);
        for (let i = 0; i < this.dim; i++) res.data[i] = this.data[i] + v.data[i];
        return res;
    }

    /**
     * Subtracts another vector from this one.
     * @param v - The vector to subtract. Must have the same `dim` as this one.
     * @returns A new vector equal to `this - v`.
     */
    sub(v: Array1D): Array1D {
        this._checkDim(v);
        const res = new Array1D(this.dim);
        for (let i = 0; i < this.dim; i++) res.data[i] = this.data[i] - v.data[i];
        return res;
    }

    /**
     * Scales this vector by a scalar.
     * @param s - The scale factor.
     * @returns A new vector equal to `this * s`.
     */
    mult(s: number): Array1D {
        const res = new Array1D(this.dim);
        for (let i = 0; i < this.dim; i++) res.data[i] = this.data[i] * s;
        return res;
    }

    /**
     * Computes the squared magnitude (length) of this vector.
     * Cheaper than `norm()` since it avoids a square root.
     * @returns The squared length of the vector.
     */
    normSq(): number {
        let sum = 0;
        for (let i = 0; i < this.dim; i++) sum += this.data[i] * this.data[i];
        return sum;
    }

    /**
     * Computes the magnitude (length) of this vector.
     * @returns The length of the vector.
     */
    norm(): number {
        return Math.sqrt(this.normSq());
    }

    /**
     * Returns a unit-length version of this vector (same direction, length 1).
     * If this vector has zero length, returns a zero vector instead of
     * dividing by zero.
     * @returns The normalized vector, or a zero vector if this vector is zero.
     */
    normalize(): Array1D {
        const m = this.norm();
        return m === 0 ? new Array1D(this.dim) : this.mult(1 / m);
    }

    /**
     * Computes the dot product of this vector with another.
     * @param v - The other vector. Must have the same `dim` as this one.
     * @returns The scalar dot product `this · v`.
     */
    dot(v: Array1D): number {
        this._checkDim(v);
        let sum = 0;
        for (let i = 0; i < this.dim; i++) sum += this.data[i] * v.data[i];
        return sum;
    }

    /**
     * Computes the Euclidean distance between this vector and another.
     * @param v - The other vector. Must have the same `dim` as this one.
     * @returns The distance between `this` and `v`.
     */
    dist(v: Array1D): number {
        return this.sub(v).norm();
    }

    /**
     * Computes the sum of this vector's components.
     * @returns The sum, or `0` if `dim === 0`.
     */
    sum(): number {
        let s = 0;
        for (let i = 0; i < this.dim; i++) s += this.data[i];
        return s;
    }

    /**
     * Finds the smallest component of this vector.
     * @returns The minimum value.
     * @throws {RangeError} If `dim === 0`.
     */
    min(): number {
        if (this.dim === 0) throw new RangeError('Array1D min() called on an empty vector');
        let m = this.data[0];
        for (let i = 1; i < this.dim; i++) if (this.data[i] < m) m = this.data[i];
        return m;
    }

    /**
     * Finds the largest component of this vector.
     * @returns The maximum value.
     * @throws {RangeError} If `dim === 0`.
     */
    max(): number {
        if (this.dim === 0) throw new RangeError('Array1D max() called on an empty vector');
        let m = this.data[0];
        for (let i = 1; i < this.dim; i++) if (this.data[i] > m) m = this.data[i];
        return m;
    }

    /**
     * Returns a copy of this vector clamped to a maximum magnitude, preserving
     * direction. If the vector's magnitude is already at or below `max`,
     * returns an unchanged copy.
     * @param max - The maximum allowed magnitude (must be >= 0).
     * @returns A new vector with magnitude at most `max`.
     */
    limit(max: number): Array1D {
        const m = this.norm();
        return m > max ? this.mult(max / m) : this.copy();
    }

    /**
     * Creates an independent copy of this vector.
     * @returns A new Array1D with the same values.
     */
    copy(): Array1D {
        return new Array1D(this.data);
    }

    /**
     * Checks whether this vector is elementwise close to another, modeled on
     * `numpy.isclose`: a component `a` is close to `b` if
     * `|a - b| <= atol + rtol * |b|`.
     * @param v - The other vector.
     * @param rtol - Relative tolerance.
     * @param atol - Absolute tolerance.
     * @returns `true` if `v` has the same `dim` and all components of `this` are close to `v`'s.
     */
    isClose(v: Array1D, rtol: number = 1e-5, atol: number = 1e-8): boolean {
        if (v.dim !== this.dim) return false;
        for (let i = 0; i < this.dim; i++) {
            if (Math.abs(this.data[i] - v.data[i]) > atol + rtol * Math.abs(v.data[i])) return false;
        }
        return true;
    }

    /**
     * Returns this vector's components as a plain array.
     * @returns The components, in order.
     */
    toArray(): number[] {
        return Array.from(this.data);
    }

    /**
     * Returns a human-readable string representation of this vector.
     * @returns e.g. `"Array1D(1, 2, 3)"`.
     */
    toString(): string {
        return `Array1D(${this.data.join(', ')})`;
    }

    /**
     * Makes Array1D iterable, e.g. `const [a, b, c] = someVector;` or `for (const x of v)`.
     */
    [Symbol.iterator](): IterableIterator<number> {
        return this.data[Symbol.iterator]();
    }

    // -----------------------------------------------------------------
    // In-place (mutating) operations.
    // -----------------------------------------------------------------

    /**
     * Sets this vector's components directly, mutating it in place.
     * @param values - Values to copy in; must have length `dim`.
     * @returns `this`, for chaining.
     */
    set(values: number[] | Float64Array): this {
        this.data.set(values);
        return this;
    }

    /**
     * Resets this vector to all zeros in place.
     * @returns `this`, for chaining.
     */
    reset(): this {
        this.data.fill(0);
        return this;
    }

    /**
     * Adds another vector to this one in place: `this += v`.
     * @param v - The vector to add. Must have the same `dim` as this one.
     * @returns `this`, for chaining.
     */
    addSelf(v: Array1D): this {
        this._checkDim(v);
        for (let i = 0; i < this.dim; i++) this.data[i] += v.data[i];
        return this;
    }

    /**
     * Subtracts another vector from this one in place: `this -= v`.
     * @param v - The vector to subtract. Must have the same `dim` as this one.
     * @returns `this`, for chaining.
     */
    subSelf(v: Array1D): this {
        this._checkDim(v);
        for (let i = 0; i < this.dim; i++) this.data[i] -= v.data[i];
        return this;
    }

    /**
     * Scales this vector in place: `this *= s`.
     * @param s - The scale factor.
     * @returns `this`, for chaining.
     */
    multSelf(s: number): this {
        for (let i = 0; i < this.dim; i++) this.data[i] *= s;
        return this;
    }

    /**
     * Adds a scaled vector to this one in place, in a single pass and
     * without an intermediate vector: `this += v * s`.
     * @param v - The vector to scale and add. Must have the same `dim` as this one.
     * @param s - The scale factor applied to `v`.
     * @returns `this`, for chaining.
     */
    addScaled(v: Array1D, s: number): this {
        this._checkDim(v);
        for (let i = 0; i < this.dim; i++) this.data[i] += v.data[i] * s;
        return this;
    }

    /**
     * Sets this vector to `a - b` in place, without allocating. Useful as a
     * reusable scratch vector inside a loop.
     * @param a
     * @param b - Must have the same `dim` as `a` and as this vector.
     * @returns `this`, set to `a - b`.
     */
    subVectors(a: Array1D, b: Array1D): this {
        this._checkDim(a);
        this._checkDim(b);
        for (let i = 0; i < this.dim; i++) this.data[i] = a.data[i] - b.data[i];
        return this;
    }

    /**
     * Creates a zero vector of the given dimension.
     * @param dim - The number of components.
     * @returns A new zero vector.
     */
    static zero(dim: number): Array1D {
        return new Array1D(dim);
    }

    /**
     * Creates a Array1D from an array or typed array.
     * @param arr - Source values.
     * @returns A new vector with dimension equal to `arr.length`.
     */
    static from(arr: number[] | Float64Array): Array1D {
        return new Array1D(arr);
    }
}