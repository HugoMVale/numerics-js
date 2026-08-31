/**
 * An N-component vector utilizing Float64Array for performance.
 */
export class Array1D {
    public data: Float64Array;

    /**
     * @param input The dimension length (initialized to 0s) or initial data.
     */
    constructor(input: number | number[] | Float64Array) {
        if (typeof input === 'number') {
            if (!Number.isInteger(input)) {
                throw new RangeError(`Array1D.constructor: dimension must be an integer, got ${input}`);
            }
            if (input < 0) {
                throw new RangeError(`Array1D.constructor: dimension must be >= 0, got ${input}`);
            }
            this.data = new Float64Array(input);
        } else {
            this.data = new Float64Array(input);
        }
    }

    /**
     * The number of components in this vector. Derived directly from
     * `data.length` so it can never desync, even if `data` is reassigned.
     */
    get size(): number {
        return this.data.length;
    }

    /**
     * Throws if `v` is not a Array1D of the same dimension as this one. Used
     * internally to guard binary operations against silent shape mismatches.
     * @param v The other vector.
     * @param caller Name of the public method invoking this check, used to
     * produce a precise error message (e.g. `"add"`).
     * @throws {RangeError} If `v.size !== this.size`.
     */
    private _checkDim(v: Array1D, caller: string): void {
        if (v.size !== this.size) {
            throw new RangeError(`Array1D.${caller}: dimension mismatch: ${this.size} vs ${v.size}`);
        }
    }

    /**
     * Throws if `i` is not a valid 0-based component index.
     * @param i Component index (0-based).
     * @param caller Name of the public method invoking this check, used to
     * produce a precise error message (e.g. `"get"`).
     * @throws {RangeError} If `i` is not an integer in `0..size-1`.
     */
    private _checkIndex(i: number, caller: string): void {
        if (!Number.isInteger(i) || i < 0 || i >= this.size) {
            throw new RangeError(`Array1D.${caller}: index ${i} out of bounds for dimension ${this.size}`);
        }
    }

    /**
     * Gets a component by its 0-based index.
     * @param i Component index (0-based).
     * @returns The component value.
     */
    get(i: number): number {
        this._checkIndex(i, 'get');
        return this.data[i];
    }

    // -----------------------------------------------------------------
    // Immutable Operations (return new instances)
    // -----------------------------------------------------------------

    /**
     * Adds another vector to this one, or adds a scalar to every component.
     * @param v The vector to add. Must have the same `size` as this one.
     * @returns A new vector equal to `this + v`.
     */
    add(v: Array1D): Array1D;

    /**
     * Adds another vector to this one, or adds a scalar to every component.
     * @param s The scalar to add to every component.
     * @returns A new vector equal to `this + s`.
     */
    add(s: number): Array1D;

    add(x: Array1D | number): Array1D {
        const res = new Array1D(this.size);
        if (x instanceof Array1D) {
            this._checkDim(x, 'add');
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] + x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] + x;
        }
        return res;
    }

    /**
     * Subtracts another vector from this one, or subtracts a scalar from
     * every component.
     * @param v The vector to subtract. Must have the same `size` as this one.
     * @returns A new vector equal to `this - v`.
     */
    sub(v: Array1D): Array1D;

    /**
     * Subtracts another vector from this one, or subtracts a scalar from
     * every component.
     * @param s The scalar to subtract from every component.
     * @returns A new vector equal to `this - s`.
     */
    sub(s: number): Array1D;

    sub(x: Array1D | number): Array1D {
        const res = new Array1D(this.size);
        if (x instanceof Array1D) {
            this._checkDim(x, 'sub');
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] - x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] - x;
        }
        return res;
    }

    /**
     * Multiplies this vector with another elementwise (Hadamard product),
     * or scales every component by a scalar.
     * @param v The vector to multiply by. Must have the same `size` as this one.
     * @returns A new vector where each component is `this[i] * v[i]`.
     */
    mult(v: Array1D): Array1D;

    /**
     * Multiplies this vector with another elementwise (Hadamard product),
     * or scales every component by a scalar.
     * @param s The scale factor.
     * @returns A new vector equal to `this * s`.
     */
    mult(s: number): Array1D;

    mult(x: Array1D | number): Array1D {
        const res = new Array1D(this.size);
        if (x instanceof Array1D) {
            this._checkDim(x, 'mult');
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] * x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] * x;
        }
        return res;
    }

    /**
     * Divides this vector by another elementwise, or by a scalar. No
     * special handling for division by zero; components divided by `0`
     * follow standard floating-point semantics (`Infinity`, `-Infinity`, or
     * `NaN`).
     * @param v The vector to divide by. Must have the same `size` as this one.
     * @returns A new vector where each component is `this[i] / v[i]`.
     */
    div(v: Array1D): Array1D;

    /**
     * Divides this vector by another elementwise, or by a scalar. No
     * special handling for division by zero; components divided by `0`
     * follow standard floating-point semantics (`Infinity`, `-Infinity`, or
     * `NaN`).
     * @param s The scalar to divide every component by.
     * @returns A new vector equal to `this / s`.
     */
    div(s: number): Array1D;

    div(x: Array1D | number): Array1D {
        const res = new Array1D(this.size);
        if (x instanceof Array1D) {
            this._checkDim(x, 'div');
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] / x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) res.data[i] = this.data[i] / x;
        }
        return res;
    }

    /**
     * Computes the elementwise absolute value of this vector.
     * @returns A new vector where each component is `|this[i]|`.
     */
    abs(): Array1D {
        const res = new Array1D(this.size);
        for (let i = 0; i < this.size; i++) res.data[i] = Math.abs(this.data[i]);
        return res;
    }

    /**
     * Raises each component of this vector to a power, elementwise.
     * @param exp The exponent.
     * @returns A new vector where each component is `this[i] ** exp`.
     */
    pow(exp: number): Array1D {
        const res = new Array1D(this.size);
        for (let i = 0; i < this.size; i++) res.data[i] = Math.pow(this.data[i], exp);
        return res;
    }

    /**
     * Computes the elementwise square root of this vector. Components that
     * are negative produce `NaN`, matching `Math.sqrt`.
     * @returns A new vector where each component is `sqrt(this[i])`.
     */
    sqrt(): Array1D {
        const res = new Array1D(this.size);
        for (let i = 0; i < this.size; i++) res.data[i] = Math.sqrt(this.data[i]);
        return res;
    }

    /**
     * Clamps each component of this vector to `[min, max]`, elementwise.
     * Unlike `limit()`, which scales the whole vector to cap its magnitude,
     * `clip` bounds each component's value independently.
     * @param min The lower bound.
     * @param max The upper bound.
     * @returns A new vector with every component clamped to `[min, max]`.
     * @throws {RangeError} If `min > max`.
     */
    clip(min: number, max: number): Array1D {
        if (min > max) throw new RangeError(`Array1D.clip: min (${min}) must be <= max (${max})`);
        const res = new Array1D(this.size);
        for (let i = 0; i < this.size; i++) res.data[i] = Math.min(Math.max(this.data[i], min), max);
        return res;
    }

    /**
     * Returns a sorted copy of this vector, leaving the original unchanged.
     * @param compareFn Optional comparator, as in `Array.prototype.sort`.
     * Defaults to ascending numeric order (unlike `Array.prototype.sort`'s
     * default, which sorts lexicographically).
     * @returns A new, sorted vector.
     */
    sort(compareFn?: (a: number, b: number) => number): Array1D {
        const res = this.copy();
        res.data.sort(compareFn);
        return res;
    }

    /**
     * Extracts a sub-vector, with the same start/end/negative-index
     * semantics as `Array.prototype.slice`.
     * @param start Start index, inclusive. Defaults to `0`. Negative values
     * count back from the end.
     * @param end End index, exclusive. Defaults to `size`. Negative values
     * count back from the end.
     * @returns A new, independent vector holding the selected components.
     */
    slice(start?: number, end?: number): Array1D {
        return new Array1D(this.data.slice(start, end));
    }

    /**
     * Applies a function to each component, elementwise.
     * @param fn Called with each component's value and index; its return
     * value becomes the corresponding component of the result.
     * @returns A new vector, the same size as this one, holding the mapped values.
     */
    map(fn: (value: number, index: number) => number): Array1D {
        const res = new Array1D(this.size);
        for (let i = 0; i < this.size; i++) res.data[i] = fn(this.data[i], i);
        return res;
    }

    /**
     * Computes the squared magnitude (length) of this vector.
     * Cheaper than `norm()` since it avoids a square root.
     * @returns The squared length of the vector.
     */
    normSq(): number {
        let sum = 0;
        for (let i = 0; i < this.size; i++) sum += this.data[i] * this.data[i];
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
        return m === 0 ? new Array1D(this.size) : this.mult(1 / m);
    }

    /**
     * Computes the dot product of this vector with another.
     * @param v The other vector. Must have the same `size` as this one.
     * @returns The scalar dot product `this · v`.
     */
    dot(v: Array1D): number {
        this._checkDim(v, 'dot');
        let sum = 0;
        for (let i = 0; i < this.size; i++) sum += this.data[i] * v.data[i];
        return sum;
    }

    /**
     * Computes the Euclidean distance between this vector and another.
     * @param v The other vector. Must have the same `size` as this one.
     * @returns The distance between `this` and `v`.
     */
    dist(v: Array1D): number {
        this._checkDim(v, 'dist');
        let sum = 0;
        for (let i = 0; i < this.size; i++) {
            const d = this.data[i] - v.data[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /**
     * Computes the sum of this vector's components.
     * @returns The sum, or `0` if `size === 0`.
     */
    sum(): number {
        let s = 0;
        for (let i = 0; i < this.size; i++) s += this.data[i];
        return s;
    }

    /**
     * Computes the arithmetic mean of this vector's components.
     * @returns The mean, or `NaN` if `size === 0`.
     */
    mean(): number {
        return this.sum() / this.size;
    }

    /**
     * Computes the variance of this vector's components: the mean of the
     * squared deviations from the mean.
     * @param ddof Delta degrees of freedom. The divisor used is `size - ddof`.
     * Defaults to `0` (population variance); pass `1` for the unbiased
     * sample variance.
     * @returns The variance. `NaN` if `size - ddof <= 0`.
     */
    variance(ddof: number = 0): number {
        const m = this.mean();
        let sq = 0;
        for (let i = 0; i < this.size; i++) {
            const d = this.data[i] - m;
            sq += d * d;
        }
        return sq / (this.size - ddof);
    }

    /**
     * Computes the standard deviation of this vector's components.
     * @param ddof Delta degrees of freedom, forwarded to `variance()`.
     * Defaults to `0` (population standard deviation); pass `1` for the
     * unbiased sample standard deviation.
     * @returns The standard deviation. `NaN` if `size - ddof <= 0`.
     */
    std(ddof: number = 0): number {
        return Math.sqrt(this.variance(ddof));
    }

    /**
     * Computes the cumulative sum of this vector's components.
     * @returns A new vector of the same size where component `i` is the
     * sum of `this[0..i]`.
     */
    cumsum(): Array1D {
        const res = new Array1D(this.size);
        let running = 0;
        for (let i = 0; i < this.size; i++) {
            running += this.data[i];
            res.data[i] = running;
        }
        return res;
    }

    /**
     * Finds the smallest component of this vector.
     * @returns The minimum value.
     * @throws {RangeError} If `size === 0`.
     */
    min(): number {
        if (this.size === 0) throw new RangeError('Array1D.min: cannot compute the min of an empty vector');
        let m = this.data[0];
        for (let i = 1; i < this.size; i++) {
            const x = this.data[i];
            // NaN must win the comparison so it propagates instead of being
            // silently skipped, matching Math.min's semantics.
            if (Number.isNaN(x) || x < m) m = x;
        }
        return m;
    }

    /**
     * Finds the index of the smallest component of this vector. If multiple
     * components tie for the minimum, the first index is returned. NaN
     * components take priority, matching `min()`'s NaN-propagation semantics.
     * @returns The index of the minimum value.
     * @throws {RangeError} If `size === 0`.
     */
    argmin(): number {
        if (this.size === 0) throw new RangeError('Array1D.argmin: cannot compute the argmin of an empty vector');
        if (Number.isNaN(this.data[0])) return 0;
        let mi = 0;
        let m = this.data[0];
        for (let i = 1; i < this.size; i++) {
            const x = this.data[i];
            if (Number.isNaN(x)) return i;
            if (x < m) {
                m = x;
                mi = i;
            }
        }
        return mi;
    }

    /**
     * Finds the largest component of this vector.
     * @returns The maximum value.
     * @throws {RangeError} If `size === 0`.
     */
    max(): number {
        if (this.size === 0) throw new RangeError('Array1D.max: cannot compute the max of an empty vector');
        let m = this.data[0];
        for (let i = 1; i < this.size; i++) {
            const x = this.data[i];
            // NaN must win the comparison so it propagates instead of being
            // silently skipped, matching Math.max's semantics.
            if (Number.isNaN(x) || x > m) m = x;
        }
        return m;
    }

    /**
     * Finds the index of the largest component of this vector. If multiple
     * components tie for the maximum, the first index is returned. NaN
     * components take priority, matching `max()`'s NaN-propagation semantics.
     * @returns The index of the maximum value.
     * @throws {RangeError} If `size === 0`.
     */
    argmax(): number {
        if (this.size === 0) throw new RangeError('Array1D.argmax: cannot compute the argmax of an empty vector');
        if (Number.isNaN(this.data[0])) return 0;
        let mi = 0;
        let m = this.data[0];
        for (let i = 1; i < this.size; i++) {
            const x = this.data[i];
            if (Number.isNaN(x)) return i;
            if (x > m) {
                m = x;
                mi = i;
            }
        }
        return mi;
    }

    /**
     * Returns a copy of this vector clamped to a maximum magnitude, preserving
     * direction. If the vector's magnitude is already at or below `max`,
     * returns an unchanged copy.
     * @param max The maximum allowed magnitude (must be >= 0).
     * @returns A new vector with magnitude at most `max`.
     */
    limit(max: number): Array1D {
        if (max < 0) throw new RangeError(`Array1D.limit: max must be >= 0, got ${max}`);
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
     * Shared implementation behind `isClose` and `allClose`. Note this is
     * **not symmetric**: `rtol` scales `v` (the argument), not `this`, so in general
     * `a.isClose(b) !== b.isClose(a)`.
     * @param v The other vector, treated as the reference/"actual" value.
     * @param rtol Relative tolerance, applied to `v`.
     * @param atol Absolute tolerance.
     * @returns A plain boolean array, one entry per component.
     * @throws {RangeError} If `v.size !== this.size`.
     */
    private _isCloseElementwise(v: Array1D, rtol: number, atol: number): boolean[] {
        this._checkDim(v, 'isClose');
        const result: boolean[] = new Array(this.size);
        for (let i = 0; i < this.size; i++) {
            const a = this.data[i];
            const b = v.data[i];
            result[i] = Math.abs(a - b) <= atol + rtol * Math.abs(b);
        }
        return result;
    }

    /**
     * Elementwise tolerance comparison. A component `a` (from `this`) is
     * close to `b` (from `v`) if
     * `|a - b| <= atol + rtol * |b|`. Note this is **not symmetric**:
     * `rtol` scales `v` only, so in general `a.isClose(b) !== b.isClose(a)`.
     * NaN components are never close to anything, including other NaNs.
     * @param v The other vector. Must have the same `size` as this one.
     * @param rtol Relative tolerance, applied to `v`'s components.
     * @param atol Absolute tolerance.
     * @returns A boolean array, one entry per component.
     * @throws {RangeError} If `v.size !== this.size`.
     */
    isClose(v: Array1D, rtol: number = 1e-5, atol: number = 1e-8): boolean[] {
        return this._isCloseElementwise(v, rtol, atol);
    }

    /**
     * Checks whether this vector is close to another across *all*
     * components. Equivalent to `isClose(v, rtol, atol).every(Boolean)`.
     * @param v The other vector. Must have the same `size` as this one.
     * @param rtol Relative tolerance, applied to `v`'s components.
     * @param atol Absolute tolerance.
     * @returns `true` if every component of `this` is close to the corresponding component of `v`.
     * @throws {RangeError} If `v.size !== this.size`.
     */
    allClose(v: Array1D, rtol: number = 1e-5, atol: number = 1e-8): boolean {
        return this._isCloseElementwise(v, rtol, atol).every(Boolean);
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
     * @param values Values to copy in; must have length `size`.
     * @returns `this`, for chaining.
     */
    set(values: number[] | Float64Array): this;

    /**
     * Sets one component by its 0-based index, mutating this vector in place.
     * @param i Component index (0-based).
     * @param value Value to store.
     * @returns `this`, for chaining.
     */
    set(i: number, value: number): this;

    set(valuesOrIndex: number[] | Float64Array | number, value?: number): this {
        if (typeof valuesOrIndex === 'number') {
            this._checkIndex(valuesOrIndex, 'set');
            this.data[valuesOrIndex] = value as number;
        } else {
            if (valuesOrIndex.length !== this.size) {
                throw new RangeError(
                    `Array1D.set: expected ${this.size} values, got ${valuesOrIndex.length}`
                );
            }
            this.data.set(valuesOrIndex);
        }
        return this;
    }

    /**
     * Sets every component of this vector to `value`, in place.
     * @param value The value to fill with.
     * @returns `this`, for chaining.
     */
    fill(value: number): this {
        this.data.fill(value);
        return this;
    }

    /**
     * Adds another vector to this one in place, or adds a scalar to every
     * component: `this += v`.
     * @param v The vector to add. Must have the same `size` as this one.
     * @returns `this`, for chaining.
     */
    addSelf(v: Array1D): this;

    /**
     * Adds another vector to this one in place, or adds a scalar to every
     * component: `this += s`.
     * @param s The scalar to add to every component.
     * @returns `this`, for chaining.
     */
    addSelf(s: number): this;

    addSelf(x: Array1D | number): this {
        if (x instanceof Array1D) {
            this._checkDim(x, 'addSelf');
            for (let i = 0; i < this.size; i++) this.data[i] += x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) this.data[i] += x;
        }
        return this;
    }

    /**
     * Subtracts another vector from this one in place, or subtracts a
     * scalar from every component: `this -= v`.
     * @param v The vector to subtract. Must have the same `size` as this one.
     * @returns `this`, for chaining.
     */
    subSelf(v: Array1D): this;

    /**
     * Subtracts another vector from this one in place, or subtracts a
     * scalar from every component: `this -= s`.
     * @param s The scalar to subtract from every component.
     * @returns `this`, for chaining.
     */
    subSelf(s: number): this;

    subSelf(x: Array1D | number): this {
        if (x instanceof Array1D) {
            this._checkDim(x, 'subSelf');
            for (let i = 0; i < this.size; i++) this.data[i] -= x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) this.data[i] -= x;
        }
        return this;
    }

    /**
     * Multiplies this vector with another in place, elementwise (Hadamard
     * product), or scales every component by a scalar: `this *= v`.
     * @param v The vector to multiply by. Must have the same `size` as this one.
     * @returns `this`, for chaining.
     */
    multSelf(v: Array1D): this;

    /**
     * Multiplies this vector with another in place, elementwise (Hadamard
     * product), or scales every component by a scalar: `this *= s`.
     * @param s The scale factor.
     * @returns `this`, for chaining.
     */
    multSelf(s: number): this;

    multSelf(x: Array1D | number): this {
        if (x instanceof Array1D) {
            this._checkDim(x, 'multSelf');
            for (let i = 0; i < this.size; i++) this.data[i] *= x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) this.data[i] *= x;
        }
        return this;
    }

    /**
     * Divides this vector by another in place, elementwise, or by a scalar:
     * `this /= v`. No special handling for division by zero; components
     * divided by `0` follow standard floating-point semantics (`Infinity`,
     * `-Infinity`, or `NaN`).
     * @param v The vector to divide by. Must have the same `size` as this one.
     * @returns `this`, for chaining.
     */
    divSelf(v: Array1D): this;

    /**
     * Divides this vector by another in place, elementwise, or by a scalar:
     * `this /= s`.
     * @param s The scalar to divide every component by.
     * @returns `this`, for chaining.
     */
    divSelf(s: number): this;

    divSelf(x: Array1D | number): this {
        if (x instanceof Array1D) {
            this._checkDim(x, 'divSelf');
            for (let i = 0; i < this.size; i++) this.data[i] /= x.data[i];
        } else {
            for (let i = 0; i < this.size; i++) this.data[i] /= x;
        }
        return this;
    }

    /**
     * Takes the elementwise absolute value of this vector in place.
     * @returns `this`, for chaining.
     */
    absSelf(): this {
        for (let i = 0; i < this.size; i++) this.data[i] = Math.abs(this.data[i]);
        return this;
    }

    /**
     * Raises each component of this vector to a power in place, elementwise.
     * @param exp The exponent.
     * @returns `this`, for chaining.
     */
    powSelf(exp: number): this {
        for (let i = 0; i < this.size; i++) this.data[i] = Math.pow(this.data[i], exp);
        return this;
    }

    /**
     * Takes the elementwise square root of this vector in place. Components
     * that are negative become `NaN`, matching `Math.sqrt`.
     * @returns `this`, for chaining.
     */
    sqrtSelf(): this {
        for (let i = 0; i < this.size; i++) this.data[i] = Math.sqrt(this.data[i]);
        return this;
    }

    /**
     * Clamps each component of this vector to `[min, max]` in place,
     * elementwise.
     * @param min The lower bound.
     * @param max The upper bound.
     * @returns `this`, for chaining.
     * @throws {RangeError} If `min > max`.
     */
    clipSelf(min: number, max: number): this {
        if (min > max) throw new RangeError(`Array1D.clipSelf: min (${min}) must be <= max (${max})`);
        for (let i = 0; i < this.size; i++) this.data[i] = Math.min(Math.max(this.data[i], min), max);
        return this;
    }

    /**
     * Adds a scaled vector to this one in place, in a single pass and
     * without an intermediate vector: `this += v * s`.
     * @param v The vector to scale and add. Must have the same `size` as this one.
     * @param s The scale factor applied to `v`.
     * @returns `this`, for chaining.
     */
    addScaled(v: Array1D, s: number): this {
        this._checkDim(v, 'addScaled');
        for (let i = 0; i < this.size; i++) this.data[i] += v.data[i] * s;
        return this;
    }

    /**
     * Sets this vector to `a - b` in place, without allocating. Useful as a
     * reusable scratch vector inside a loop.
     * @param a
     * @param b Must have the same `size` as `a` and as this vector.
     * @returns `this`, set to `a - b`.
     */
    subVectors(a: Array1D, b: Array1D): this {
        this._checkDim(a, 'subVectors');
        this._checkDim(b, 'subVectors');
        for (let i = 0; i < this.size; i++) this.data[i] = a.data[i] - b.data[i];
        return this;
    }

    /**
     * Creates a zero vector of the given dimension.
     * @param dim The number of components.
     * @returns A new zero vector.
     */
    static zero(dim: number): Array1D {
        return new Array1D(dim);
    }

    /**
     * Creates a vector of the given dimension filled with `1`.
     * @param dim The number of components.
     * @returns A new vector of all ones.
     */
    static ones(dim: number): Array1D {
        return new Array1D(dim).fill(1);
    }

    /**
     * Creates a vector of the given dimension filled with `value`.
     * @param dim The number of components.
     * @param value The value to fill with.
     * @returns A new vector with every component equal to `value`.
     */
    static full(dim: number, value: number): Array1D {
        return new Array1D(dim).fill(value);
    }

    /**
     * Creates a Array1D from an array or typed array.
     * @param arr Source values.
     * @returns A new vector with dimension equal to `arr.length`.
     */
    static from(arr: number[] | Float64Array): Array1D {
        return new Array1D(arr);
    }

    /**
     * Creates a vector of evenly spaced values within a half-open interval
     * `[start, stop)`, stepping by `step`. Distinct from `linspace`, which
     * takes a sample *count* instead of a step size.
     * @param stop Exclusive upper bound, with `start` defaulting to `0` and `step` to `1`.
     * @returns A new vector of values `0, 1, 2, ..., stop - 1`.
     */
    static arange(stop: number): Array1D;

    /**
     * Creates a vector of evenly spaced values within a half-open interval
     * `[start, stop)`, stepping by `step`. Distinct from `linspace`, which
     * takes a sample *count* instead of a step size.
     * @param start Inclusive lower bound.
     * @param stop Exclusive upper bound.
     * @param step The spacing between values. May be negative to count down
     * (in which case `start` should be greater than `stop`). Defaults to `1`.
     * @returns A new vector of values `start, start + step, start + 2*step, ...` up to (excluding) `stop`.
     * @throws {RangeError} If `step === 0`.
     */
    static arange(start: number, stop: number, step?: number): Array1D;

    static arange(startOrStop: number, stop?: number, step: number = 1): Array1D {
        const start = stop === undefined ? 0 : startOrStop;
        const actualStop = stop === undefined ? startOrStop : stop;
        if (step === 0) throw new RangeError('Array1D.arange: step must not be 0');
        const n = Math.max(0, Math.ceil((actualStop - start) / step));
        const res = new Array1D(n);
        for (let i = 0; i < n; i++) res.data[i] = start + i * step;
        return res;
    }

    /**
     * Creates a vector of `num` evenly spaced samples over `[start, stop]`.
     * @param start The first sample.
     * @param stop The last sample (included when `endpoint` is `true`).
     * @param num The number of samples. Defaults to `50`.
     * @param endpoint Whether to include `stop` as the last sample. Defaults to `true`.
     * @returns A new vector of length `num`.
     * @throws {RangeError} If `num` is not a non-negative integer.
     */
    static linspace(start: number, stop: number, num: number = 50, endpoint: boolean = true): Array1D {
        if (!Number.isInteger(num) || num < 0) {
            throw new RangeError(`Array1D.linspace: num must be a non-negative integer, got ${num}`);
        }
        const res = new Array1D(num);
        if (num === 0) return res;
        if (num === 1) {
            res.data[0] = start;
            return res;
        }
        const denom = endpoint ? num - 1 : num;
        const step = (stop - start) / denom;
        for (let i = 0; i < num; i++) res.data[i] = start + step * i;
        // Force the exact endpoint instead of relying on the accumulated
        // step multiplication, which can drift slightly due to floating
        // point error.
        if (endpoint) res.data[num - 1] = stop;
        return res;
    }

    /**
     * Creates a vector of `num` samples evenly spaced on a log scale over
     * `[base**start, base**stop]`. Equivalent to
     * `base ** linspace(start, stop, num, endpoint)`.
     * @param start `base**start` is the first sample.
     * @param stop `base**stop` is the last sample (included when `endpoint` is `true`).
     * @param num The number of samples. Defaults to `50`.
     * @param endpoint Whether to include `base**stop` as the last sample. Defaults to `true`.
     * @param base The base of the log space. Defaults to `10`.
     * @returns A new vector of length `num`.
     * @throws {RangeError} If `num` is not a non-negative integer.
     */
    static logspace(
        start: number,
        stop: number,
        num: number = 50,
        endpoint: boolean = true,
        base: number = 10
    ): Array1D {
        const exponents = Array1D.linspace(start, stop, num, endpoint);
        const res = new Array1D(num);
        for (let i = 0; i < num; i++) res.data[i] = Math.pow(base, exponents.data[i]);
        return res;
    }
}