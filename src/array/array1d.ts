import { ArrayND } from './arraynd.js';

/**
 * An N-component vector utilizing Float64Array for performance.
 *
 * Elementwise arithmetic (`add`/`sub`/`mult`/`div` + `Self` variants,
 * `abs`/`pow`/`sqrt`/`clip` + `Self` variants), tolerance comparisons
 * (`isClose`/`allClose`), `normSq`/`norm`/`dot`/`dist`, and `copy`/`fill`
 * are inherited from `ArrayND` unchanged; see that class for their docs.
 * `toArray` is not inherited (its natural shape differs per subclass) and
 * is defined here directly.
 */
export class Array1D extends ArrayND {
    public data: Float64Array;

    /**
     * @param input The dimension length (initialized to 0s) or initial data.
     */
    constructor(input: number | number[] | Float64Array) {
        super();
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
     * Throws if `other` is not shape-compatible with this instance: for
     * Array1D, "compatible" means the same `size`. Used internally
     * (via `ArrayND`'s arithmetic/isClose/dot/dist methods) to guard
     * against silent dimension mismatches.
     * @param other The other vector.
     * @param caller Name of the public method invoking this check, used to
     * produce a precise error message (e.g. `"add"`).
     * @throws {RangeError} If `other.size !== this.size`.
     */
    protected _checkSameShape(other: this, caller: string): void {
        if (other.size !== this.size) {
            throw new RangeError(`Array1D.${caller}: dimension mismatch: ${this.size} vs ${other.size}`);
        }
    }

    /**
     * Internal-only fast constructor: wraps `data` directly as a new
     * Array1D, with no copying and no validation whatsoever — `data` must
     * already be a fresh `Float64Array` of the right length. Used by
     * `_create()` (so `ArrayND`'s arithmetic doesn't pay for a second
     * allocation+copy on top of the buffer it already built) and by
     * `Matrix.row()` (which already owns a freshly sliced, independent
     * buffer by the time it gets here). Not part of the public API —
     * despite being a public static method (TypeScript has no
     * package-private), treat the leading underscore as a hard "don't call
     * this from outside the array module." Like `_create`'s `as this`
     * cast, this assumes Array1D is never itself subclassed.
     * @param data The buffer to wrap directly. Not copied.
     * @returns A new Array1D wrapping `data`.
     */
    static _wrapUnchecked(data: Float64Array): Array1D {
        const v = Object.create(Array1D.prototype) as Array1D;
        v.data = data;
        return v;
    }

    /**
     * Constructs a new Array1D wrapping the given buffer directly.
     * @param data The buffer for the new vector to wrap.
     * @returns A new Array1D of the same dimension as `data.length`.
     */
    protected _create(data: Float64Array): this {
        return Array1D._wrapUnchecked(data) as this;
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
    // Operations that stay vector-specific: arity/shape differ too much
    // from ArrayND's shared logic to be shared directly (e.g. `get`/`set`
    // take one index here vs two on Matrix). Matrix has its own,
    // axis-aware versions of sort/slice/cumsum/argmin/argmax — these
    // aren't shared with those, each class's version is its own design.
    // -----------------------------------------------------------------

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
     * Computes the sum of this vector's components.
     * @returns The sum, or `0` if `size === 0`.
     */
    sum(): number {
        return this._sumAll();
    }

    /**
     * Computes the arithmetic mean of this vector's components.
     * @returns The mean, or `NaN` if `size === 0`.
     */
    mean(): number {
        return this._meanAll();
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
        return this._varianceAll(ddof);
    }

    /**
     * Computes the standard deviation of this vector's components.
     * @param ddof Delta degrees of freedom, forwarded to `variance()`.
     * Defaults to `0` (population standard deviation); pass `1` for the
     * unbiased sample standard deviation.
     * @returns The standard deviation. `NaN` if `size - ddof <= 0`.
     */
    std(ddof: number = 0): number {
        return this._stdAll(ddof);
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
        return this._minAll();
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
        return ArrayND._argMinArr(this.data);
    }

    /**
     * Finds the largest component of this vector.
     * @returns The maximum value.
     * @throws {RangeError} If `size === 0`.
     */
    max(): number {
        return this._maxAll();
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
        return ArrayND._argMaxArr(this.data);
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
    // In-place (mutating) operations that stay vector-specific.
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
     * Adds a scaled vector to this one in place, in a single pass and
     * without an intermediate vector: `this += v * s`.
     * @param v The vector to scale and add. Must have the same `size` as this one.
     * @param s The scale factor applied to `v`.
     * @returns `this`, for chaining.
     */
    addScaled(v: Array1D, s: number): this {
        this._checkSameShape(v as this, 'addScaled');
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
        this._checkSameShape(a as this, 'subVectors');
        this._checkSameShape(b as this, 'subVectors');
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
