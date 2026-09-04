/**
 * Abstract base class for N-dimensional arrays backed by a flat
 * `Float64Array`. Holds every operation that needs no knowledge of the
 * concrete shape (elementwise arithmetic, reductions over the whole
 * buffer, tolerance comparisons, copying). Shape-specific behavior
 * (element access, `transpose`, `matmul`, `sort`, `slice`, `cumsum`,
 * `argmin`/`argmax`, axis-aware reductions, etc.) belongs on the
 * concrete subclasses (`Array1D`, `Matrix`), not here.
 *
 * Subclasses provide two hooks so the arithmetic family can be written
 * once, using TypeScript's polymorphic `this` type, and still return the
 * correct concrete type with no casting:
 * - `_checkSameShape`: validates that another instance has a compatible
 *   shape (not just the same `data.length`), throwing a shape-specific,
 *   precisely worded error otherwise.
 * - `_create`: constructs a new instance of the concrete subclass wrapping
 *   a given `Float64Array`, without re-running the public constructor's
 *   argument-parsing logic.
 */
export abstract class ArrayND {
    /** The flat, row-major (for multi-dimensional subclasses) backing buffer. */
    public abstract data: Float64Array;

    /**
     * The total number of elements in the underlying flat buffer. For
     * `Array1D` this is the same as `size`/dimension; for `Matrix` this is
     * `rows * cols`.
     */
    get size(): number {
        return this.data.length;
    }

    /**
     * Throws if `other` is not shape-compatible with this instance. "Shape
     * compatible" is stricter than "same `data.length`" for some
     * subclasses (e.g. a 2x3 and a 3x2 matrix have equal length but
     * incompatible shape) so this is left to the subclass to define.
     * @param other The other instance.
     * @param caller Name of the public method invoking this check, used to
     * produce a precise, class-specific error message.
     * @throws {RangeError} If `other`'s shape is not compatible with this one.
     */
    protected abstract _checkSameShape(other: this, caller: string): void;

    /**
     * Constructs a new instance of the concrete subclass, wrapping the
     * given buffer directly (no copying, no validation) and preserving
     * this instance's shape (e.g. an `Matrix`'s `rows`/`cols`).
     * @param data The buffer for the new instance to wrap. Must already be
     * the correct length for this instance's shape.
     * @returns A new instance of the same concrete type as `this`.
     */
    protected abstract _create(data: Float64Array): this;

    // -----------------------------------------------------------------
    // Immutable elementwise arithmetic (return new instances)
    // -----------------------------------------------------------------

    /**
     * Adds another instance to this one elementwise, or adds a scalar to
     * every element.
     * @param x The instance to add. Must be shape-compatible with this one.
     * @returns A new instance equal to `this + x`.
     */
    add(x: this): this;

    /**
     * Adds a scalar to every element.
     * @param s The scalar to add to every element.
     * @returns A new instance equal to `this + s`.
     */
    add(s: number): this;

    add(x: this | number): this {
        const res = this._create(new Float64Array(this.data.length));
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] + x;
        } else {
            this._checkSameShape(x, 'add');
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] + x.data[i];
        }
        return res;
    }

    /**
     * Subtracts another instance from this one elementwise, or subtracts a
     * scalar from every element.
     * @param x The instance to subtract. Must be shape-compatible with this one.
     * @returns A new instance equal to `this - x`.
     */
    sub(x: this): this;

    /**
     * Subtracts a scalar from every element.
     * @param s The scalar to subtract from every element.
     * @returns A new instance equal to `this - s`.
     */
    sub(s: number): this;

    sub(x: this | number): this {
        const res = this._create(new Float64Array(this.data.length));
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] - x;
        } else {
            this._checkSameShape(x, 'sub');
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] - x.data[i];
        }
        return res;
    }

    /**
     * Multiplies this instance with another elementwise (Hadamard
     * product), or scales every element by a scalar.
     * @param x The instance to multiply by. Must be shape-compatible with this one.
     * @returns A new instance where each element is `this[i] * x[i]`.
     */
    mult(x: this): this;

    /**
     * Scales every element by a scalar.
     * @param s The scale factor.
     * @returns A new instance equal to `this * s`.
     */
    mult(s: number): this;

    mult(x: this | number): this {
        const res = this._create(new Float64Array(this.data.length));
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] * x;
        } else {
            this._checkSameShape(x, 'mult');
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] * x.data[i];
        }
        return res;
    }

    /**
     * Divides this instance by another elementwise, or by a scalar. No
     * special handling for division by zero; elements divided by `0`
     * follow standard floating-point semantics (`Infinity`, `-Infinity`,
     * or `NaN`).
     * @param x The instance to divide by. Must be shape-compatible with this one.
     * @returns A new instance where each element is `this[i] / x[i]`.
     */
    div(x: this): this;

    /**
     * Divides every element by a scalar. No special handling for division
     * by zero; elements divided by `0` follow standard floating-point
     * semantics (`Infinity`, `-Infinity`, or `NaN`).
     * @param s The scalar to divide every element by.
     * @returns A new instance equal to `this / s`.
     */
    div(s: number): this;

    div(x: this | number): this {
        const res = this._create(new Float64Array(this.data.length));
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] / x;
        } else {
            this._checkSameShape(x, 'div');
            for (let i = 0; i < this.data.length; i++) res.data[i] = this.data[i] / x.data[i];
        }
        return res;
    }

    /**
     * Computes the elementwise absolute value.
     * @returns A new instance where each element is `|this[i]|`.
     */
    abs(): this {
        const res = this._create(new Float64Array(this.data.length));
        for (let i = 0; i < this.data.length; i++) res.data[i] = Math.abs(this.data[i]);
        return res;
    }

    /**
     * Raises each element to a power, elementwise.
     * @param exp The exponent.
     * @returns A new instance where each element is `this[i] ** exp`.
     */
    pow(exp: number): this {
        const res = this._create(new Float64Array(this.data.length));
        for (let i = 0; i < this.data.length; i++) res.data[i] = Math.pow(this.data[i], exp);
        return res;
    }

    /**
     * Computes the elementwise square root. Elements that are negative
     * produce `NaN`, matching `Math.sqrt`.
     * @returns A new instance where each element is `sqrt(this[i])`.
     */
    sqrt(): this {
        const res = this._create(new Float64Array(this.data.length));
        for (let i = 0; i < this.data.length; i++) res.data[i] = Math.sqrt(this.data[i]);
        return res;
    }

    /**
     * Clamps each element to `[min, max]`, elementwise.
     * @param min The lower bound.
     * @param max The upper bound.
     * @returns A new instance with every element clamped to `[min, max]`.
     * @throws {RangeError} If `min > max`.
     */
    clip(min: number, max: number): this {
        if (min > max) {
            throw new RangeError(`${this._className()}.clip: min (${min}) must be <= max (${max})`);
        }
        const res = this._create(new Float64Array(this.data.length));
        for (let i = 0; i < this.data.length; i++) res.data[i] = Math.min(Math.max(this.data[i], min), max);
        return res;
    }

    // -----------------------------------------------------------------
    // In-place (mutating) elementwise arithmetic
    // -----------------------------------------------------------------

    /**
     * Adds another instance to this one in place, elementwise, or adds a
     * scalar to every element: `this += x`.
     * @param x The instance to add. Must be shape-compatible with this one.
     * @returns `this`, for chaining.
     */
    addSelf(x: this): this;

    /**
     * Adds a scalar to every element in place: `this += s`.
     * @param s The scalar to add to every element.
     * @returns `this`, for chaining.
     */
    addSelf(s: number): this;

    addSelf(x: this | number): this {
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) this.data[i] += x;
        } else {
            this._checkSameShape(x, 'addSelf');
            for (let i = 0; i < this.data.length; i++) this.data[i] += x.data[i];
        }
        return this;
    }

    /**
     * Subtracts another instance from this one in place, elementwise, or
     * subtracts a scalar from every element: `this -= x`.
     * @param x The instance to subtract. Must be shape-compatible with this one.
     * @returns `this`, for chaining.
     */
    subSelf(x: this): this;

    /**
     * Subtracts a scalar from every element in place: `this -= s`.
     * @param s The scalar to subtract from every element.
     * @returns `this`, for chaining.
     */
    subSelf(s: number): this;

    subSelf(x: this | number): this {
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) this.data[i] -= x;
        } else {
            this._checkSameShape(x, 'subSelf');
            for (let i = 0; i < this.data.length; i++) this.data[i] -= x.data[i];
        }
        return this;
    }

    /**
     * Multiplies this instance with another in place, elementwise
     * (Hadamard product), or scales every element by a scalar: `this *= x`.
     * @param x The instance to multiply by. Must be shape-compatible with this one.
     * @returns `this`, for chaining.
     */
    multSelf(x: this): this;

    /**
     * Scales every element by a scalar in place: `this *= s`.
     * @param s The scale factor.
     * @returns `this`, for chaining.
     */
    multSelf(s: number): this;

    multSelf(x: this | number): this {
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) this.data[i] *= x;
        } else {
            this._checkSameShape(x, 'multSelf');
            for (let i = 0; i < this.data.length; i++) this.data[i] *= x.data[i];
        }
        return this;
    }

    /**
     * Divides this instance by another in place, elementwise, or by a
     * scalar: `this /= x`. No special handling for division by zero;
     * elements divided by `0` follow standard floating-point semantics
     * (`Infinity`, `-Infinity`, or `NaN`).
     * @param x The instance to divide by. Must be shape-compatible with this one.
     * @returns `this`, for chaining.
     */
    divSelf(x: this): this;

    /**
     * Divides every element by a scalar in place: `this /= s`. No special
     * handling for division by zero; elements divided by `0` follow
     * standard floating-point semantics (`Infinity`, `-Infinity`, or `NaN`).
     * @param s The scalar to divide every element by.
     * @returns `this`, for chaining.
     */
    divSelf(s: number): this;

    divSelf(x: this | number): this {
        if (typeof x === 'number') {
            for (let i = 0; i < this.data.length; i++) this.data[i] /= x;
        } else {
            this._checkSameShape(x, 'divSelf');
            for (let i = 0; i < this.data.length; i++) this.data[i] /= x.data[i];
        }
        return this;
    }

    /**
     * Takes the elementwise absolute value in place.
     * @returns `this`, for chaining.
     */
    absSelf(): this {
        for (let i = 0; i < this.data.length; i++) this.data[i] = Math.abs(this.data[i]);
        return this;
    }

    /**
     * Raises each element to a power in place, elementwise.
     * @param exp The exponent.
     * @returns `this`, for chaining.
     */
    powSelf(exp: number): this {
        for (let i = 0; i < this.data.length; i++) this.data[i] = Math.pow(this.data[i], exp);
        return this;
    }

    /**
     * Takes the elementwise square root in place. Elements that are
     * negative become `NaN`, matching `Math.sqrt`.
     * @returns `this`, for chaining.
     */
    sqrtSelf(): this {
        for (let i = 0; i < this.data.length; i++) this.data[i] = Math.sqrt(this.data[i]);
        return this;
    }

    /**
     * Clamps each element to `[min, max]` in place, elementwise.
     * @param min The lower bound.
     * @param max The upper bound.
     * @returns `this`, for chaining.
     * @throws {RangeError} If `min > max`.
     */
    clipSelf(min: number, max: number): this {
        if (min > max) {
            throw new RangeError(`${this._className()}.clipSelf: min (${min}) must be <= max (${max})`);
        }
        for (let i = 0; i < this.data.length; i++) this.data[i] = Math.min(Math.max(this.data[i], min), max);
        return this;
    }

    // -----------------------------------------------------------------
    // Tolerance comparisons
    // -----------------------------------------------------------------

    /**
     * Tests whether a single pair of elements is "close": `a` (from
     * `this`) is close to `b` (from the argument) if
     * `|a - b| <= atol + rtol * |b|`. Shared by `isClose` (which needs
     * every element's result) and `allClose` (which only needs the first
     * failure, if any, and stops there).
     */
    private static _closeAt(a: number, b: number, rtol: number, atol: number): boolean {
        return Math.abs(a - b) <= atol + rtol * Math.abs(b);
    }

    /**
     * Elementwise tolerance comparison. An element `a` (from `this`) is
     * close to `b` (from `x`) if `|a - b| <= atol + rtol * |b|`. Note this
     * is **not symmetric**: `rtol` scales `x`'s elements only, so in
     * general `a.isClose(b) !== b.isClose(a)`. NaN elements are never
     * close to anything, including other NaNs.
     * @param x The other instance. Must be shape-compatible with this one.
     * @param rtol Relative tolerance, applied to `x`'s elements.
     * @param atol Absolute tolerance.
     * @returns A boolean array, one entry per element of the flat buffer.
     * @throws {RangeError} If `x` is not shape-compatible with this instance.
     */
    isClose(x: this, rtol: number = 1e-5, atol: number = 1e-8): boolean[] {
        this._checkSameShape(x, 'isClose');
        const result: boolean[] = new Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            result[i] = ArrayND._closeAt(this.data[i], x.data[i], rtol, atol);
        }
        return result;
    }

    /**
     * Checks whether this instance is close to another across *every*
     * element, using the same `|a - b| <= atol + rtol * |b|` test as
     * `isClose` (same asymmetry, same NaN handling). Unlike
     * `isClose(x, rtol, atol).every(Boolean)`, this stops at the first
     * non-close pair instead of computing and allocating a full result
     * array first.
     * @param x The other instance. Must be shape-compatible with this one.
     * @param rtol Relative tolerance, applied to `x`'s elements.
     * @param atol Absolute tolerance.
     * @returns `true` if every element of `this` is close to the corresponding element of `x`.
     * @throws {RangeError} If `x` is not shape-compatible with this instance.
     */
    allClose(x: this, rtol: number = 1e-5, atol: number = 1e-8): boolean {
        this._checkSameShape(x, 'allClose');
        for (let i = 0; i < this.data.length; i++) {
            if (!ArrayND._closeAt(this.data[i], x.data[i], rtol, atol)) return false;
        }
        return true;
    }

    // -----------------------------------------------------------------
    // Norms, inner product, distance (whole-buffer, no axis concept)
    // -----------------------------------------------------------------

    /**
     * Computes the squared Euclidean norm of the flat buffer (for
     * `Array1D`, the squared vector length; for `Matrix`, the squared
     * Frobenius norm). Cheaper than `norm()` since it avoids a square root.
     * @returns The sum of squares of every element.
     */
    normSq(): number {
        let sum = 0;
        for (let i = 0; i < this.data.length; i++) sum += this.data[i] * this.data[i];
        return sum;
    }

    /**
     * Computes the Euclidean norm of the flat buffer (for `Array1D`, the
     * vector length; for `Matrix`, the Frobenius norm).
     * @returns `sqrt(normSq())`.
     */
    norm(): number {
        return Math.sqrt(this.normSq());
    }

    /**
     * Computes the elementwise (Frobenius, for `Matrix`) inner product of
     * this instance with another.
     * @param x The other instance. Must be shape-compatible with this one.
     * @returns The scalar `this · x`, summed over every element of the flat buffer.
     * @throws {RangeError} If `x` is not shape-compatible with this instance.
     */
    dot(x: this): number {
        this._checkSameShape(x, 'dot');
        let sum = 0;
        for (let i = 0; i < this.data.length; i++) sum += this.data[i] * x.data[i];
        return sum;
    }

    /**
     * Computes the Euclidean distance between this instance and another,
     * treating both as flat buffers (for `Matrix`, this is the Frobenius
     * distance).
     * @param x The other instance. Must be shape-compatible with this one.
     * @returns The distance between `this` and `x`.
     * @throws {RangeError} If `x` is not shape-compatible with this instance.
     */
    dist(x: this): number {
        this._checkSameShape(x, 'dist');
        let sum = 0;
        for (let i = 0; i < this.data.length; i++) {
            const d = this.data[i] - x.data[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    // -----------------------------------------------------------------
    // Pure, array-level reducers: operate on any Float64Array, not just
    // this instance's own `data`. This is what lets Matrix reuse the
    // exact same reduction/argmin/argmax logic for its axis-aware
    // methods (reducing one row or column at a time), instead of
    // duplicating it. The `_xAll`/instance-level methods below are thin
    // wrappers over these, applied to `this.data`.
    // -----------------------------------------------------------------

    /**
     * Sums every element of `a`.
     * @returns The sum, or `0` if `a` is empty.
     */
    protected static _sumArr(a: Float64Array): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i];
        return s;
    }

    /**
     * Computes the arithmetic mean of every element of `a`.
     * @returns The mean, or `NaN` if `a` is empty.
     */
    protected static _meanArr(a: Float64Array): number {
        return ArrayND._sumArr(a) / a.length;
    }

    /**
     * Finds the smallest element of `a`. NaN wins the comparison so it
     * propagates, matching `Math.min`'s semantics. Assumes `a` is
     * non-empty; callers are responsible for that check (so they can
     * produce a class-specific error message).
     */
    protected static _minArr(a: Float64Array): number {
        let m = a[0];
        for (let i = 1; i < a.length; i++) {
            const x = a[i];
            if (Number.isNaN(x) || x < m) m = x;
        }
        return m;
    }

    /**
     * Finds the largest element of `a`. NaN wins the comparison so it
     * propagates, matching `Math.max`'s semantics. Assumes `a` is
     * non-empty; callers are responsible for that check (so they can
     * produce a class-specific error message).
     */
    protected static _maxArr(a: Float64Array): number {
        let m = a[0];
        for (let i = 1; i < a.length; i++) {
            const x = a[i];
            if (Number.isNaN(x) || x > m) m = x;
        }
        return m;
    }

    /**
     * Computes the variance of `a`'s elements: the mean of the squared
     * deviations from the mean.
     * @param ddof Delta degrees of freedom. The divisor used is `a.length - ddof`.
     * @returns The variance. `NaN` if `a.length - ddof <= 0`.
     */
    protected static _varianceArr(a: Float64Array, ddof: number): number {
        const m = ArrayND._meanArr(a);
        let sq = 0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - m;
            sq += d * d;
        }
        return sq / (a.length - ddof);
    }

    /**
     * Computes the standard deviation of `a`'s elements.
     * @param ddof Delta degrees of freedom, forwarded to `_varianceArr()`.
     * @returns The standard deviation. `NaN` if `a.length - ddof <= 0`.
     */
    protected static _stdArr(a: Float64Array, ddof: number): number {
        return Math.sqrt(ArrayND._varianceArr(a, ddof));
    }

    /**
     * Finds the index of the smallest value in `a`. If multiple values tie
     * for the minimum, the first index is returned. A NaN value takes
     * priority and its index is returned immediately, matching
     * `_minArr`'s NaN-propagation semantics. Assumes `a` is non-empty;
     * callers are responsible for that check (so they can produce a
     * class-specific error message).
     */
    protected static _argMinArr(a: Float64Array): number {
        if (Number.isNaN(a[0])) return 0;
        let mi = 0;
        let m = a[0];
        for (let i = 1; i < a.length; i++) {
            const x = a[i];
            if (Number.isNaN(x)) return i;
            if (x < m) { m = x; mi = i; }
        }
        return mi;
    }

    /**
     * Finds the index of the largest value in `a`. If multiple values tie
     * for the maximum, the first index is returned. A NaN value takes
     * priority and its index is returned immediately, matching
     * `_maxArr`'s NaN-propagation semantics. Assumes `a` is non-empty;
     * callers are responsible for that check (so they can produce a
     * class-specific error message).
     */
    protected static _argMaxArr(a: Float64Array): number {
        if (Number.isNaN(a[0])) return 0;
        let mi = 0;
        let m = a[0];
        for (let i = 1; i < a.length; i++) {
            const x = a[i];
            if (Number.isNaN(x)) return i;
            if (x > m) { m = x; mi = i; }
        }
        return mi;
    }

    // -----------------------------------------------------------------
    // Whole-buffer reductions, applying the reducers above to `this.data`.
    // Protected: subclasses expose these as public methods, adding an
    // `axis` parameter where numpy would (only meaningful for shapes with
    // more than one axis, i.e. not Array1D).
    // -----------------------------------------------------------------

    /**
     * Sums every element of the flat buffer.
     * @returns The sum, or `0` if the buffer is empty.
     */
    protected _sumAll(): number {
        return ArrayND._sumArr(this.data);
    }

    /**
     * Computes the arithmetic mean of every element of the flat buffer.
     * @returns The mean, or `NaN` if the buffer is empty.
     */
    protected _meanAll(): number {
        return ArrayND._meanArr(this.data);
    }

    /**
     * Finds the smallest element of the flat buffer. NaN wins the
     * comparison so it propagates, matching `Math.min`'s semantics.
     * @returns The minimum value.
     * @throws {RangeError} If the buffer is empty.
     */
    protected _minAll(): number {
        if (this.data.length === 0) {
            throw new RangeError(`${this._className()}.min: cannot compute the min of an empty array`);
        }
        return ArrayND._minArr(this.data);
    }

    /**
     * Finds the largest element of the flat buffer. NaN wins the
     * comparison so it propagates, matching `Math.max`'s semantics.
     * @returns The maximum value.
     * @throws {RangeError} If the buffer is empty.
     */
    protected _maxAll(): number {
        if (this.data.length === 0) {
            throw new RangeError(`${this._className()}.max: cannot compute the max of an empty array`);
        }
        return ArrayND._maxArr(this.data);
    }

    /**
     * Computes the variance of every element of the flat buffer: the mean
     * of the squared deviations from the mean.
     * @param ddof Delta degrees of freedom. The divisor used is
     * `size - ddof`. Defaults to `0` (population variance); pass `1` for
     * the unbiased sample variance.
     * @returns The variance. `NaN` if `size - ddof <= 0`.
     */
    protected _varianceAll(ddof: number = 0): number {
        return ArrayND._varianceArr(this.data, ddof);
    }

    /**
     * Computes the standard deviation of every element of the flat buffer.
     * @param ddof Delta degrees of freedom, forwarded to `_varianceAll()`.
     * @returns The standard deviation. `NaN` if `size - ddof <= 0`.
     */
    protected _stdAll(ddof: number = 0): number {
        return Math.sqrt(this._varianceAll(ddof));
    }

    // -----------------------------------------------------------------
    // Copying / filling
    // -----------------------------------------------------------------

    /**
     * Creates an independent copy of this instance.
     * @returns A new instance of the same concrete type, with the same values.
     */
    copy(): this {
        return this._create(this.data.slice());
    }

    /**
     * Sets every element of the flat buffer to `value`, in place.
     * @param value The value to fill with.
     * @returns `this`, for chaining.
     */
    fill(value: number): this {
        this.data.fill(value);
        return this;
    }

    /**
     * The concrete subclass's name, used to produce precise, class-specific
     * error messages from methods implemented here.
     */
    private _className(): string {
        return this.constructor.name;
    }
}