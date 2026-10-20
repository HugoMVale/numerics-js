import { ArrayND } from './ArrayND.js';
import { Matrix } from './Matrix.js';

/**
 * Anything that can be used as a numeric vector input: a {@link Vector}, a plain (possibly
 * read-only) `number[]`, or a `Float64Array`. Public functions that take a numeric sequence
 * accept this type and normalize it once with {@link Vector.asVector}.
 */
export type VectorLike = Vector | readonly number[] | Float64Array;

/**
 * An N-component vector utilizing Float64Array for performance.
 *
 */
export class Vector extends ArrayND {
  public data: Float64Array;

  /**
   * @param input The dimension length (initialized to 0s) or initial data.
   */
  constructor(input: number | number[] | Float64Array) {
    super();
    if (typeof input === 'number') {
      if (!Number.isInteger(input)) {
        throw new RangeError(`Vector.constructor: dimension must be an integer, got ${input}`);
      }
      if (input < 0) {
        throw new RangeError(`Vector.constructor: dimension must be >= 0, got ${input}`);
      }
      this.data = new Float64Array(input);
    } else {
      this.data = new Float64Array(input);
    }
  }

  /**
   * Throws if `other` is not shape-compatible with this instance: for
   * Vector, "compatible" means the same `size`. Used internally
   * (via `ArrayND`'s arithmetic/isClose/dot/dist methods) to guard
   * against silent dimension mismatches.
   * @param other The other vector.
   * @param caller Name of the public method invoking this check, used to
   * produce a precise error message (e.g. `"add"`).
   * @throws {RangeError} If `other.size !== this.size`.
   */
  protected _checkSameShape(other: this, caller: string): void {
    if (other.size !== this.size) {
      throw new RangeError(`Vector.${caller}: dimension mismatch: ${this.size} vs ${other.size}`);
    }
  }

  /**
   * Internal-only fast constructor: wraps `data` directly as a new
   * Vector, with no copying and no validation whatsoever — `data` must
   * already be a fresh `Float64Array` of the right length. Used by
   * `_create()` (so `ArrayND`'s arithmetic doesn't pay for a second
   * allocation+copy on top of the buffer it already built) and by
   * `Matrix.row()` (which already owns a freshly sliced, independent
   * buffer by the time it gets here). `public` only because TypeScript
   * has no package-private — the name (rather than a leading
   * underscore) is the actual "don't call this unless you already hold
   * a validated, freshly-built buffer" signal: skip it and you can
   * construct a Vector whose declared size doesn't match its buffer's
   * length. Like `_create`'s `as this` cast, this assumes Vector is
   * never itself subclassed.
   * @param data The buffer to wrap directly. Not copied.
   * @returns A new Vector wrapping `data`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const buf = new Float64Array([1, 2, 3]);
   * const v = Vector.wrapUnchecked(buf);
   * buf[0] = 10;
   * console.log(v.toString());
   * console.log(v.size);
   * console.log(v.data === buf);
   * ```
   *
   * Output:
   * ```text
   * Vector(10, 2, 3)
   * 3
   * true
   * ```
   */
  static wrapUnchecked(data: Float64Array): Vector {
    const v = Object.create(Vector.prototype) as Vector;
    v.data = data;
    return v;
  }

  /**
   * Constructs a new Vector wrapping the given buffer directly.
   * @param data The buffer for the new vector to wrap.
   * @returns A new Vector of the same dimension as `data.length`.
   */
  protected _create(data: Float64Array): this {
    return Vector.wrapUnchecked(data) as this;
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
      throw new RangeError(`Vector.${caller}: index ${i} out of bounds for dimension ${this.size}`);
    }
  }

  /**
   * Gets a component by its 0-based index.
   * @param i Component index (0-based).
   * @returns The component value.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([10, 20, 30]);
   * console.log(v.get(0));
   * console.log(v.get(2));
   * ```
   *
   * Output:
   * ```text
   * 10
   * 30
   * ```
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([3, 1, 10, 2]);
   * console.log(v.sort().toString());
   * console.log(v.sort((a, b) => b - a).toString());
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3, 10)
   * Vector(10, 3, 2, 1)
   * Vector(3, 1, 10, 2)
   * ```
   */
  sort(compareFn?: (a: number, b: number) => number): Vector {
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([0, 1, 2, 3, 4, 5]);
   * console.log(v.slice(1, 4).toString());
   * console.log(v.slice(-2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3)
   * Vector(4, 5)
   * ```
   */
  slice(start?: number, end?: number): Vector {
    return new Vector(this.data.slice(start, end));
  }

  /**
   * Repeats each component `repeats` times in place, like
   * `numpy.repeat(a, repeats)`: `[1, 2, 3].repeat(2)` is
   * `[1, 1, 2, 2, 3, 3]`. Contrast with `tile()`, which repeats the
   * whole vector. Only a scalar count is supported (no per-element counts).
   * @param repeats How many times to repeat each component. Must be a
   * non-negative integer; `0` yields an empty vector.
   * @returns A new, independent vector of size `size * repeats`.
   * @throws {RangeError} If `repeats` is not a non-negative integer.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(v.repeat(2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 1, 2, 2, 3, 3)
   * ```
   */
  repeat(repeats: number): Vector {
    ArrayND._checkCount(repeats, 'Vector.repeat');
    const res = new Vector(this.size * repeats);
    ArrayND._repeatElements(this.data, repeats, res.data, 0);
    return res;
  }

  /**
   * Concatenates this vector with itself `reps` times, like
   * `numpy.tile(a, reps)`: `[1, 2, 3].tile(2)` is `[1, 2, 3, 1, 2, 3]`.
   * Contrast with `repeat()`, which repeats each component in place.
   * @param reps How many copies to concatenate. Must be a non-negative
   * integer; `0` yields an empty vector.
   * @returns A new, independent vector of size `size * reps`.
   * @throws {RangeError} If `reps` is not a non-negative integer.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(v.tile(2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3, 1, 2, 3)
   * ```
   */
  tile(reps: number): Vector {
    ArrayND._checkCount(reps, 'Vector.tile', 'reps');
    const res = new Vector(this.size * reps);
    ArrayND._tileArray(this.data, reps, res.data, 0);
    return res;
  }

  /**
   * Applies a function to each component, elementwise.
   * @param fn Called with each component's value and index; its return
   * value becomes the corresponding component of the result.
   * @returns A new vector, the same size as this one, holding the mapped values.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(v.map((x, i) => x * x + i).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 5, 11)
   * ```
   */
  map(fn: (value: number, index: number) => number): Vector {
    const res = new Vector(this.size);
    for (let i = 0; i < this.size; i++) res.data[i] = fn(this.data[i], i);
    return res;
  }

  /**
   * Returns a unit-length version of this vector (same direction, length 1).
   * If this vector has zero length, returns a zero vector instead of
   * dividing by zero.
   * @returns The normalized vector, or a zero vector if this vector is zero.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([3, 4]);
   * console.log(v.normalize().toString());
   * console.log(Vector.zero(2).normalize().toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0.6000000000000001, 0.8)
   * Vector(0, 0)
   * ```
   */
  normalize(): Vector {
    const m = this.norm();
    return m === 0 ? new Vector(this.size) : this.mult(1 / m);
  }

  /**
   * Computes the outer product of this vector and another: a new
   * `this.size x v.size` matrix `M` with `M[i][j] === this[i] * v[j]`.
   * Unlike `dot` (inherited from `ArrayND`), the two vectors don't need
   * matching sizes — `this` runs down the rows, `v` runs across the
   * columns. Used, e.g., to build a rank-1 modification `u * v^T` for
   * `Matrix.qrUpdate`/`qrUpdateSelf`.
   * @param v The other vector. May have a different `size` than this one.
   * @returns A new `this.size x v.size` matrix.
   * @throws {RangeError} If this vector or `v` is empty (`size === 0`)
   * — same restriction `Matrix` places on its own dimensions.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const u = Vector.from([1, 2]);
   * const v = Vector.from([3, 4, 5]);
   * console.log(u.outer(v).toString());
   * ```
   *
   * Output:
   * ```text
   * Matrix[[3, 4, 5], [6, 8, 10]]
   * ```
   */
  outer(v: Vector): Matrix {
    const n = v.size;
    const data = new Float64Array(this.size * n);
    for (let i = 0; i < this.size; i++) {
      const ui = this.data[i];
      if (ui === 0) continue;
      const offset = i * n;
      for (let j = 0; j < n; j++) data[offset + j] = ui * v.data[j];
    }
    return new Matrix(this.size, n, data);
  }

  /**
   * Computes the sum of this vector's components.
   * @returns The sum, or `0` if `size === 0`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3, 4]);
   * console.log(v.sum());
   * ```
   *
   * Output:
   * ```text
   * 10
   * ```
   */
  sum(): number {
    return this._sumAll();
  }

  /**
   * Computes the arithmetic mean of this vector's components.
   * @returns The mean, or `NaN` if `size === 0`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3, 4]);
   * console.log(v.mean());
   * ```
   *
   * Output:
   * ```text
   * 2.5
   * ```
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3, 4]);
   * console.log(v.variance());
   * console.log(v.variance(1));
   * ```
   *
   * Output:
   * ```text
   * 1.25
   * 1.6666666666666667
   * ```
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3, 4]);
   * console.log(v.std());
   * console.log(v.std(1));
   * ```
   *
   * Output:
   * ```text
   * 1.118033988749895
   * 1.2909944487358056
   * ```
   */
  std(ddof: number = 0): number {
    return this._stdAll(ddof);
  }

  /**
   * Computes the cumulative sum of this vector's components.
   * @returns A new vector of the same size where component `i` is the
   * sum of `this[0..i]`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3, 4]);
   * console.log(v.cumsum().toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 3, 6, 10)
   * ```
   */
  cumsum(): Vector {
    const res = new Vector(this.size);
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([4, -1, 7, 2]);
   * console.log(v.min());
   * ```
   *
   * Output:
   * ```text
   * -1
   * ```
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([4, -1, 7, 2]);
   * console.log(v.argmin());
   * ```
   *
   * Output:
   * ```text
   * 1
   * ```
   */
  argmin(): number {
    if (this.size === 0)
      throw new RangeError('Vector.argmin: cannot compute the argmin of an empty vector');
    return ArrayND._argMinArr(this.data);
  }

  /**
   * Finds the largest component of this vector.
   * @returns The maximum value.
   * @throws {RangeError} If `size === 0`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([4, -1, 7, 2]);
   * console.log(v.max());
   * ```
   *
   * Output:
   * ```text
   * 7
   * ```
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([4, -1, 7, 2]);
   * console.log(v.argmax());
   * ```
   *
   * Output:
   * ```text
   * 2
   * ```
   */
  argmax(): number {
    if (this.size === 0)
      throw new RangeError('Vector.argmax: cannot compute the argmax of an empty vector');
    return ArrayND._argMaxArr(this.data);
  }

  /**
   * Returns this vector's components as a plain array.
   * @returns The components, in order.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(v.toArray());
   * ```
   *
   * Output:
   * ```text
   * [ 1, 2, 3 ]
   * ```
   */
  toArray(): number[] {
    return Array.from(this.data);
  }

  /**
   * Returns a human-readable string representation of this vector.
   * @returns e.g. `"Vector(1, 2, 3)"`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(v.toString());
   * console.log(`v = ${v}`);
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3)
   * v = Vector(1, 2, 3)
   * ```
   */
  toString(): string {
    return `Vector(${this.data.join(', ')})`;
  }

  /**
   * Makes Vector iterable, e.g. `const [a, b, c] = someVector;` or `for (const x of v)`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * for (const x of v) console.log(x);
   * const [a, b] = v;
   * console.log(a + b);
   * ```
   *
   * Output:
   * ```text
   * 1
   * 2
   * 3
   * 3
   * ```
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.zero(3);
   * v.set([1, 2, 3]);
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3)
   * ```
   */
  set(values: number[] | Float64Array): this;

  /**
   * Sets one component by its 0-based index, mutating this vector in place.
   * @param i Component index (0-based).
   * @param value Value to store.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.zero(3);
   * v.set(1, 20);
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0, 20, 0)
   * ```
   */
  set(i: number, value: number): this;

  set(valuesOrIndex: number[] | Float64Array | number, value?: number): this {
    if (typeof valuesOrIndex === 'number') {
      this._checkIndex(valuesOrIndex, 'set');
      this.data[valuesOrIndex] = value as number;
    } else {
      if (valuesOrIndex.length !== this.size) {
        throw new RangeError(
          `Vector.set: expected ${this.size} values, got ${valuesOrIndex.length}`
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const x = Vector.from([1, 2, 3]);
   * const d = Vector.from([1, 0, -1]);
   * x.addScaled(d, 0.5);
   * console.log(x.toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1.5, 2, 2.5)
   * ```
   */
  addScaled(v: Vector, s: number): this {
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const a = Vector.from([5, 7, 9]);
   * const b = Vector.from([1, 2, 3]);
   * const out = Vector.zero(3);
   * out.subVectors(a, b);
   * console.log(out.toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(4, 5, 6)
   * ```
   */
  subVectors(a: Vector, b: Vector): this {
    this._checkSameShape(a as this, 'subVectors');
    this._checkSameShape(b as this, 'subVectors');
    for (let i = 0; i < this.size; i++) this.data[i] = a.data[i] - b.data[i];
    return this;
  }

  /**
   * Creates a zero vector of the given dimension.
   * @param dim The number of components.
   * @returns A new zero vector.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.zero(3).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0, 0, 0)
   * ```
   */
  static zero(dim: number): Vector {
    return new Vector(dim);
  }

  /**
   * Creates a vector of the given dimension filled with `1`.
   * @param dim The number of components.
   * @returns A new vector of all ones.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.ones(3).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 1, 1)
   * ```
   */
  static ones(dim: number): Vector {
    return new Vector(dim).fill(1);
  }

  /**
   * Creates a vector of the given dimension filled with `value`.
   * @param dim The number of components.
   * @param value The value to fill with.
   * @returns A new vector with every component equal to `value`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.full(3, 2.5).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(2.5, 2.5, 2.5)
   * ```
   */
  static full(dim: number, value: number): Vector {
    return new Vector(dim).fill(value);
  }

  /**
   * Creates a Vector from an array or typed array.
   * @param arr Source values.
   * @returns A new vector with dimension equal to `arr.length`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 2, 3)
   * ```
   */
  static from(arr: number[] | Float64Array): Vector {
    return new Vector(arr);
  }

  /**
   * Normalizes a {@link VectorLike} input to a `Vector`, in the spirit of `numpy.asarray`.
   * A `Vector` is returned as is (the same instance, no copy); a `number[]`, read-only array or
   * `Float64Array` is copied into a new `Vector`. Contrast with {@link Vector.from}, which always
   * returns a fresh copy. Only the container type is validated, not the individual elements.
   * @param x A `Vector`, a plain `number[]` (possibly read-only) or a `Float64Array`.
   * @returns `x` itself if it is a `Vector`, otherwise a new vector with the same components.
   * @throws {TypeError} If `x` is not a `Vector`, `number[]` or `Float64Array`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * const v = Vector.from([1, 2, 3]);
   * console.log(Vector.asVector(v) === v);
   * console.log(Vector.asVector([4, 5, 6]).toString());
   * ```
   *
   * Output:
   * ```text
   * true
   * Vector(4, 5, 6)
   * ```
   */
  static asVector(x: VectorLike): Vector {
    if (x instanceof Vector) return x;
    if (Array.isArray(x) || x instanceof Float64Array) {
      const v = new Vector(x.length);
      v.data.set(x);
      return v;
    }
    const got = x === null ? 'null' : typeof x === 'object' ? x.constructor?.name : typeof x;
    throw new TypeError(
      `Vector.asVector: expected a Vector, number[] or Float64Array, got ${got ?? 'object'}`
    );
  }

  /**
   * Creates a vector of evenly spaced values within a half-open interval
   * `[start, stop)`, stepping by `step`. Distinct from `linspace`, which
   * takes a sample *count* instead of a step size.
   * @param stop Exclusive upper bound, with `start` defaulting to `0` and `step` to `1`.
   * @returns A new vector of values `0, 1, 2, ..., stop - 1`.
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.arange(4).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0, 1, 2, 3)
   * ```
   */
  static arange(stop: number): Vector;

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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.arange(0, 1, 0.25).toString());
   * console.log(Vector.arange(5, 0, -2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0, 0.25, 0.5, 0.75)
   * Vector(5, 3, 1)
   * ```
   */
  static arange(start: number, stop: number, step?: number): Vector;

  static arange(startOrStop: number, stop?: number, step: number = 1): Vector {
    const start = stop === undefined ? 0 : startOrStop;
    const actualStop = stop === undefined ? startOrStop : stop;
    if (step === 0) throw new RangeError('Vector.arange: step must not be 0');
    const n = Math.max(0, Math.ceil((actualStop - start) / step));
    const res = new Vector(n);
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.linspace(0, 1, 5).toString());
   * console.log(Vector.linspace(0, 1, 4, false).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(0, 0.25, 0.5, 0.75, 1)
   * Vector(0, 0.25, 0.5, 0.75)
   * ```
   */
  static linspace(start: number, stop: number, num: number = 50, endpoint: boolean = true): Vector {
    if (!Number.isInteger(num) || num < 0) {
      throw new RangeError(`Vector.linspace: num must be a non-negative integer, got ${num}`);
    }
    const res = new Vector(num);
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
   *
   * @example
   * ```ts
   * import { Vector } from 'numerics-js/linalg';
   *
   * console.log(Vector.logspace(0, 2, 3).toString());
   * console.log(Vector.logspace(0, 3, 4, true, 2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vector(1, 10, 100)
   * Vector(1, 2, 4, 8)
   * ```
   */
  static logspace(
    start: number,
    stop: number,
    num: number = 50,
    endpoint: boolean = true,
    base: number = 10
  ): Vector {
    const exponents = Vector.linspace(start, stop, num, endpoint);
    const res = new Vector(num);
    for (let i = 0; i < num; i++) res.data[i] = Math.pow(base, exponents.data[i]);
    return res;
  }
}
