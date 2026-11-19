import { isCloseScalar } from './scalar.js';

/**
 * A 3-component vector of `number`s, used for positions, directions,
 * velocities and the like. It is a small, allocation-friendly companion to
 * the general-purpose {@link Vector}, with the components exposed as the
 * public fields `x`, `y` and `z`.
 *
 * Most operations are non-mutating: they return a new `Vec3` and leave `this`
 * untouched. The exceptions are the methods ending in `Self`
 * (`addSelf`, `subSelf`, `multSelf`, `addScaledSelf`) and the setters
 * `set`, `setSub` and `reset`, which modify `this` in place and return it
 * for chaining.
 *
 * @example
 * ```ts
 * import { Vec3 } from 'numerics-js/linalg';
 *
 * const p = new Vec3(1, 2, 3);
 * const q = p.add(new Vec3(1, 1, 1)).mult(2);
 * console.log(q.toString());
 * p.addSelf(q);
 * console.log(p.toString());
 * ```
 *
 * Output:
 * ```text
 * Vec3(4, 6, 8)
 * Vec3(5, 8, 11)
 * ```
 */
export class Vec3 {
  /** The x component. */
  x: number;
  /** The y component. */
  y: number;
  /** The z component. */
  z: number;

  /**
   * Creates a vector from its components.
   * @param x The x component. Defaults to `0`.
   * @param y The y component. Defaults to `0`.
   * @param z The z component. Defaults to `0`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(1, 2, 3);
   * const b = new Vec3();
   * console.log(a.toString());
   * console.log(b.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(1, 2, 3)
   * Vec3(0, 0, 0)
   * ```
   */
  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  /**
   * Adds another vector to this one.
   * @param v The vector to add.
   * @returns A new vector equal to `this + v`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(1, 2, 3);
   * const b = new Vec3(4, 5, 6);
   * console.log(a.add(b).toString());
   * console.log(a.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(5, 7, 9)
   * Vec3(1, 2, 3)
   * ```
   */
  add(v: Vec3): Vec3 {
    return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z);
  }

  /**
   * Subtracts another vector from this one.
   * @param v The vector to subtract.
   * @returns A new vector equal to `this - v`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(5, 7, 9);
   * const b = new Vec3(1, 2, 3);
   * console.log(a.sub(b).toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(4, 5, 6)
   * ```
   */
  sub(v: Vec3): Vec3 {
    return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z);
  }

  /**
   * Scales this vector by a scalar.
   * @param s The scale factor.
   * @returns A new vector equal to `this * s`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(1, 2, 3);
   * console.log(v.mult(2).toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(2, 4, 6)
   * ```
   */
  mult(s: number): Vec3 {
    return new Vec3(this.x * s, this.y * s, this.z * s);
  }

  /**
   * Computes the squared magnitude (length) of this vector.
   * Cheaper than `norm()` since it avoids a square root — useful for
   * comparisons where the exact length isn't needed.
   * @returns The squared length of the vector.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * console.log(new Vec3(1, 2, 2).normSq());
   * ```
   *
   * Output:
   * ```text
   * 9
   * ```
   */
  normSq(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  /**
   * Computes the magnitude (length) of this vector.
   * @returns The length of the vector.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * console.log(new Vec3(1, 2, 2).norm());
   * ```
   *
   * Output:
   * ```text
   * 3
   * ```
   */
  norm(): number {
    return Math.sqrt(this.normSq());
  }

  /**
   * Returns a unit-length version of this vector (same direction, length 1).
   * If this vector has zero length, returns a zero vector instead of
   * dividing by zero.
   * @returns The normalized vector, or `(0, 0, 0)` if this vector is zero.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * console.log(new Vec3(0, 3, 4).normalize().toString());
   * console.log(new Vec3().normalize().toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(0, 0.6000000000000001, 0.8)
   * Vec3(0, 0, 0)
   * ```
   */
  normalize(): Vec3 {
    const m = this.norm();
    return m === 0 ? new Vec3() : this.mult(1 / m);
  }

  /**
   * Computes the dot product of this vector with another.
   * @param v The other vector.
   * @returns The scalar dot product `this · v`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(1, 2, 3);
   * const b = new Vec3(4, 5, 6);
   * console.log(a.dot(b));
   * ```
   *
   * Output:
   * ```text
   * 32
   * ```
   */
  dot(v: Vec3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  /**
   * Computes the cross product of this vector with another.
   * @param v The other vector.
   * @returns A new vector equal to `this x v`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const x = new Vec3(1, 0, 0);
   * const y = new Vec3(0, 1, 0);
   * console.log(x.cross(y).toString());
   * console.log(y.cross(x).toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(0, 0, 1)
   * Vec3(0, 0, -1)
   * ```
   */
  cross(v: Vec3): Vec3 {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }

  /**
   * Computes the Euclidean distance between this vector and another.
   * @param v The other vector.
   * @returns The distance between `this` and `v`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(1, 2, 3);
   * const b = new Vec3(4, 6, 3);
   * console.log(a.dist(b));
   * ```
   *
   * Output:
   * ```text
   * 5
   * ```
   */
  dist(v: Vec3): number {
    return this.sub(v).norm();
  }

  /**
   * Returns a copy of this vector clamped to a maximum magnitude, preserving
   * direction. If the vector's magnitude is already at or below `max`,
   * returns an unchanged copy.
   * @param max The maximum allowed magnitude. Must be >= 0.
   * @returns A new vector with magnitude at most `max`.
   * @throws {RangeError} If `max` is negative or NaN.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(3, 4, 0);
   * console.log(v.limit(2.5).toString());
   * console.log(v.limit(10).toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(1.5, 2, 0)
   * Vec3(3, 4, 0)
   * ```
   */
  limit(max: number): Vec3 {
    if (!(max >= 0)) {
      throw new RangeError(`Vec3.limit: max must be >= 0, got ${max}.`);
    }
    const m = this.norm();
    return m > max ? this.mult(max / m) : this.copy();
  }

  /**
   * Creates an independent copy of this vector.
   * @returns A new Vec3 with the same x, y, z values.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(1, 2, 3);
   * const b = a.copy();
   * b.x = 99;
   * console.log(a.toString());
   * console.log(b.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(1, 2, 3)
   * Vec3(99, 2, 3)
   * ```
   */
  copy(): Vec3 {
    return new Vec3(this.x, this.y, this.z);
  }

  /**
   * Checks whether every component of this vector is close to the
   * corresponding component of another, with the same test as
   * `Vector.allClose`: a component `a` is close to `b` if `a === b` or
   * `|a - b| <= atol + rtol * |b|`. The test is not symmetric (`rtol` scales
   * `v`'s components only), equal infinities are close, an infinite
   * component of `v` is close to nothing but itself, and NaN is never close
   * to anything.
   * @param v The other vector.
   * @param rtol Relative tolerance, applied to `v`'s components.
   * @param atol Absolute tolerance.
   * @returns `true` if all components of `this` are close to `v`'s.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(1, 2, 3);
   * console.log(a.allClose(new Vec3(1.000001, 2, 3)));
   * console.log(a.allClose(new Vec3(1.1, 2, 3)));
   * ```
   *
   * Output:
   * ```text
   * true
   * false
   * ```
   */
  allClose(v: Vec3, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    return (
      isCloseScalar(this.x, v.x, rtol, atol) &&
      isCloseScalar(this.y, v.y, rtol, atol) &&
      isCloseScalar(this.z, v.z, rtol, atol)
    );
  }

  /**
   * Returns this vector's components as a tuple.
   * @returns `[x, y, z]`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * console.log(new Vec3(1, 2, 3).toArray());
   * ```
   *
   * Output:
   * ```text
   * [ 1, 2, 3 ]
   * ```
   */
  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  /**
   * Returns a human-readable string representation of this vector.
   * @returns e.g. `"Vec3(1, 2, 3)"`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(1, 2, 3);
   * console.log(v.toString());
   * console.log(`v = ${v}`);
   * ```
   *
   * Output:
   * ```text
   * Vec3(1, 2, 3)
   * v = Vec3(1, 2, 3)
   * ```
   */
  toString(): string {
    return `Vec3(${this.x}, ${this.y}, ${this.z})`;
  }

  /**
   * Makes Vec3 iterable, e.g. `const [x, y, z] = someVec3;`.
   * @returns An iterator over `x`, `y` and `z`, in that order.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(1, 2, 3);
   * const [x, y, z] = v;
   * console.log(x + y + z);
   * console.log(Math.max(...v));
   * ```
   *
   * Output:
   * ```text
   * 6
   * 3
   * ```
   */
  [Symbol.iterator](): IterableIterator<number> {
    return [this.x, this.y, this.z][Symbol.iterator]();
  }

  // -----------------------------------------------------------------
  // In-place (mutating) operations.
  // -----------------------------------------------------------------

  /**
   * Sets this vector's components directly, mutating it in place.
   * @param x The new x component.
   * @param y The new y component.
   * @param z The new z component. Defaults to `0`.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3();
   * v.set(1, 2, 3);
   * console.log(v.toString());
   * v.set(4, 5);
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(1, 2, 3)
   * Vec3(4, 5, 0)
   * ```
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
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(1, 2, 3);
   * v.reset();
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(0, 0, 0)
   * ```
   */
  reset(): Vec3 {
    return this.set(0, 0, 0);
  }

  /**
   * Adds another vector to this one in place: `this += v`.
   * @param v The vector to add.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(1, 2, 3);
   * v.addSelf(new Vec3(10, 20, 30));
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(11, 22, 33)
   * ```
   */
  addSelf(v: Vec3): Vec3 {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }

  /**
   * Subtracts another vector from this one in place: `this -= v`.
   * @param v The vector to subtract.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(11, 22, 33);
   * v.subSelf(new Vec3(10, 20, 30));
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(1, 2, 3)
   * ```
   */
  subSelf(v: Vec3): Vec3 {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }

  /**
   * Scales this vector in place: `this *= s`.
   * @param s The scale factor.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const v = new Vec3(1, 2, 3);
   * v.multSelf(3);
   * console.log(v.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(3, 6, 9)
   * ```
   */
  multSelf(s: number): Vec3 {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }

  /**
   * Adds a scaled vector to this one in place, in a single pass and
   * without an intermediate vector: `this += v * s`.
   * @param v The vector to scale and add.
   * @param s The scale factor applied to `v`.
   * @returns `this`, for chaining.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const position = new Vec3(0, 0, 0);
   * const velocity = new Vec3(1, 2, 3);
   * position.addScaledSelf(velocity, 0.5);
   * console.log(position.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(0.5, 1, 1.5)
   * ```
   */
  addScaledSelf(v: Vec3, s: number): Vec3 {
    this.x += v.x * s;
    this.y += v.y * s;
    this.z += v.z * s;
    return this;
  }

  /**
   * Sets this vector to `a - b` in place, without allocating. Useful as a
   * reusable scratch vector inside a loop.
   * @param a The vector to subtract from.
   * @param b The vector to subtract.
   * @returns `this`, set to `a - b`.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * const a = new Vec3(5, 7, 9);
   * const b = new Vec3(1, 2, 3);
   * const scratch = new Vec3();
   * scratch.setSub(a, b);
   * console.log(scratch.toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(4, 5, 6)
   * ```
   */
  setSub(a: Vec3, b: Vec3): Vec3 {
    this.x = a.x - b.x;
    this.y = a.y - b.y;
    this.z = a.z - b.z;
    return this;
  }

  /**
   * Creates a zero vector `(0, 0, 0)`.
   * @returns A new zero vector.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * console.log(Vec3.zero().toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(0, 0, 0)
   * ```
   */
  static zero(): Vec3 {
    return new Vec3();
  }

  /**
   * Creates a Vec3 from an array or array-like of `[x, y, z]`.
   * @param arr Source values; missing entries default to 0.
   * @returns A new vector.
   *
   * @example
   * ```ts
   * import { Vec3 } from 'numerics-js/linalg';
   *
   * console.log(Vec3.from([1, 2, 3]).toString());
   * console.log(Vec3.from([1, 2]).toString());
   * ```
   *
   * Output:
   * ```text
   * Vec3(1, 2, 3)
   * Vec3(1, 2, 0)
   * ```
   */
  static from(arr: ArrayLike<number>): Vec3 {
    return new Vec3(arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0);
  }
}
