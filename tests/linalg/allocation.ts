/**
 * Counts how many `Float64Array`s `fn` creates with `new Float64Array(...)`, by wrapping the
 * global constructor while it runs. `Vector`, `Matrix` and the working buffers of the library are
 * built that way, so `0` means that the call built none. Views made by `subarray` and copies made
 * by `slice` are not counted (the engine does not go through the global), so the code under test
 * must not rely on them for the property being checked.
 *
 * @param fn The code to observe.
 * @returns The number of `new Float64Array(...)` calls made while `fn` ran.
 */
export function countFloat64ArrayAllocations(fn: () => void): number {
  const Original = globalThis.Float64Array;
  let count = 0;
  const Counting: Float64ArrayConstructor = new Proxy(Original, {
    construct(target, args, newTarget): object {
      count++;
      return Reflect.construct(target, args, newTarget === Counting ? target : newTarget);
    }
  });
  globalThis.Float64Array = Counting;
  try {
    fn();
  } finally {
    globalThis.Float64Array = Original;
  }
  return count;
}
