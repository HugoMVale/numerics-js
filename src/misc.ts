/**
 * Restricts a number to the inclusive range between `lo` and `hi`.
 *
 * @param x The number to clip.
 * @param lo The lower bound.
 * @param hi The upper bound.
 * @returns The clipped number.
 */
export function clip(x: number, lo: number, hi: number): number {
    return x < lo ? lo : x > hi ? hi : x;
}