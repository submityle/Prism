//! Ordered (Bayer) dithering — CPU golden reference.
//!
//! Ordered dithering hides low-bit-depth quantization banding by adding a
//! small, *spatially structured*, signal-independent threshold pattern before
//! quantization. The classic pattern is the **Bayer / dispersed-dot** matrix,
//! a recursively defined `N×N` arrangement of the integers `0 .. N²` whose
//! value grows as "dispersed" as possible so the resulting dither reads as a
//! fine, non-clumping texture rather than visible contours.
//!
//! The recursion is the standard dispersed-dot construction. Starting from the
//! `1×1` matrix `M₁ = [0]`, each doubling applies:
//!
//! ```text
//! M_{2n} = | 4·M_n + 0   4·M_n + 2 |
//!          | 4·M_n + 3   4·M_n + 1 |
//! ```
//!
//! which yields the familiar `2×2`, `4×4`, and `8×8` matrices. A tile value
//! `M_N(x, y) ∈ [0, N²)` is normalized to a zero-mean threshold in
//! `[-0.5, 0.5)` via `(M_N(x, y) + 0.5) / N² − 0.5`; averaged over a full tile
//! the thresholds sum to exactly zero, so ordered dither introduces no DC bias.
//!
//! It is deliberately distinct from the stochastic grain in
//! [`crate::gi::film_grain`]: that pass models signal-dependent photographic
//! noise, whereas this pass is a deterministic, periodic threshold map used to
//! break up quantization steps.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Tile sizes are powers of two; a requested size is clamped to the largest
//!   valid power of two in `[1, MAX_BAYER_SIZE]`.
//! * Pixel coordinates are reduced modulo the tile size, so the pattern tiles
//!   the plane seamlessly.
//! * Thresholds are finite and lie in `[-0.5, 0.5)`; a full tile is zero-mean.
//! * Transcendental-free; only integer arithmetic plus one float divide.
//!
//! # References
//! * B. E. Bayer, "An Optimum Method for Two-Level Rendition of
//!   Continuous-Tone Pictures", IEEE ICC, 1973.
//! * R. Ulichney, *Digital Halftoning*, MIT Press, 1987 (dispersed-dot
//!   ordered dither).

/// Largest Bayer tile edge supported by the runtime generator (`2⁸`).
///
/// The recursive generator descends one level per factor of two, so this bounds
/// the recursion depth at eight. Requested sizes above this are clamped down.
pub const MAX_BAYER_SIZE: u32 = 256;

/// The `2×2` dispersed-dot base cell used by the doubling recursion.
///
/// Row-major `[y][x]`. Every larger matrix is `4·M_{N/2}` plus the entry of
/// this cell selected by the high bit of each coordinate.
const BAYER_BASE: [[u32; 2]; 2] = [[0, 2], [3, 1]];

/// Canonical `2×2` Bayer matrix (`M₂`), row-major `[y][x]`.
pub const BAYER_2: [[u32; 2]; 2] = [[0, 2], [3, 1]];

/// Canonical `4×4` Bayer matrix (`M₄`), row-major `[y][x]`.
pub const BAYER_4: [[u32; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

/// Canonical `8×8` Bayer matrix (`M₈`), row-major `[y][x]`.
pub const BAYER_8: [[u32; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// Reduces a requested tile `size` to a valid Bayer edge.
///
/// Returns the largest power of two that is `≤ size` and within
/// `[1, MAX_BAYER_SIZE]`. A `size` of `0` maps to `1` (a degenerate `1×1`
/// tile whose single threshold is `0`), guaranteeing the generator always has a
/// well-defined power-of-two edge.
#[inline]
#[must_use]
pub fn normalize_size(size: u32) -> u32 {
    let clamped = size.clamp(1, MAX_BAYER_SIZE);
    if clamped.is_power_of_two() {
        clamped
    } else {
        // Highest power of two not exceeding `clamped`. `leading_zeros` is in
        // `[0, 31]` here because `clamped >= 1`, so the shift is well-defined.
        1u32 << (31 - clamped.leading_zeros())
    }
}

/// Computes the raw Bayer tile value `M_size(x, y) ∈ [0, size²)`.
///
/// `size` must already be a power of two (see [`normalize_size`]); coordinates
/// are reduced modulo `size` so the pattern tiles. The value is built by the
/// dispersed-dot doubling recursion: at each level the high bit of each
/// coordinate selects an entry of [`BAYER_BASE`], which is added on top of
/// `4×` the finer-level value.
///
/// Returns `0` for the degenerate `size <= 1` tile.
#[must_use]
pub fn bayer_value(x: u32, y: u32, size: u32) -> u32 {
    let size = normalize_size(size);
    bayer_recursive(x % size, y % size, size)
}

/// Recursive dispersed-dot kernel; `size` is a power of two, `x`/`y < size`.
#[must_use]
fn bayer_recursive(x: u32, y: u32, size: u32) -> u32 {
    if size <= 1 {
        return 0;
    }
    let half = size / 2;
    let col = (x >= half) as usize;
    let row = (y >= half) as usize;
    let base = BAYER_BASE[row][col];
    4 * bayer_recursive(x % half, y % half, half) + base
}

/// Returns the zero-mean ordered-dither threshold at pixel `(x, y)`.
///
/// The raw tile value `M_N(x, y) ∈ [0, N²)` is normalized to
/// `(M_N(x, y) + 0.5) / N² − 0.5`, giving a value in `[-0.5, 0.5)`. The `+0.5`
/// centers each quantization bin, and subtracting `0.5` makes the pattern
/// symmetric about zero so a full tile has exactly zero mean (no DC shift).
///
/// `size` is normalized to a power of two in `[1, MAX_BAYER_SIZE]`; coordinates
/// wrap, so the threshold map tiles the plane seamlessly. The result is always
/// finite and within `[-0.5, 0.5)`.
#[inline]
#[must_use]
pub fn bayer_threshold(x: u32, y: u32, size: u32) -> f32 {
    let size = normalize_size(size);
    let value = bayer_recursive(x % size, y % size, size);
    let n2 = (size as f32) * (size as f32);
    (value as f32 + 0.5) / n2 - 0.5
}

/// Returns the ordered-dither threshold mapped into `[0, 1)` instead of the
/// zero-mean `[-0.5, 0.5)` of [`bayer_threshold`].
///
/// This is `(M_N(x, y) + 0.5) / N²`, the raw normalized tile value, convenient
/// where a `[0, 1)` comparison threshold is wanted directly.
#[inline]
#[must_use]
pub fn bayer_threshold01(x: u32, y: u32, size: u32) -> f32 {
    bayer_threshold(x, y, size) + 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runtime generator reproduces the canonical `2×2` matrix.
    #[test]
    fn generator_matches_bayer_2() {
        for y in 0..2u32 {
            for x in 0..2u32 {
                assert_eq!(bayer_value(x, y, 2), BAYER_2[y as usize][x as usize]);
            }
        }
    }

    /// The runtime generator reproduces the canonical `4×4` matrix.
    #[test]
    fn generator_matches_bayer_4() {
        for y in 0..4u32 {
            for x in 0..4u32 {
                assert_eq!(bayer_value(x, y, 4), BAYER_4[y as usize][x as usize]);
            }
        }
    }

    /// The runtime generator reproduces the canonical `8×8` matrix.
    #[test]
    fn generator_matches_bayer_8() {
        for y in 0..8u32 {
            for x in 0..8u32 {
                assert_eq!(bayer_value(x, y, 8), BAYER_8[y as usize][x as usize]);
            }
        }
    }

    /// Each tile is a permutation of `0 .. N²` (every value appears once).
    #[test]
    fn tile_is_a_permutation() {
        // Fixed scratch buffer sized for the largest tile exercised here
        // (16×16 = 256), avoiding any heap allocation in the test.
        for &size in &[2u32, 4, 8, 16] {
            let n2 = (size * size) as usize;
            let mut seen = [false; 256];
            for y in 0..size {
                for x in 0..size {
                    let v = bayer_value(x, y, size) as usize;
                    assert!(v < n2, "value {v} out of range for size {size}");
                    assert!(!seen[v], "duplicate value {v} for size {size}");
                    seen[v] = true;
                }
            }
            assert!(
                seen.iter().take(n2).all(|&s| s),
                "missing values for size {size}"
            );
        }
    }

    /// Thresholds stay within the half-open `[-0.5, 0.5)` interval.
    #[test]
    fn threshold_range_is_bounded() {
        for &size in &[1u32, 2, 4, 8, 16, 32] {
            for y in 0..size.max(1) {
                for x in 0..size.max(1) {
                    let t = bayer_threshold(x, y, size);
                    assert!(t.is_finite(), "non-finite threshold");
                    assert!((-0.5..0.5).contains(&t), "t={t} size={size}");
                }
            }
        }
    }

    /// A full tile of thresholds is exactly zero-mean (no DC bias).
    #[test]
    fn full_tile_is_zero_mean() {
        for &size in &[2u32, 4, 8] {
            let mut sum = 0.0_f64;
            for y in 0..size {
                for x in 0..size {
                    sum += bayer_threshold(x, y, size) as f64;
                }
            }
            assert!(sum.abs() < 1.0e-6, "tile mean {sum} for size {size}");
        }
    }

    /// The `[0, 1)` variant is exactly the zero-mean threshold plus `0.5`.
    #[test]
    fn threshold01_is_offset_of_centered() {
        for y in 0..8u32 {
            for x in 0..8u32 {
                let a = bayer_threshold01(x, y, 8);
                let b = bayer_threshold(x, y, 8) + 0.5;
                assert!((a - b).abs() < 1.0e-7, "a={a} b={b}");
                assert!((0.0..1.0).contains(&a), "a={a}");
            }
        }
    }

    /// Coordinates wrap modulo the tile size.
    #[test]
    fn coordinates_tile() {
        for &size in &[2u32, 4, 8] {
            for y in 0..size {
                for x in 0..size {
                    assert_eq!(bayer_value(x, y, size), bayer_value(x + size, y, size));
                    assert_eq!(bayer_value(x, y, size), bayer_value(x, y + 3 * size, size));
                }
            }
        }
    }

    /// Non-power-of-two and out-of-range sizes clamp to a valid power of two.
    #[test]
    fn size_normalization() {
        assert_eq!(normalize_size(0), 1);
        assert_eq!(normalize_size(1), 1);
        assert_eq!(normalize_size(3), 2);
        assert_eq!(normalize_size(5), 4);
        assert_eq!(normalize_size(7), 4);
        assert_eq!(normalize_size(8), 8);
        assert_eq!(normalize_size(1000), MAX_BAYER_SIZE);
    }

    /// A `1×1` tile has a single, finite, in-range threshold.
    #[test]
    fn degenerate_tile_is_finite() {
        let t = bayer_threshold(5, 9, 1);
        assert!(t.is_finite());
        assert!((-0.5..0.5).contains(&t));
    }
}
