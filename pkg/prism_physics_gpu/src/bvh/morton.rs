//! 30-bit Morton (Z-order) codes for the `LBVH` centroid sort.
//!
//! Each leaf centroid is quantised into a `[0, 1023]` integer per axis relative
//! to the [`SceneBounds`] cube, then the three 10-bit coordinates are
//! bit-interleaved into a single 30-bit key. Sorting leaves by this key lays
//! spatially near primitives next to each other, which is the property Karras'
//! binary radix tree turns into a hierarchy in one parallel pass.
//!
//! Every function here is the exact scalar the `bvh_morton.wgsl` kernel runs,
//! so the `CPU` twin and the `GPU` build produce identical keys and therefore
//! identical sort orders and trees.
//!
//! # Provenance
//!
//! Bit-interleaving via the classic "magic number" shift-and-mask expansion and
//! Morton/Z-order curve indexing are long-established public techniques. No
//! Unreal Engine source or derived code.

use glam::Vec3;

use super::config::SceneBounds;

/// The number of quantisation buckets per axis (`2^10`).
pub const AXIS_BUCKETS: f32 = 1024.0;

/// The largest quantised coordinate per axis (`AXIS_BUCKETS - 1`).
pub const AXIS_MAX: f32 = 1023.0;

/// Spreads the low 10 bits of `v` so bit `i` lands in bit `3 * i`.
///
/// This is the per-axis half of the 30-bit interleave: after expansion the
/// three axes can be OR-ed together at offsets 0, 1, and 2.
#[must_use]
pub fn cpu_expand_bits_10(v: u32) -> u32 {
    let mut x = v & 0x0000_03ff;
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// Interleaves three 10-bit coordinates into a 30-bit Morton code.
#[must_use]
pub fn cpu_morton3d(x: u32, y: u32, z: u32) -> u32 {
    cpu_expand_bits_10(x) | (cpu_expand_bits_10(y) << 1) | (cpu_expand_bits_10(z) << 2)
}

/// The reciprocal of an axis extent, or zero for a non-positive extent.
///
/// The `GPU` kernel cannot compute this: `WGSL` division carries a 2.5-`ULP`
/// tolerance, so a device-side `(c - lo) / extent` can land in a different
/// bucket than the host at a boundary. Instead the host computes this
/// correctly-rounded reciprocal once per axis and both sides quantise with a
/// single correctly-rounded multiply, which keeps the codes bit-for-bit equal.
/// A flat or single-point axis (non-positive extent) yields zero so every
/// coordinate maps to bucket zero.
#[must_use]
pub fn cpu_inv_extent(extent: f32) -> f32 {
    if extent > 0.0 {
        1.0 / extent
    } else {
        0.0
    }
}

/// Quantises one axis coordinate `c` into `[0, 1023]`.
///
/// `lo` is the axis minimum and `inv_extent` the [`cpu_inv_extent`] reciprocal
/// of its span. The normalised coordinate is a single correctly-rounded
/// multiply, matching the `WGSL` kernel exactly so host and device never
/// diverge; a zero `inv_extent` (degenerate axis) maps everything to bucket
/// zero.
#[must_use]
pub fn cpu_quantize(c: f32, lo: f32, inv_extent: f32) -> u32 {
    let norm = (c - lo) * inv_extent;
    (norm * AXIS_BUCKETS).floor().clamp(0.0, AXIS_MAX) as u32
}

/// Computes the 30-bit Morton code of `centroid` within `bounds`.
#[must_use]
pub fn cpu_morton_code(centroid: Vec3, bounds: &SceneBounds) -> u32 {
    let extent = bounds.extent();
    let x = cpu_quantize(centroid.x, bounds.min.x, cpu_inv_extent(extent.x));
    let y = cpu_quantize(centroid.y, bounds.min.y, cpu_inv_extent(extent.y));
    let z = cpu_quantize(centroid.z, bounds.min.z, cpu_inv_extent(extent.z));
    cpu_morton3d(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::{cpu_expand_bits_10, cpu_inv_extent, cpu_morton3d, cpu_morton_code, cpu_quantize};
    use crate::bvh::config::SceneBounds;
    use glam::Vec3;

    #[test]
    fn expand_bits_spreads_by_three() {
        // All ten input bits set spread to the 0x09249249 mask.
        assert_eq!(cpu_expand_bits_10(0x3ff), 0x0924_9249);
        // A single low bit stays in place; the next input bit lands three over.
        assert_eq!(cpu_expand_bits_10(0b01), 0b001);
        assert_eq!(cpu_expand_bits_10(0b10), 0b001_000);
        // Only the low ten bits participate.
        assert_eq!(cpu_expand_bits_10(0xffff_fc00), 0);
    }

    #[test]
    fn morton3d_interleaves_axes() {
        // x contributes bit 0, y bit 1, z bit 2 of the least-significant triple.
        assert_eq!(cpu_morton3d(1, 0, 0), 0b001);
        assert_eq!(cpu_morton3d(0, 1, 0), 0b010);
        assert_eq!(cpu_morton3d(0, 0, 1), 0b100);
        assert_eq!(cpu_morton3d(1, 1, 1), 0b111);
        // The maximum code sets every one of the 30 used bits.
        assert_eq!(cpu_morton3d(1023, 1023, 1023), 0x3fff_ffff);
    }

    #[test]
    fn quantize_maps_span_to_buckets() {
        // Minimum maps to 0, the maximum clamps to 1023, the midpoint to 512.
        let inv = cpu_inv_extent(4.0);
        assert_eq!(cpu_quantize(0.0, 0.0, inv), 0);
        assert_eq!(cpu_quantize(4.0, 0.0, inv), 1023);
        assert_eq!(cpu_quantize(2.0, 0.0, inv), 512);
    }

    #[test]
    fn quantize_degenerate_axis_is_zero() {
        // A zero (or negative) extent yields a zero reciprocal, putting every
        // coordinate in bucket zero.
        assert_eq!(cpu_inv_extent(0.0), 0.0);
        assert_eq!(cpu_inv_extent(-1.0), 0.0);
        assert_eq!(cpu_quantize(7.0, 7.0, cpu_inv_extent(0.0)), 0);
        assert_eq!(cpu_quantize(9.0, 7.0, cpu_inv_extent(-1.0)), 0);
    }

    #[test]
    fn morton_code_of_bounds_corners() {
        let bounds = SceneBounds {
            min: Vec3::new(0.0, 0.0, 0.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        // The minimum corner is code zero; the maximum corner is all 30 bits.
        assert_eq!(cpu_morton_code(Vec3::new(0.0, 0.0, 0.0), &bounds), 0);
        assert_eq!(
            cpu_morton_code(Vec3::new(1.0, 1.0, 1.0), &bounds),
            0x3fff_ffff
        );
    }
}
