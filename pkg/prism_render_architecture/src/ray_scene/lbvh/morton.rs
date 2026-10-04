//! 30-bit 3D `Morton` encoding and centroid quantization for the linear `BVH`.
//!
//! The parallel linear-`BVH` builder of Karras (2012) orders primitives along a
//! space-filling `Morton` curve so that spatially adjacent primitives are
//! adjacent in memory, then builds a binary radix tree over the sorted keys.
//! This module owns the two pieces that must agree bit-for-bit with a `GPU`
//! build kernel: the fixed-point centroid quantization and the bit-interleave
//! that forms the key.
//!
//! Each axis is quantized to 10 bits (`[0, 1023]`), so a key packs 30 bits into
//! a `u32`. The quantization is a pure multiply-add followed by a clamp, which a
//! `GPU` lane reproduces exactly, and the interleave is integer bit magic with
//! no data-dependent branches.

/// Largest quantized coordinate per axis (`2^10 - 1`).
pub const MORTON_GRID_MAX: u32 = 1023;

/// Spreads the low 10 bits of `v` so two zero bits sit between each, the
/// per-axis step of a 30-bit 3D `Morton` interleave.
///
/// Input bits above bit 9 are ignored. The constants are the standard
/// "magic number" bit-splitting masks for a 10-bit field.
#[must_use]
pub const fn expand_bits10(v: u32) -> u32 {
    let mut x = v & 0x0000_03ff;
    x = (x | (x << 16)) & 0xff00_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// Interleaves three 10-bit coordinates into a 30-bit `Morton` key.
///
/// `x` occupies the least-significant bit of each triple, matching the common
/// `GPU` convention; only the low 10 bits of each argument are used.
#[must_use]
pub const fn morton3d(x: u32, y: u32, z: u32) -> u32 {
    (expand_bits10(x) << 2) | (expand_bits10(y) << 1) | expand_bits10(z)
}

/// Maps a scene-space centroid into the `[0, 1]^3` unit cube using the scene
/// centroid bounds, used to produce the per-axis grid coordinate.
///
/// `inv_extent` is the reciprocal of the centroid-bounds extent per axis, with
/// a zero entry wherever the extent is zero (a degenerate axis); such an axis
/// maps every centroid to coordinate `0`, which keeps the key well defined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MortonQuantizer {
    /// Minimum corner of the centroid bounds.
    min: [f32; 3],
    /// Reciprocal extent per axis; `0` on a degenerate axis.
    inv_extent: [f32; 3],
}

impl MortonQuantizer {
    /// Builds a quantizer from the inclusive centroid-bounds corners.
    ///
    /// A zero-width axis yields a zero reciprocal so the mapped coordinate is
    /// pinned to `0` instead of producing a non-finite value.
    #[must_use]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        let mut inv_extent = [0.0_f32; 3];
        let mut axis = 0;
        while axis < 3 {
            let extent = max[axis] - min[axis];
            inv_extent[axis] = if extent > 0.0 { 1.0 / extent } else { 0.0 };
            axis += 1;
        }
        Self { min, inv_extent }
    }

    /// Quantizes one centroid to a 30-bit `Morton` key.
    ///
    /// Non-finite inputs are clamped into the valid grid range, so the builder
    /// never produces an out-of-range coordinate even for degenerate geometry.
    #[must_use]
    pub fn key(&self, centroid: [f32; 3]) -> u32 {
        let gx = Self::grid_coord((centroid[0] - self.min[0]) * self.inv_extent[0]);
        let gy = Self::grid_coord((centroid[1] - self.min[1]) * self.inv_extent[1]);
        let gz = Self::grid_coord((centroid[2] - self.min[2]) * self.inv_extent[2]);
        morton3d(gx, gy, gz)
    }

    /// Clamps a unit-cube coordinate to `[0, MORTON_GRID_MAX]`.
    ///
    /// A non-finite `unit` (from a degenerate axis or bad input) collapses to
    /// `0` through the ordered comparisons below.
    fn grid_coord(unit: f32) -> u32 {
        let scaled = unit * MORTON_GRID_MAX as f32;
        if scaled > 0.0 {
            if scaled < MORTON_GRID_MAX as f32 {
                // `scaled` is finite and in range here.
                scaled as u32
            } else {
                MORTON_GRID_MAX
            }
        } else {
            0
        }
    }
}
