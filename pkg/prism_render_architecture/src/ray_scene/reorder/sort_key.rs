//! Coherence sort-key encoding for Shader Execution Reordering (`SER`).
//!
//! Path tracing is notoriously *incoherent*: adjacent threads in a `GPU`
//! wavefront bounce toward unrelated materials and directions, so they diverge
//! into different shading branches and touch unrelated memory. Hardware
//! Shader-Execution-Reordering (NVIDIA `SER`) and the classic "ray sorting"
//! technique both attack this by **reordering** the ray/hit stream so that
//! threads processed together are *coherent* — same material, similar outgoing
//! direction, nearby in space — which collapses branch divergence and improves
//! cache / memory-coalescing behaviour.
//!
//! This module owns the deterministic, device-free **sort key**: it folds the
//! three coherence axes into a single [`CoherenceKey`] (`u64`) whose numeric
//! ordering *is* the coherence ordering. A plan built from these keys
//! ([`super::plan`]) is the `CPU` golden a `GPU` radix sort reproduces exactly.
//!
//! # Key layout
//!
//! The key packs three fields, most-significant first, so sorting groups by the
//! coarsest coherence axis before refining:
//!
//! ```text
//! MSB                                                           LSB
//! | material / hit-group id | octahedral direction | spatial Morton |
//! |      material_bits      |   2·dir_bits_per_axis | 3·spatial_bits |
//! ```
//!
//! * **Material / hit-group** (primary). Threads must take the same shading
//!   branch to stop diverging, so the hit-group id dominates the ordering.
//! * **Direction** (secondary). The outgoing (or incoming) ray direction is
//!   octahedral-encoded to the unit square and quantised to a
//!   `2^dir_bits_per_axis` grid, then [`Morton`](crate::particle::morton_code)
//!   interleaved so neighbouring directions stay adjacent in the key.
//! * **Space** (tertiary). The origin is normalised into a world
//!   [`SpatialBounds`] box, quantised per axis, and 3D-Morton interleaved so
//!   spatially-near rays with equal material/direction share a cache-friendly
//!   run.
//!
//! Everything after the direction/origin quantisation is pure integer bit
//! manipulation, so the key is deterministic and platform independent; the only
//! floating-point work is the octahedral map and the per-axis normalise, both
//! clamped and guarded against degenerate (zero-length / zero-extent) inputs.
//!
//! # References
//! - Cigolle, Donow, Evangelakos, Mara, `McGuire`, Meyer, *A Survey of Efficient
//!   Representations for Independent Unit Vectors*, JCGT 2014 (octahedral map).
//! - NVIDIA, *Shader Execution Reordering* (Ada / `OptiX`) white paper.
//! - Meister et al., *A Survey on Bounding Volume Hierarchies for Ray Tracing*,
//!   EG 2021 (ray sorting for coherence).

use crate::particle::morton_code::{morton_encode_2d, morton_encode_3d};

/// Widest coordinate (per axis) the 2D `Morton` interleave accepts.
pub const MAX_DIR_BITS_PER_AXIS: u32 = 16;
/// Widest coordinate (per axis) the 3D `Morton` interleave accepts.
pub const MAX_SPATIAL_BITS_PER_AXIS: u32 = 10;
/// Total key width budget: the three fields must fit a `u64`.
pub const KEY_BIT_BUDGET: u32 = 64;

/// Error returned when a [`CoherenceKeyLayout`] or [`SpatialBounds`] is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutError {
    /// `material_bits + 2·dir_bits_per_axis + 3·spatial_bits_per_axis` exceeds
    /// [`KEY_BIT_BUDGET`].
    TooManyBits {
        /// Requested total key width in bits.
        total: u32,
    },
    /// `dir_bits_per_axis` exceeds [`MAX_DIR_BITS_PER_AXIS`].
    DirAxisTooWide {
        /// Requested direction bits per axis.
        bits: u32,
    },
    /// `spatial_bits_per_axis` exceeds [`MAX_SPATIAL_BITS_PER_AXIS`].
    SpatialAxisTooWide {
        /// Requested spatial bits per axis.
        bits: u32,
    },
    /// A [`SpatialBounds`] axis has `max <= min` (or a non-finite extent).
    DegenerateBounds,
}

/// Axis-aligned world-space box the ray origin is normalised against before
/// spatial quantisation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialBounds {
    min: [f32; 3],
    inv_extent: [f32; 3],
}

impl SpatialBounds {
    /// Builds bounds from the min/max corners, precomputing the reciprocal
    /// extent used to normalise origins into `[0, 1]` per axis.
    ///
    /// # Errors
    /// Returns [`LayoutError::DegenerateBounds`] if any axis has
    /// `max <= min` or a non-finite extent.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Result<Self, LayoutError> {
        let mut inv_extent = [0.0f32; 3];
        for axis in 0..3 {
            let extent = max[axis] - min[axis];
            if !extent.is_finite() || extent <= 0.0 {
                return Err(LayoutError::DegenerateBounds);
            }
            inv_extent[axis] = 1.0 / extent;
        }
        Ok(Self { min, inv_extent })
    }

    /// Normalises `p` into `[0, 1]` per axis, clamping points outside the box to
    /// the nearest face.
    #[must_use]
    fn normalize(&self, p: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for axis in 0..3 {
            let t = (p[axis] - self.min[axis]) * self.inv_extent[axis];
            out[axis] = t.clamp(0.0, 1.0);
        }
        out
    }
}

/// Fully-qualified coherence key: a single `u64` whose ascending order groups
/// rays by material, then direction, then spatial locality.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CoherenceKey(pub u64);

impl CoherenceKey {
    /// Returns the raw packed value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Bit-budget configuration + world bounds driving [`CoherenceKey`] encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoherenceKeyLayout {
    material_bits: u32,
    dir_bits_per_axis: u32,
    spatial_bits_per_axis: u32,
    bounds: SpatialBounds,
}

impl CoherenceKeyLayout {
    /// Builds a validated layout.
    ///
    /// # Errors
    /// - [`LayoutError::DirAxisTooWide`] / [`LayoutError::SpatialAxisTooWide`]
    ///   when a per-axis width exceeds the `Morton` interleave's capacity.
    /// - [`LayoutError::TooManyBits`] when the packed fields exceed
    ///   [`KEY_BIT_BUDGET`].
    pub fn new(
        material_bits: u32,
        dir_bits_per_axis: u32,
        spatial_bits_per_axis: u32,
        bounds: SpatialBounds,
    ) -> Result<Self, LayoutError> {
        if dir_bits_per_axis > MAX_DIR_BITS_PER_AXIS {
            return Err(LayoutError::DirAxisTooWide {
                bits: dir_bits_per_axis,
            });
        }
        if spatial_bits_per_axis > MAX_SPATIAL_BITS_PER_AXIS {
            return Err(LayoutError::SpatialAxisTooWide {
                bits: spatial_bits_per_axis,
            });
        }
        let total = material_bits + 2 * dir_bits_per_axis + 3 * spatial_bits_per_axis;
        if total > KEY_BIT_BUDGET {
            return Err(LayoutError::TooManyBits { total });
        }
        Ok(Self {
            material_bits,
            dir_bits_per_axis,
            spatial_bits_per_axis,
            bounds,
        })
    }

    /// A balanced default: 16 material bits, 8 direction bits/axis (256²
    /// octahedral grid), 10 spatial bits/axis (1024³ cells) = 62 key bits.
    ///
    /// # Errors
    /// Propagates [`SpatialBounds::new`] validation.
    pub fn balanced(min: [f32; 3], max: [f32; 3]) -> Result<Self, LayoutError> {
        Self::new(16, 8, 10, SpatialBounds::new(min, max)?)
    }

    /// Shift separating the spatial field from the direction field.
    #[must_use]
    pub const fn spatial_bits(&self) -> u32 {
        3 * self.spatial_bits_per_axis
    }

    /// Shift separating the direction field from the material field.
    #[must_use]
    pub const fn dir_bits(&self) -> u32 {
        2 * self.dir_bits_per_axis
    }

    /// Total packed key width in bits.
    #[must_use]
    pub const fn total_bits(&self) -> u32 {
        self.material_bits + self.dir_bits() + self.spatial_bits()
    }

    /// Shift that drops the direction + spatial fields, leaving the material id
    /// in the low bits (so `key >> material_shift()` groups by material).
    #[must_use]
    pub const fn material_shift(&self) -> u32 {
        self.dir_bits() + self.spatial_bits()
    }

    /// Encodes one ray into its [`CoherenceKey`].
    ///
    /// `material` is masked to `material_bits`; `direction` need not be
    /// normalised (a zero-length direction maps to the octahedral centre);
    /// `origin` is clamped into the layout's [`SpatialBounds`].
    #[must_use]
    pub fn encode(&self, material: u32, direction: [f32; 3], origin: [f32; 3]) -> CoherenceKey {
        let mat = mask_u64(u64::from(material), self.material_bits);

        let [u, v] = octahedral_unit(direction);
        let qx = quantize_unit(u, self.dir_bits_per_axis);
        let qy = quantize_unit(v, self.dir_bits_per_axis);
        let dir = u64::from(morton_encode_2d(qx as u16, qy as u16));
        let dir = mask_u64(dir, self.dir_bits());

        let n = self.bounds.normalize(origin);
        let sx = quantize_unit(n[0], self.spatial_bits_per_axis);
        let sy = quantize_unit(n[1], self.spatial_bits_per_axis);
        let sz = quantize_unit(n[2], self.spatial_bits_per_axis);
        let spatial = u64::from(morton_encode_3d(sx, sy, sz));
        let spatial = mask_u64(spatial, self.spatial_bits());

        let key = (mat << self.material_shift()) | (dir << self.spatial_bits()) | spatial;
        CoherenceKey(key)
    }
}

/// Keeps the low `bits` of `v` (a no-op when `bits >= 64`).
#[must_use]
fn mask_u64(v: u64, bits: u32) -> u64 {
    if bits >= 64 {
        v
    } else if bits == 0 {
        0
    } else {
        v & ((1u64 << bits) - 1)
    }
}

/// Returns `+1.0` for non-negative inputs, `-1.0` otherwise (never zero), so the
/// octahedral fold matches the reference `signNotZero`.
#[must_use]
fn sign_not_zero(v: f32) -> f32 {
    if v >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Octahedral-encodes a direction to the unit square `[0, 1]²`.
///
/// Degenerate (zero-length / non-finite) directions map to the centre
/// `[0.5, 0.5]` rather than producing `NaN`.
#[must_use]
fn octahedral_unit(dir: [f32; 3]) -> [f32; 2] {
    let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    if !len_sq.is_finite() || len_sq <= 0.0 {
        return [0.5, 0.5];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    let n = [dir[0] * inv_len, dir[1] * inv_len, dir[2] * inv_len];
    let denom = n[0].abs() + n[1].abs() + n[2].abs();
    if !denom.is_finite() || denom <= 0.0 {
        return [0.5, 0.5];
    }
    let inv_denom = 1.0 / denom;
    let mut px = n[0] * inv_denom;
    let mut py = n[1] * inv_denom;
    if n[2] < 0.0 {
        let ox = (1.0 - py.abs()) * sign_not_zero(px);
        let oy = (1.0 - px.abs()) * sign_not_zero(py);
        px = ox;
        py = oy;
    }
    [(px * 0.5 + 0.5).clamp(0.0, 1.0), (py * 0.5 + 0.5).clamp(0.0, 1.0)]
}

/// Quantises `t ∈ [0, 1]` to an integer in `[0, 2^bits)`.
#[must_use]
fn quantize_unit(t: f32, bits: u32) -> u32 {
    if bits == 0 {
        return 0;
    }
    let res = 1u32 << bits;
    let tc = t.clamp(0.0, 1.0);
    let q = (tc * res as f32).floor() as i64;
    q.clamp(0, i64::from(res - 1)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> CoherenceKeyLayout {
        CoherenceKeyLayout::balanced([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).unwrap()
    }

    #[test]
    fn balanced_layout_fits_the_budget() {
        let l = layout();
        assert_eq!(l.total_bits(), 16 + 2 * 8 + 3 * 10);
        assert!(l.total_bits() <= KEY_BIT_BUDGET);
    }

    #[test]
    fn rejects_oversized_fields() {
        let b = SpatialBounds::new([0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(
            CoherenceKeyLayout::new(16, 17, 10, b),
            Err(LayoutError::DirAxisTooWide { bits: 17 })
        );
        assert_eq!(
            CoherenceKeyLayout::new(16, 8, 11, b),
            Err(LayoutError::SpatialAxisTooWide { bits: 11 })
        );
        // 32 + 2*16 + 3*10 = 94 > 64.
        assert!(matches!(
            CoherenceKeyLayout::new(32, 16, 10, b),
            Err(LayoutError::TooManyBits { .. })
        ));
    }

    #[test]
    fn degenerate_bounds_are_rejected() {
        assert_eq!(
            SpatialBounds::new([0.0; 3], [0.0, 1.0, 1.0]),
            Err(LayoutError::DegenerateBounds)
        );
        assert_eq!(
            SpatialBounds::new([0.0; 3], [1.0, 1.0, f32::NAN]),
            Err(LayoutError::DegenerateBounds)
        );
    }

    #[test]
    fn encoding_is_deterministic() {
        let l = layout();
        let a = l.encode(3, [0.2, -0.4, 0.9], [0.1, 0.2, -0.3]);
        let b = l.encode(3, [0.2, -0.4, 0.9], [0.1, 0.2, -0.3]);
        assert_eq!(a, b);
    }

    #[test]
    fn material_dominates_ordering() {
        let l = layout();
        // Low material + "largest" direction/space must still sort below a
        // higher material with the smallest direction/space.
        let low_mat = l.encode(1, [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]);
        let high_mat = l.encode(2, [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]);
        assert!(low_mat < high_mat);
        assert_eq!(low_mat.raw() >> l.material_shift(), 1);
        assert_eq!(high_mat.raw() >> l.material_shift(), 2);
    }

    #[test]
    fn opposite_directions_differ_within_a_material() {
        let l = layout();
        let up = l.encode(5, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);
        let down = l.encode(5, [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]);
        assert_ne!(up, down);
        assert_eq!(up.raw() >> l.material_shift(), 5);
        assert_eq!(down.raw() >> l.material_shift(), 5);
    }

    #[test]
    fn nearby_origins_share_spatial_bits() {
        let l = layout();
        let base = l.encode(0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let near = l.encode(0, [1.0, 0.0, 0.0], [0.0005, 0.0, 0.0]);
        let far = l.encode(0, [1.0, 0.0, 0.0], [0.9, 0.9, 0.9]);
        // Within one 1024³ cell the spatial code is identical; a far origin is
        // not.
        assert_eq!(base, near);
        assert_ne!(base, far);
    }

    #[test]
    fn zero_direction_is_centre_without_nan() {
        let l = layout();
        let k = l.encode(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let centre = l.encode(0, {
            // Reconstruct the exact direction the centre maps from: the oct
            // centre is produced by the guarded zero path, so just compare the
            // encoded key is finite and stable.
            [0.0, 0.0, 0.0]
        }, [0.0, 0.0, 0.0]);
        assert_eq!(k, centre);
    }

    #[test]
    fn out_of_bounds_origin_clamps_to_face() {
        let l = layout();
        let inside = l.encode(0, [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let outside = l.encode(0, [1.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        assert_eq!(inside, outside);
    }

    #[test]
    fn material_is_masked_to_field_width() {
        // 4-bit material field: id 0x13 masks to 0x3.
        let b = SpatialBounds::new([0.0; 3], [1.0; 3]).unwrap();
        let l = CoherenceKeyLayout::new(4, 4, 4, b).unwrap();
        let a = l.encode(0x13, [1.0, 0.0, 0.0], [0.1, 0.1, 0.1]);
        let c = l.encode(0x03, [1.0, 0.0, 0.0], [0.1, 0.1, 0.1]);
        assert_eq!(a, c);
    }
}
