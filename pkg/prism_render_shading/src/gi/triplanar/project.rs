//! Triplanar and biplanar projection weights — CPU golden reference.
//!
//! A single UV parameterisation cannot be stretched across arbitrary geometry
//! (cliffs, terrain, procedurally fused meshes) without seams or smearing.
//! Triplanar mapping sidesteps the problem by projecting the surface three
//! times — once down each world axis — and blending the three projections by
//! how strongly the surface normal faces that axis.  The result looks seamless
//! on any slope because every texel is dominated by whichever projection is
//! least foreshortened there.
//!
//! The blend weight for axis `k` is `|n_k| ^ sharpness`, renormalised so the
//! three weights sum to one.  The `sharpness` exponent sets how tightly each
//! projection hugs its axis: `1` is a soft, broad blend; large values collapse
//! toward a hard "dominant axis wins" selection.  Biplanar mapping (Quilez)
//! keeps only the two strongest axes, which halves the texture fetches at a
//! small quality cost on the diagonal.
//!
//! Each axis projects the *world position* onto the plane perpendicular to it:
//!
//! | axis | sampled plane | projected UV |
//! |------|---------------|--------------|
//! | X    | `yz`          | `(world.z, world.y)` |
//! | Y    | `xz`          | `(world.x, world.z)` |
//! | Z    | `xy`          | `(world.x, world.y)` |
//!
//! The `z`-first ordering on the X projection keeps all three UVs right-handed
//! when viewed from the positive side of their axis, matching the GPU twin.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; `sqrt` uses the inherent
//!   method.  No `f32::powf()`-style free functions.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit ordering).
//! * Weights are non-negative and sum to exactly one (within rounding).
//! * Defensive clamping everywhere: a degenerate (zero / non-finite) normal
//!   falls back to the up axis, `sharpness` is clamped to a sane finite range,
//!   and no `NaN`/`inf` ever escapes.

use bevy_math::{ops, Vec2, Vec3};

/// Smallest normal length treated as a usable direction.  Below this the input
/// carries no orientation and the reference falls back to the world up axis.
const MIN_NORMAL_LEN: f32 = 1.0e-6;

/// Lowest exponent allowed for the weight blend.  Zero would make every axis
/// weigh the same regardless of orientation; we keep a tiny positive floor so
/// the blend still responds to the normal.
const MIN_SHARPNESS: f32 = 1.0e-3;

/// Highest exponent allowed, bounding `powf` growth and keeping the weights
/// representable even for near-axis-aligned normals.
const MAX_SHARPNESS: f32 = 256.0;

/// One of the three world-space projection axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Project along world X; sample the `yz` plane.
    X,
    /// Project along world Y; sample the `xz` plane.
    Y,
    /// Project along world Z; sample the `xy` plane.
    Z,
}

/// Per-axis triplanar blend weights.
///
/// All three fields are non-negative and sum to one (within floating-point
/// rounding).  A weight of `1` on one axis means the surface faces that axis
/// head-on and that projection fully dominates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriplanarWeights {
    /// Weight of the X projection (`yz` plane).
    pub x: f32,
    /// Weight of the Y projection (`xz` plane).
    pub y: f32,
    /// Weight of the Z projection (`xy` plane).
    pub z: f32,
}

impl TriplanarWeights {
    /// Returns the weights as an `[x, y, z]` array.
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Sum of the three weights (one for any valid result).
    pub fn sum(self) -> f32 {
        self.x + self.y + self.z
    }

    /// Weight associated with `axis`.
    pub fn weight(self, axis: Axis) -> f32 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }
}

/// A full triplanar projection: per-axis UVs together with their blend weights.
///
/// `uv_x`/`uv_y`/`uv_z` are the texture coordinates for the three projections
/// and `weights` tells the shader how to mix the three samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriplanarProjection {
    /// UV for the X projection (`yz` plane).
    pub uv_x: Vec2,
    /// UV for the Y projection (`xz` plane).
    pub uv_y: Vec2,
    /// UV for the Z projection (`xy` plane).
    pub uv_z: Vec2,
    /// Blend weights for the three projections.
    pub weights: TriplanarWeights,
}

/// A biplanar projection: the two dominant axes with their renormalised weights.
///
/// `weight0 + weight1 == 1`, with `weight0 >= weight1` so `axis0` is always the
/// stronger (major) axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiplanarProjection {
    /// Dominant (major) axis.
    pub axis0: Axis,
    /// UV for the dominant axis.
    pub uv0: Vec2,
    /// Weight of the dominant axis, in `[0.5, 1]`.
    pub weight0: f32,
    /// Second-strongest (median) axis.
    pub axis1: Axis,
    /// UV for the median axis.
    pub uv1: Vec2,
    /// Weight of the median axis, in `[0, 0.5]`.
    pub weight1: f32,
}

/// Computes normalised triplanar blend weights from a surface normal.
///
/// The raw per-axis weight is `|n_k| ^ sharpness`; the three are renormalised
/// to sum to one.  `normal` need not be unit length (it is normalised here);
/// a degenerate normal falls back to the world up axis (`+Y`) and `sharpness`
/// is clamped to `[MIN_SHARPNESS, MAX_SHARPNESS]`.
pub fn triplanar_weights(normal: Vec3, sharpness: f32) -> TriplanarWeights {
    let n = sanitize_normal(normal);
    let sharpness = sanitize_sharpness(sharpness);

    // Blend on the absolute normal: both facings of an axis share a projection.
    let a = n.abs();
    let wx = ops::powf(a.x, sharpness);
    let wy = ops::powf(a.y, sharpness);
    let wz = ops::powf(a.z, sharpness);

    let sum = wx + wy + wz;
    if sum > f32::MIN_POSITIVE && sum.is_finite() {
        let inv = 1.0 / sum;
        TriplanarWeights {
            x: wx * inv,
            y: wy * inv,
            z: wz * inv,
        }
    } else {
        // powf underflowed to zero on every axis (only possible for an all-zero
        // abs normal after sanitising, which cannot happen) — fall back to up.
        TriplanarWeights {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }
}

/// Projects a world position onto the plane perpendicular to `axis`.
///
/// See the module table for the exact component ordering.
pub fn project_uv(world: Vec3, axis: Axis) -> Vec2 {
    let w = sanitize_vec3(world);
    match axis {
        Axis::X => Vec2::new(w.z, w.y),
        Axis::Y => Vec2::new(w.x, w.z),
        Axis::Z => Vec2::new(w.x, w.y),
    }
}

/// Builds the full triplanar projection (three UVs + weights) for a surface.
///
/// `world` is the shaded point in world space (scaled by any desired tiling
/// frequency before the call); `normal` is the surface normal; `sharpness`
/// tunes the blend hardness.
pub fn triplanar_projection(world: Vec3, normal: Vec3, sharpness: f32) -> TriplanarProjection {
    TriplanarProjection {
        uv_x: project_uv(world, Axis::X),
        uv_y: project_uv(world, Axis::Y),
        uv_z: project_uv(world, Axis::Z),
        weights: triplanar_weights(normal, sharpness),
    }
}

/// Builds a biplanar projection, keeping only the two strongest axes.
///
/// The weights are the two largest `|n_k| ^ sharpness` values renormalised to
/// sum to one, so the discarded (weakest) axis contributes nothing.  This is
/// the Quilez biplanar scheme: two fetches instead of three, seamless except
/// on the exact body diagonal where all axes tie.
pub fn biplanar_projection(world: Vec3, normal: Vec3, sharpness: f32) -> BiplanarProjection {
    let w = triplanar_weights(normal, sharpness);
    let entries = [(Axis::X, w.x), (Axis::Y, w.y), (Axis::Z, w.z)];

    // Find the major and median axes by weight (stable on ties: X > Y > Z).
    let mut major = entries[0];
    for &e in &entries[1..] {
        if e.1 > major.1 {
            major = e;
        }
    }
    let mut median = (Axis::X, -1.0f32);
    for &e in &entries {
        if e.0 != major.0 && e.1 > median.1 {
            median = e;
        }
    }

    let sum = major.1 + median.1;
    let (w0, w1) = if sum > f32::MIN_POSITIVE {
        let inv = 1.0 / sum;
        (major.1 * inv, median.1 * inv)
    } else {
        (1.0, 0.0)
    };

    BiplanarProjection {
        axis0: major.0,
        uv0: project_uv(world, major.0),
        weight0: w0,
        axis1: median.0,
        uv1: project_uv(world, median.0),
        weight1: w1,
    }
}

/// Normalises a normal, falling back to world up (`+Y`) when degenerate.
fn sanitize_normal(n: Vec3) -> Vec3 {
    let n = sanitize_vec3(n);
    let len = n.length();
    if len >= MIN_NORMAL_LEN && len.is_finite() {
        n / len
    } else {
        Vec3::Y
    }
}

/// Clamps the sharpness exponent to a finite, positive, bounded range.
fn sanitize_sharpness(s: f32) -> f32 {
    if s.is_finite() {
        s.clamp(MIN_SHARPNESS, MAX_SHARPNESS)
    } else {
        1.0
    }
}

/// Replaces any non-finite component of a `Vec3` with `0`.
fn sanitize_vec3(v: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(v.x),
        finite_or_zero(v.y),
        finite_or_zero(v.z),
    )
}

/// Returns `x` when finite, otherwise `0`.
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every weight set must be a partition of unity.
    #[test]
    fn weights_sum_to_one() {
        let normals = [
            Vec3::new(0.3, 0.6, 0.1),
            Vec3::new(-0.7, 0.2, 0.9),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-2.0, 5.0, -3.0),
        ];
        for &n in &normals {
            for &s in &[0.5f32, 1.0, 4.0, 16.0] {
                let w = triplanar_weights(n, s);
                assert!((w.sum() - 1.0).abs() < 1e-5, "sum {} for n {:?} s {}", w.sum(), n, s);
                assert!(w.x >= 0.0 && w.y >= 0.0 && w.z >= 0.0, "negative weight {:?}", w);
            }
        }
    }

    /// A normal aligned with an axis puts almost all weight on that axis.
    #[test]
    fn axis_aligned_normal_dominates() {
        let wx = triplanar_weights(Vec3::X, 8.0);
        assert!(wx.x > 0.999, "x weight {}", wx.x);
        let wy = triplanar_weights(Vec3::new(0.0, -1.0, 0.0), 8.0);
        assert!(wy.y > 0.999, "y weight {}", wy.y);
        let wz = triplanar_weights(Vec3::Z, 8.0);
        assert!(wz.z > 0.999, "z weight {}", wz.z);
    }

    /// Raising sharpness pushes more weight onto the dominant axis (monotone).
    #[test]
    fn sharpness_monotonically_concentrates_weight() {
        // A normal that leans toward +Y but is not axis aligned.
        let n = Vec3::new(0.4, 0.8, 0.3);
        let mut prev = -1.0f32;
        for &s in &[0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0] {
            let w = triplanar_weights(n, s);
            // Y is the dominant axis here.
            assert!(w.y >= w.x && w.y >= w.z, "Y should dominate: {:?}", w);
            assert!(w.y >= prev - 1e-6, "Y weight must not decrease: {} -> {}", prev, w.y);
            prev = w.y;
        }
        assert!(prev > 0.9, "high sharpness should nearly saturate: {}", prev);
    }

    /// Projected UVs follow the documented component ordering.
    #[test]
    fn projection_uv_ordering() {
        let world = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(project_uv(world, Axis::X), Vec2::new(3.0, 2.0));
        assert_eq!(project_uv(world, Axis::Y), Vec2::new(1.0, 3.0));
        assert_eq!(project_uv(world, Axis::Z), Vec2::new(1.0, 2.0));
    }

    /// A degenerate normal must not produce NaNs; it falls back to up.
    #[test]
    fn degenerate_normal_falls_back_to_up() {
        let w = triplanar_weights(Vec3::ZERO, 4.0);
        assert!((w.sum() - 1.0).abs() < 1e-5);
        assert!(w.y > 0.999, "zero normal should map to up axis: {:?}", w);

        let nan = triplanar_weights(Vec3::new(f32::NAN, 1.0, f32::INFINITY), 4.0);
        assert!(nan.x.is_finite() && nan.y.is_finite() && nan.z.is_finite());
        assert!((nan.sum() - 1.0).abs() < 1e-5);
    }

    /// Biplanar keeps the two strongest axes and renormalises to one.
    #[test]
    fn biplanar_selects_two_dominant_axes() {
        // Weights order: Y (0.8) > X (0.4) > Z (0.3) before exponent.
        let n = Vec3::new(0.4, 0.8, 0.3);
        let bp = biplanar_projection(Vec3::new(1.0, 2.0, 3.0), n, 2.0);
        assert_eq!(bp.axis0, Axis::Y, "major axis");
        assert_eq!(bp.axis1, Axis::X, "median axis");
        assert!((bp.weight0 + bp.weight1 - 1.0).abs() < 1e-5, "weights sum");
        assert!(bp.weight0 >= bp.weight1, "major >= median");
        assert_eq!(bp.uv0, Vec2::new(1.0, 3.0));
        assert_eq!(bp.uv1, Vec2::new(3.0, 2.0));
    }

    /// On an axis-aligned normal biplanar collapses to a single projection.
    #[test]
    fn biplanar_axis_aligned_is_single_projection() {
        let bp = biplanar_projection(Vec3::new(5.0, 6.0, 7.0), Vec3::Z, 8.0);
        assert_eq!(bp.axis0, Axis::Z);
        assert!(bp.weight0 > 0.999, "weight0 {}", bp.weight0);
        assert!(bp.weight1 < 1e-3, "weight1 {}", bp.weight1);
    }

    /// The full projection bundles the three UVs with the shared weights.
    #[test]
    fn full_projection_bundles_uvs_and_weights() {
        let world = Vec3::new(2.0, 4.0, 6.0);
        let n = Vec3::new(0.1, 0.2, 0.9);
        let p = triplanar_projection(world, n, 3.0);
        assert_eq!(p.uv_x, project_uv(world, Axis::X));
        assert_eq!(p.uv_y, project_uv(world, Axis::Y));
        assert_eq!(p.uv_z, project_uv(world, Axis::Z));
        assert!((p.weights.sum() - 1.0).abs() < 1e-5);
        assert!(p.weights.z > p.weights.x && p.weights.z > p.weights.y);
    }
}
