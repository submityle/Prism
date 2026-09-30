//! Analytic closed-form signed-distance-field (`SDF`) primitives for the
//! particle subsystem's collision, ribbon-capsule, and metaball authoring
//! contracts (design §10, §14).
//!
//! This module is the *analytic* half of the subsystem's distance work. Its
//! siblings differ in kind:
//!
//! * [`crate::particle::sdf`] is a **3D `SDF` texture-field sampler**: it owns an
//!   `SdfField`, an `SdfTransform`, and the trilinear machinery that reads a
//!   *baked* distance/gradient volume produced from a mesh. It has voxels and a
//!   texture.
//! * **This module** owns the *analytic closed-form* primitive functions — a
//!   sphere, box, rounded box, capsule, line segment, infinite cylinder, torus,
//!   and plane — plus the polynomial smooth-boolean operators that combine them.
//!   There is no voxel grid and no texture: every distance is evaluated in
//!   closed form from the point and the primitive parameters, exactly as an
//!   `iq`-style raymarch scene or a `GPU` metaball kernel would.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file, and every distance uses only `+ - * /`, `f32::sqrt`, `f32::abs`,
//! `f32::min`, `f32::max`, and `f32::clamp`. No transcendental function is ever
//! called, so the `CPU` reference here agrees bit for bit with a future `GPU`
//! (`WESL`) kernel that packs the same primitives through the `std430` helpers
//! in [`crate::particle::gpu_layout`].
//!
//! Sign convention (the universal `SDF` rule): the returned distance is
//! **negative inside** the solid, **zero on the surface**, and **positive
//! outside**, and its magnitude is the Euclidean distance to the surface (exact
//! for the round primitives, a conservative under-estimate only at the box's
//! concave-free exterior, which is still a valid Lipschitz-1 `SDF`).

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Epsilon used to guard divisions and to compare magnitudes without ever
/// writing an exact `==` / `!=` on a production `f32`.
const CMP_EPS: f32 = 1.0e-6;

/// A hand-rolled three-component vector, kept local so the module stays a
/// zero-dependency contract and its vector math is auditable in one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with all three lanes set to `s`.
    #[must_use]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
    }

    /// Component-wise sum `self + rhs`.
    ///
    /// Named `plus` (not the `Add` operator) so the whole module keeps a single
    /// uniform call-site style and never trips the operator-trait lint.
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self × rhs` (right-handed).
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only a
    /// comparison is needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector in the same direction, or the zero vector when the input is
    /// shorter than [`CMP_EPS`] (so it never yields a `NaN`).
    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len <= CMP_EPS {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }

    /// Component-wise absolute value.
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// The largest of the three components.
    #[must_use]
    pub fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Component-wise minimum of `self` and `rhs`.
    #[must_use]
    pub fn min_with(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    /// Component-wise maximum of `self` and `rhs`.
    #[must_use]
    pub fn max_with(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }
}

// ---------------------------------------------------------------------------
// Primitive signed-distance functions.
// ---------------------------------------------------------------------------

/// Signed distance to a sphere of radius `r` centered at the origin.
///
/// Negative inside, zero on the shell, positive outside; exact everywhere.
#[must_use]
pub fn sd_sphere(p: Vec3, r: f32) -> f32 {
    p.length() - r
}

/// Signed distance to an axis-aligned box of half-extents `half_extent`
/// centered at the origin (the classic `iq` box `SDF`).
///
/// The exterior term is the length of the positive part of `|p| - he`; the
/// interior term pulls the distance negative by the largest overshoot when the
/// point is fully inside.
#[must_use]
pub fn sd_box(p: Vec3, half_extent: Vec3) -> f32 {
    let q = p.abs().minus(half_extent);
    let outside = q.max_with(Vec3::ZERO).length();
    let inside = q.max_component().min(0.0);
    outside + inside
}

/// Signed distance to a box of half-extents `he` with its edges rounded by
/// radius `r` (a valid `SDF` obtained by subtracting `r` from the box field).
#[must_use]
pub fn sd_round_box(p: Vec3, he: Vec3, r: f32) -> f32 {
    let q = p.abs().minus(he);
    let outside = q.max_with(Vec3::ZERO).length();
    let inside = q.max_component().min(0.0);
    outside + inside - r
}

/// Unsigned distance from `p` to the line segment `a`-`b`.
///
/// The projection parameter is clamped to `[0, 1]` so points beyond either end
/// measure to the nearer endpoint. A degenerate segment (`a == b`) safely
/// collapses to the point distance instead of dividing by zero.
#[must_use]
pub fn sd_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let pa = p.minus(a);
    let ba = b.minus(a);
    let denom = ba.length_squared();
    let h = if denom <= CMP_EPS {
        0.0
    } else {
        (pa.dot(ba) / denom).clamp(0.0, 1.0)
    };
    pa.minus(ba.scale(h)).length()
}

/// Signed distance to a capsule: the line segment `a`-`b` inflated by radius
/// `r`. Negative inside the swept volume, positive outside; exact everywhere.
#[must_use]
pub fn sd_capsule(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    sd_segment(p, a, b) - r
}

/// Signed distance to an infinite cylinder of the given `radius` whose axis is
/// the world `Y` axis. Independent of `p.y`.
#[must_use]
pub fn sd_cylinder_infinite(p: Vec3, radius: f32) -> f32 {
    (p.x * p.x + p.z * p.z).sqrt() - radius
}

/// Signed distance to a torus lying in the `XZ` plane, with `major` ring radius
/// and `minor` tube radius.
///
/// `q = (length(p.xz) - major, p.y)`, and the distance is `length(q) - minor`.
#[must_use]
pub fn sd_torus(p: Vec3, major: f32, minor: f32) -> f32 {
    let radial = (p.x * p.x + p.z * p.z).sqrt() - major;
    (radial * radial + p.y * p.y).sqrt() - minor
}

/// Signed distance to a plane with unit normal `n` and offset `d`
/// (`dot(p, n) + d`). `n` is assumed already normalized by the caller.
#[must_use]
pub fn sd_plane(p: Vec3, n: Vec3, d: f32) -> f32 {
    p.dot(n) + d
}

// ---------------------------------------------------------------------------
// Boolean combinators.
// ---------------------------------------------------------------------------

/// Hard union of two fields: the point belongs to whichever solid is nearer.
#[must_use]
pub fn op_union(d1: f32, d2: f32) -> f32 {
    d1.min(d2)
}

/// Hard subtraction `d2 \ d1`: carve the first solid out of the second.
#[must_use]
pub fn op_subtract(d1: f32, d2: f32) -> f32 {
    (-d1).max(d2)
}

/// Hard intersection: the point must be inside both solids.
#[must_use]
pub fn op_intersect(d1: f32, d2: f32) -> f32 {
    d1.max(d2)
}

/// Linear interpolation `a + (b - a) * t`, kept local so no external math crate
/// is pulled in. Used only by the smooth combinators.
#[must_use]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Polynomial smooth union with blend radius `k`.
///
/// Falls back to the hard [`op_union`] when `k` is at or below [`CMP_EPS`], so a
/// zero blend radius is well defined instead of dividing by zero.
#[must_use]
pub fn op_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    if k <= CMP_EPS {
        return op_union(d1, d2);
    }
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    mix(d2, d1, h) - k * h * (1.0 - h)
}

/// Polynomial smooth subtraction `d2 \ d1` with blend radius `k`.
///
/// Falls back to the hard [`op_subtract`] when `k` is at or below [`CMP_EPS`].
#[must_use]
pub fn op_smooth_subtract(d1: f32, d2: f32, k: f32) -> f32 {
    if k <= CMP_EPS {
        return op_subtract(d1, d2);
    }
    let h = (0.5 - 0.5 * (d2 + d1) / k).clamp(0.0, 1.0);
    mix(d2, -d1, h) + k * h * (1.0 - h)
}

/// Polynomial smooth intersection with blend radius `k`.
///
/// Falls back to the hard [`op_intersect`] when `k` is at or below [`CMP_EPS`].
#[must_use]
pub fn op_smooth_intersect(d1: f32, d2: f32, k: f32) -> f32 {
    if k <= CMP_EPS {
        return op_intersect(d1, d2);
    }
    let h = (0.5 - 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    mix(d2, d1, h) + k * h * (1.0 - h)
}

/// Estimates the outward surface normal at `p` by central differences of an
/// arbitrary field `sample_fn`, using the finite step `eps`.
///
/// The six-tap central difference is transcendental-free and returns a unit
/// vector (or the zero vector for a perfectly flat neighborhood, via
/// [`Vec3::normalized`]).
#[must_use]
pub fn gradient_normal<F>(sample_fn: F, p: Vec3, eps: f32) -> Vec3
where
    F: Fn(Vec3) -> f32,
{
    let ex = Vec3::new(eps, 0.0, 0.0);
    let ey = Vec3::new(0.0, eps, 0.0);
    let ez = Vec3::new(0.0, 0.0, eps);
    let gx = sample_fn(p.plus(ex)) - sample_fn(p.minus(ex));
    let gy = sample_fn(p.plus(ey)) - sample_fn(p.minus(ey));
    let gz = sample_fn(p.plus(ez)) - sample_fn(p.minus(ez));
    Vec3::new(gx, gy, gz).normalized()
}

// ---------------------------------------------------------------------------
// Optional GPU packing.
// ---------------------------------------------------------------------------

/// `std430` byte size of one packed [`Capsule`]: two `vec4` slots
/// (`a.xyz + r`, then `b.xyz + pad`).
pub const CAPSULE_SDF_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// A capsule primitive ready to be uploaded to a `GPU` `SDF` kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// First segment endpoint.
    pub a: Vec3,
    /// Second segment endpoint.
    pub b: Vec3,
    /// Sweep radius around the segment.
    pub r: f32,
}

impl Capsule {
    /// Builds a capsule from its two endpoints and sweep radius.
    #[must_use]
    pub const fn new(a: Vec3, b: Vec3, r: f32) -> Self {
        Self { a, b, r }
    }

    /// Evaluates this capsule's signed distance at `p` (convenience wrapper
    /// over [`sd_capsule`]).
    #[must_use]
    pub fn distance(&self, p: Vec3) -> f32 {
        sd_capsule(p, self.a, self.b, self.r)
    }

    /// Serializes the capsule to its little-endian `std430` byte image.
    ///
    /// Layout: `[a.x, a.y, a.z, r]` in the first `vec4` slot, then
    /// `[b.x, b.y, b.z, 0.0]` in the second, matching how a `WESL` kernel would
    /// read two `vec4<f32>` loads.
    #[must_use]
    pub fn to_std430(&self) -> [u8; CAPSULE_SDF_STD430_SIZE] {
        let mut bytes = [0u8; CAPSULE_SDF_STD430_SIZE];
        let words = [
            self.a.x, self.a.y, self.a.z, self.r, self.b.x, self.b.y, self.b.z, 0.0,
        ];
        for (i, word) in words.iter().enumerate() {
            let start = i * U32_STRIDE;
            bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Total `std430` byte size for a storage buffer of `count` capsules, clamped
/// up to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(CAPSULE_SDF_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Absolute tolerance for distance comparisons that are not bit-exact.
    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    #[test]
    fn vec3_plus_minus_scale_are_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.plus(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.minus(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(Vec3::splat(2.0), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn vec3_dot_and_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(x.dot(y), 0.0);
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).dot(Vec3::new(4.0, 5.0, 6.0)), 32.0);
    }

    #[test]
    fn vec3_length_and_normalized() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
        let n = a.normalized();
        approx(n.length(), 1.0);
        assert_eq!(Vec3::ZERO.normalized(), Vec3::ZERO);
    }

    #[test]
    fn vec3_abs_max_component_and_component_min_max() {
        let a = Vec3::new(-1.0, 2.0, -3.0);
        assert_eq!(a.abs(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.max_component(), 2.0);
        let b = Vec3::new(0.0, 5.0, -4.0);
        assert_eq!(a.min_with(b), Vec3::new(-1.0, 2.0, -4.0));
        assert_eq!(a.max_with(b), Vec3::new(0.0, 5.0, -3.0));
    }

    #[test]
    fn sphere_sign_inside_outside_surface() {
        approx(sd_sphere(Vec3::new(2.0, 0.0, 0.0), 1.0), 1.0);
        approx(sd_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0), -0.5);
        approx(sd_sphere(Vec3::new(1.0, 0.0, 0.0), 1.0), 0.0);
        assert!(sd_sphere(Vec3::ZERO, 1.0) < 0.0);
    }

    #[test]
    fn box_faces_corner_and_interior() {
        let he = Vec3::splat(1.0);
        // Outside a face along +x.
        approx(sd_box(Vec3::new(2.0, 0.0, 0.0), he), 1.0);
        // On a face.
        approx(sd_box(Vec3::new(1.0, 0.0, 0.0), he), 0.0);
        // Outside a corner: distance to (1,1,1).
        approx(sd_box(Vec3::new(2.0, 2.0, 2.0), he), 3.0_f32.sqrt());
        // Deep interior center: negative, magnitude = nearest face.
        approx(sd_box(Vec3::ZERO, he), -1.0);
    }

    #[test]
    fn round_box_is_box_minus_radius() {
        let he = Vec3::splat(1.0);
        let p = Vec3::new(2.0, 0.0, 0.0);
        approx(sd_round_box(p, he, 0.25), sd_box(p, he) - 0.25);
    }

    #[test]
    fn segment_projects_and_clamps() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        // Above the middle.
        approx(sd_segment(Vec3::new(0.0, 2.0, 0.0), a, b), 2.0);
        // Beyond the +x end clamps to endpoint b.
        approx(sd_segment(Vec3::new(3.0, 0.0, 0.0), a, b), 2.0);
        // On the segment.
        approx(sd_segment(Vec3::new(0.5, 0.0, 0.0), a, b), 0.0);
    }

    #[test]
    fn segment_degenerate_collapses_to_point() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        // a == b: distance to the point, no divide-by-zero.
        approx(sd_segment(Vec3::new(1.0, 1.0, 4.0), a, a), 3.0);
    }

    #[test]
    fn capsule_is_segment_minus_radius() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        // Point 2 above the axis, radius 0.5 -> 1.5 outside.
        approx(sd_capsule(Vec3::new(0.0, 2.0, 0.0), a, b, 0.5), 1.5);
        // Inside the tube.
        assert!(sd_capsule(Vec3::new(0.0, 0.2, 0.0), a, b, 0.5) < 0.0);
    }

    #[test]
    fn cylinder_infinite_ignores_y() {
        approx(sd_cylinder_infinite(Vec3::new(3.0, 0.0, 4.0), 1.0), 4.0);
        approx(sd_cylinder_infinite(Vec3::new(3.0, 100.0, 4.0), 1.0), 4.0);
        assert!(sd_cylinder_infinite(Vec3::new(0.2, 5.0, 0.0), 1.0) < 0.0);
    }

    #[test]
    fn torus_distance_and_sign() {
        // major 2, minor 0.5: point on the +x ring center line at radius 2.
        approx(sd_torus(Vec3::new(2.0, 0.0, 0.0), 2.0, 0.5), -0.5);
        // Just outside the tube at radius 2 + 0.5 + 0.5.
        approx(sd_torus(Vec3::new(3.0, 0.0, 0.0), 2.0, 0.5), 0.5);
        // Above the ring line by 1 unit in y.
        approx(sd_torus(Vec3::new(2.0, 1.0, 0.0), 2.0, 0.5), 0.5);
    }

    #[test]
    fn plane_is_signed_offset() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        approx(sd_plane(Vec3::new(0.0, 3.0, 0.0), n, 0.0), 3.0);
        approx(sd_plane(Vec3::new(0.0, -2.0, 0.0), n, 0.0), -2.0);
        // With an offset d the surface shifts.
        approx(sd_plane(Vec3::new(0.0, 0.0, 0.0), n, 1.5), 1.5);
    }

    #[test]
    fn hard_union_is_min() {
        approx(op_union(2.0, -1.0), -1.0);
        approx(op_union(0.5, 3.0), 0.5);
    }

    #[test]
    fn hard_subtract_carves() {
        // Subtract solid d1 (inside, negative) from d2.
        approx(op_subtract(-0.5, 1.0), 1.0);
        // d2 is inside (-0.5) and d1 is outside (1.0): the point survives the
        // carve and stays inside at -0.5 (max(-1.0, -0.5)).
        approx(op_subtract(1.0, -0.5), -0.5);
    }

    #[test]
    fn hard_intersect_is_max() {
        approx(op_intersect(-1.0, 0.5), 0.5);
        approx(op_intersect(-2.0, -0.5), -0.5);
    }

    #[test]
    fn smooth_union_reduces_to_min_at_zero_k() {
        approx(op_smooth_union(2.0, -1.0, 0.0), op_union(2.0, -1.0));
        approx(
            op_smooth_union(2.0, -1.0, CMP_EPS * 0.5),
            op_union(2.0, -1.0),
        );
    }

    #[test]
    fn smooth_union_never_exceeds_min_and_is_bounded() {
        let d1 = 0.3;
        let d2 = 0.4;
        let k = 0.5;
        let s = op_smooth_union(d1, d2, k);
        // The blended result dips at or below the hard min.
        assert!(s <= op_union(d1, d2) + EPS);
        // But stays within k of it.
        assert!(s >= op_union(d1, d2) - k);
    }

    #[test]
    fn smooth_union_is_continuous_across_the_seam() {
        let k = 0.4;
        let mut prev = op_smooth_union(-1.0, 1.0, k);
        let mut t = -1.0_f32;
        // Sweep d1 while holding d2 fixed; consecutive samples stay close.
        while t <= 1.0 {
            let cur = op_smooth_union(t, 1.0, k);
            assert!((cur - prev).abs() < 0.05);
            prev = cur;
            t += 0.02;
        }
    }

    #[test]
    fn smooth_subtract_reduces_to_hard_at_zero_k() {
        approx(op_smooth_subtract(-0.5, 1.0, 0.0), op_subtract(-0.5, 1.0));
    }

    #[test]
    fn smooth_intersect_reduces_to_hard_at_zero_k() {
        approx(op_smooth_intersect(-1.0, 0.5, 0.0), op_intersect(-1.0, 0.5));
    }

    #[test]
    fn smooth_intersect_is_bounded_by_hard() {
        let d1 = -0.2;
        let d2 = 0.1;
        let k = 0.5;
        let s = op_smooth_intersect(d1, d2, k);
        assert!(s >= op_intersect(d1, d2) - EPS);
        assert!(s <= op_intersect(d1, d2) + k);
    }

    #[test]
    fn gradient_normal_points_radially_out_of_a_sphere() {
        let field = |q: Vec3| sd_sphere(q, 1.0);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let n = gradient_normal(field, p, 1.0e-3);
        approx(n.length(), 1.0);
        approx(n.x, 1.0);
        assert!(n.y.abs() < EPS);
        assert!(n.z.abs() < EPS);
    }

    #[test]
    fn gradient_normal_matches_plane_normal() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let field = move |q: Vec3| sd_plane(q, n, 0.0);
        let g = gradient_normal(field, Vec3::new(2.0, 3.0, -1.0), 1.0e-3);
        approx(g.y, 1.0);
        assert!(g.x.abs() < EPS);
        assert!(g.z.abs() < EPS);
    }

    #[test]
    fn gradient_normal_flat_field_is_zero() {
        let field = |_q: Vec3| 7.0_f32;
        assert_eq!(gradient_normal(field, Vec3::ZERO, 1.0e-3), Vec3::ZERO);
    }

    #[test]
    fn capsule_distance_wrapper_matches_free_function() {
        let cap = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        let p = Vec3::new(0.0, 2.0, 0.0);
        approx(cap.distance(p), sd_capsule(p, cap.a, cap.b, cap.r));
    }

    #[test]
    fn std430_size_is_two_vec4_slots() {
        let cap = Capsule::new(Vec3::ZERO, Vec3::ZERO, 0.0);
        let bytes = cap.to_std430();
        assert_eq!(bytes.len(), 32);
        assert_eq!(CAPSULE_SDF_STD430_SIZE, bytes.len());
    }

    #[test]
    fn std430_roundtrip_decodes_endpoints_and_radius() {
        let cap = Capsule::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, -5.0, -6.0), 0.75);
        let bytes = cap.to_std430();
        let word = |i: usize| {
            let start = i * U32_STRIDE;
            let mut w = [0u8; U32_STRIDE];
            w.copy_from_slice(&bytes[start..start + U32_STRIDE]);
            f32::from_le_bytes(w)
        };
        approx(word(0), 1.0);
        approx(word(1), 2.0);
        approx(word(2), 3.0);
        approx(word(3), 0.75);
        approx(word(4), -4.0);
        approx(word(5), -5.0);
        approx(word(6), -6.0);
        approx(word(7), 0.0);
    }

    #[test]
    fn gpu_storage_bytes_clamps_and_scales() {
        assert_eq!(gpu_storage_bytes(0), CAPSULE_SDF_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), 4 * CAPSULE_SDF_STD430_SIZE);
    }

    #[test]
    fn packing_many_capsules_is_contiguous() {
        let caps = [
            Capsule::new(Vec3::ZERO, Vec3::splat(1.0), 0.1),
            Capsule::new(Vec3::splat(-1.0), Vec3::splat(2.0), 0.2),
        ];
        let mut blob: Vec<u8> = Vec::new();
        for c in &caps {
            blob.extend_from_slice(&c.to_std430());
        }
        assert_eq!(blob.len(), gpu_storage_bytes(caps.len()));
    }
}
