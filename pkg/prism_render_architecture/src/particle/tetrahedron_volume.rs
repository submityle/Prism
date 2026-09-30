//! Signed tetrahedron volume, the `orient3d` orientation predicate, tetrahedron
//! centroid, and closed-triangle-mesh volume / volume-centroid via the
//! divergence theorem, for the particle geometry contracts (design §8.2, §11).
//!
//! Several particle stages need to reason about *solid* volumes rather than
//! surface areas: a collision-proxy pass wants the enclosed volume (and volume
//! centroid) of a convex particle hull so it can distribute mass; a
//! spawn-in-volume emitter wants the signed volume of a tetrahedron so it can
//! reject a degenerate seed cell; and a mesh-emitter samples a closed triangle
//! shell and must know its total enclosed volume and center of mass. This
//! module owns the small, `CPU`-verifiable contract those stages share: the
//! scalar triple-product signed volume of a tetrahedron, the 3D orientation
//! predicate built on its sign, the arithmetic-mean centroid of four points,
//! and the divergence-theorem accumulation that turns a closed triangle shell
//! into a total signed volume and its volume-weighted centroid.
//!
//! # Strict scope
//! This module only computes tetrahedron volume / orientation / centroid and
//! the closed-mesh volume built from them. It deliberately does not construct
//! circumspheres or 2D circumcircles ([`super::triangle_circumcircle`]), it
//! does not assemble an inertia tensor ([`super::inertia_tensor`]), and it does
//! not build 2D convex hulls ([`super::convex_hull_2d`]); it neither imports nor
//! reconstructs any of those contracts, and it reuses the crate's own
//! [`Vec3`](crate::particle::Vec3) instead of a private point type.
//!
//! # No transcendental math
//! Every quantity here is a ratio of scalar triple products, so the whole
//! module is pure `+`, `-`, `*`, `/`. There is no `sqrt` (volume and centroid
//! need none), no `sin`, `cos`, `powf`, `ceil`, `round`, `cbrt`, or any other
//! transcendental call, and no `f32` equality: a near-zero volume or
//! determinant is always compared against [`CMP_EPS`], never `== 0.0`.

use crate::particle::gpu_layout::VEC4_STRIDE;
use crate::particle::Vec3;

/// Magnitude below which a signed volume or an orientation determinant is
/// treated as zero.
///
/// This is the comparison rule used throughout instead of `==` on `f32`: a
/// scalar whose absolute value does not exceed this bound is considered zero, so
/// a tetrahedron whose signed volume is within it is degenerate (its four
/// vertices are coplanar or coincident), and a closed mesh whose accumulated
/// volume is within it has no reliably-defined volume centroid.
pub const CMP_EPS: f32 = 1e-6;

/// Byte stride of one packed [`MeshVolume`] record in a `std430` storage buffer.
///
/// The record is four `f32` words (`volume`, then the three centroid
/// components), so it fills exactly one [`VEC4_STRIDE`] slot and is a multiple
/// of 16 as `std430` requires for a `vec4`-aligned element.
pub const MESH_VOLUME_STRIDE: usize = VEC4_STRIDE;

/// Returns `true` when `x` is within [`CMP_EPS`] of zero.
#[must_use]
fn approx_zero(x: f32) -> bool {
    x.abs() <= CMP_EPS
}

/// Six times the signed volume of the tetrahedron `(a, b, c, d)`, i.e. the
/// scalar triple product `dot(b - a, cross(c - a, d - a))`.
///
/// This is the raw orientation determinant: its sign classifies whether `d`
/// lies above, below, or on the plane of the oriented triangle `(a, b, c)`, and
/// [`signed_volume`] is simply this value divided by six.
#[must_use]
pub fn orient3d_det(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    b.sub(a).dot(c.sub(a).cross(d.sub(a)))
}

/// Signed volume of the tetrahedron with vertices `(a, b, c, d)`.
///
/// Defined as `dot(b - a, cross(c - a, d - a)) / 6`, the standard scalar
/// triple product. The sign follows the winding of the base triangle
/// `(a, b, c)` seen from `d`: swapping any two vertices negates it, and four
/// coplanar vertices give (numerically) zero.
#[must_use]
pub fn signed_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    orient3d_det(a, b, c, d) / 6.0
}

/// The centroid (arithmetic mean) of a tetrahedron's four vertices,
/// `(a + b + c + d) / 4`.
#[must_use]
pub fn centroid(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec3 {
    a.add(b).add(c).add(d).scale(0.25)
}

/// Where the point `d` lies relative to the oriented plane of triangle
/// `(a, b, c)`, i.e. the sign of the tetrahedron's signed volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// `d` lies on the positive side of `(a, b, c)` (strictly positive signed
    /// volume): the four vertices form a positively-oriented tetrahedron.
    Positive,
    /// `d` lies on the negative side (strictly negative signed volume).
    Negative,
    /// The four vertices are coplanar (signed volume within [`CMP_EPS`]).
    Coplanar,
}

/// Classifies the orientation of the tetrahedron `(a, b, c, d)` from the sign of
/// its volume determinant, treating a near-zero determinant as coplanar.
#[must_use]
pub fn orient3d(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Orientation {
    let det = orient3d_det(a, b, c, d);
    if approx_zero(det) {
        Orientation::Coplanar
    } else if det > 0.0 {
        Orientation::Positive
    } else {
        Orientation::Negative
    }
}

/// The enclosed signed volume of a closed triangle mesh together with its
/// volume-weighted centroid (center of mass of the solid it bounds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVolume {
    /// Total signed volume. Positive when the faces wind counter-clockwise as
    /// seen from outside (outward normals); the magnitude is the enclosed
    /// volume, and the sign flips if every face winding is reversed.
    pub volume: f32,
    /// Volume-weighted centroid of the enclosed solid. When the total volume is
    /// within [`CMP_EPS`] of zero the centroid is not well defined and is
    /// reported as [`Vec3::ZERO`].
    pub centroid: Vec3,
}

impl MeshVolume {
    /// Packs the record into its `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[volume, centroid.x, centroid.y, centroid.z]` as raw `u32`
    /// words (each `f32` via [`f32::to_bits`]), filling one `vec4` slot that
    /// matches [`MESH_VOLUME_STRIDE`].
    #[must_use]
    pub fn to_std430(&self) -> [u32; 4] {
        [
            self.volume.to_bits(),
            self.centroid.x.to_bits(),
            self.centroid.y.to_bits(),
            self.centroid.z.to_bits(),
        ]
    }

    /// Whether the accumulated volume is (numerically) zero, in which case the
    /// centroid is degenerate and reported as [`Vec3::ZERO`].
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        approx_zero(self.volume)
    }
}

/// Accumulates the signed volume and volume-weighted centroid of a closed
/// triangle mesh via the divergence theorem.
///
/// Each face `(p0, p1, p2)` forms a tetrahedron with the origin; the mesh's
/// total signed volume is the sum of those tetrahedra's signed volumes, and its
/// centroid is their volume-weighted average of per-tetrahedron centroids. The
/// origin cancels out for a *closed* shell, so the result is independent of
/// where the origin sits relative to the mesh.
///
/// When the total volume is within [`CMP_EPS`] of zero (an empty, open, or
/// self-cancelling shell) the centroid is undefined and returned as
/// [`Vec3::ZERO`], guarding the division.
#[must_use]
pub fn closed_mesh_volume(faces: &[(Vec3, Vec3, Vec3)]) -> MeshVolume {
    let origin = Vec3::ZERO;
    let mut total = 0.0f32;
    let mut weighted = Vec3::ZERO;
    for &(p0, p1, p2) in faces {
        let vol = signed_volume(origin, p0, p1, p2);
        let c = centroid(origin, p0, p1, p2);
        total += vol;
        weighted = weighted.add(c.scale(vol));
    }
    let centroid = if approx_zero(total) {
        Vec3::ZERO
    } else {
        weighted.scale(1.0 / total)
    };
    MeshVolume {
        volume: total,
        centroid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-difference float comparison for assertions (never `==`).
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn close_vec(a: Vec3, b: Vec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    fn unit_tetra() -> (Vec3, Vec3, Vec3, Vec3) {
        (
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Twelve outward-facing (CCW-from-outside) triangles of the axis-aligned
    /// cube `[0, s]^3`, offset by `origin`.
    fn cube_mesh(origin: Vec3, s: f32) -> [(Vec3, Vec3, Vec3); 12] {
        let v = |x: f32, y: f32, z: f32| origin.add(Vec3::new(x * s, y * s, z * s));
        let v0 = v(0.0, 0.0, 0.0);
        let v1 = v(1.0, 0.0, 0.0);
        let v2 = v(1.0, 1.0, 0.0);
        let v3 = v(0.0, 1.0, 0.0);
        let v4 = v(0.0, 0.0, 1.0);
        let v5 = v(1.0, 0.0, 1.0);
        let v6 = v(1.0, 1.0, 1.0);
        let v7 = v(0.0, 1.0, 1.0);
        [
            // -Z bottom
            (v0, v2, v1),
            (v0, v3, v2),
            // +Z top
            (v4, v5, v6),
            (v4, v6, v7),
            // -Y front
            (v0, v1, v5),
            (v0, v5, v4),
            // +Y back
            (v3, v7, v6),
            (v3, v6, v2),
            // -X left
            (v0, v4, v7),
            (v0, v7, v3),
            // +X right
            (v1, v2, v6),
            (v1, v6, v5),
        ]
    }

    #[test]
    fn unit_tetrahedron_volume_is_one_sixth() {
        let (a, b, c, d) = unit_tetra();
        assert!(close(signed_volume(a, b, c, d), 1.0 / 6.0));
    }

    #[test]
    fn orient3d_det_is_six_times_volume() {
        let (a, b, c, d) = unit_tetra();
        assert!(close(
            orient3d_det(a, b, c, d),
            6.0 * signed_volume(a, b, c, d)
        ));
    }

    #[test]
    fn swapping_two_vertices_negates_volume() {
        let (a, b, c, d) = unit_tetra();
        let v = signed_volume(a, b, c, d);
        let swapped = signed_volume(a, c, b, d);
        assert!(close(swapped, -v));
        assert!(!close(v, 0.0));
    }

    #[test]
    fn rotating_base_vertices_preserves_volume() {
        let (a, b, c, d) = unit_tetra();
        // An even permutation (cyclic on the base) keeps the sign.
        assert!(close(signed_volume(a, b, c, d), signed_volume(b, c, a, d)));
    }

    #[test]
    fn reflected_tetra_has_negative_volume() {
        let a = Vec3::ZERO;
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let d = Vec3::new(0.0, 0.0, -1.0);
        assert!(signed_volume(a, b, c, d) < 0.0);
    }

    #[test]
    fn coplanar_tetra_volume_is_zero() {
        let a = Vec3::ZERO;
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let d = Vec3::new(1.0, 1.0, 0.0);
        assert!(approx_zero(signed_volume(a, b, c, d)));
    }

    #[test]
    fn coincident_vertices_volume_is_zero() {
        let a = Vec3::new(2.0, 3.0, 4.0);
        assert!(approx_zero(signed_volume(a, a, a, a)));
        let b = Vec3::new(5.0, 6.0, 7.0);
        assert!(approx_zero(signed_volume(a, a, b, b)));
    }

    #[test]
    fn orient3d_positive_side() {
        let (a, b, c, d) = unit_tetra();
        assert_eq!(orient3d(a, b, c, d), Orientation::Positive);
    }

    #[test]
    fn orient3d_negative_side() {
        let a = Vec3::ZERO;
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let d = Vec3::new(0.0, 0.0, -1.0);
        assert_eq!(orient3d(a, b, c, d), Orientation::Negative);
    }

    #[test]
    fn orient3d_coplanar_case() {
        let a = Vec3::ZERO;
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(2.0, 0.0, 0.0);
        let d = Vec3::new(3.0, 0.0, 0.0);
        assert_eq!(orient3d(a, b, c, d), Orientation::Coplanar);
    }

    #[test]
    fn orient3d_three_states_are_distinct() {
        let (a, b, c, d) = unit_tetra();
        let below = Vec3::new(0.0, 0.0, -1.0);
        let on = Vec3::new(1.0, 1.0, 0.0);
        assert_eq!(orient3d(a, b, c, d), Orientation::Positive);
        assert_eq!(orient3d(a, b, c, below), Orientation::Negative);
        assert_eq!(orient3d(a, b, c, on), Orientation::Coplanar);
    }

    #[test]
    fn centroid_of_unit_tetra_is_quarter_sum() {
        let (a, b, c, d) = unit_tetra();
        assert!(close_vec(centroid(a, b, c, d), Vec3::new(0.25, 0.25, 0.25)));
    }

    #[test]
    fn centroid_is_mean_of_arbitrary_points() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(5.0, 6.0, 7.0);
        let c = Vec3::new(-1.0, 0.0, 1.0);
        let d = Vec3::new(3.0, -4.0, 5.0);
        let expected = Vec3::new(
            (1.0 + 5.0 - 1.0 + 3.0) / 4.0,
            (2.0 + 6.0 + 0.0 - 4.0) / 4.0,
            (3.0 + 7.0 + 1.0 + 5.0) / 4.0,
        );
        assert!(close_vec(centroid(a, b, c, d), expected));
    }

    #[test]
    fn translation_leaves_volume_invariant() {
        let (a, b, c, d) = unit_tetra();
        let t = Vec3::new(10.0, -3.0, 7.5);
        let v0 = signed_volume(a, b, c, d);
        let v1 = signed_volume(a.add(t), b.add(t), c.add(t), d.add(t));
        assert!(close(v0, v1));
    }

    #[test]
    fn translation_shifts_centroid_by_offset() {
        let (a, b, c, d) = unit_tetra();
        let t = Vec3::new(-2.0, 4.0, 6.0);
        let c0 = centroid(a, b, c, d);
        let c1 = centroid(a.add(t), b.add(t), c.add(t), d.add(t));
        assert!(close_vec(c1, c0.add(t)));
    }

    #[test]
    fn uniform_scale_cubes_the_volume() {
        let (a, b, c, d) = unit_tetra();
        let s = 3.0f32;
        let base = signed_volume(a, b, c, d);
        let scaled = signed_volume(a.scale(s), b.scale(s), c.scale(s), d.scale(s));
        assert!(close(scaled, base * s * s * s));
    }

    #[test]
    fn negative_scale_flips_and_cubes_volume() {
        let (a, b, c, d) = unit_tetra();
        let s = -2.0f32;
        let base = signed_volume(a, b, c, d);
        let scaled = signed_volume(a.scale(s), b.scale(s), c.scale(s), d.scale(s));
        assert!(close(scaled, base * s * s * s));
        assert!(scaled < 0.0);
    }

    #[test]
    fn closed_cube_volume_equals_edge_cubed() {
        let faces = cube_mesh(Vec3::ZERO, 1.0);
        let mv = closed_mesh_volume(&faces);
        assert!(close(mv.volume, 1.0));
    }

    #[test]
    fn scaled_cube_volume_equals_edge_cubed() {
        let s = 2.0f32;
        let faces = cube_mesh(Vec3::ZERO, s);
        let mv = closed_mesh_volume(&faces);
        assert!(close(mv.volume, s * s * s));
    }

    #[test]
    fn cube_centroid_is_geometric_center() {
        let faces = cube_mesh(Vec3::ZERO, 1.0);
        let mv = closed_mesh_volume(&faces);
        assert!(close_vec(mv.centroid, Vec3::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn translated_cube_centroid_tracks_translation() {
        let origin = Vec3::new(5.0, -2.0, 3.0);
        let faces = cube_mesh(origin, 2.0);
        let mv = closed_mesh_volume(&faces);
        // Center of a size-2 cube offset by `origin` is origin + (1,1,1).
        assert!(close_vec(mv.centroid, origin.add(Vec3::new(1.0, 1.0, 1.0))));
        assert!(close(mv.volume, 8.0));
    }

    #[test]
    fn reversed_winding_flips_mesh_volume_sign() {
        let faces = cube_mesh(Vec3::ZERO, 1.0);
        let reversed = faces.map(|(a, b, c)| (a, c, b));
        let outward = closed_mesh_volume(&faces);
        let inward = closed_mesh_volume(&reversed);
        assert!(close(inward.volume, -outward.volume));
        // Centroid is orientation-independent (sign cancels in the ratio).
        assert!(close_vec(inward.centroid, outward.centroid));
    }

    #[test]
    fn empty_mesh_is_degenerate() {
        let mv = closed_mesh_volume(&[]);
        assert!(mv.is_degenerate());
        assert!(approx_zero(mv.volume));
        assert_eq!(mv.centroid, Vec3::ZERO);
    }

    #[test]
    fn self_cancelling_shell_guards_centroid() {
        // A face and its exact reverse cancel: zero volume, guarded centroid.
        let p0 = Vec3::new(1.0, 0.0, 0.0);
        let p1 = Vec3::new(0.0, 1.0, 0.0);
        let p2 = Vec3::new(0.0, 0.0, 1.0);
        let faces = [(p0, p1, p2), (p0, p2, p1)];
        let mv = closed_mesh_volume(&faces);
        assert!(mv.is_degenerate());
        assert_eq!(mv.centroid, Vec3::ZERO);
    }

    #[test]
    fn mesh_volume_matches_origin_tetra_sum() {
        // For a single face, closed_mesh_volume must equal the origin tetra.
        let p0 = Vec3::new(1.0, 0.0, 0.0);
        let p1 = Vec3::new(0.0, 2.0, 0.0);
        let p2 = Vec3::new(0.0, 0.0, 3.0);
        let mv = closed_mesh_volume(&[(p0, p1, p2)]);
        assert!(close(mv.volume, signed_volume(Vec3::ZERO, p0, p1, p2)));
    }

    #[test]
    fn cube_volume_is_translation_invariant() {
        let a = closed_mesh_volume(&cube_mesh(Vec3::ZERO, 1.5));
        let b = closed_mesh_volume(&cube_mesh(Vec3::new(100.0, -50.0, 25.0), 1.5));
        assert!(close(a.volume, b.volume));
    }

    #[test]
    fn determinism_bit_for_bit() {
        let faces = cube_mesh(Vec3::new(0.3, 0.7, -1.1), 1.25);
        let first = closed_mesh_volume(&faces);
        let second = closed_mesh_volume(&faces);
        assert_eq!(first.volume.to_bits(), second.volume.to_bits());
        assert_eq!(first.centroid.x.to_bits(), second.centroid.x.to_bits());
        assert_eq!(first.centroid.y.to_bits(), second.centroid.y.to_bits());
        assert_eq!(first.centroid.z.to_bits(), second.centroid.z.to_bits());
    }

    #[test]
    fn std430_stride_is_multiple_of_sixteen() {
        assert_eq!(MESH_VOLUME_STRIDE, VEC4_STRIDE);
        assert_eq!(MESH_VOLUME_STRIDE % 16, 0);
    }

    #[test]
    fn std430_roundtrip_recovers_fields() {
        let mv = closed_mesh_volume(&cube_mesh(Vec3::ZERO, 1.0));
        let words = mv.to_std430();
        assert!(close(f32::from_bits(words[0]), mv.volume));
        assert!(close(f32::from_bits(words[1]), mv.centroid.x));
        assert!(close(f32::from_bits(words[2]), mv.centroid.y));
        assert!(close(f32::from_bits(words[3]), mv.centroid.z));
    }

    #[test]
    fn std430_storage_bytes_is_vec4_aligned() {
        use crate::particle::gpu_layout::storage_bytes;
        assert_eq!(storage_bytes(MESH_VOLUME_STRIDE, 4), 64);
        assert_eq!(storage_bytes(MESH_VOLUME_STRIDE, 0) % 16, 0);
    }
}
