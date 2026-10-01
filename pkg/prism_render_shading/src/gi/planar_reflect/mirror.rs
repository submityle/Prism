//! Exact planar mirror reflection geometry — CPU golden.
//!
//! A planar reflector (a mirror, a sheet of still water, a polished floor) maps
//! every point of the scene to its mirror image across a single infinite plane.
//! Unlike the stochastic screen-space tracer in [`crate::gi::reflect`], this is
//! an *exact* geometric reflection: there is no ray marching, no sampling, and
//! no approximation beyond floating-point rounding.  The reflection of a point
//! `p` across the plane `n·p + d = 0` (with `n` a unit normal) is
//!
//! ```text
//! reflect(p) = p - 2 (n·p + d) n
//! ```
//!
//! and the reflection of a free vector `v` drops the plane offset `d`:
//!
//! ```text
//! reflect(v) = v - 2 (n·v) n
//! ```
//!
//! Both are instances of a **Householder reflection**.  Packed into a 4×4
//! homogeneous matrix the transform is
//!
//! ```text
//! R = | I - 2 n nᵀ   -2 d n |
//!     |    0ᵀ            1   |
//! ```
//!
//! whose upper-left 3×3 block is the classic Householder matrix `I - 2 n nᵀ`
//! and whose translation column `-2 d n` accounts for a plane that does not pass
//! through the origin.  `R` is an *improper* orthogonal transform: `det R = -1`,
//! so it reverses handedness.  A camera reflected by `R` therefore sees a
//! mirror-flipped world and front faces become back faces — the caller must
//! invert triangle winding (or the cull mode) when rendering from the mirrored
//! camera.  This module exposes the reflected camera basis so the caller can do
//! exactly that.
//!
//! # Conventions
//! * `no_std`: math via `bevy_math`; no `alloc` is needed here.  Transcendental
//!   functions would go through [`bevy_math::ops`]; this module only needs
//!   `sqrt`, taken via the inherent method.
//! * A plane is stored as a unit normal `n` and a scalar offset `d` with the
//!   implicit equation `n·p + d = 0`.  The **positive** half-space is `n·p + d >
//!   0` (the side the normal points into).
//! * [`Plane::new`] renormalises the normal (scaling `d` to match) so the stored
//!   plane is always well-formed; a (near-)zero normal falls back to the plane
//!   `y = 0` (`n = +Y`, `d = 0`) so results stay finite and never `NaN`.
//! * `bevy_math` matrices are column-major; [`Plane::reflection_matrix`] returns
//!   a [`Mat4`] whose `mul_vec4(vec4(p, 1))` equals [`Plane::reflect_point`].
//! * Every function is deterministic and pure: no RNG, no I/O, no GPU, no global
//!   state, and no `unsafe`.

use bevy_math::{Mat4, Vec3, Vec4};

/// Smallest squared normal length treated as a usable plane normal; anything at
/// or below this collapses to the default `y = 0` fallback plane.
const MIN_NORMAL_LEN_SQ: f32 = 1.0e-12;
/// Smallest squared length a camera forward/up vector may have before the
/// mirrored basis falls back to a canonical axis instead of normalising noise.
const MIN_BASIS_LEN_SQ: f32 = 1.0e-12;

/// An oriented infinite plane `n·p + d = 0` with a unit normal `n`.
///
/// The normal is kept normalised by every constructor so geometric queries
/// ([`Plane::signed_distance`], [`Plane::reflect_point`]) are metric-correct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit-length plane normal; the positive half-space is `n·p + d > 0`.
    pub normal: Vec3,
    /// Signed plane offset: the plane is the locus of `n·p + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Builds a plane from a raw `(normal, d)` pair, renormalising the normal.
    ///
    /// Scaling the normal to unit length also scales `d` by the same factor so
    /// the represented plane is unchanged.  A degenerate (near-zero) normal
    /// falls back to the plane `y = 0` so the result is always finite.
    #[inline]
    pub fn new(normal: Vec3, d: f32) -> Self {
        let len_sq = normal.length_squared();
        if !len_sq.is_finite() || len_sq <= MIN_NORMAL_LEN_SQ || !d.is_finite() {
            return Plane {
                normal: Vec3::Y,
                d: 0.0,
            };
        }
        let inv_len = 1.0 / len_sq.sqrt();
        Plane {
            normal: normal * inv_len,
            d: d * inv_len,
        }
    }

    /// Builds the plane that passes through `point` with the given `normal`.
    ///
    /// Solves `n·point + d = 0` for `d = -n·point` after normalising `n`.
    #[inline]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        let unit = Self::new(normal, 0.0).normal;
        Plane {
            normal: unit,
            d: -unit.dot(sanitize(point)),
        }
    }

    /// The signed distance from `point` to the plane: `n·point + d`.
    ///
    /// Positive on the side the normal points toward, negative behind it, and
    /// zero on the plane.  Because `n` is unit length this is a true Euclidean
    /// distance, not merely proportional to one.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(sanitize(point)) + self.d
    }

    /// Reflects a *position* across the plane: `p - 2 (n·p + d) n`.
    ///
    /// A point on the plane is returned unchanged (its signed distance is zero);
    /// applying the reflection twice returns the original point.
    #[inline]
    pub fn reflect_point(&self, point: Vec3) -> Vec3 {
        let p = sanitize(point);
        p - 2.0 * self.signed_distance(p) * self.normal
    }

    /// Reflects a *free vector* (direction) across the plane: `v - 2 (n·v) n`.
    ///
    /// The plane offset `d` is irrelevant for directions, so only the normal
    /// component is flipped.  Length is preserved (a Householder reflection is
    /// an isometry) and a vector lying in the plane is unchanged.
    #[inline]
    pub fn reflect_vector(&self, v: Vec3) -> Vec3 {
        let v = sanitize(v);
        v - 2.0 * self.normal.dot(v) * self.normal
    }

    /// The 4×4 homogeneous Householder reflection matrix for this plane.
    ///
    /// Column-major, so `matrix.mul_vec4(p.extend(1.0)).truncate()` equals
    /// [`Plane::reflect_point`]`(p)` and `matrix.mul_vec4(v.extend(0.0))`
    /// equals [`Plane::reflect_vector`]`(v)`.  The matrix is an involution
    /// (`R·R = I`) with determinant `-1`.
    pub fn reflection_matrix(&self) -> Mat4 {
        let n = self.normal;
        let d = self.d;
        // Column j holds entries (δ_ij - 2 n_i n_j) for i = 0..3 and a w term.
        let col0 = Vec4::new(1.0 - 2.0 * n.x * n.x, -2.0 * n.y * n.x, -2.0 * n.z * n.x, 0.0);
        let col1 = Vec4::new(-2.0 * n.x * n.y, 1.0 - 2.0 * n.y * n.y, -2.0 * n.z * n.y, 0.0);
        let col2 = Vec4::new(-2.0 * n.x * n.z, -2.0 * n.y * n.z, 1.0 - 2.0 * n.z * n.z, 0.0);
        let col3 = Vec4::new(-2.0 * d * n.x, -2.0 * d * n.y, -2.0 * d * n.z, 1.0);
        Mat4::from_cols(col0, col1, col2, col3)
    }

    /// Flips the plane to face the opposite half-space (`n → -n`, `d → -d`).
    ///
    /// Represents the same set of points but swaps which side is "positive",
    /// which is useful when orienting the reflector toward the camera.
    #[inline]
    pub fn flipped(&self) -> Self {
        Plane {
            normal: -self.normal,
            d: -self.d,
        }
    }
}

/// The pinhole basis of a camera: eye position plus an orthonormal frame.
///
/// `forward` points along the viewing direction, `up` is the camera's up axis,
/// and `right = forward × up` (a right-handed frame).  All three are unit
/// length; `eye` is a world-space position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraBasis {
    /// World-space eye (camera) position.
    pub eye: Vec3,
    /// Unit forward (view) direction.
    pub forward: Vec3,
    /// Unit up direction, orthogonal to `forward`.
    pub up: Vec3,
    /// Unit right direction, `forward × up`.
    pub right: Vec3,
}

impl CameraBasis {
    /// Builds an orthonormal camera basis from an eye, forward, and up hint.
    ///
    /// `forward` and `up` are orthonormalised (Gram–Schmidt); degenerate inputs
    /// (zero-length or parallel) fall back to a canonical axis so the frame is
    /// always valid and right-handed.
    pub fn new(eye: Vec3, forward: Vec3, up_hint: Vec3) -> Self {
        let fwd = normalize_or(sanitize(forward), Vec3::NEG_Z);
        let up_in = sanitize(up_hint);
        // Remove the forward component from the up hint, then renormalise.
        let up_proj = up_in - fwd * fwd.dot(up_in);
        let up = normalize_or(up_proj, fallback_up(fwd));
        let right = normalize_or(fwd.cross(up), Vec3::X);
        // Re-derive up so the frame is exactly orthonormal despite rounding.
        let up = right.cross(fwd).normalize();
        CameraBasis {
            eye: sanitize(eye),
            forward: fwd,
            up,
            right,
        }
    }

    /// Returns `true` when the frame is right-handed (`right·(forward×up) > 0`).
    #[inline]
    pub fn is_right_handed(&self) -> bool {
        self.right.dot(self.forward.cross(self.up)) > 0.0
    }
}

/// The camera basis as seen through a planar mirror.
///
/// The mirrored eye is the reflected eye position; the mirrored forward and up
/// are the reflected view/up directions.  Because planar reflection reverses
/// handedness, a frame built directly from the reflected axes would be
/// left-handed, so [`mirror_camera`] rebuilds a consistent right-handed frame
/// (`right = forward × up`) and reports the handedness flip via
/// [`MirroredCamera::handedness_flipped`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirroredCamera {
    /// The right-handed basis to render the reflection pass with.
    pub basis: CameraBasis,
    /// `true` when reflection reversed handedness (it always does for a real
    /// plane): the caller must invert triangle winding / cull mode.
    pub handedness_flipped: bool,
}

/// Reflects a camera basis across `plane`, producing the mirror-pass camera.
///
/// The eye is reflected as a point and the forward/up as vectors; the returned
/// [`CameraBasis`] is re-orthonormalised into a right-handed frame.  The naive
/// reflected frame is left-handed, so [`MirroredCamera::handedness_flipped`] is
/// set whenever the reflection genuinely flipped orientation.
pub fn mirror_camera(plane: &Plane, camera: &CameraBasis) -> MirroredCamera {
    let eye = plane.reflect_point(camera.eye);
    let forward = normalize_or(plane.reflect_vector(camera.forward), Vec3::NEG_Z);
    let up = normalize_or(plane.reflect_vector(camera.up), fallback_up(forward));
    // The reflected (forward, up, right=forward×up) frame is left-handed; detect
    // that flip relative to the source frame.
    let reflected_right = forward.cross(up);
    let flipped = reflected_right.dot(plane.reflect_vector(camera.right)) < 0.0;
    MirroredCamera {
        basis: CameraBasis::new(eye, forward, up),
        handedness_flipped: flipped,
    }
}

/// Clamps a position/vector to all-finite components, mapping non-finite to 0.
#[inline]
fn sanitize(v: Vec3) -> Vec3 {
    if v.is_finite() {
        v
    } else {
        Vec3::new(
            if v.x.is_finite() { v.x } else { 0.0 },
            if v.y.is_finite() { v.y } else { 0.0 },
            if v.z.is_finite() { v.z } else { 0.0 },
        )
    }
}

/// Normalises `v`, or returns `fallback` when `v` is too short to normalise.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > MIN_BASIS_LEN_SQ {
        v * (1.0 / len_sq.sqrt())
    } else {
        fallback
    }
}

/// A canonical up axis not parallel to `forward` (for degenerate up hints).
#[inline]
fn fallback_up(forward: Vec3) -> Vec3 {
    if forward.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A general, non-axis-aligned plane for stress tests.
    fn tilted_plane() -> Plane {
        Plane::new(Vec3::new(0.3, 1.0, -0.4), 2.5)
    }

    #[test]
    fn constructor_normalises_normal_and_scales_d() {
        let p = Plane::new(Vec3::new(0.0, 2.0, 0.0), 6.0);
        assert!((p.normal - Vec3::Y).length() < 1e-6);
        // d scaled by 1/|n| = 1/2.
        assert!((p.d - 3.0).abs() < 1e-6);
        // Same plane: a point on it still has zero signed distance.
        let on = Vec3::new(5.0, -3.0, 1.0); // y = -3 satisfies y + 3 = 0.
        assert!(p.signed_distance(on).abs() < 1e-6);
    }

    #[test]
    fn degenerate_normal_falls_back_to_y_plane() {
        let p = Plane::new(Vec3::ZERO, 7.0);
        assert_eq!(p.normal, Vec3::Y);
        assert_eq!(p.d, 0.0);
        let bad = Plane::new(Vec3::new(f32::NAN, 0.0, 0.0), 1.0);
        assert!(bad.normal.is_finite());
    }

    #[test]
    fn point_on_plane_is_unchanged() {
        let p = tilted_plane();
        // Build a point exactly on the plane: start anywhere, project onto it.
        let anywhere = Vec3::new(1.0, 2.0, 3.0);
        let on = anywhere - p.normal * p.signed_distance(anywhere);
        assert!(p.signed_distance(on).abs() < 1e-5);
        let r = p.reflect_point(on);
        assert!((r - on).length() < 1e-5);
    }

    #[test]
    fn reflecting_twice_is_identity() {
        let p = tilted_plane();
        for q in [
            Vec3::new(4.0, -2.0, 1.0),
            Vec3::new(-7.0, 3.5, 9.0),
            Vec3::ZERO,
        ] {
            let back = p.reflect_point(p.reflect_point(q));
            assert!((back - q).length() < 1e-4, "double reflect != identity");
        }
    }

    #[test]
    fn reflection_flips_signed_distance_sign() {
        let p = tilted_plane();
        let q = Vec3::new(1.0, 8.0, -2.0);
        let dq = p.signed_distance(q);
        let dr = p.signed_distance(p.reflect_point(q));
        // Reflected point sits the same distance on the opposite side.
        assert!((dq + dr).abs() < 1e-4);
    }

    #[test]
    fn vector_reflection_preserves_length_and_tangent() {
        let p = tilted_plane();
        let v = Vec3::new(2.0, -5.0, 3.0);
        let r = p.reflect_vector(v);
        assert!((r.length() - v.length()).abs() < 1e-4);
        // A vector tangent to the plane is unchanged.
        let tangent = p.normal.cross(Vec3::X);
        let rt = p.reflect_vector(tangent);
        assert!((rt - tangent).length() < 1e-5);
        // The normal direction is exactly negated.
        let rn = p.reflect_vector(p.normal);
        assert!((rn + p.normal).length() < 1e-5);
    }

    #[test]
    fn matrix_matches_pointwise_reflection() {
        let p = tilted_plane();
        let m = p.reflection_matrix();
        for q in [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 0.5, 6.0),
            Vec3::new(10.0, -10.0, 2.0),
        ] {
            let via_matrix = m.mul_vec4(q.extend(1.0)).truncate();
            let via_fn = p.reflect_point(q);
            assert!((via_matrix - via_fn).length() < 1e-4, "matrix != reflect_point");
        }
        // The matrix applied to a direction (w = 0) matches reflect_vector.
        let v = Vec3::new(3.0, -2.0, 5.0);
        let mv = m.mul_vec4(v.extend(0.0)).truncate();
        assert!((mv - p.reflect_vector(v)).length() < 1e-4);
    }

    #[test]
    fn matrix_is_an_involution_with_negative_determinant() {
        let p = tilted_plane();
        let m = p.reflection_matrix();
        let mm = m * m;
        let id = Mat4::IDENTITY;
        // R·R = I.
        for c in 0..4 {
            assert!((mm.col(c) - id.col(c)).length() < 1e-4);
        }
        // Improper transform: determinant is -1.
        assert!((m.determinant() + 1.0).abs() < 1e-4);
    }

    #[test]
    fn mirror_camera_flips_handedness_and_reflects_eye() {
        let plane = Plane::new(Vec3::Y, 0.0); // ground mirror at y = 0.
        let cam = CameraBasis::new(
            Vec3::new(0.0, 3.0, 5.0),
            Vec3::new(0.0, -0.3, -1.0),
            Vec3::Y,
        );
        assert!(cam.is_right_handed());
        let mirrored = mirror_camera(&plane, &cam);
        // Eye reflects below the floor.
        assert!((mirrored.basis.eye - Vec3::new(0.0, -3.0, 5.0)).length() < 1e-5);
        // Reflection reversed handedness.
        assert!(mirrored.handedness_flipped);
        // The rebuilt basis is itself a valid right-handed frame.
        assert!(mirrored.basis.is_right_handed());
        // Forward's vertical component flipped sign (reflected across y = 0).
        assert!(mirrored.basis.forward.y > 0.0);
    }

    #[test]
    fn camera_basis_orthonormal_even_with_parallel_up_hint() {
        // up hint parallel to forward -> fallback keeps the frame valid.
        let cam = CameraBasis::new(Vec3::ZERO, Vec3::Y, Vec3::Y);
        assert!(cam.forward.is_finite() && cam.up.is_finite() && cam.right.is_finite());
        assert!(cam.forward.dot(cam.up).abs() < 1e-5);
        assert!(cam.is_right_handed());
    }
}
