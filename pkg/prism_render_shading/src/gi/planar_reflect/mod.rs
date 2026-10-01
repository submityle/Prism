//! Planar mirror reflection CPU golden references.
//!
//! Deterministic, GPU-free planar reflection math for mirrors / still water,
//! distinct from the stochastic SSR + reflection-probe blend in
//! [`crate::gi::reflect`]: here reflection is an exact mirror across a plane.
//!
//! * [`mirror`] — reflection of points/vectors across an arbitrary plane and
//!   the mirrored camera basis.
//! * [`clip`] — oblique near-plane clipping of the mirrored projection so the
//!   reflection is not drawn behind the reflector.
//! * [`sample`] — screen-space UV projection of the reflection buffer plus a
//!   roughness/distance-driven blur and edge fade.

pub mod clip;
pub mod mirror;
pub mod sample;

use bevy_math::{Mat4, Vec3};

use clip::oblique_near_clip;
use mirror::Plane;
use sample::{sample_reflection, ReflectionParams, ReflectionSample};

/// The view-projection to render the mirror pass with, for a given reflector.
///
/// Rendering the scene with `mirrored_view_projection(vp, plane)` is exactly
/// equivalent to projecting each world point's planar reflection with the real
/// `vp`: since `vp · R · x = vp · reflect(x)` for the reflection matrix `R`, the
/// mirrored camera and the per-point reflection are the same operation.  This is
/// the identity that lets a reflection buffer be sampled with ordinary screen
/// UVs.
///
/// Composition order matters: the reflection is applied to world points *first*,
/// then the camera projection, hence `vp * reflection_matrix`.
#[inline]
pub fn mirrored_view_projection(clip_from_world: &Mat4, plane: &Plane) -> Mat4 {
    *clip_from_world * plane.reflection_matrix()
}

/// The mirror-pass view-projection with the reflector installed as the oblique
/// near clip plane, so geometry behind the reflector is culled.
///
/// `plane_view` is the reflector expressed in the **mirrored camera's view
/// space** as a clip 4-vector (see [`clip::clip_plane_from_view_plane`]).  The
/// oblique clip is applied to the mirrored projection; pass the mirrored
/// projection matrix (not the full world VP) as `mirror_proj`.
#[inline]
pub fn mirrored_projection_with_clip(
    mirror_proj: &Mat4,
    plane_view: bevy_math::Vec4,
) -> Mat4 {
    oblique_near_clip(*mirror_proj, plane_view)
}

/// A full planar-reflection tap for a scene point seen in a reflector.
///
/// Bundles the reflected world position with the [`ReflectionSample`] that says
/// where (UV), how blurry (LOD), and how trustworthy (weight) its mirror image
/// is on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarTap {
    /// The scene point reflected across the reflector plane.
    pub reflected_point: Vec3,
    /// Where/how to read the reflection buffer for this point.
    pub sample: ReflectionSample,
}

/// Resolves the on-screen mirror image of a world-space scene point.
///
/// Reflects `scene_point` across `plane`, projects the reflection with the real
/// camera `clip_from_world`, and derives the UV / LOD / fade weight.  A point
/// that is *behind* the reflector (on the non-visible side, signed distance
/// `≤ 0`) cannot appear in the mirror and is returned with zero weight; the
/// reflected position is still reported for debugging.
///
/// * `plane` — the reflector plane (mirror / water surface).
/// * `clip_from_world` — the real camera view-projection.
/// * `scene_point` — a world-space point above/in front of the reflector.
/// * `roughness` — perceptual roughness of the reflector in `[0, 1]`.
/// * `params` — fade/blur tuning (see [`ReflectionParams`]).
pub fn resolve_planar_tap(
    plane: &Plane,
    clip_from_world: &Mat4,
    scene_point: Vec3,
    roughness: f32,
    params: &ReflectionParams,
) -> PlanarTap {
    let height = plane.signed_distance(scene_point);
    let reflected = plane.reflect_point(scene_point);
    // Travel distance of the reflected ray is twice the point's height above
    // the reflector (the reflection is that far "below" the surface).
    let distance = 2.0 * height.max(0.0);
    let mut sample = sample_reflection(clip_from_world, reflected, distance, roughness, params);
    if height <= 0.0 {
        // Below the mirror plane: not a real reflection, reject the tap.
        sample.weight = 0.0;
        sample.on_screen = false;
    }
    PlanarTap {
        reflected_point: reflected,
        sample,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-handed `[0, 1]`-depth perspective, built explicitly (no
    /// deprecated helpers).  Looks down `-Z`; near -> 0, far -> 1.
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let r = far / (near - far);
        Mat4::from_cols(
            bevy_math::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            bevy_math::Vec4::new(0.0, f, 0.0, 0.0),
            bevy_math::Vec4::new(0.0, 0.0, r, -1.0),
            bevy_math::Vec4::new(0.0, 0.0, near * far / (near - far), 0.0),
        )
    }

    /// A right-handed look-at view matrix, built explicitly (no deprecated
    /// helpers).  Mirrors `glam`'s `look_at_rh` convention (forward = `-Z`).
    fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
        let fwd = (center - eye).normalize();
        let right = fwd.cross(up).normalize();
        let u = right.cross(fwd);
        Mat4::from_cols(
            bevy_math::Vec4::new(right.x, u.x, -fwd.x, 0.0),
            bevy_math::Vec4::new(right.y, u.y, -fwd.y, 0.0),
            bevy_math::Vec4::new(right.z, u.z, -fwd.z, 0.0),
            bevy_math::Vec4::new(-right.dot(eye), -u.dot(eye), fwd.dot(eye), 1.0),
        )
    }

    /// A real camera view-projection looking down toward a `y = 0` floor.
    fn camera_vp() -> Mat4 {
        let proj = perspective(core::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        // Eye at (0, 2, 6) looking toward the origin, up = +Y.
        let view = look_at(Vec3::new(0.0, 2.0, 6.0), Vec3::ZERO, Vec3::Y);
        proj * view
    }

    #[test]
    fn mirrored_vp_matches_pointwise_reflection() {
        let vp = camera_vp();
        let plane = Plane::new(Vec3::Y, 0.0);
        let mvp = mirrored_view_projection(&vp, &plane);
        for x in [
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-2.0, 4.0, 2.0),
        ] {
            // Projecting x with the mirrored VP equals projecting reflect(x) with
            // the real VP.
            let a = mvp.mul_vec4(x.extend(1.0));
            let b = vp.mul_vec4(plane.reflect_point(x).extend(1.0));
            assert!((a - b).length() < 1e-3, "mirrored VP identity broke");
        }
    }

    #[test]
    fn point_above_mirror_has_a_valid_tap() {
        let vp = camera_vp();
        let plane = Plane::new(Vec3::Y, 0.0);
        let tap = resolve_planar_tap(
            &plane,
            &vp,
            Vec3::new(0.0, 3.0, 0.0),
            0.1,
            &ReflectionParams::DEFAULT,
        );
        // Reflection of a point 3 above the floor is 3 below.
        assert!((tap.reflected_point - Vec3::new(0.0, -3.0, 0.0)).length() < 1e-5);
        assert!(tap.sample.on_screen);
        assert!(tap.sample.weight > 0.0);
        assert!((0.0..=1.0).contains(&tap.sample.uv.x));
        assert!((0.0..=1.0).contains(&tap.sample.uv.y));
    }

    #[test]
    fn point_below_mirror_is_rejected() {
        let vp = camera_vp();
        let plane = Plane::new(Vec3::Y, 0.0);
        let tap = resolve_planar_tap(
            &plane,
            &vp,
            Vec3::new(0.0, -1.0, 0.0),
            0.1,
            &ReflectionParams::DEFAULT,
        );
        assert_eq!(tap.sample.weight, 0.0);
        assert!(!tap.sample.on_screen);
    }

    #[test]
    fn tap_lod_is_monotonic_in_roughness() {
        let vp = camera_vp();
        let plane = Plane::new(Vec3::Y, 0.0);
        let p = ReflectionParams::DEFAULT;
        let mut prev = -1.0;
        for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let tap = resolve_planar_tap(&plane, &vp, Vec3::new(0.0, 3.0, 0.0), r, &p);
            assert!(tap.sample.lod >= prev - 1e-6, "lod not monotonic in roughness");
            prev = tap.sample.lod;
        }
    }

    #[test]
    fn mirrored_projection_with_clip_is_finite_and_noops_on_bad_plane() {
        let proj = perspective(core::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let good = clip::clip_plane_from_view_plane(Vec3::new(0.0, 1.0, 0.0), 2.0);
        let clipped = mirrored_projection_with_clip(&proj, good);
        for c in 0..4 {
            assert!(clipped.col(c).is_finite());
        }
        // Non-finite plane -> unchanged projection.
        let bad = bevy_math::Vec4::new(f32::NAN, 0.0, 1.0, 1.0);
        let same = mirrored_projection_with_clip(&proj, bad);
        for c in 0..4 {
            assert!((same.col(c) - proj.col(c)).length() < 1e-6);
        }
    }

    #[test]
    fn degenerate_scene_point_stays_finite() {
        let vp = camera_vp();
        let plane = Plane::new(Vec3::Y, 0.0);
        let tap = resolve_planar_tap(
            &plane,
            &vp,
            Vec3::new(f32::NAN, 3.0, 0.0),
            0.5,
            &ReflectionParams::DEFAULT,
        );
        assert!(tap.reflected_point.is_finite());
        assert!(tap.sample.uv.is_finite());
        assert!(tap.sample.lod.is_finite());
    }
}
