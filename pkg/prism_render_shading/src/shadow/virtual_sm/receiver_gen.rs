//! Receiver generation: turn a shaded G-buffer sample into a virtual-shadow-map
//! [`Receiver`].
//!
//! Before the page-request pass ([`super::request`]) can decide which pages a
//! frame needs, every visible surface has to be expressed as a *receiver* in the
//! light's clipmap plane.  On the GPU this happens in a compute pass that reads
//! the depth buffer, reconstructs each pixel's world position from the camera's
//! inverse view-projection, projects it onto the light's ground plane and writes
//! a receiver record.  This module is the CPU golden twin of that pass: the two
//! must agree bit-for-bit so the CPU allocator reference and the GPU feedback
//! buffer request the same page set.
//!
//! The work splits into two pure steps:
//!
//! * [`reconstruct_world_position`] — unprojects a screen UV + non-linear depth
//!   back to a world position, following Bevy's screen conventions (UV origin
//!   top-left with `y` growing downward, NDC `z` taken straight from the depth
//!   buffer so any depth convention — including reverse-Z — round-trips through
//!   the matrix it was produced with).
//! * [`ReceiverProjection::project`] — projects a world position onto the light's
//!   orthonormal clipmap basis to get the absolute `light_space_xy`, and measures
//!   the camera view distance that selects the clip level.
//!
//! Keeping `light_space_xy` an *absolute* world-plane projection (never
//! camera-relative) is what preserves the clipmap's absolute page addressing:
//! a static surface keeps the same [`super::ShadowPageKey`] no matter where the
//! camera is, so the residency cache never churns on camera motion.

use bevy_math::{Vec2, Vec3};

use crate::shadow::math::{transform_point, Mat4};
use crate::shadow::virtual_sm::request::Receiver;

/// Reconstructs the world-space position of a screen pixel from its UV and the
/// depth-buffer value, using the camera's inverse view-projection matrix.
///
/// * `uv` follows Bevy's framebuffer convention: `(0, 0)` is the top-left corner
///   and `y` increases downward, so it is flipped to the `y`-up NDC the matrix
///   expects.
/// * `ndc_depth` is the raw depth-buffer sample, passed through unchanged. Because
///   it is fed back through the same matrix that produced it, this round-trips
///   for any projection — perspective or orthographic, standard or reverse-Z.
///
/// `inverse_view_proj` is column-major (see [`Mat4`]), byte-identical to the
/// matrix uploaded to the GPU, so this stays a twin of the shader unprojection.
pub fn reconstruct_world_position(inverse_view_proj: &Mat4, uv: Vec2, ndc_depth: f32) -> Vec3 {
    // Bevy UV (y-down, [0,1]) -> NDC (y-up, [-1,1]).
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    // Unproject the clip-space point and apply the perspective divide.
    let clip = transform_point(inverse_view_proj, [ndc_x, ndc_y, ndc_depth]);
    let inv_w = 1.0 / clip[3];
    Vec3::new(clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w)
}

/// The per-light constants needed to turn reconstructed world positions into
/// [`Receiver`]s: an orthonormal basis spanning the light's clipmap plane, the
/// camera position that measures view distance and the soft-shadow filter width
/// every receiver inherits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReceiverProjection {
    /// First (right) axis of the light's clipmap plane; unit length, perpendicular
    /// to the light direction and to [`light_up`].
    pub light_right: Vec3,
    /// Second (up) axis of the light's clipmap plane; unit length, perpendicular
    /// to the light direction and to [`light_right`].
    pub light_up: Vec3,
    /// Camera world position; a receiver's view distance is measured from here.
    pub camera_world: Vec3,
    /// Soft-shadow filter kernel half-width in shadow texels, copied onto every
    /// receiver so [`super::generate_page_requests`] can expand its footprint.
    pub filter_radius_texels: f32,
}

impl ReceiverProjection {
    /// Builds a projection from a (not necessarily normalised) light direction,
    /// deriving a stable orthonormal clipmap basis perpendicular to it.
    ///
    /// The basis is chosen deterministically: [`light_right`] is the light
    /// direction crossed with the world up axis (falling back to the world `x`
    /// axis when the light points nearly straight up or down so the cross product
    /// stays well-conditioned), and [`light_up`] completes the right-handed frame.
    /// Two frames with the same light direction always produce the same basis, so
    /// the clipmap page grid is stable.
    pub fn from_light_direction(
        light_direction: Vec3,
        camera_world: Vec3,
        filter_radius_texels: f32,
    ) -> Self {
        let dir = light_direction.normalize_or_zero();
        let dir = if dir == Vec3::ZERO { Vec3::NEG_Y } else { dir };
        // Pick a world axis that is not parallel to the light to seed the basis.
        let up_hint = if dir.y.abs() > 0.999 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let light_right = dir.cross(up_hint).normalize();
        let light_up = light_right.cross(dir).normalize();
        Self {
            light_right,
            light_up,
            camera_world,
            filter_radius_texels: filter_radius_texels.max(0.0),
        }
    }

    /// Projects a world position onto the clipmap plane and measures its camera
    /// view distance, producing the [`Receiver`] the request pass consumes.
    pub fn project(&self, world: Vec3) -> Receiver {
        let light_space_xy = Vec2::new(world.dot(self.light_right), world.dot(self.light_up));
        let view_distance = (world - self.camera_world).length();
        Receiver {
            light_space_xy,
            view_distance,
            filter_radius_texels: self.filter_radius_texels.max(0.0),
        }
    }
}

/// Convenience end-to-end helper: reconstruct a pixel's world position from its
/// screen sample and immediately project it into a [`Receiver`].
///
/// This is the exact composition the GPU pass performs per pixel and keeps the
/// two-step golden callable as one function in the scene-side tests.
pub fn generate_receiver(
    projection: &ReceiverProjection,
    inverse_view_proj: &Mat4,
    uv: Vec2,
    ndc_depth: f32,
) -> Receiver {
    let world = reconstruct_world_position(inverse_view_proj, uv, ndc_depth);
    projection.project(world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::math::{invert, look_at_rh, mul, perspective_rh_01, transform_point};

    /// Builds a camera view-projection and its inverse for round-trip tests.
    fn camera() -> (Mat4, Mat4) {
        let view = look_at_rh([4.0, 3.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_rh_01(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let view_proj = mul(&proj, &view);
        let inv = invert(&view_proj).expect("camera view-projection is invertible");
        (view_proj, inv)
    }

    /// Forward-projects `world` to the screen UV + depth the GPU would store.
    fn project_to_screen(view_proj: &Mat4, world: Vec3) -> (Vec2, f32) {
        let clip = transform_point(view_proj, [world.x, world.y, world.z]);
        let inv_w = 1.0 / clip[3];
        let ndc_x = clip[0] * inv_w;
        let ndc_y = clip[1] * inv_w;
        let ndc_z = clip[2] * inv_w;
        // NDC (y-up) -> Bevy UV (y-down).
        let uv = Vec2::new(ndc_x * 0.5 + 0.5, 0.5 - ndc_y * 0.5);
        (uv, ndc_z)
    }

    /// Reconstruction is the exact inverse of the camera's forward projection:
    /// a world point projected to screen and back lands on itself.
    #[test]
    fn reconstruct_round_trips_the_projection() {
        let (view_proj, inv) = camera();
        for world in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.5, -2.0, 3.0),
            Vec3::new(-3.0, 4.0, -1.0),
            Vec3::new(2.0, 2.0, 2.0),
        ] {
            let (uv, depth) = project_to_screen(&view_proj, world);
            let back = reconstruct_world_position(&inv, uv, depth);
            assert!(
                (back - world).length() < 1.0e-3,
                "world {world:?} round-tripped to {back:?}"
            );
        }
    }

    /// A pixel at the top-left corner (UV origin) unprojects to a different point
    /// than the bottom-left, proving the `y`-down flip is applied (a missing flip
    /// would mirror the two vertically).
    #[test]
    fn uv_origin_is_top_left() {
        let (_, inv) = camera();
        let top = reconstruct_world_position(&inv, Vec2::new(0.0, 0.0), 0.5);
        let bottom = reconstruct_world_position(&inv, Vec2::new(0.0, 1.0), 0.5);
        assert!((top.y - bottom.y).abs() > 1.0e-3);
        // Top-of-screen pixel reconstructs above the bottom-of-screen pixel.
        assert!(top.y > bottom.y);
    }

    /// The derived clipmap basis is orthonormal and perpendicular to the light.
    #[test]
    fn basis_is_orthonormal_and_perpendicular_to_light() {
        for dir in [
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(-0.3, -0.9, 0.4),
            Vec3::new(0.0, -0.999_5, 0.03),
        ] {
            let p = ReceiverProjection::from_light_direction(dir, Vec3::ZERO, 0.0);
            assert!((p.light_right.length() - 1.0).abs() < 1.0e-5);
            assert!((p.light_up.length() - 1.0).abs() < 1.0e-5);
            assert!(p.light_right.dot(p.light_up).abs() < 1.0e-5);
            let unit_dir = dir.normalize();
            assert!(p.light_right.dot(unit_dir).abs() < 1.0e-5);
            assert!(p.light_up.dot(unit_dir).abs() < 1.0e-5);
        }
    }

    /// A degenerate (zero) light direction still yields a valid orthonormal basis
    /// rather than NaNs, defaulting to straight down.
    #[test]
    fn zero_light_direction_is_handled() {
        let p = ReceiverProjection::from_light_direction(Vec3::ZERO, Vec3::ZERO, 0.0);
        assert!(p.light_right.is_finite());
        assert!(p.light_up.is_finite());
        assert!((p.light_right.length() - 1.0).abs() < 1.0e-5);
        assert!((p.light_up.length() - 1.0).abs() < 1.0e-5);
    }

    /// `light_space_xy` depends only on world position, never on the camera, so a
    /// static surface projects identically as the camera moves — the property the
    /// clipmap relies on for cache stability.
    #[test]
    fn projection_is_camera_independent() {
        let world = Vec3::new(5.0, 1.0, -2.0);
        let a = ReceiverProjection::from_light_direction(
            Vec3::new(0.2, -1.0, 0.1),
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
        );
        let b = ReceiverProjection {
            camera_world: Vec3::new(100.0, 50.0, -80.0),
            ..a
        };
        assert_eq!(a.project(world).light_space_xy, b.project(world).light_space_xy);
    }

    /// View distance is the Euclidean camera-to-world distance and grows as the
    /// surface recedes, which is what drives coarser clip-level selection.
    #[test]
    fn view_distance_measures_from_the_camera() {
        let p = ReceiverProjection::from_light_direction(
            Vec3::NEG_Y,
            Vec3::new(0.0, 0.0, 0.0),
            0.5,
        );
        let near = p.project(Vec3::new(0.0, 0.0, 3.0));
        let far = p.project(Vec3::new(0.0, 0.0, 30.0));
        assert!((near.view_distance - 3.0).abs() < 1.0e-4);
        assert!((far.view_distance - 30.0).abs() < 1.0e-4);
        assert!(far.view_distance > near.view_distance);
    }

    /// The filter radius is carried onto every receiver and clamped non-negative.
    #[test]
    fn filter_radius_is_propagated_and_clamped() {
        let p = ReceiverProjection::from_light_direction(Vec3::NEG_Y, Vec3::ZERO, 2.5);
        assert_eq!(p.project(Vec3::ZERO).filter_radius_texels, 2.5);
        let clamped = ReceiverProjection::from_light_direction(Vec3::NEG_Y, Vec3::ZERO, -1.0);
        assert_eq!(clamped.filter_radius_texels, 0.0);
    }

    /// The end-to-end helper equals the two steps composed by hand.
    #[test]
    fn generate_receiver_composes_reconstruct_and_project() {
        let (view_proj, inv) = camera();
        let projection = ReceiverProjection::from_light_direction(
            Vec3::new(0.1, -1.0, 0.2),
            Vec3::new(4.0, 3.0, 10.0),
            1.0,
        );
        let world = Vec3::new(1.0, -1.0, 2.0);
        let (uv, depth) = project_to_screen(&view_proj, world);
        let one_shot = generate_receiver(&projection, &inv, uv, depth);
        let reconstructed = reconstruct_world_position(&inv, uv, depth);
        let by_hand = projection.project(reconstructed);
        assert_eq!(one_shot, by_hand);
    }
}
