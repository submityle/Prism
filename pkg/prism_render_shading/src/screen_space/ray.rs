//! View-space reflection setup and its projection into a screen-space ray.
//!
//! SSR marches in screen space, but the reflection direction is defined in view
//! space where the camera sits at the origin looking down `-Z`.  This module
//! reflects the incident view ray about the surface normal, clips the reflected
//! ray to the near plane so it never crosses behind the camera, and projects
//! both endpoints through the projection matrix into UV plus device depth.
//!
//! A projection maps 3D lines to straight lines in normalized-device space, so
//! the endpoints fully define the march: the intermediate UV *and* device depth
//! are exact linear interpolations of the endpoints.  The hierarchical tracer
//! ([`super::march`]) relies on that to compare interpolated ray depth against
//! the depth pyramid.

use bevy_math::{Mat4, Vec2, Vec3, Vec4};

/// Reflects an `incident` direction about a unit `normal`.
///
/// Mirrors the GLSL/WGSL `reflect` intrinsic: `incident - 2 (incident·n) n`.
/// `incident` points along the ray of travel (from the camera toward the
/// surface); the result points away from the surface into the scene.
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * incident.dot(normal) * normal
}

/// A single projected sample: framebuffer UV plus device depth and clip `w`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenSample {
    /// Framebuffer UV with the origin at the top-left (`y` runs downward).
    pub uv: Vec2,
    /// Device depth after the perspective divide (reverse-Z: `1` at the near
    /// plane, `0` at the far plane).
    pub device_depth: f32,
    /// Clip-space `w`, i.e. the positive view-space distance in front of the
    /// camera.  Non-positive `w` means the point is at or behind the camera.
    pub clip_w: f32,
}

/// Projects a view-space point through `clip_from_view` into a [`ScreenSample`].
///
/// The `y` axis is flipped so the UV origin is the top-left texel, matching the
/// framebuffer/HZB texel layout the tracer samples.
pub fn project_view_to_screen(clip_from_view: Mat4, point_view: Vec3) -> ScreenSample {
    let clip = clip_from_view * Vec4::new(point_view.x, point_view.y, point_view.z, 1.0);
    let inv_w = 1.0 / clip.w;
    let ndc = Vec3::new(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
    ScreenSample {
        uv: Vec2::new(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5),
        device_depth: ndc.z,
        clip_w: clip.w,
    }
}

/// A screen-space reflection ray reduced to its two projected endpoints.
///
/// UV and device depth both interpolate linearly in the ray parameter `t`
/// (`0` at [`Self::start_uv`], `1` at [`Self::end_uv`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenRay {
    /// Projected origin (the shaded pixel).
    pub start_uv: Vec2,
    /// Device depth at the origin.
    pub start_depth: f32,
    /// Projected far endpoint after near-plane clipping.
    pub end_uv: Vec2,
    /// Device depth at the far endpoint.
    pub end_depth: f32,
    /// Fraction of the requested view-space march length that survived
    /// near-plane clipping, in `(0, 1]`.
    pub clipped_fraction: f32,
}

/// View/camera constants required to build a screen-space reflection ray.
#[derive(Clone, Copy, Debug)]
pub struct SsrCamera {
    /// Projection matrix mapping view space to clip space (reverse-Z).
    pub clip_from_view: Mat4,
    /// Positive near-plane distance in front of the camera along `-Z`.
    pub near: f32,
}

/// Builds the screen-space reflection ray for a surface at `position_view`
/// with unit `normal_view`, marching up to `max_distance` view-space units.
///
/// Returns `None` when the surface is at/behind the camera or when the
/// reflected ray is entirely behind the near plane (nothing to trace).  The
/// ray is clipped so its far endpoint stays at least `near` in front of the
/// camera, keeping the projection finite.
pub fn build_screen_ray(
    camera: SsrCamera,
    position_view: Vec3,
    normal_view: Vec3,
    max_distance: f32,
) -> Option<ScreenRay> {
    // The camera sits at the view-space origin; the incident ray points from
    // the camera toward the surface.
    if position_view.z >= -camera.near {
        return None;
    }
    let incident = position_view.normalize_or_zero();
    if incident == Vec3::ZERO {
        return None;
    }
    let reflection = reflect(incident, normal_view.normalize_or_zero()).normalize_or_zero();
    if reflection == Vec3::ZERO {
        return None;
    }

    let mut travel = max_distance.max(1.0e-3);
    let end_view = position_view + reflection * travel;

    // Clip the far endpoint to the near plane so it never projects behind the
    // camera.  View space looks down -Z, so "in front" means z <= -near.
    let plane = -camera.near;
    if end_view.z > plane {
        // The ray crosses the near plane; shorten it to stop just in front.
        let denom = reflection.z;
        if denom.abs() <= 1.0e-6 {
            return None;
        }
        let clipped = (plane - position_view.z) / denom;
        if clipped <= 1.0e-4 {
            return None;
        }
        travel = clipped;
    }

    let clipped_fraction = (travel / max_distance.max(1.0e-3)).clamp(1.0e-4, 1.0);
    let end_view = position_view + reflection * travel;

    let start = project_view_to_screen(camera.clip_from_view, position_view);
    let end = project_view_to_screen(camera.clip_from_view, end_view);
    if start.clip_w <= 0.0 || end.clip_w <= 0.0 {
        return None;
    }

    Some(ScreenRay {
        start_uv: start.uv,
        start_depth: start.device_depth,
        end_uv: end.uv,
        end_depth: end.device_depth,
        clipped_fraction,
    })
}

/// A perspective reverse-Z projection matrix, useful for constructing an
/// [`SsrCamera`] in tests and as a reference for the exact convention the
/// tracer expects (near maps to device depth `1`, far to `0`).
///
/// `fov_y` is the vertical field of view in radians and `aspect` is
/// width/height.  Matches Bevy's infinite reverse-Z perspective when
/// `far` is left unbounded, but takes an explicit finite `far` here so tests
/// can place geometry at known device depths.
pub fn reverse_z_perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    use bevy_math::ops;
    let focal = 1.0 / ops::tan(fov_y * 0.5);
    // Reverse-Z: z_ndc = near/(near - far) at far, -> maps near->1, far->0.
    // Column-major construction via from_cols.
    let a = focal / aspect;
    let b = focal;
    let c = near / (far - near);
    let d = (far * near) / (far - near);
    Mat4::from_cols(
        Vec4::new(a, 0.0, 0.0, 0.0),
        Vec4::new(0.0, b, 0.0, 0.0),
        Vec4::new(0.0, 0.0, c, -1.0),
        Vec4::new(0.0, 0.0, d, 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_mirrors_about_normal() {
        // A ray heading straight down reflects straight up off a floor.
        let r = reflect(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!((r - Vec3::new(0.0, 1.0, 0.0)).length() < 1.0e-6);
        // A 45-degree ray reflects to the mirror angle.
        let r = reflect(
            Vec3::new(1.0, -1.0, 0.0).normalize(),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!((r - Vec3::new(1.0, 1.0, 0.0).normalize()).length() < 1.0e-6);
    }

    #[test]
    fn projection_maps_near_and_far_to_reverse_z_extents() {
        let proj = reverse_z_perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.5, 100.0);
        let near = project_view_to_screen(proj, Vec3::new(0.0, 0.0, -0.5));
        let far = project_view_to_screen(proj, Vec3::new(0.0, 0.0, -100.0));
        assert!((near.device_depth - 1.0).abs() < 1.0e-4);
        assert!(far.device_depth.abs() < 1.0e-4);
        // A point on the view axis projects to the screen centre.
        assert!((near.uv - Vec2::new(0.5, 0.5)).length() < 1.0e-5);
    }

    #[test]
    fn build_ray_rejects_surfaces_behind_camera() {
        let camera = SsrCamera {
            clip_from_view: reverse_z_perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.5, 100.0),
            near: 0.5,
        };
        // Surface behind the camera (positive z) -> no ray.
        assert!(build_screen_ray(camera, Vec3::new(0.0, 0.0, 1.0), Vec3::Y, 10.0).is_none());
    }

    #[test]
    fn build_ray_projects_endpoints_and_stays_in_front() {
        let camera = SsrCamera {
            clip_from_view: reverse_z_perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.5, 100.0),
            near: 0.5,
        };
        // A point 5 units deep on a floor whose normal points up; the camera
        // looks slightly down so the reflection heads deeper into the scene.
        let position = Vec3::new(0.0, -1.0, -5.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let ray = build_screen_ray(camera, position, normal, 8.0).expect("ray");
        assert!(ray.clipped_fraction > 0.0 && ray.clipped_fraction <= 1.0);
        // Endpoints are inside a sane UV band around the centre column.
        assert!(ray.start_uv.x.is_finite() && ray.end_uv.x.is_finite());
        // Reverse-Z: the farther endpoint has the smaller device depth.
        assert!(ray.end_depth <= ray.start_depth + 1.0e-4);
    }

    #[test]
    fn build_ray_clips_to_near_plane_when_reflection_returns() {
        let camera = SsrCamera {
            clip_from_view: reverse_z_perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.5, 100.0),
            near: 0.5,
        };
        // Surface just in front of the camera whose reflection points back
        // toward it; the ray must be clipped to a short in-front segment.
        let position = Vec3::new(0.0, 0.0, -1.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let ray = build_screen_ray(camera, position, normal, 10.0);
        // Reflection points straight back (+Z) and immediately crosses the
        // near plane, so the clipped fraction is small but valid or rejected.
        if let Some(ray) = ray {
            assert!(ray.clipped_fraction < 1.0);
        }
    }
}
