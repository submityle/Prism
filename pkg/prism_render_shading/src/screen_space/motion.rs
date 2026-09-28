//! Backend-neutral CPU reference for the per-pixel **motion-vector G-buffer**.
//!
//! A motion vector stores, for the surface visible at a pixel this frame, the
//! screen-space displacement from where that *same surface point* appeared last
//! frame: `motion_uv = current_uv - previous_uv`.  Temporal passes (TAA, the
//! SSR history reprojection) add it back to the current UV to look the surface
//! up in the previous frame's target, so history follows both camera motion and
//! per-object motion without the ghosting a depth-only camera reprojection
//! suffers on animated or skinned geometry.
//!
//! Both endpoints are projected from the *geometry* world position - never from
//! the pixel centre - so a static object under a static camera yields an exactly
//! zero vector (no sub-pixel residual that would smear a still image).  The
//! current endpoint uses this frame's `world_from_local` and `clip_from_world`;
//! the previous endpoint uses the instance's previous-frame `world_from_local`
//! and the previous-frame `clip_from_world`.  For a rigid (unmoved) instance the
//! two world positions coincide and the vector reduces to pure camera motion.
//!
//! Mirrored bit-for-bit by the `shading_resolve.wesl` motion export.

use bevy_math::{Mat4, Vec2, Vec3, Vec4};

/// A world point projected to framebuffer UV space (top-left origin, reverse-Z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSample {
    /// Framebuffer UV: `x` right, `y` down, both nominally in `[0, 1]` on
    /// screen (values outside that range are kept so callers can decide how to
    /// treat off-screen history).
    pub uv: Vec2,
    /// Device depth after the perspective divide (reverse-Z: `1` near, `0` far).
    pub device_depth: f32,
    /// Clip-space `w`, the positive view-space distance in front of the camera.
    /// Non-positive `w` means the point is at or behind the camera.
    pub clip_w: f32,
}

/// Projects a **world** point through a full `clip_from_world` matrix into
/// framebuffer UV space, flipping `y` so the origin is the top-left texel to
/// match the framebuffer layout (identical convention to
/// [`super::project_view_to_screen`], only starting from world instead of view
/// space).  Returns `None` when the point is at or behind the camera
/// (`clip.w <= 0`), where the perspective divide is undefined.
pub fn project_world_to_screen(clip_from_world: Mat4, world: Vec3) -> Option<MotionSample> {
    let clip: Vec4 = clip_from_world * Vec4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let inv_w = 1.0 / clip.w;
    let ndc = Vec3::new(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
    Some(MotionSample {
        uv: Vec2::new(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5),
        device_depth: ndc.z,
        clip_w: clip.w,
    })
}

/// Computes the screen-space motion vector `current_uv - previous_uv` for a
/// surface point.
///
/// * `clip_from_world_cur` / `world_cur` - this frame's view-projection and the
///   surface's current world position (current `world_from_local * local`).
/// * `clip_from_world_prev` / `world_prev` - last frame's view-projection and
///   the surface's world position *last* frame (previous `world_from_local *
///   local`).  Pass `world_prev == world_cur` for a rigid instance so the
///   result captures pure camera motion.
///
/// Returns `None` when either endpoint is at or behind its camera, where the
/// projection is undefined; callers should fall back to a zero vector (treat
/// history as unavailable) in that case.
pub fn motion_vector(
    clip_from_world_cur: Mat4,
    clip_from_world_prev: Mat4,
    world_cur: Vec3,
    world_prev: Vec3,
) -> Option<Vec2> {
    let cur = project_world_to_screen(clip_from_world_cur, world_cur)?;
    let prev = project_world_to_screen(clip_from_world_prev, world_prev)?;
    Some(cur.uv - prev.uv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reverse-Z perspective `clip_from_world` with the camera at
    /// `[0, 0, eye_z]` looking down `-Z` (column-major like `glam`/WGSL).
    fn clip_from_world(eye_z: f32) -> Mat4 {
        let near = 0.1_f32;
        let far = 100.0_f32;
        let f = 1.0 / (0.5_f32).tan();
        let proj = Mat4::from_cols(
            Vec4::new(f, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, near / (far - near), -1.0),
            Vec4::new(0.0, 0.0, (far * near) / (far - near), 0.0),
        );
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -eye_z));
        proj * view
    }

    #[test]
    fn static_surface_and_camera_yield_zero_motion() {
        let cur = clip_from_world(0.0);
        let world = Vec3::new(0.3, -0.2, -5.0);
        let motion = motion_vector(cur, cur, world, world).unwrap();
        assert!(motion.length() < 1.0e-6, "static config must be zero: {motion:?}");
    }

    #[test]
    fn projected_uv_origin_is_top_left() {
        let cur = clip_from_world(0.0);
        let sample = project_world_to_screen(cur, Vec3::new(0.5, 0.5, -5.0)).unwrap();
        assert!(sample.uv.x > 0.5, "right of axis => u>0.5: {:?}", sample.uv);
        assert!(sample.uv.y < 0.5, "above axis => v<0.5: {:?}", sample.uv);
    }

    #[test]
    fn camera_pan_moves_history_consistently_with_reprojection() {
        let world = Vec3::new(0.4, 0.0, -5.0);
        let cur = clip_from_world(0.0);
        let prev = clip_from_world(-0.5);
        let motion = motion_vector(cur, prev, world, world).unwrap();
        assert!(motion.length() > 1.0e-4, "camera motion must be non-zero: {motion:?}");
        let cur_uv = project_world_to_screen(cur, world).unwrap().uv;
        let prev_uv = project_world_to_screen(prev, world).unwrap().uv;
        assert!((cur_uv - motion - prev_uv).length() < 1.0e-6);
    }

    #[test]
    fn object_motion_registers_even_with_a_static_camera() {
        let cam = clip_from_world(0.0);
        let world_prev = Vec3::new(0.0, 0.0, -5.0);
        let world_cur = Vec3::new(0.6, 0.0, -5.0);
        let motion = motion_vector(cam, cam, world_cur, world_prev).unwrap();
        assert!(motion.x.abs() > 1.0e-4, "moved object must register: {motion:?}");
        assert!(motion.y.abs() < 1.0e-6, "no vertical move => no vertical motion");
    }

    #[test]
    fn point_behind_camera_has_no_motion_vector() {
        let cam = clip_from_world(0.0);
        let behind = Vec3::new(0.0, 0.0, 5.0);
        assert!(project_world_to_screen(cam, behind).is_none());
        assert!(motion_vector(cam, cam, behind, behind).is_none());
    }
}
