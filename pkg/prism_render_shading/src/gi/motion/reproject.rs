//! Screen-space motion-vector generation via view-projection reprojection — CPU golden.
//!
//! A *motion vector* records, per pixel, where the surface currently under that
//! pixel sat in the previous frame's image.  TAA, temporal super-resolution and
//! object motion blur all consume it to fetch history / integrate blur along the
//! correct trajectory.  This module derives that vector purely from camera and
//! object transforms — classical projective geometry, no learned priors.
//!
//! Two paths are covered:
//!
//! * **Camera motion** — a world-static surface whose only apparent movement
//!   comes from the camera.  The same world position is projected through the
//!   current and previous `clip_from_world` (view-projection) matrices; the
//!   screen-space difference is the motion vector.
//! * **Object motion** — a rigid/animated surface that also moved in world
//!   space.  The current world position is projected with the current matrix and
//!   the *previous* world position with the previous matrix, so per-object
//!   animation and skinning are captured alongside the camera term.
//!
//! Temporal anti-aliasing jitters the projection by a sub-pixel NDC offset each
//! frame.  That offset must **not** leak into the motion vector (it would bias
//! history reprojection), so both endpoints are *dejittered* with their own
//! frame's offset before the screen-space difference is taken; equal jitter on a
//! static scene therefore cancels exactly.
//!
//! # Conventions
//! * `clip_from_world` is the full view-projection `Mat4`; a world point `p` maps
//!   to clip space as `clip = clip_from_world * vec4(p, 1)` (column-vector,
//!   `glam` / `bevy_math` convention).
//! * Clip → NDC divides by `clip.w`; when `|w|` is at or below `w_epsilon`
//!   (point on/behind the camera plane) the projection is *degenerate* and the
//!   endpoint is reported invalid rather than dividing by (near-)zero.
//! * NDC is right-handed with `x,y ∈ [-1, 1]`.  UV uses a **top-left origin**:
//!   `u = x*0.5 + 0.5`, `v = 0.5 - y*0.5`, matching the sampling convention in
//!   the sibling `temporal` reprojection reference.
//! * The emitted velocity is `current_uv - previous_uv` (UV units) — the
//!   displacement from the previous to the current frame — so history lives at
//!   `current_uv - velocity`.
//! * A jitter offset is expressed in **NDC units** and is *subtracted* from the
//!   post-divide NDC of its own frame.
//! * Every helper is deterministic and allocation-free (no RNG/IO/GPU/unsafe)
//!   and defends against degenerate inputs: non-finite coordinates are
//!   sanitized, `|w| <= w_epsilon` falls back to an invalid/zero result, and no
//!   path can emit a `NaN`.

use bevy_math::{Mat4, Vec2, Vec3, Vec4};

/// Default clip-space `w` magnitude below which a projection is treated as
/// degenerate (point on or behind the camera plane).
pub const DEFAULT_W_EPSILON: f32 = 1.0e-6;

/// Replaces any non-finite component of a world position with `0.0` so a bad
/// input degrades to the origin instead of propagating a `NaN`.
#[inline]
fn sanitize_point(p: Vec3) -> Vec3 {
    Vec3::new(
        if p.x.is_finite() { p.x } else { 0.0 },
        if p.y.is_finite() { p.y } else { 0.0 },
        if p.z.is_finite() { p.z } else { 0.0 },
    )
}

/// Replaces any non-finite component of a 2-D NDC/UV offset with `0.0`.
#[inline]
fn sanitize_offset(v: Vec2) -> Vec2 {
    Vec2::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
    )
}

/// Clamps a user-supplied `w_epsilon` to a small positive value.
///
/// A zero or negative epsilon would re-admit a divide-by-zero, so it is floored
/// to [`DEFAULT_W_EPSILON`]; non-finite requests fall back the same way.
#[inline]
fn sanitize_w_epsilon(w_epsilon: f32) -> f32 {
    if w_epsilon.is_finite() && w_epsilon > 0.0 {
        w_epsilon
    } else {
        DEFAULT_W_EPSILON
    }
}

/// Converts a right-handed NDC `xy` (`[-1, 1]`) to a top-left-origin UV
/// (`[0, 1]`), flipping the vertical axis.
#[inline]
pub fn ndc_to_uv(ndc: Vec2) -> Vec2 {
    let ndc = sanitize_offset(ndc);
    Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5)
}

/// Converts a top-left-origin UV (`[0, 1]`) back to right-handed NDC `xy`
/// (`[-1, 1]`), the inverse of [`ndc_to_uv`].
#[inline]
pub fn uv_to_ndc(uv: Vec2) -> Vec2 {
    let uv = sanitize_offset(uv);
    Vec2::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0)
}

/// Transforms a world position into homogeneous clip space.
///
/// Pure matrix–vector product `clip_from_world * vec4(p, 1)`; the perspective
/// divide is deferred to [`clip_to_ndc`] so callers can inspect `w` first.
#[inline]
pub fn project_to_clip(clip_from_world: &Mat4, p_world: Vec3) -> Vec4 {
    clip_from_world.mul_vec4(sanitize_point(p_world).extend(1.0))
}

/// Performs the perspective divide, returning NDC when `|w| > w_epsilon`.
///
/// Returns `None` for a degenerate projection (`w` non-finite or at/below the
/// epsilon), signalling the caller to reject the endpoint rather than emit a
/// division by (near-)zero.  A finite `w` that nonetheless produces a non-finite
/// NDC (overflow) is likewise rejected.
#[inline]
pub fn clip_to_ndc(clip: Vec4, w_epsilon: f32) -> Option<Vec3> {
    let eps = sanitize_w_epsilon(w_epsilon);
    let w = clip.w;
    if !w.is_finite() || w.abs() <= eps {
        return None;
    }
    let inv_w = 1.0 / w;
    let ndc = Vec3::new(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
    if ndc.is_finite() {
        Some(ndc)
    } else {
        None
    }
}

/// Projects a world position all the way to UV, or `None` if degenerate.
///
/// Composition of [`project_to_clip`], [`clip_to_ndc`] and [`ndc_to_uv`]; the
/// returned UV is **not** clamped to `[0, 1]` so callers can detect off-frame
/// reprojections themselves.
#[inline]
pub fn project_world_to_uv(clip_from_world: &Mat4, p_world: Vec3, w_epsilon: f32) -> Option<Vec2> {
    let clip = project_to_clip(clip_from_world, p_world);
    let ndc = clip_to_ndc(clip, w_epsilon)?;
    Some(ndc_to_uv(ndc.truncate()))
}

/// Projects a world position to UV with a per-frame jitter offset removed.
///
/// `jitter_ndc` is the sub-pixel projection jitter applied to this frame
/// (NDC units); it is subtracted from the post-divide NDC before the UV
/// mapping, so the returned UV describes the *un-jittered* screen position.
#[inline]
pub fn project_world_to_uv_dejittered(
    clip_from_world: &Mat4,
    p_world: Vec3,
    jitter_ndc: Vec2,
    w_epsilon: f32,
) -> Option<Vec2> {
    let clip = project_to_clip(clip_from_world, p_world);
    let ndc = clip_to_ndc(clip, w_epsilon)?;
    let jitter = sanitize_offset(jitter_ndc);
    let dejittered = Vec2::new(ndc.x - jitter.x, ndc.y - jitter.y);
    Some(ndc_to_uv(dejittered))
}

/// Per-frame sub-pixel jitter offsets, in NDC units.
///
/// `current` and `previous` are each subtracted from their own frame's NDC so a
/// static scene with matched jitter yields exactly zero motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JitterNdc {
    /// Current-frame projection jitter (NDC units).
    pub current: Vec2,
    /// Previous-frame projection jitter (NDC units).
    pub previous: Vec2,
}

impl JitterNdc {
    /// No jitter on either frame (both offsets zero).
    pub const NONE: Self = Self {
        current: Vec2::ZERO,
        previous: Vec2::ZERO,
    };
}

impl Default for JitterNdc {
    #[inline]
    fn default() -> Self {
        Self::NONE
    }
}

/// Converts a pixel-space jitter offset to the NDC offset added by the
/// projection matrix.
///
/// A shift of `jitter_px` pixels corresponds to `2 * jitter_px / resolution` in
/// NDC along each axis (the full NDC range `2` spans `resolution` pixels).  The
/// vertical axis is negated to match the top-left-origin UV convention, so a
/// positive `jitter_px.y` (downward in pixels) maps to a negative NDC `y`.
/// Non-positive or non-finite resolutions floor to one pixel to avoid a
/// divide-by-zero.
#[inline]
pub fn jitter_pixels_to_ndc(jitter_px: Vec2, resolution: Vec2) -> Vec2 {
    let jitter = sanitize_offset(jitter_px);
    let res_x = if resolution.x.is_finite() {
        resolution.x.max(1.0)
    } else {
        1.0
    };
    let res_y = if resolution.y.is_finite() {
        resolution.y.max(1.0)
    } else {
        1.0
    };
    Vec2::new(2.0 * jitter.x / res_x, -2.0 * jitter.y / res_y)
}

/// The result of a motion-vector computation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionVector {
    /// Screen-space velocity in UV units: `current_uv - previous_uv`.
    ///
    /// Always finite; forced to [`Vec2::ZERO`] when the vector is not
    /// [`valid`](MotionVector::valid).
    pub velocity: Vec2,
    /// `true` when **both** endpoints projected to a well-defined (non-degenerate)
    /// screen position; `false` means at least one endpoint was behind/near the
    /// camera plane and the velocity was forced to zero.
    pub valid: bool,
}

impl MotionVector {
    /// A zero, invalid motion vector used as the degenerate fallback.
    pub const INVALID: Self = Self {
        velocity: Vec2::ZERO,
        valid: false,
    };
}

/// Builds a [`MotionVector`] from two optional UV endpoints.
///
/// Returns a valid velocity only when both endpoints projected; otherwise the
/// degenerate [`MotionVector::INVALID`] (zero, invalid) is returned.
#[inline]
fn motion_from_uvs(current_uv: Option<Vec2>, previous_uv: Option<Vec2>) -> MotionVector {
    match (current_uv, previous_uv) {
        (Some(cur), Some(prev)) => {
            let velocity = cur - prev;
            if velocity.is_finite() {
                MotionVector {
                    velocity,
                    valid: true,
                }
            } else {
                MotionVector::INVALID
            }
        }
        _ => MotionVector::INVALID,
    }
}

/// Motion vector for a **world-static** surface (camera-only motion).
///
/// The same world position `p_world` is projected through the current and
/// previous view-projection matrices (each dejittered with its own frame's
/// offset); the UV difference is the apparent screen motion caused by the camera
/// alone.  Identical matrices and jitter yield exactly zero velocity.
#[inline]
pub fn camera_motion_vector(
    current_clip_from_world: &Mat4,
    previous_clip_from_world: &Mat4,
    p_world: Vec3,
    jitter: JitterNdc,
    w_epsilon: f32,
) -> MotionVector {
    object_motion_vector(
        current_clip_from_world,
        previous_clip_from_world,
        p_world,
        p_world,
        jitter,
        w_epsilon,
    )
}

/// Motion vector for a **dynamic** surface with a per-object previous transform.
///
/// The current world position is projected with the current matrix and the
/// previous world position with the previous matrix, so per-object animation
/// (translation, rotation, skinning baked into the world positions) is captured
/// in addition to the camera term.  Each endpoint is dejittered with its own
/// frame's offset before differencing.
#[inline]
pub fn object_motion_vector(
    current_clip_from_world: &Mat4,
    previous_clip_from_world: &Mat4,
    current_world_pos: Vec3,
    previous_world_pos: Vec3,
    jitter: JitterNdc,
    w_epsilon: f32,
) -> MotionVector {
    let current_uv = project_world_to_uv_dejittered(
        current_clip_from_world,
        current_world_pos,
        jitter.current,
        w_epsilon,
    );
    let previous_uv = project_world_to_uv_dejittered(
        previous_clip_from_world,
        previous_world_pos,
        jitter.previous,
        w_epsilon,
    );
    motion_from_uvs(current_uv, previous_uv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An orthographic-like identity projection: with `w == 1`, world `xy`
    /// passes straight through to NDC `xy`, giving fully predictable UVs.
    fn identity_vp() -> Mat4 {
        Mat4::IDENTITY
    }

    /// A pure camera translation expressed as a view matrix `translate(-cam)`.
    fn camera_at(cam: Vec3) -> Mat4 {
        Mat4::from_translation(-cam)
    }

    #[test]
    fn ndc_uv_round_trip() {
        for &p in &[
            Vec2::new(-1.0, -1.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.3, -0.7),
        ] {
            let uv = ndc_to_uv(p);
            let back = uv_to_ndc(uv);
            assert!((back - p).length() < 1e-6, "round trip {p:?} -> {back:?}");
        }
        // Top-left origin: NDC (-1,+1) is the top-left corner -> UV (0,0).
        assert!((ndc_to_uv(Vec2::new(-1.0, 1.0)) - Vec2::new(0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn static_camera_static_object_is_zero() {
        let vp = identity_vp();
        let mv = camera_motion_vector(&vp, &vp, Vec3::new(0.1, -0.2, 0.0), JitterNdc::NONE, DEFAULT_W_EPSILON);
        assert!(mv.valid);
        assert!(mv.velocity.length() < 1e-7, "velocity {:?}", mv.velocity);
    }

    #[test]
    fn pure_translation_camera_sign_and_magnitude() {
        // Camera moves +x by 0.2 world units; a world-static point at the origin
        // should slide in the -x direction in UV by 0.1 (NDC 0.2 -> half in UV).
        let prev = camera_at(Vec3::ZERO);
        let cur = camera_at(Vec3::new(0.2, 0.0, 0.0));
        let mv = camera_motion_vector(&cur, &prev, Vec3::ZERO, JitterNdc::NONE, DEFAULT_W_EPSILON);
        assert!(mv.valid);
        assert!(mv.velocity.x < 0.0, "expected leftward motion, got {:?}", mv.velocity);
        assert!((mv.velocity.x + 0.1).abs() < 1e-6, "velocity.x {:?}", mv.velocity.x);
        assert!(mv.velocity.y.abs() < 1e-7, "velocity.y {:?}", mv.velocity.y);
    }

    #[test]
    fn object_translation_sign_and_magnitude() {
        // Static (identity) camera; object moves +x by 0.2 -> UV velocity +0.1.
        let vp = identity_vp();
        let mv = object_motion_vector(
            &vp,
            &vp,
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::ZERO,
            JitterNdc::NONE,
            DEFAULT_W_EPSILON,
        );
        assert!(mv.valid);
        assert!(mv.velocity.x > 0.0, "expected rightward motion, got {:?}", mv.velocity);
        assert!((mv.velocity.x - 0.1).abs() < 1e-6, "velocity.x {:?}", mv.velocity.x);
    }

    #[test]
    fn matched_jitter_cancels_on_static_scene() {
        // Bake equal jitter into both matrices as a clip-space (NDC, since w==1)
        // translation; removing it with matched JitterNdc must yield zero motion.
        let j = Vec2::new(0.137, -0.091);
        let vp = Mat4::from_translation(Vec3::new(j.x, j.y, 0.0));
        let jitter = JitterNdc {
            current: j,
            previous: j,
        };
        let mv = camera_motion_vector(&vp, &vp, Vec3::ZERO, jitter, DEFAULT_W_EPSILON);
        assert!(mv.valid);
        assert!(mv.velocity.length() < 1e-6, "velocity {:?}", mv.velocity);
    }

    #[test]
    fn dejitter_matches_unjittered_projection() {
        let j = Vec2::new(0.05, 0.08);
        let jittered = Mat4::from_translation(Vec3::new(j.x, j.y, 0.0));
        let plain = Mat4::IDENTITY;
        let p = Vec3::new(0.2, -0.3, 0.0);
        let dejit = project_world_to_uv_dejittered(&jittered, p, j, DEFAULT_W_EPSILON).unwrap();
        let plain_uv = project_world_to_uv(&plain, p, DEFAULT_W_EPSILON).unwrap();
        assert!((dejit - plain_uv).length() < 1e-6, "{dejit:?} vs {plain_uv:?}");
    }

    #[test]
    fn degenerate_w_is_rejected_and_finite() {
        // A matrix whose w-row annihilates the point gives w == 0 -> invalid.
        let mut m = Mat4::IDENTITY;
        // Set the homogeneous (w) output to constant 0 regardless of input.
        m.w_axis = Vec4::new(0.0, 0.0, 0.0, 0.0);
        m.x_axis = Vec4::new(1.0, 0.0, 0.0, 0.0);
        assert!(clip_to_ndc(project_to_clip(&m, Vec3::ZERO), DEFAULT_W_EPSILON).is_none());
        let mv = camera_motion_vector(&m, &Mat4::IDENTITY, Vec3::ZERO, JitterNdc::NONE, DEFAULT_W_EPSILON);
        assert!(!mv.valid);
        assert_eq!(mv.velocity, Vec2::ZERO);
        assert!(mv.velocity.is_finite());
    }

    #[test]
    fn non_finite_inputs_never_nan() {
        let vp = Mat4::IDENTITY;
        let p = Vec3::new(f32::NAN, f32::INFINITY, 0.0);
        let mv = object_motion_vector(&vp, &vp, p, Vec3::ZERO, JitterNdc::NONE, DEFAULT_W_EPSILON);
        assert!(mv.velocity.is_finite());
        // Sanitized point collapses to origin, matching the static endpoint.
        assert!(mv.valid);
        assert!(mv.velocity.length() < 1e-7);
    }

    #[test]
    fn jitter_pixels_to_ndc_scales_and_flips() {
        let ndc = jitter_pixels_to_ndc(Vec2::new(0.5, 0.5), Vec2::new(100.0, 200.0));
        assert!((ndc.x - 0.01).abs() < 1e-7, "x {:?}", ndc.x);
        assert!((ndc.y + 0.005).abs() < 1e-7, "y flipped {:?}", ndc.y);
        // Degenerate resolution floors to 1 and stays finite.
        let safe = jitter_pixels_to_ndc(Vec2::new(1.0, 1.0), Vec2::new(0.0, -4.0));
        assert!(safe.is_finite());
    }

    #[test]
    fn w_epsilon_sanitized() {
        assert_eq!(sanitize_w_epsilon(0.0), DEFAULT_W_EPSILON);
        assert_eq!(sanitize_w_epsilon(-1.0), DEFAULT_W_EPSILON);
        assert_eq!(sanitize_w_epsilon(f32::NAN), DEFAULT_W_EPSILON);
        assert_eq!(sanitize_w_epsilon(0.25), 0.25);
    }
}
