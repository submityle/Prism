//! Camera and object reprojection: mapping a current-frame pixel back to where
//! its surface was in the previous frame.
//!
//! Reprojection is the heart of the motion subsystem. For *static* geometry the
//! only thing that moves between frames is the camera, so a pixel's history
//! location is found purely from the previous/current view-projection
//! transforms: reconstruct the world position from the current `NDC` position
//! and device depth, project it with the previous frame's transform, and take
//! the screen-space difference. For *dynamic* geometry (rigid, skinned, morph)
//! the surface point itself moved, so the caller supplies the point's previous
//! and current world positions and this module projects both.
//!
//! Coordinate conventions:
//! - `NDC` is the clip-space-after-perspective-divide cube with `x, y` in
//!   `[-1, 1]`, `y` pointing *up*.
//! - Device depth is the post-divide `z` the depth buffer stores. A reversed-Z
//!   `[0, 1]` or a `[0, 1]` forward range both work because the value is only
//!   fed back through the same transforms; the code never assumes a specific
//!   range.
//! - Pixel space has its origin at the top-left, `x` to the right and `y`
//!   *down*, matching the velocity target every temporal consumer samples.
//!
//! The `GPU` velocity-write kernel that runs this per pixel is pending the
//! `GPU` backend; this module is the `CPU`-verifiable reference.

use super::{clamp01, Mat4, MotionSource, ScreenDims, Vec2, Vec4, EPS};

/// Converts an `NDC` coordinate (`x, y` in `[-1, 1]`, `y` up) to a pixel
/// coordinate (origin top-left, `y` down). Values outside `[-1, 1]` map to
/// pixels outside the frame, which the confidence logic then penalizes.
#[must_use]
pub fn ndc_to_pixel(ndc: Vec2, dims: ScreenDims) -> Vec2 {
    let px = (ndc.x * 0.5 + 0.5) * dims.width_f32();
    // Flip Y: NDC +1 (top) maps to pixel row 0.
    let py = (0.5 - ndc.y * 0.5) * dims.height_f32();
    Vec2::new(px, py)
}

/// Converts a pixel coordinate (origin top-left, `y` down) to an `NDC`
/// coordinate (`x, y` in `[-1, 1]`, `y` up). The inverse of [`ndc_to_pixel`].
#[must_use]
pub fn pixel_to_ndc(pixel: Vec2, dims: ScreenDims) -> Vec2 {
    let nx = (pixel.x / dims.width_f32()) * 2.0 - 1.0;
    let ny = 1.0 - (pixel.y / dims.height_f32()) * 2.0;
    Vec2::new(nx, ny)
}

/// The precomputed transforms for a reprojection pass over one view.
///
/// The current view-projection inverse is computed once and cached so the
/// per-pixel path only does forward matrix multiplies. When the current
/// transform is singular (a degenerate projection), the inverse is `None` and
/// camera reprojection reports zero confidence for every pixel rather than
/// producing garbage.
#[derive(Clone, Copy, Debug)]
pub struct ReprojectionContext {
    /// Previous frame's world-to-clip transform.
    pub prev_view_proj: Mat4,
    /// Current frame's world-to-clip transform.
    pub curr_view_proj: Mat4,
    /// Cached inverse of [`ReprojectionContext::curr_view_proj`].
    inv_curr_view_proj: Option<Mat4>,
    /// Render-target dimensions.
    pub dims: ScreenDims,
}

/// The result of reprojecting a single pixel or surface point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reprojected {
    /// Where the surface was last frame, in pixel space.
    pub prev_pixel: Vec2,
    /// Screen-space velocity in pixels (`prev_pixel - curr_pixel`); adding it to
    /// the current pixel yields the history fetch location.
    pub velocity_pixels: Vec2,
    /// Reprojection confidence in `[0, 1]`; `0` when the history location is
    /// behind the previous camera or well outside the frame.
    pub confidence: f32,
    /// Whether the previous location was inside the view frustum (in front of
    /// the camera and within the frame bounds).
    pub prev_in_view: bool,
}

impl ReprojectionContext {
    /// Builds a context, caching the inverse of the current transform.
    #[must_use]
    pub fn new(prev_view_proj: Mat4, curr_view_proj: Mat4, dims: ScreenDims) -> Self {
        Self {
            prev_view_proj,
            curr_view_proj,
            inv_curr_view_proj: curr_view_proj.inverse(),
            dims,
        }
    }

    /// Whether the current transform was invertible; camera reprojection only
    /// produces confident results when this is `true`.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        self.inv_curr_view_proj.is_some()
    }

    /// Reconstructs the world position of a pixel from its `NDC` position and
    /// device depth, or `None` when the current transform is singular or the
    /// reconstructed point is degenerate (zero `w`).
    #[must_use]
    pub fn world_from_pixel(&self, pixel: Vec2, device_depth: f32) -> Option<Vec4> {
        let inv = self.inv_curr_view_proj?;
        let ndc = pixel_to_ndc(pixel, self.dims);
        let clip = Vec4::new(ndc.x, ndc.y, device_depth, 1.0);
        let world = inv.mul_vec4(clip).perspective_divide();
        // `perspective_divide` returns ZERO (w == 0) for a degenerate divide.
        if world.w > EPS {
            Some(world)
        } else {
            None
        }
    }

    /// Camera reprojection for *static* geometry: given a current pixel and its
    /// device depth, find where that surface projected in the previous frame.
    ///
    /// This is the workhorse for [`MotionSource::Camera`]. Dynamic sources must
    /// use [`ReprojectionContext::reproject_world_point`] with their own
    /// previous/current world positions instead.
    #[must_use]
    pub fn reproject_static_pixel(&self, pixel: Vec2, device_depth: f32) -> Reprojected {
        match self.world_from_pixel(pixel, device_depth) {
            Some(world) => self.reproject_world_point_from(pixel, world),
            None => Reprojected {
                prev_pixel: pixel,
                velocity_pixels: Vec2::ZERO,
                confidence: 0.0,
                prev_in_view: false,
            },
        }
    }

    /// Object reprojection: given the same surface point's previous and current
    /// world positions, compute the screen-space motion. Handles rigid, skinned,
    /// and morph sources where per-vertex motion is known.
    #[must_use]
    pub fn reproject_world_point(&self, world_prev: Vec4, world_curr: Vec4) -> Reprojected {
        let curr_clip = self.curr_view_proj.mul_vec4(world_curr);
        if !curr_clip.is_in_front() {
            // The point is behind the current camera; it should have been
            // culled. Report zero confidence and no motion.
            return Reprojected {
                prev_pixel: Vec2::ZERO,
                velocity_pixels: Vec2::ZERO,
                confidence: 0.0,
                prev_in_view: false,
            };
        }
        let curr_ndc = curr_clip.perspective_divide();
        let curr_pixel = ndc_to_pixel(Vec2::new(curr_ndc.x, curr_ndc.y), self.dims);
        self.reproject_world_point_from(curr_pixel, world_prev)
    }

    /// Shared tail: project a previous-frame world position and difference it
    /// against a known current pixel.
    fn reproject_world_point_from(&self, curr_pixel: Vec2, world_prev: Vec4) -> Reprojected {
        let prev_clip = self.prev_view_proj.mul_vec4(world_prev);
        if !prev_clip.is_in_front() {
            return Reprojected {
                prev_pixel: curr_pixel,
                velocity_pixels: Vec2::ZERO,
                confidence: 0.0,
                prev_in_view: false,
            };
        }
        let prev_ndc = prev_clip.perspective_divide();
        let prev_pixel = ndc_to_pixel(Vec2::new(prev_ndc.x, prev_ndc.y), self.dims);
        let velocity = prev_pixel.sub(curr_pixel);
        let confidence = self.frame_bounds_confidence(prev_pixel);
        Reprojected {
            prev_pixel,
            velocity_pixels: velocity,
            confidence,
            prev_in_view: confidence > 0.0,
        }
    }

    /// Confidence from how far a previous-frame pixel lands outside the frame.
    ///
    /// Fully inside → `1`. As the pixel leaves the frame the confidence ramps
    /// linearly to `0` over a one-pixel-wide guard band on each edge, so a
    /// history fetch that samples just past the border still contributes a
    /// little rather than snapping off. Coordinates far outside clamp to `0`.
    fn frame_bounds_confidence(&self, prev_pixel: Vec2) -> f32 {
        let w = self.dims.width_f32();
        let h = self.dims.height_f32();
        let inside_x = edge_confidence(prev_pixel.x, w);
        let inside_y = edge_confidence(prev_pixel.y, h);
        clamp01(inside_x.min(inside_y))
    }
}

/// One-axis edge confidence: `1` when `coord` is within `[0, extent]`, ramping
/// to `0` across a one-pixel guard band just outside each edge.
fn edge_confidence(coord: f32, extent: f32) -> f32 {
    const GUARD: f32 = 1.0;
    if coord < 0.0 {
        clamp01(1.0 + coord / GUARD)
    } else if coord > extent {
        clamp01(1.0 - (coord - extent) / GUARD)
    } else {
        1.0
    }
}

/// Whether a motion source should use [`ReprojectionContext::reproject_static_pixel`]
/// (camera-only) or must supply its own world positions.
#[must_use]
pub fn uses_camera_reprojection(source: MotionSource) -> bool {
    source.is_camera_reprojectable()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3
    }

    fn approx_vec(a: Vec2, b: Vec2) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y)
    }

    /// A simple symmetric perspective projection (column-major), looking down
    /// -Z, right-handed, mapping to a `[-1, 1]` depth range. Only used to build
    /// realistic transforms for the tests.
    fn perspective(fov_scale: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = fov_scale;
        let nf = 1.0 / (near - far);
        Mat4::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, (far + near) * nf, -1.0),
            Vec4::new(0.0, 0.0, 2.0 * far * near * nf, 0.0),
        )
    }

    /// A translation transform (column-major); the translation lives in the
    /// last column.
    fn translate(x: f32, y: f32, z: f32) -> Mat4 {
        Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(x, y, z, 1.0),
        )
    }

    #[test]
    fn ndc_pixel_round_trip() {
        let dims = ScreenDims::new(1920, 1080);
        for &(nx, ny) in &[(0.0, 0.0), (-1.0, 1.0), (1.0, -1.0), (0.25, -0.5)] {
            let ndc = Vec2::new(nx, ny);
            let back = pixel_to_ndc(ndc_to_pixel(ndc, dims), dims);
            assert!(approx_vec(back, ndc), "{ndc:?} -> {back:?}");
        }
    }

    #[test]
    fn ndc_center_is_screen_center() {
        let dims = ScreenDims::new(800, 600);
        let center = ndc_to_pixel(Vec2::ZERO, dims);
        assert!(approx_vec(center, Vec2::new(400.0, 300.0)));
        // NDC top-left (-1, 1) maps to pixel (0, 0).
        assert!(approx_vec(
            ndc_to_pixel(Vec2::new(-1.0, 1.0), dims),
            Vec2::ZERO
        ));
    }

    #[test]
    fn static_geometry_with_no_camera_motion_has_zero_velocity() {
        let proj = perspective(1.5, 16.0 / 9.0, 0.1, 100.0);
        let view = translate(0.0, 0.0, -5.0); // camera 5 units back
        let vp = proj.mul_mat4(&view);
        let dims = ScreenDims::new(1920, 1080);
        let ctx = ReprojectionContext::new(vp, vp, dims);
        assert!(ctx.is_invertible());

        // Pick a pixel and reconstruct/reproject it; with identical transforms
        // the velocity must be (numerically) zero and confidence full.
        let pixel = Vec2::new(960.0, 540.0);
        // Reconstruct a plausible device depth by projecting a known point.
        let world = Vec4::point(0.0, 0.0, 0.0);
        let clip = vp.mul_vec4(world);
        let ndc = clip.perspective_divide();
        let depth = ndc.z;
        let center_pixel = ndc_to_pixel(Vec2::new(ndc.x, ndc.y), dims);
        let r = ctx.reproject_static_pixel(center_pixel, depth);
        assert!(approx_vec(r.velocity_pixels, Vec2::ZERO), "{r:?}");
        assert!(approx(r.confidence, 1.0));
        assert!(r.prev_in_view);
        let _ = pixel;
    }

    #[test]
    fn camera_pan_moves_static_geometry_opposite_direction() {
        let proj = perspective(1.5, 1.0, 0.1, 100.0);
        let dims = ScreenDims::new(1000, 1000);
        // A point straight ahead of the origin camera.
        let world = Vec4::point(0.0, 0.0, -10.0);

        let view_prev = translate(0.0, 0.0, 0.0);
        // Current frame: camera slides +X, so the world point appears to move -X
        // on screen (its pixel moves left, velocity points right toward history).
        let view_curr = translate(-0.5, 0.0, 0.0); // moving camera +X == translating world -X
        let vp_prev = proj.mul_mat4(&view_prev);
        let vp_curr = proj.mul_mat4(&view_curr);
        let ctx = ReprojectionContext::new(vp_prev, vp_curr, dims);

        // Where is the point now?
        let curr_ndc = vp_curr.mul_vec4(world).perspective_divide();
        let curr_pixel = ndc_to_pixel(Vec2::new(curr_ndc.x, curr_ndc.y), dims);
        let depth = curr_ndc.z;
        let r = ctx.reproject_static_pixel(curr_pixel, depth);

        // History is to the right of the current pixel (camera moved +X), so the
        // velocity toward history has positive X.
        assert!(r.velocity_pixels.x > 1.0, "{r:?}");
        assert!(approx(r.velocity_pixels.y, 0.0), "{r:?}");
        assert!(r.confidence > 0.0);
    }

    #[test]
    fn object_motion_matches_manual_projection() {
        let proj = perspective(1.5, 1.0, 0.1, 100.0);
        let dims = ScreenDims::new(1280, 720);
        let vp = proj.mul_mat4(&translate(0.0, 0.0, 0.0));
        let ctx = ReprojectionContext::new(vp, vp, dims);

        let world_prev = Vec4::point(-1.0, 0.0, -10.0);
        let world_curr = Vec4::point(1.0, 0.0, -10.0);
        let r = ctx.reproject_world_point(world_prev, world_curr);

        let prev_pixel = ndc_to_pixel(vp.mul_vec4(world_prev).perspective_divide().xy(), dims);
        assert!(
            approx_vec(r.prev_pixel, prev_pixel),
            "{r:?} vs {prev_pixel:?}"
        );
        // Moving right in world → previous pixel is to the left → velocity.x < 0.
        assert!(r.velocity_pixels.x < 0.0, "{r:?}");
    }

    #[test]
    fn behind_camera_history_has_zero_confidence() {
        let proj = perspective(1.5, 1.0, 0.1, 100.0);
        let dims = ScreenDims::new(640, 480);
        let vp = proj.mul_mat4(&translate(0.0, 0.0, 0.0));
        let ctx = ReprojectionContext::new(vp, vp, dims);
        // A point behind the camera (positive Z in a look-down-−Z setup).
        let world_prev = Vec4::point(0.0, 0.0, 5.0);
        let world_curr = Vec4::point(0.0, 0.0, -10.0);
        let r = ctx.reproject_world_point(world_prev, world_curr);
        assert_eq!(r.confidence, 0.0);
        assert!(!r.prev_in_view);
    }

    #[test]
    fn offscreen_history_confidence_falls_to_zero() {
        let dims = ScreenDims::new(100, 100);
        assert_eq!(edge_confidence(-5.0, 100.0), 0.0);
        assert_eq!(edge_confidence(50.0, 100.0), 1.0);
        assert_eq!(edge_confidence(105.0, 100.0), 0.0);
        assert!(edge_confidence(-0.5, 100.0) > 0.0);
        let _ = dims;
    }

    #[test]
    fn singular_current_transform_yields_no_confidence() {
        let dims = ScreenDims::new(256, 256);
        let singular = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        let ctx = ReprojectionContext::new(Mat4::IDENTITY, singular, dims);
        assert!(!ctx.is_invertible());
        let r = ctx.reproject_static_pixel(Vec2::new(128.0, 128.0), 0.5);
        assert_eq!(r.confidence, 0.0);
        assert!(ctx.world_from_pixel(Vec2::new(128.0, 128.0), 0.5).is_none());
    }

    #[test]
    fn camera_reprojection_source_gate() {
        assert!(uses_camera_reprojection(MotionSource::Camera));
        assert!(!uses_camera_reprojection(MotionSource::Skinned));
    }
}
