//! A pinhole camera that generates primary rays for the reference tracer.
//!
//! The camera is a classical thin-pinhole model: every primary ray leaves the
//! eye point and passes through a point on a virtual image plane. To honour the
//! crate's determinism policy (no `sin`/`cos`/`tan`), the vertical field of view
//! is supplied as the *tangent* of its half-angle rather than the angle itself,
//! so ray generation needs only multiplies, adds, and a single `sqrt` for
//! direction normalization. Pixel coordinates use the usual convention: `x`
//! grows to the right and `y` grows downward, while the generated rays place
//! `+up` toward the top of the image.

use super::sampler::Sample2;
use super::Vec3;
use crate::ray_scene::traversal::Ray;

/// A pinhole camera defined by an eye point and an orthonormal view frame.
#[derive(Clone, Copy, Debug)]
pub struct PinholeCamera {
    /// Eye (pinhole) position in world space.
    origin: Vec3,
    /// Unit forward (viewing) direction.
    forward: Vec3,
    /// Unit right direction of the image plane.
    right: Vec3,
    /// Unit up direction of the image plane.
    up: Vec3,
    /// Tangent of half the vertical field of view.
    tan_half_fov_y: f32,
    /// Tangent of half the horizontal field of view (`tan_half_fov_y * aspect`).
    tan_half_fov_x: f32,
}

impl PinholeCamera {
    /// Builds a camera looking from `eye` toward `target`, with `up_hint` giving
    /// the approximate world up, a vertical half-`FOV` of `tan_half_fov_y`
    /// (expressed as a tangent), and an image `aspect` ratio (width / height).
    ///
    /// Returns [`None`] when the frame is degenerate: `eye` coincides with
    /// `target`, `up_hint` is parallel to the view direction, or `aspect` /
    /// `tan_half_fov_y` are non-positive, so no valid frame can be formed.
    #[must_use]
    pub fn look_at(
        eye: Vec3,
        target: Vec3,
        up_hint: Vec3,
        tan_half_fov_y: f32,
        aspect: f32,
    ) -> Option<Self> {
        if tan_half_fov_y <= 0.0 || aspect <= 0.0 {
            return None;
        }
        let forward = target.sub(eye).normalize_or_zero();
        if forward.length_squared() <= 0.0 {
            return None;
        }
        let right = forward.cross(up_hint).normalize_or_zero();
        if right.length_squared() <= 0.0 {
            return None;
        }
        // Recompute an exactly orthonormal up from the completed frame.
        let up = right.cross(forward);
        Some(Self {
            origin: eye,
            forward,
            right,
            up,
            tan_half_fov_y,
            tan_half_fov_x: tan_half_fov_y * aspect,
        })
    }

    /// The eye position.
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Generates the primary ray through pixel `(px, py)` of a `width`×`height`
    /// image, offset within the pixel by the stratified `sample` in `[0, 1)^2`.
    ///
    /// Pixels outside `[0, width) × [0, height)` are still mapped linearly (no
    /// panic); callers normally pass in-range coordinates. Returns a ray with an
    /// infinite `t` interval anchored at the eye.
    #[must_use]
    pub fn primary_ray(&self, px: u32, py: u32, width: u32, height: u32, sample: Sample2) -> Ray {
        // Guard against a zero-sized film so the mapping never divides by zero.
        let inv_w = if width == 0 {
            1.0
        } else {
            1.0 / (width as f32)
        };
        let inv_h = if height == 0 {
            1.0
        } else {
            1.0 / (height as f32)
        };
        let u = ((px as f32) + sample.x) * inv_w;
        let v = ((py as f32) + sample.y) * inv_h;
        // Normalized device coordinates in `[-1, 1]`, with `+y` pointing up.
        let ndc_x = 2.0 * u - 1.0;
        let ndc_y = 1.0 - 2.0 * v;
        let direction = self
            .forward
            .add(self.right.scale(ndc_x * self.tan_half_fov_x))
            .add(self.up.scale(ndc_y * self.tan_half_fov_y))
            .normalize_or_zero();
        Ray::new(
            self.origin.to_array(),
            direction.to_array(),
            0.0,
            f32::INFINITY,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_camera() -> PinholeCamera {
        PinholeCamera::look_at(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            1.0,
        )
        .expect("valid camera")
    }

    #[test]
    fn degenerate_configurations_are_rejected() {
        // Eye == target.
        assert!(
            PinholeCamera::look_at(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0, 1.0)
                .is_none()
        );
        // Up parallel to view direction.
        assert!(PinholeCamera::look_at(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            1.0,
        )
        .is_none());
        // Non-positive FOV / aspect.
        assert!(PinholeCamera::look_at(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
            1.0,
        )
        .is_none());
    }

    #[test]
    fn center_ray_points_forward() {
        let cam = basic_camera();
        // The pixel center of a 1x1 image is the optical axis.
        let ray = cam.primary_ray(0, 0, 1, 1, Sample2 { x: 0.5, y: 0.5 });
        let dir = Vec3::from_array(ray.direction());
        assert!(dir.sub(Vec3::new(0.0, 0.0, -1.0)).length_squared() < 1e-10);
        assert_eq!(ray.origin(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn primary_rays_are_unit_length() {
        let cam = basic_camera();
        for py in 0..16 {
            for px in 0..16 {
                let ray = cam.primary_ray(px, py, 16, 16, Sample2 { x: 0.25, y: 0.75 });
                let dir = Vec3::from_array(ray.direction());
                assert!((dir.length() - 1.0).abs() < 1e-5);
                assert!(dir.is_finite());
            }
        }
    }

    #[test]
    fn image_corners_spread_symmetrically() {
        let cam = basic_camera();
        // Opposite corners mirror through the optical axis.
        let tl = Vec3::from_array(
            cam.primary_ray(0, 0, 2, 2, Sample2 { x: 0.0, y: 0.0 })
                .direction(),
        );
        let br = Vec3::from_array(
            cam.primary_ray(1, 1, 2, 2, Sample2 { x: 1.0, y: 1.0 })
                .direction(),
        );
        assert!((tl.x + br.x).abs() < 1e-6);
        assert!((tl.y + br.y).abs() < 1e-6);
    }

    #[test]
    fn zero_sized_film_does_not_panic() {
        let cam = basic_camera();
        let ray = cam.primary_ray(0, 0, 0, 0, Sample2 { x: 0.5, y: 0.5 });
        assert!(Vec3::from_array(ray.direction()).is_finite());
    }
}
