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

use super::sampler::{uniform_disk, Rng, Sample2};
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
    /// Lens aperture radius in world units; `0` models an ideal pinhole and
    /// yields an everywhere-sharp image.
    aperture_radius: f32,
    /// Distance along `forward` to the plane that is rendered in perfect focus.
    focus_distance: f32,
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
            aperture_radius: 0.0,
            focus_distance: 1.0,
        })
    }

    /// The eye position.
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Enables a physically based thin-lens model on this camera, giving it a
    /// finite aperture so that only surfaces near the focus plane stay sharp
    /// while nearer and farther geometry blurs into a circle of confusion.
    ///
    /// `aperture_radius` is the lens radius in world units (`0` keeps the ideal
    /// pinhole behaviour), and `focus_distance` is the distance along `forward`
    /// to the plane that is rendered in perfect focus. Returns [`None`] when the
    /// parameters are not usable: a non-finite or negative aperture, or a
    /// non-finite or non-positive focus distance.
    #[must_use]
    pub fn with_thin_lens(mut self, aperture_radius: f32, focus_distance: f32) -> Option<Self> {
        if !aperture_radius.is_finite() || aperture_radius < 0.0 {
            return None;
        }
        if !focus_distance.is_finite() || focus_distance <= 0.0 {
            return None;
        }
        self.aperture_radius = aperture_radius;
        self.focus_distance = focus_distance;
        Some(self)
    }

    /// The lens aperture radius in world units; `0` for an ideal pinhole.
    #[must_use]
    pub fn aperture_radius(&self) -> f32 {
        self.aperture_radius
    }

    /// The focus distance along `forward`, i.e. the depth of the sharp plane.
    #[must_use]
    pub fn focus_distance(&self) -> f32 {
        self.focus_distance
    }

    /// The pinhole image-plane direction for pixel `(px, py)`, expressed in the
    /// view frame so its `forward` component is exactly `1`. This is the shared
    /// core of both the pinhole and thin-lens ray generators.
    fn image_plane_direction(
        &self,
        px: u32,
        py: u32,
        width: u32,
        height: u32,
        sample: Sample2,
    ) -> Vec3 {
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
        self.forward
            .add(self.right.scale(ndc_x * self.tan_half_fov_x))
            .add(self.up.scale(ndc_y * self.tan_half_fov_y))
    }

    /// Generates the primary ray through pixel `(px, py)` of a `width`×`height`
    /// image, offset within the pixel by the stratified `sample` in `[0, 1)^2`.
    ///
    /// Pixels outside `[0, width) × [0, height)` are still mapped linearly (no
    /// panic); callers normally pass in-range coordinates. Returns a ray with an
    /// infinite `t` interval anchored at the eye.
    #[must_use]
    pub fn primary_ray(&self, px: u32, py: u32, width: u32, height: u32, sample: Sample2) -> Ray {
        let direction = self
            .image_plane_direction(px, py, width, height, sample)
            .normalize_or_zero();
        Ray::new(
            self.origin.to_array(),
            direction.to_array(),
            0.0,
            f32::INFINITY,
        )
    }

    /// Generates a primary ray through pixel `(px, py)` using the thin-lens
    /// model, so the camera's finite aperture produces depth-of-field blur.
    ///
    /// `sample` jitters the sub-pixel position in `[0, 1)^2` exactly as
    /// [`primary_ray`](Self::primary_ray), while `rng` is used to draw a uniform
    /// point on the lens disk. The pinhole ray is intersected with the focus
    /// plane at [`focus_distance`](Self::focus_distance) to find the focal point;
    /// the ray is then re-anchored at the sampled lens point and re-aimed at that
    /// focal point. Every ray through a given pixel therefore converges on the
    /// same focal point, keeping surfaces on the focus plane perfectly sharp
    /// regardless of aperture, while off-plane geometry blurs. With a zero
    /// aperture this reduces exactly to the pinhole ray.
    #[must_use]
    pub fn primary_ray_lens(
        &self,
        px: u32,
        py: u32,
        width: u32,
        height: u32,
        sample: Sample2,
        rng: &mut Rng,
    ) -> Ray {
        // The image-plane direction has a unit `forward` component, so scaling
        // it by `focus_distance` lands exactly on the focus plane (the plane
        // perpendicular to `forward` at that depth).
        let pinhole_dir = self.image_plane_direction(px, py, width, height, sample);
        let focal_point = self.origin.add(pinhole_dir.scale(self.focus_distance));
        // Sample a uniform point on the aperture disk (trig-free rejection).
        let (lens_x, lens_y) = uniform_disk(rng);
        let lens_offset = self
            .right
            .scale(lens_x * self.aperture_radius)
            .add(self.up.scale(lens_y * self.aperture_radius));
        let lens_origin = self.origin.add(lens_offset);
        let direction = focal_point.sub(lens_origin).normalize_or_zero();
        Ray::new(
            lens_origin.to_array(),
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

    #[test]
    fn with_thin_lens_rejects_invalid_parameters() {
        let cam = basic_camera();
        // Negative aperture, non-positive focus distance, and non-finite inputs.
        assert!(cam.with_thin_lens(-0.1, 5.0).is_none());
        assert!(cam.with_thin_lens(0.5, 0.0).is_none());
        assert!(cam.with_thin_lens(0.5, -5.0).is_none());
        assert!(cam.with_thin_lens(f32::NAN, 5.0).is_none());
        assert!(cam.with_thin_lens(0.5, f32::INFINITY).is_none());
        // A pinhole-equivalent zero aperture with a positive focus is accepted.
        let pinhole = cam.with_thin_lens(0.0, 5.0).expect("valid lens");
        assert_eq!(pinhole.aperture_radius(), 0.0);
        assert_eq!(pinhole.focus_distance(), 5.0);
    }

    #[test]
    fn all_lens_rays_converge_on_the_focus_point() {
        // Looking down -z, the focus point of the central pixel is on the axis.
        let focus_distance = 4.0;
        let cam = basic_camera()
            .with_thin_lens(0.75, focus_distance)
            .expect("valid lens");
        let focal_point = Vec3::new(0.0, 0.0, -focus_distance);
        let mut rng = Rng::seed(7);
        for _ in 0..256 {
            let ray = cam.primary_ray_lens(0, 0, 1, 1, Sample2 { x: 0.5, y: 0.5 }, &mut rng);
            let origin = Vec3::from_array(ray.origin());
            let dir = Vec3::from_array(ray.direction());
            // March to the focus plane (perpendicular to -z at the focus depth).
            let t = (focal_point.z - origin.z) / dir.z;
            let hit = origin.add(dir.scale(t));
            assert!(hit.sub(focal_point).length_squared() < 1e-6);
            // The lens origin stays within the aperture disk around the eye.
            assert!(origin.sub(cam.origin()).length() <= 0.75 + 1e-4);
            assert!((dir.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn wider_aperture_spreads_the_lens_origin() {
        let narrow = basic_camera().with_thin_lens(0.1, 4.0).expect("valid lens");
        let wide = basic_camera().with_thin_lens(1.0, 4.0).expect("valid lens");
        let mut rng_n = Rng::seed(11);
        let mut rng_w = Rng::seed(11);
        let mut spread_n = 0.0_f32;
        let mut spread_w = 0.0_f32;
        for _ in 0..512 {
            let rn = narrow.primary_ray_lens(0, 0, 1, 1, Sample2 { x: 0.5, y: 0.5 }, &mut rng_n);
            let rw = wide.primary_ray_lens(0, 0, 1, 1, Sample2 { x: 0.5, y: 0.5 }, &mut rng_w);
            spread_n = spread_n.max(Vec3::from_array(rn.origin()).sub(narrow.origin()).length());
            spread_w = spread_w.max(Vec3::from_array(rw.origin()).sub(wide.origin()).length());
        }
        // The wider lens scatters origins over a visibly larger disk.
        assert!(spread_w > spread_n * 5.0);
    }

    #[test]
    fn zero_aperture_matches_the_pinhole_ray() {
        let cam = basic_camera().with_thin_lens(0.0, 3.0).expect("valid lens");
        let mut rng = Rng::seed(3);
        let jitter = Sample2 { x: 0.3, y: 0.6 };
        let lens = cam.primary_ray_lens(5, 7, 16, 16, jitter, &mut rng);
        let pin = cam.primary_ray(5, 7, 16, 16, jitter);
        let dl = Vec3::from_array(lens.direction());
        let dp = Vec3::from_array(pin.direction());
        assert!(dl.sub(dp).length_squared() < 1e-10);
        assert_eq!(lens.origin(), pin.origin());
    }
}
