//! Backend-neutral CPU golden for a physically-based depth of field (`DoF`).
//!
//! Depth of field is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: it runs on the resolved, pre-exposed HDR radiance (see
//! [`crate::exposure`]) plus the scene depth, and blurs each pixel by the size
//! of the lens *circle of confusion* (`CoC`) at that pixel's distance. Every
//! illumination model — physically based or stylized — writes into the same
//! HDR buffer this pass reads, so one implementation serves them all.
//!
//! The optics follow the thin-lens model used by UE's `FDiaphragmDOF` and the
//! classic Riguer/Potmesil references:
//!
//! * A camera focused at distance `d_f` images an object at distance `d_o`
//!   onto a blur disc whose diameter (the **circle of confusion**) is
//!
//!   ```text
//!   `CoC` = | A * f * (d_o - d_f) / (d_o * (d_f - f)) |
//!   ```
//!
//!   where `f` is the focal length and `A = f / N` is the aperture diameter for
//!   an f-number (f-stop) `N`. At the focus plane (`d_o == d_f`) the `CoC` is
//!   exactly `0`; it grows toward a bounded asymptote for far objects and grows
//!   without bound as a near object approaches the lens.
//! * The **sign** of `d_o - d_f` separates the *near* field (objects closer
//!   than focus, which must be scattered *over* the sharp midground) from the
//!   *far* field (objects beyond focus, which the midground occludes). AAA `DoF`
//!   composites the two layers separately, so the golden exposes signed and
//!   clipped near/far `CoCs`.
//! * The physical `CoC` is a length on the sensor (mm); the gather kernel needs a
//!   **pixel radius**, so `coc_to_pixels` rescales by the sensor width and image
//!   width. A **bokeh weight** then softens the disc edge for each gather tap,
//!   and a **blend** eases the sharp image toward the blurred one by the `CoC`.
//!
//! Everything is pure `f32` maths — no transcendental calls — mirrored
//! arm-for-arm by `shaders/dof.wesl`, so the CPU golden and the GPU twin agree.
//!
//! The physical camera aperture is reused from [`crate::exposure::PhysicalCamera`]
//! (which already models f-stop / shutter / ISO); `DoF` additionally needs the
//! focal length, focus distance and sensor width, so [`DofCamera`] composes the
//! exposure camera with those extra optical fields rather than duplicating or
//! mutating the exposure type.

use crate::exposure::PhysicalCamera;

/// Aperture (entrance-pupil) diameter `A = f / N` for a focal length `f` (mm)
/// and f-number `N` (f-stop). A non-positive f-stop returns `0` so the `CoC`
/// maths never divides light through a degenerate lens.
#[must_use]
pub fn aperture_diameter(focal_length_mm: f32, aperture_f_stop: f32) -> f32 {
    if aperture_f_stop <= 0.0 {
        return 0.0;
    }
    focal_length_mm / aperture_f_stop
}

/// Circle-of-confusion **diameter** (mm on the sensor) for an object at
/// `object_distance` imaged by a lens of focal length `focal_length` focused at
/// `focus_distance`, with f-number `aperture_f_stop`:
///
/// ```text
/// A   = focal_length / aperture_f_stop
/// `CoC` = | A * f * (d_o - d_f) / (d_o * (d_f - f)) |
/// ```
///
/// All distances share one unit (mm). The result is the *unsigned* blur-disc
/// diameter, clamped to `[0, sensor_size]` because a blur circle wider than the
/// frame is capped in practice. At the focus plane (`d_o == d_f`) the `CoC` is
/// `0`. Degenerate geometry (`d_o <= 0`, or `d_f == f` which puts the image at
/// infinity) returns `0` rather than a NaN/inf.
#[must_use]
pub fn circle_of_confusion(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    let denom = object_distance * (focus_distance - focal_length);
    if object_distance <= 0.0 || denom == 0.0 {
        return 0.0;
    }
    let a = aperture_diameter(focal_length, aperture_f_stop);
    let coc = a * focal_length * (object_distance - focus_distance) / denom;
    coc.abs().clamp(0.0, sensor_size.max(0.0))
}

/// Signed circle-of-confusion diameter (mm): negative for the **near** field
/// (objects closer than focus, `d_o < d_f`) and positive for the **far** field
/// (objects beyond focus). The magnitude equals [`circle_of_confusion`] (same
/// clamp to `[0, sensor_size]`); only the sign of `d_o - d_f` is carried so the
/// compositor can layer near over far.
#[must_use]
pub fn signed_coc(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    let magnitude = circle_of_confusion(
        focus_distance,
        object_distance,
        focal_length,
        aperture_f_stop,
        sensor_size,
    );
    if object_distance < focus_distance {
        -magnitude
    } else {
        magnitude
    }
}

/// Near-field `CoC` diameter (mm): the blur magnitude for objects *closer* than
/// focus, and `0` for anything at or beyond the focus plane. This is the
/// foreground layer that scatters over the sharp midground.
#[must_use]
pub fn near_field_coc(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    (-signed_coc(
        focus_distance,
        object_distance,
        focal_length,
        aperture_f_stop,
        sensor_size,
    ))
    .max(0.0)
}

/// Far-field `CoC` diameter (mm): the blur magnitude for objects *beyond* focus,
/// and `0` for anything at or nearer than the focus plane. This is the
/// background layer the midground occludes.
#[must_use]
pub fn far_field_coc(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    signed_coc(
        focus_distance,
        object_distance,
        focal_length,
        aperture_f_stop,
        sensor_size,
    )
    .max(0.0)
}

/// Converts a `CoC` **diameter** in mm to a gather **radius** in pixels:
///
/// ```text
/// radius_px = coc_mm / sensor_width_mm * image_width_px * 0.5
/// ```
///
/// (Half the diameter, scaled by the sensor-to-image pixel density.) A
/// non-positive sensor width returns `0` so the conversion never divides by a
/// degenerate sensor.
#[must_use]
pub fn coc_to_pixels(coc_mm: f32, sensor_width_mm: f32, image_width_px: f32) -> f32 {
    if sensor_width_mm <= 0.0 {
        return 0.0;
    }
    coc_mm / sensor_width_mm * image_width_px * 0.5
}

/// Bokeh gather weight for a tap `sample_offset` pixels from the disc centre
/// inside a blur disc of radius `coc_radius` pixels. The weight is `1` at the
/// centre, falls off with a soft quadratic edge and is exactly `0` at and
/// beyond the disc boundary:
///
/// ```text
/// t = clamp(sample_offset / coc_radius, 0, 1)
/// w = (1 - t)^2
/// ```
///
/// A non-positive radius (`DoF` effectively off for this pixel) collapses to a
/// point sample: weight `1` only at the exact centre, `0` otherwise, so the
/// gather reduces to the identity.
#[must_use]
pub fn bokeh_weight(sample_offset: f32, coc_radius: f32) -> f32 {
    if coc_radius <= 0.0 {
        return if sample_offset <= 0.0 { 1.0 } else { 0.0 };
    }
    let t = (sample_offset / coc_radius).clamp(0.0, 1.0);
    let e = 1.0 - t;
    e * e
}

/// Blend factor in `[0, 1]` easing the sharp image toward the blurred one for a
/// `CoC` radius (pixels), normalised by `max_coc_pixels` and scaled by a global
/// `enabled_scale` (0 disables `DoF` entirely). Both inputs are clamped so the
/// factor never leaves `[0, 1]`.
#[must_use]
pub fn dof_blend_factor(coc_radius_px: f32, max_coc_pixels: f32, enabled_scale: f32) -> f32 {
    if max_coc_pixels <= 0.0 {
        return 0.0;
    }
    let normalised = (coc_radius_px / max_coc_pixels).clamp(0.0, 1.0);
    normalised * enabled_scale.clamp(0.0, 1.0)
}

/// Per-channel lerp of the `sharp` sample toward the `blurred` sample by
/// `factor`: `sharp + (blurred - sharp) * factor`. Written as an explicit lerp
/// (not `mix`) to stay bit-identical with the GPU twin.
#[must_use]
pub fn apply_dof(sharp: [f32; 3], blurred: [f32; 3], factor: f32) -> [f32; 3] {
    [
        sharp[0] + (blurred[0] - sharp[0]) * factor,
        sharp[1] + (blurred[1] - sharp[1]) * factor,
        sharp[2] + (blurred[2] - sharp[2]) * factor,
    ]
}

/// A physical camera extended with the optical fields `DoF` needs on top of the
/// exposure triangle. Reuses [`PhysicalCamera`] for the aperture f-stop
/// (shutter / ISO are ignored here) and adds the focal length, focus distance
/// and sensor width. Defaults to a 50 mm lens on a full-frame (36 mm) sensor
/// focused at 2 m.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofCamera {
    /// Exposure camera; only its `aperture` (f-stop) is consumed by the `DoF`
    /// optics, keeping the aperture definition shared with [`crate::exposure`].
    pub camera: PhysicalCamera,
    /// Lens focal length `f` in millimetres.
    pub focal_length_mm: f32,
    /// Distance the lens is focused at, in millimetres.
    pub focus_distance_mm: f32,
    /// Sensor width in millimetres (full-frame is 36 mm).
    pub sensor_width_mm: f32,
}

impl Default for DofCamera {
    fn default() -> Self {
        Self {
            camera: PhysicalCamera::default(),
            focal_length_mm: 50.0,
            focus_distance_mm: 2000.0,
            sensor_width_mm: 36.0,
        }
    }
}

impl DofCamera {
    /// `CoC` diameter (mm) for an object at `object_distance_mm`, using this
    /// camera's focal length, focus distance, aperture f-stop and sensor width.
    #[must_use]
    pub fn coc(&self, object_distance_mm: f32) -> f32 {
        circle_of_confusion(
            self.focus_distance_mm,
            object_distance_mm,
            self.focal_length_mm,
            self.camera.aperture,
            self.sensor_width_mm,
        )
    }

    /// Signed `CoC` diameter (mm) for an object at `object_distance_mm` (negative
    /// near, positive far).
    #[must_use]
    pub fn signed_coc(&self, object_distance_mm: f32) -> f32 {
        signed_coc(
            self.focus_distance_mm,
            object_distance_mm,
            self.focal_length_mm,
            self.camera.aperture,
            self.sensor_width_mm,
        )
    }
}

/// Artist controls for the `DoF` pass. Defaults to *disabled* (`enabled_scale`
/// `0`), so [`DofParams::apply`] is the identity — no blur — until `DoF` is
/// explicitly turned on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofParams {
    /// Optical camera (focal length / focus distance / aperture / sensor).
    pub camera: DofCamera,
    /// Rendered image width in pixels (for the mm -> pixel `CoC` conversion).
    pub image_width_px: f32,
    /// Largest gather radius (pixels) the blend saturates at.
    pub max_coc_pixels: f32,
    /// Global effect scale in `[0, 1]`; `0` disables `DoF` (identity blend).
    pub enabled_scale: f32,
}

impl Default for DofParams {
    fn default() -> Self {
        Self {
            camera: DofCamera::default(),
            image_width_px: 1920.0,
            max_coc_pixels: 32.0,
            enabled_scale: 0.0,
        }
    }
}

impl DofParams {
    /// Gather radius (pixels) for an object at `object_distance_mm`.
    #[must_use]
    pub fn coc_radius_px(&self, object_distance_mm: f32) -> f32 {
        let coc_mm = self.camera.coc(object_distance_mm);
        coc_to_pixels(coc_mm, self.camera.sensor_width_mm, self.image_width_px)
    }

    /// Blends `sharp` toward `blurred` for an object at `object_distance_mm`.
    /// With the default (`enabled_scale == 0`) this returns `sharp` unchanged.
    #[must_use]
    pub fn apply(&self, object_distance_mm: f32, sharp: [f32; 3], blurred: [f32; 3]) -> [f32; 3] {
        let radius = self.coc_radius_px(object_distance_mm);
        let factor = dof_blend_factor(radius, self.max_coc_pixels, self.enabled_scale);
        apply_dof(sharp, blurred, factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} !~= {b}");
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    #[test]
    fn aperture_diameter_is_focal_over_fstop() {
        approx(aperture_diameter(50.0, 2.0), 25.0);
        approx(aperture_diameter(100.0, 4.0), 25.0);
        // Degenerate f-stop is safe.
        approx(aperture_diameter(50.0, 0.0), 0.0);
    }

    #[test]
    fn coc_is_zero_at_focus_plane() {
        // Object exactly at the focus distance -> perfectly sharp.
        approx(circle_of_confusion(2000.0, 2000.0, 50.0, 2.8, 36.0), 0.0);
    }

    #[test]
    fn coc_positive_for_near_object() {
        // Closer than focus -> a real blur disc.
        let coc = circle_of_confusion(2000.0, 1000.0, 50.0, 2.8, 36.0);
        assert!(coc > 0.0, "near CoC should be > 0, got {coc}");
    }

    #[test]
    fn coc_positive_for_far_object() {
        // Beyond focus -> a real blur disc.
        let coc = circle_of_confusion(2000.0, 5000.0, 50.0, 2.8, 36.0);
        assert!(coc > 0.0, "far CoC should be > 0, got {coc}");
    }

    #[test]
    fn coc_grows_as_far_object_recedes() {
        // Far field: monotonically increasing with object distance.
        let mut prev = -1.0_f32;
        for d in [2500.0, 3000.0, 5000.0, 10_000.0, 50_000.0] {
            let coc = circle_of_confusion(2000.0, d, 50.0, 2.8, 1000.0);
            assert!(coc > prev, "far CoC not increasing at {d}: {coc} <= {prev}");
            prev = coc;
        }
    }

    #[test]
    fn coc_grows_as_near_object_approaches() {
        // Near field: CoC grows as the object moves toward the lens.
        let mut prev = -1.0_f32;
        for d in [1900.0, 1500.0, 1000.0, 500.0, 200.0] {
            let coc = circle_of_confusion(2000.0, d, 50.0, 2.8, 1000.0);
            assert!(
                coc > prev,
                "near CoC not increasing at {d}: {coc} <= {prev}"
            );
            prev = coc;
        }
    }

    #[test]
    fn coc_grows_with_wider_aperture() {
        // A smaller f-number is a wider aperture -> a larger CoC.
        let wide = circle_of_confusion(2000.0, 5000.0, 50.0, 1.4, 1000.0);
        let narrow = circle_of_confusion(2000.0, 5000.0, 50.0, 8.0, 1000.0);
        assert!(
            wide > narrow,
            "wider aperture should blur more: {wide} <= {narrow}"
        );
    }

    #[test]
    fn coc_grows_with_longer_focal_length() {
        // A longer lens (kept below the sensor clamp) blurs more.
        let long = circle_of_confusion(2000.0, 5000.0, 85.0, 2.8, 1000.0);
        let short = circle_of_confusion(2000.0, 5000.0, 35.0, 2.8, 1000.0);
        assert!(
            long > short,
            "longer focal length should blur more: {long} <= {short}"
        );
    }

    #[test]
    fn coc_clamped_to_sensor_size() {
        // A huge geometric CoC is capped at the sensor extent.
        let coc = circle_of_confusion(2000.0, 100.0, 200.0, 1.0, 36.0);
        approx(coc, 36.0);
    }

    #[test]
    fn coc_safe_for_degenerate_geometry() {
        // Object at/behind the lens, and focus at the focal length (image at
        // infinity): both return 0 rather than NaN/inf.
        approx(circle_of_confusion(2000.0, 0.0, 50.0, 2.8, 36.0), 0.0);
        approx(circle_of_confusion(50.0, 5000.0, 50.0, 2.8, 36.0), 0.0);
    }

    #[test]
    fn coc_to_pixels_matches_formula() {
        // 0.36 mm CoC on a 36 mm sensor imaged 1920 px wide.
        // radius = 0.36 / 36 * 1920 * 0.5 = 9.6 px.
        approx(coc_to_pixels(0.36, 36.0, 1920.0), 9.6);
    }

    #[test]
    fn coc_to_pixels_safe_for_zero_sensor() {
        approx(coc_to_pixels(1.0, 0.0, 1920.0), 0.0);
    }

    #[test]
    fn signed_coc_is_negative_near_positive_far() {
        let near = signed_coc(2000.0, 1000.0, 50.0, 2.8, 1000.0);
        let far = signed_coc(2000.0, 5000.0, 50.0, 2.8, 1000.0);
        assert!(near < 0.0, "near signed CoC should be negative, got {near}");
        assert!(far > 0.0, "far signed CoC should be positive, got {far}");
    }

    #[test]
    fn near_field_coc_isolates_foreground() {
        // Near object -> positive near CoC, zero far CoC.
        approx(far_field_coc(2000.0, 1000.0, 50.0, 2.8, 1000.0), 0.0);
        assert!(near_field_coc(2000.0, 1000.0, 50.0, 2.8, 1000.0) > 0.0);
    }

    #[test]
    fn far_field_coc_isolates_background() {
        // Far object -> positive far CoC, zero near CoC.
        approx(near_field_coc(2000.0, 5000.0, 50.0, 2.8, 1000.0), 0.0);
        assert!(far_field_coc(2000.0, 5000.0, 50.0, 2.8, 1000.0) > 0.0);
    }

    #[test]
    fn bokeh_weight_is_max_at_center() {
        approx(bokeh_weight(0.0, 8.0), 1.0);
    }

    #[test]
    fn bokeh_weight_is_zero_at_and_beyond_edge() {
        approx(bokeh_weight(8.0, 8.0), 0.0);
        approx(bokeh_weight(20.0, 8.0), 0.0);
    }

    #[test]
    fn bokeh_weight_is_monotonic_decreasing() {
        let mut prev = f32::INFINITY;
        for i in 0..=16 {
            let w = bokeh_weight(i as f32 * 0.5, 8.0);
            assert!(
                w <= prev,
                "bokeh weight not decreasing at {i}: {w} > {prev}"
            );
            prev = w;
        }
    }

    #[test]
    fn bokeh_weight_zero_radius_is_point_sample() {
        // No blur: only the exact centre contributes.
        approx(bokeh_weight(0.0, 0.0), 1.0);
        approx(bokeh_weight(0.5, 0.0), 0.0);
    }

    #[test]
    fn blend_factor_respects_max_and_scale() {
        // Half the max radius, fully enabled -> 0.5.
        approx(dof_blend_factor(16.0, 32.0, 1.0), 0.5);
        // Saturates at 1 past the max.
        approx(dof_blend_factor(64.0, 32.0, 1.0), 1.0);
        // Disabled scale -> 0.
        approx(dof_blend_factor(16.0, 32.0, 0.0), 0.0);
        // Degenerate max -> 0.
        approx(dof_blend_factor(16.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn apply_dof_lerps_channels() {
        let sharp = [0.2, 0.4, 0.6];
        let blurred = [1.0, 0.8, 0.0];
        approx3(apply_dof(sharp, blurred, 0.0), sharp);
        approx3(apply_dof(sharp, blurred, 1.0), blurred);
        approx3(apply_dof(sharp, blurred, 0.5), [0.6, 0.6, 0.3]);
    }

    #[test]
    fn dof_camera_reuses_physical_aperture() {
        // The DoF camera's CoC is driven by the exposure PhysicalCamera aperture.
        let cam = DofCamera::default();
        let direct = circle_of_confusion(
            cam.focus_distance_mm,
            5000.0,
            cam.focal_length_mm,
            cam.camera.aperture,
            cam.sensor_width_mm,
        );
        approx(cam.coc(5000.0), direct);
        // Focus plane is sharp.
        approx(cam.coc(cam.focus_distance_mm), 0.0);
    }

    #[test]
    fn dof_params_default_is_identity() {
        let p = DofParams::default();
        approx(p.enabled_scale, 0.0);
        // Disabled DoF leaves the sharp sample untouched at any distance.
        let sharp = [0.3, 0.5, 0.7];
        let blurred = [1.0, 1.0, 1.0];
        approx3(p.apply(5000.0, sharp, blurred), sharp);
        approx3(p.apply(500.0, sharp, blurred), sharp);
    }

    #[test]
    fn dof_params_blurs_when_enabled() {
        let p = DofParams {
            enabled_scale: 1.0,
            ..DofParams::default()
        };
        let sharp = [0.0, 0.0, 0.0];
        let blurred = [1.0, 1.0, 1.0];
        // A far, strongly-defocused object pulls the result toward blurred.
        let out = p.apply(50_000.0, sharp, blurred);
        assert!(
            out[0] > 0.0,
            "enabled DoF should blend some blur, got {out:?}"
        );
    }
}
