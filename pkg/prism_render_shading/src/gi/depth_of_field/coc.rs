//! Thin-lens circle-of-confusion (CoC) model for physical depth of field.
//!
//! A real camera focuses a single object plane sharply; points nearer or
//! farther than that plane project onto the sensor as a blur disk — the
//! *circle of confusion*.  Its diameter follows directly from the thin-lens
//! equation and the geometry of the aperture.  This module is the
//! backend-neutral CPU reference for that geometry: given a lens description
//! (f-number, focal length, focus distance, sensor size) it returns the signed
//! CoC diameter on the sensor, in sensor millimetres or in output pixels, for
//! any object depth.
//!
//! # Model
//! With aperture diameter `A = f / N` (focal length `f`, f-number `N`), focus
//! distance `S1`, and object distance `S2` — all distances in the *same* unit —
//! the signed sensor-plane CoC diameter is
//!
//! ```text
//! c = A * f * (S2 - S1) / (S2 * (S1 - f)).
//! ```
//!
//! The sign is the physically useful part: `S2 > S1` (behind focus, "far")
//! yields a positive CoC, `S2 < S1` (in front of focus, "near") yields a
//! negative CoC, and `S2 == S1` yields exactly zero.  As `S2 -> infinity` the
//! CoC approaches the finite far-field asymptote `A * f / (S1 - f)`, which this
//! module exposes as the natural normalisation scale.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, I/O, GPU, allocation, or `unsafe`.
//! * `LensParams` distances mix units deliberately: optical lengths
//!   (`focal_length_mm`, `sensor_height_mm`) are millimetres to match lens data
//!   sheets, while scene distances (`focus_distance_m`, per-sample depth) are
//!   metres to match world space.  Conversions happen internally.
//! * Signed CoC: negative = near / foreground, positive = far / background,
//!   zero = exactly on the focus plane.
//! * No transcendental functions are needed here; the formula is rational, so
//!   there is no [`bevy_math::ops`] dependency.  Only inherent `f32` methods are
//!   used (`abs`, `clamp`, `is_finite`, ...).
//! * Defensive clamping everywhere: a pinhole aperture (`N <= 0`), a
//!   non-positive focal length, a focus distance at or inside the focal length
//!   (`S1 <= f`), a non-positive object depth, or any non-finite input all fall
//!   back to a perfectly sharp CoC of `0` rather than emitting `NaN`/`inf`.

/// Smallest denominator magnitude treated as non-degenerate.
///
/// Guards the `S2 * (S1 - f)` divide so a near-cancelling denominator cannot
/// blow the CoC up toward infinity.
const MIN_DENOM: f32 = 1.0e-6;

/// Metres-to-millimetres conversion for scene distances.
const M_TO_MM: f32 = 1000.0;

/// Thin-lens description of a physical camera, in GPU-twin storage layout.
///
/// All fields are `f32` and ordered to mirror the eventual uniform block.  Use
/// [`LensParams::sanitized`] (applied automatically by every query) to collapse
/// degenerate configurations to a sharp pinhole.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensParams {
    /// Aperture f-number `N` (dimensionless, e.g. `2.8`).
    ///
    /// Larger values mean a smaller aperture and shallower blur.  Values `<= 0`
    /// or non-finite are treated as a pinhole (zero aperture, no blur).
    pub aperture_f_stop: f32,
    /// Focal length `f` in millimetres (e.g. `50.0`).
    ///
    /// Values `<= 0` or non-finite disable blur.
    pub focal_length_mm: f32,
    /// Focus distance `S1` in metres, measured from the lens to the plane of
    /// sharp focus.
    ///
    /// Must exceed the focal length (`S1 > f`) for a real image; otherwise blur
    /// is disabled.
    pub focus_distance_m: f32,
    /// Physical sensor height in millimetres (e.g. full-frame `24.0`).
    ///
    /// Used only to convert the sensor-plane CoC into output pixels.  Values
    /// `<= 0` or non-finite yield a zero pixel CoC.
    pub sensor_height_mm: f32,
}

impl Default for LensParams {
    /// A 50 mm full-frame lens at f/2.8 focused on 2 m.
    #[inline]
    fn default() -> Self {
        Self {
            aperture_f_stop: 2.8,
            focal_length_mm: 50.0,
            focus_distance_m: 2.0,
            sensor_height_mm: 24.0,
        }
    }
}

impl LensParams {
    /// Returns `true` when the configuration can produce a real, blurred image.
    ///
    /// False when any optical quantity is non-finite or degenerate (pinhole
    /// aperture, non-positive focal length, or a focus plane at/inside the
    /// focal length).  Such configurations render perfectly sharp.
    #[inline]
    pub fn is_blurring(self) -> bool {
        let s = self.sanitized();
        s.aperture_f_stop > 0.0
            && s.focal_length_mm > 0.0
            && s.focus_distance_m * M_TO_MM > s.focal_length_mm
    }

    /// Replaces any non-finite field with a safe sentinel.
    ///
    /// Non-finite numbers become `0`, which the downstream guards read as
    /// "disable blur".  Finite but physically degenerate values (e.g. a tiny
    /// f-stop) are preserved and handled by the per-query clamps.
    #[inline]
    fn sanitized(self) -> Self {
        #[inline]
        fn finite_or_zero(x: f32) -> f32 {
            if x.is_finite() { x } else { 0.0 }
        }
        Self {
            aperture_f_stop: finite_or_zero(self.aperture_f_stop),
            focal_length_mm: finite_or_zero(self.focal_length_mm),
            focus_distance_m: finite_or_zero(self.focus_distance_m),
            sensor_height_mm: finite_or_zero(self.sensor_height_mm),
        }
    }

    /// Aperture diameter `A = f / N` in millimetres.
    ///
    /// Returns `0` for a pinhole or any degenerate optical input.
    #[inline]
    pub fn aperture_diameter_mm(self) -> f32 {
        let s = self.sanitized();
        if s.aperture_f_stop > 0.0 && s.focal_length_mm > 0.0 {
            s.focal_length_mm / s.aperture_f_stop
        } else {
            0.0
        }
    }

    /// Far-field CoC asymptote `A * f / (S1 - f)` in sensor millimetres.
    ///
    /// This is the limiting CoC diameter of an infinitely distant object and
    /// the natural scale for [`LensParams::coc_normalized`].  Returns `0` for a
    /// degenerate configuration.
    #[inline]
    pub fn max_far_coc_mm(self) -> f32 {
        let s = self.sanitized();
        let a = s.aperture_diameter_mm();
        let f = s.focal_length_mm;
        let s1 = s.focus_distance_m * M_TO_MM;
        let denom = s1 - f;
        if a <= 0.0 || f <= 0.0 || denom <= MIN_DENOM {
            return 0.0;
        }
        let c = a * f / denom;
        if c.is_finite() { c.max(0.0) } else { 0.0 }
    }

    /// Signed sensor-plane CoC diameter in millimetres for an object at
    /// `depth_m` metres.
    ///
    /// Negative = nearer than focus (foreground), positive = farther
    /// (background), `0` = on the focus plane or any degenerate input.  The
    /// result is always finite.
    pub fn coc_diameter_mm(self, depth_m: f32) -> f32 {
        let s = self.sanitized();
        let a = s.aperture_diameter_mm();
        let f = s.focal_length_mm;
        let s1 = s.focus_distance_m * M_TO_MM;
        let s2 = depth_m * M_TO_MM;

        // Degenerate optics or geometry -> perfectly sharp.
        if !s2.is_finite() || a <= 0.0 || f <= 0.0 || s2 <= 0.0 {
            return 0.0;
        }
        let focus_term = s1 - f;
        if focus_term <= MIN_DENOM {
            return 0.0;
        }
        let denom = s2 * focus_term;
        if denom.abs() <= MIN_DENOM {
            return 0.0;
        }
        let c = a * f * (s2 - s1) / denom;
        if c.is_finite() { c } else { 0.0 }
    }

    /// Signed CoC *radius* in sensor millimetres (half the diameter).
    #[inline]
    pub fn coc_radius_mm(self, depth_m: f32) -> f32 {
        0.5 * self.coc_diameter_mm(depth_m)
    }

    /// Signed CoC diameter in output pixels for an image `image_height_px`
    /// pixels tall.
    ///
    /// Scales the sensor-plane CoC by the pixels-per-millimetre of the output
    /// (`image_height_px / sensor_height_mm`).  Returns `0` when the sensor
    /// height or pixel height is non-positive.
    pub fn coc_diameter_pixels(self, depth_m: f32, image_height_px: f32) -> f32 {
        let s = self.sanitized();
        if !(s.sensor_height_mm > 0.0) || !(image_height_px > 0.0) || !image_height_px.is_finite() {
            return 0.0;
        }
        let c_mm = self.coc_diameter_mm(depth_m);
        let px = c_mm * (image_height_px / s.sensor_height_mm);
        if px.is_finite() { px } else { 0.0 }
    }

    /// Signed CoC *radius* in output pixels (half the pixel diameter).
    #[inline]
    pub fn coc_radius_pixels(self, depth_m: f32, image_height_px: f32) -> f32 {
        0.5 * self.coc_diameter_pixels(depth_m, image_height_px)
    }

    /// Signed CoC normalised to `[-1, 1]` against the far-field asymptote.
    ///
    /// `coc / max_far_coc_mm`, clamped so a very near object (whose raw CoC can
    /// exceed the far-field scale) saturates at `-1` instead of running away.
    /// Returns `0` when the asymptote is degenerate.
    pub fn coc_normalized(self, depth_m: f32) -> f32 {
        let scale = self.max_far_coc_mm();
        if scale <= MIN_DENOM {
            return 0.0;
        }
        let n = self.coc_diameter_mm(depth_m) / scale;
        if n.is_finite() { n.clamp(-1.0, 1.0) } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Shared reference lens used by the hand-computed checks below:
    /// 50 mm, f/2.0 (A = 25 mm), focused on 2 m, 24 mm sensor.
    fn ref_lens() -> LensParams {
        LensParams {
            aperture_f_stop: 2.0,
            focal_length_mm: 50.0,
            focus_distance_m: 2.0,
            sensor_height_mm: 24.0,
        }
    }

    #[test]
    fn focus_plane_is_sharp() {
        let lens = ref_lens();
        // Exactly on the focus plane: CoC must vanish.
        assert!(approx(lens.coc_diameter_mm(2.0), 0.0, 1.0e-7));
        assert!(approx(lens.coc_normalized(2.0), 0.0, 1.0e-7));
        assert!(approx(lens.coc_radius_mm(2.0), 0.0, 1.0e-7));
    }

    #[test]
    fn aperture_diameter_is_focal_over_fstop() {
        let lens = ref_lens();
        assert!(approx(lens.aperture_diameter_mm(), 25.0, 1.0e-6));
    }

    #[test]
    fn far_object_matches_hand_computation() {
        // f=50, A=25, S1=2000mm, S2=4000mm:
        // c = 25*50*(2000) / (4000*1950) = 2_500_000 / 7_800_000 = 0.320513 mm.
        let lens = ref_lens();
        let c = lens.coc_diameter_mm(4.0);
        assert!(c > 0.0, "far object must have positive (far) CoC");
        assert!(approx(c, 0.320_513, 1.0e-4), "c = {c}");
    }

    #[test]
    fn near_object_matches_hand_computation() {
        // S2=1000mm: c = 1250*(1000-2000)/(1000*1950) = -1_250_000/1_950_000
        //            = -0.641026 mm.
        let lens = ref_lens();
        let c = lens.coc_diameter_mm(1.0);
        assert!(c < 0.0, "near object must have negative (near) CoC");
        assert!(approx(c, -0.641_026, 1.0e-4), "c = {c}");
    }

    #[test]
    fn far_field_asymptote_matches() {
        // A*f/(S1-f) = 1250/1950 = 0.641026 mm.
        let lens = ref_lens();
        assert!(approx(lens.max_far_coc_mm(), 0.641_026, 1.0e-4));
    }

    #[test]
    fn normalized_far_is_half_at_double_focus() {
        // At 4 m the far CoC (0.3205) is exactly half the asymptote (0.6410).
        let lens = ref_lens();
        assert!(approx(lens.coc_normalized(4.0), 0.5, 1.0e-3));
    }

    #[test]
    fn far_coc_is_monotonic_in_depth() {
        // Beyond the focus plane, farther objects blur more (CoC increases),
        // approaching but never exceeding the far-field asymptote.
        let lens = ref_lens();
        let asym = lens.max_far_coc_mm();
        let mut prev = lens.coc_diameter_mm(2.0); // 0 at focus
        for k in 1..=20 {
            let depth = 2.0 + k as f32 * 0.5;
            let c = lens.coc_diameter_mm(depth);
            assert!(c >= prev - 1.0e-6, "far CoC must grow: {c} < {prev}");
            assert!(c <= asym + 1.0e-4, "far CoC must stay below asymptote");
            prev = c;
        }
    }

    #[test]
    fn near_coc_magnitude_is_monotonic_toward_lens() {
        // In front of focus, moving the object toward the lens deepens the
        // (negative) CoC magnitude.
        let lens = ref_lens();
        let mut prev_mag = 0.0_f32;
        for k in 0..=18 {
            let depth = 1.9 - k as f32 * 0.1; // 1.9 m down to 0.1 m
            if depth <= 0.0 {
                break;
            }
            let c = lens.coc_diameter_mm(depth);
            assert!(c <= 1.0e-6, "near side must be negative/zero: {c}");
            let mag = c.abs();
            assert!(mag >= prev_mag - 1.0e-5, "near magnitude must grow: {mag} < {prev_mag}");
            prev_mag = mag;
        }
    }

    #[test]
    fn pixel_conversion_scales_sensor_coc() {
        // 0.320513 mm on a 24 mm sensor rendered at 1080 px:
        // px = 0.320513 / 24 * 1080 = 14.423 px.
        let lens = ref_lens();
        let px = lens.coc_diameter_pixels(4.0, 1080.0);
        assert!(approx(px, 14.423, 2.0e-2), "px = {px}");
        // Radius is half the diameter.
        assert!(approx(lens.coc_radius_pixels(4.0, 1080.0), 0.5 * px, 1.0e-4));
    }

    #[test]
    fn pinhole_aperture_is_perfectly_sharp() {
        let lens = LensParams {
            aperture_f_stop: 0.0,
            ..ref_lens()
        };
        assert!(!lens.is_blurring());
        assert_eq!(lens.coc_diameter_mm(10.0), 0.0);
        assert_eq!(lens.coc_normalized(10.0), 0.0);
        assert_eq!(lens.aperture_diameter_mm(), 0.0);
    }

    #[test]
    fn focus_inside_focal_length_is_sharp() {
        // S1 (0.01 m = 10 mm) < f (50 mm): no real image, blur disabled.
        let lens = LensParams {
            focus_distance_m: 0.01,
            ..ref_lens()
        };
        assert!(!lens.is_blurring());
        assert_eq!(lens.coc_diameter_mm(5.0), 0.0);
        assert_eq!(lens.max_far_coc_mm(), 0.0);
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let bad = [
            LensParams { aperture_f_stop: f32::NAN, ..ref_lens() },
            LensParams { focal_length_mm: -5.0, ..ref_lens() },
            LensParams { focus_distance_m: f32::INFINITY, ..ref_lens() },
            LensParams { sensor_height_mm: 0.0, ..ref_lens() },
        ];
        for lens in bad {
            for &d in &[-1.0, 0.0, 0.5, 2.0, 1.0e6] {
                let c = lens.coc_diameter_mm(d);
                assert!(c.is_finite(), "CoC must be finite, got {c}");
                let px = lens.coc_diameter_pixels(d, 1080.0);
                assert!(px.is_finite(), "pixel CoC must be finite, got {px}");
                let n = lens.coc_normalized(d);
                assert!(n.is_finite() && (-1.0..=1.0).contains(&n), "norm out of range: {n}");
            }
        }
    }

    #[test]
    fn non_positive_depth_falls_back_to_sharp() {
        let lens = ref_lens();
        assert_eq!(lens.coc_diameter_mm(0.0), 0.0);
        assert_eq!(lens.coc_diameter_mm(-3.0), 0.0);
    }
}
