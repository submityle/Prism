//! A ray cone: a scalar footprint (`width`) that grows along a path at a fixed
//! half-angle (`spread_angle`). Propagating one cone per ray gives a cheap,
//! bounce-stable texture-LOD estimate in the absence of hardware derivatives.
//!
//! # Conventions
//! * `width` is the cone diameter at the current point, in world units
//!   (meters). It starts at `0` for a pinhole primary ray.
//! * `spread_angle` is the growth half-angle in radians. For a primary ray it
//!   is the angular size of one pixel; it is updated at each scattering event
//!   by surface curvature and BSDF roughness.
//! * Growth uses the standard small-angle linearization `width += t *
//!   spread_angle` (RTGems ch. 20), exact to first order for the sub-degree
//!   pixel angles seen in practice.
//!
//! # References
//! Ray Tracing Gems 2019, ch. 20, "Texture Level of Detail Strategies for
//! Real-Time Ray Tracing": cone construction (eq. for primary spread),
//! propagation, and curvature/roughness widening on scatter.

/// Maximum spread half-angle (radians) a cone may accumulate; prevents a long
/// chain of rough bounces from driving `spread_angle` to a nonsensical value.
use bevy_math::ops;

const MAX_SPREAD_ANGLE: f32 = core::f32::consts::FRAC_PI_2;

/// A propagated ray cone (footprint diameter + growth half-angle).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCone {
    width: f32,
    spread_angle: f32,
}

impl RayCone {
    /// Construct an arbitrary cone from an explicit width and spread angle.
    /// Both are clamped to the valid non-negative, bounded range.
    #[inline]
    #[must_use]
    pub fn new(width: f32, spread_angle: f32) -> Self {
        Self {
            width: width.max(0.0),
            spread_angle: spread_angle.clamp(0.0, MAX_SPREAD_ANGLE),
        }
    }

    /// Primary-ray cone for a pinhole camera: zero width at the eye, spreading
    /// by the angular size of a single pixel.
    ///
    /// `vertical_fov` is the full vertical field of view in radians and
    /// `screen_height_px` the render-target height in pixels. The exact
    /// per-pixel angle is `atan(2 * tan(fov / 2) / height)`.
    #[must_use]
    pub fn from_pinhole_pixel(vertical_fov: f32, screen_height_px: f32) -> Self {
        let height = screen_height_px.max(1.0);
        let half = ops::tan(vertical_fov.max(0.0) * 0.5);
        let spread = ops::atan(2.0 * half / height);
        Self::new(0.0, spread)
    }

    /// Current cone diameter in world units.
    #[inline]
    #[must_use]
    pub fn width(self) -> f32 {
        self.width
    }

    /// Current spread half-angle in radians.
    #[inline]
    #[must_use]
    pub fn spread_angle(self) -> f32 {
        self.spread_angle
    }

    /// Advance the cone by `hit_distance` world units toward the next surface,
    /// widening the footprint by `distance * spread_angle`.
    #[must_use]
    pub fn advanced(self, hit_distance: f32) -> Self {
        let t = hit_distance.max(0.0);
        Self {
            width: self.width + t * self.spread_angle,
            spread_angle: self.spread_angle,
        }
    }

    /// Update the spread angle at a scattering event.
    ///
    /// `curvature_spread` is the surface's additional spread half-angle from
    /// local curvature (`2 * beta` in RTGems; pass the already-doubled value or
    /// use [`Self::reflected`] which doubles it). `roughness_spread` is an
    /// extra isotropic widening from BSDF roughness. Both are additive and the
    /// result is clamped.
    #[must_use]
    pub fn scattered(self, curvature_spread: f32, roughness_spread: f32) -> Self {
        let extra = curvature_spread.max(0.0) + roughness_spread.max(0.0);
        Self {
            width: self.width,
            spread_angle: (self.spread_angle + extra).clamp(0.0, MAX_SPREAD_ANGLE),
        }
    }

    /// Specialized reflection update: adds `2 * surface_curvature_spread` (the
    /// RTGems reflection term) plus a roughness widening.
    #[inline]
    #[must_use]
    pub fn reflected(self, surface_curvature_spread: f32, roughness_spread: f32) -> Self {
        self.scattered(2.0 * surface_curvature_spread.max(0.0), roughness_spread)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinhole_spread_matches_closed_form() {
        let fov = core::f32::consts::FRAC_PI_2; // 90 deg
        let h = 1080.0;
        let cone = RayCone::from_pinhole_pixel(fov, h);
        let expected = ops::atan(2.0 * ops::tan(fov * 0.5) / h);
        assert!((cone.spread_angle() - expected).abs() < 1.0e-9);
        assert_eq!(cone.width(), 0.0);
    }

    #[test]
    fn advance_grows_width_linearly() {
        let cone = RayCone::new(0.0, 0.01).advanced(10.0);
        assert!((cone.width() - 0.1).abs() < 1.0e-6);
        // A second equal advance doubles the footprint.
        let cone2 = cone.advanced(10.0);
        assert!((cone2.width() - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn reflection_doubles_curvature_term() {
        let cone = RayCone::new(0.1, 0.01).reflected(0.02, 0.0);
        assert!((cone.spread_angle() - (0.01 + 0.04)).abs() < 1.0e-6);
        // Width is unchanged by a scatter event (only angle widens).
        assert!((cone.width() - 0.1).abs() < 1.0e-9);
    }

    #[test]
    fn spread_angle_is_clamped() {
        let cone = RayCone::new(0.0, 10.0);
        assert!(cone.spread_angle() <= core::f32::consts::FRAC_PI_2 + 1.0e-6);
    }

    #[test]
    fn negative_inputs_are_sanitized() {
        let cone = RayCone::new(-1.0, -1.0).advanced(-5.0);
        assert_eq!(cone.width(), 0.0);
        assert_eq!(cone.spread_angle(), 0.0);
    }
}
