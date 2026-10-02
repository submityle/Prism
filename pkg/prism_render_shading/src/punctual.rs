//! Backend-neutral punctual (point and spot) light contracts.
//!
//! Directional lights model an infinitely distant emitter and reach the
//! surface with constant illuminance.  Punctual lights instead live at a world
//! position and fall off with distance and, for spot lights, with the angle to
//! the cone axis.  This module converts a punctual light plus a shaded world
//! position into the same [`DirectLightSample`] the analytic BSDFs already
//! consume, so the lighting integrator stays light-type agnostic.
//!
//! The attenuation model mirrors Unreal's `GetLocalLightAttenuation`
//! (`DeferredLightingCommon.ush`) and the Frostbite course notes: an inverse
//! square law bounded by a smooth range window, and a spot cone encoded with a
//! precomputed scale/offset pair (Unreal's `FDeferredLightData::SpotAngles`).

use crate::DirectLightSample;

/// Smallest squared distance used to bound the inverse-square singularity when
/// the shaded point coincides with the light position.
const MIN_DISTANCE_SQUARED: f32 = 1.0e-4;

/// A point or spot light expressed in world space.
///
/// The struct is intentionally flat and `Copy` so the same layout can later be
/// uploaded to a GPU light buffer.  Point lights leave the spot fields at their
/// neutral values (`spot_scale == 0`), which makes the angular term evaluate to
/// `1` everywhere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PunctualLight {
    /// World-space position of the emitter.
    pub position: [f32; 3],
    /// Linear, pre-exposed radiant intensity radiated toward the surface.
    pub intensity: [f32; 3],
    /// Influence radius in world units.  `0` (or non-finite) disables the
    /// range window and the light only falls off by inverse square.
    pub range: f32,
    /// Unit cone axis pointing from the light out into the scene.  Unused by
    /// point lights.
    pub direction: [f32; 3],
    /// Precomputed `1 / (cos(inner) - cos(outer))`; `0` marks a point light.
    pub spot_scale: f32,
    /// Precomputed `-cos(outer) * spot_scale`.
    pub spot_offset: f32,
    /// Analytic shadow visibility term in `[0, 1]`.
    pub visibility: f32,
}

impl Default for PunctualLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            intensity: [0.0; 3],
            range: 0.0,
            direction: [0.0, 0.0, -1.0],
            spot_scale: 0.0,
            spot_offset: 1.0,
            visibility: 1.0,
        }
    }
}

impl PunctualLight {
    /// Builds an omnidirectional point light.
    pub fn point(position: [f32; 3], intensity: [f32; 3], range: f32) -> Self {
        Self {
            position,
            intensity,
            range,
            ..Self::default()
        }
    }

    /// Builds a spot light from the cosines of the inner and outer cone
    /// half-angles.
    ///
    /// Cosines are taken directly (rather than angles) so this backend-neutral
    /// crate never depends on platform transcendental functions, and because it
    /// matches the packed `SpotAngles` encoding used by GPU light buffers.
    /// `direction` is normalized and points from the light into the scene; the
    /// inner cosine is clamped strictly above the outer cosine so the scale
    /// term never divides by zero.
    pub fn spot(
        position: [f32; 3],
        intensity: [f32; 3],
        range: f32,
        direction: [f32; 3],
        cos_inner: f32,
        cos_outer: f32,
    ) -> Self {
        let axis = normalize_or(direction, [0.0, 0.0, -1.0]);
        let cos_outer = cos_outer.clamp(-1.0, 1.0);
        // Keep the inner cone strictly inside the outer cone.
        let cos_inner = cos_inner.clamp(-1.0, 1.0).max(cos_outer + 1.0e-3);
        let scale = 1.0 / (cos_inner - cos_outer);
        Self {
            position,
            intensity,
            range,
            direction: axis,
            spot_scale: scale,
            spot_offset: -cos_outer * scale,
            visibility: 1.0,
        }
    }

    /// Overrides the analytic shadow visibility term.
    pub fn with_visibility(mut self, visibility: f32) -> Self {
        self.visibility = visibility.clamp(0.0, 1.0);
        self
    }

    /// Converts the light into a [`DirectLightSample`] at `world_position`.
    ///
    /// Returns `None` when the surface receives no energy: it sits at the light
    /// position, beyond the range window, or outside the spot cone.  Callers
    /// skip the BSDF evaluation entirely in that case.
    pub fn sample(&self, world_position: [f32; 3]) -> Option<DirectLightSample> {
        let to_light = sub(self.position, world_position);
        let distance_squared = dot(to_light, to_light);
        if !distance_squared.is_finite() || distance_squared <= 0.0 {
            return None;
        }
        let distance = distance_squared.sqrt();
        let direction = mul_scalar(to_light, distance.recip());

        // Inverse-square law, clamped near the source to avoid a singularity.
        let mut attenuation = 1.0 / distance_squared.max(MIN_DISTANCE_SQUARED);

        // Smooth range window: fades to zero at `range` with a squared falloff
        // so the derivative also vanishes at the boundary.
        if self.range.is_finite() && self.range > 0.0 {
            let range_squared = self.range * self.range;
            let ratio = (distance_squared / range_squared).min(1.0);
            let window = (1.0 - ratio * ratio).max(0.0);
            attenuation *= window * window;
        }

        // Spot angular falloff.  `spot_scale == 0` (point light) leaves the
        // clamped term at `1`, so the branch is cheap and allocation-free.
        if self.spot_scale != 0.0 {
            let cos_angle = dot(self.direction, mul_scalar(direction, -1.0));
            let cone = (cos_angle * self.spot_scale + self.spot_offset).clamp(0.0, 1.0);
            attenuation *= cone * cone;
        }

        if !attenuation.is_finite() || attenuation <= 0.0 {
            return None;
        }

        Some(DirectLightSample {
            direction,
            illuminance: mul_scalar(self.intensity, attenuation),
            visibility: self.visibility.clamp(0.0, 1.0),
        })
    }
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        mul_scalar(value, length_squared.sqrt().recip())
    } else {
        fallback
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul_scalar(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_light_falls_off_with_inverse_square() {
        let light = PunctualLight::point([0.0, 0.0, 0.0], [4.0; 3], 0.0);
        let near = light.sample([0.0, 0.0, 1.0]).unwrap();
        let far = light.sample([0.0, 0.0, 2.0]).unwrap();
        // Twice the distance -> a quarter of the illuminance.
        for channel in 0..3 {
            assert!((near.illuminance[channel] - 4.0).abs() < 1.0e-4);
            assert!((far.illuminance[channel] - 1.0).abs() < 1.0e-4);
        }
        assert_eq!(near.direction, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn range_window_reaches_zero_at_the_boundary() {
        let light = PunctualLight::point([0.0, 0.0, 0.0], [10.0; 3], 5.0);
        assert!(light.sample([0.0, 0.0, 5.0]).is_none());
        // Just inside the range still contributes.
        let inside = light.sample([0.0, 0.0, 4.9]).unwrap();
        assert!(inside.illuminance.iter().all(|c| *c > 0.0));
    }

    #[test]
    fn coincident_sample_is_rejected() {
        let light = PunctualLight::point([1.0, 2.0, 3.0], [1.0; 3], 0.0);
        assert!(light.sample([1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn spot_cone_is_bright_on_axis_and_dark_outside() {
        // Cone points down -Z; cosines encode a ~53 deg outer half-angle.
        let light = PunctualLight::spot([0.0, 0.0, 0.0], [8.0; 3], 0.0, [0.0, 0.0, -1.0], 0.8, 0.5);
        // On-axis: full angular term.
        let on_axis = light.sample([0.0, 0.0, -1.0]).unwrap();
        assert!(on_axis.illuminance.iter().all(|c| *c > 0.0));
        // Perpendicular to the axis: outside the cone -> no contribution.
        assert!(light.sample([1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn spot_falloff_is_monotonic_between_inner_and_outer() {
        let light =
            PunctualLight::spot([0.0, 0.0, 0.0], [1.0; 3], 0.0, [0.0, 0.0, -1.0], 0.99, 0.1);
        // Points at increasing angle from the axis should not brighten.
        let mut previous = f32::INFINITY;
        for offset in [0.0_f32, 0.2, 0.4, 0.6] {
            let position = [offset, 0.0, -1.0];
            let value = light
                .sample(position)
                .map(|sample| sample.illuminance[0])
                .unwrap_or(0.0);
            assert!(value <= previous + 1.0e-4, "spot term must be monotonic");
            previous = value;
        }
    }

    #[test]
    fn visibility_scales_but_direction_is_preserved() {
        let light = PunctualLight::point([0.0, 0.0, 0.0], [4.0; 3], 0.0).with_visibility(0.25);
        let sample = light.sample([0.0, 0.0, 1.0]).unwrap();
        assert!((sample.visibility - 0.25).abs() < 1.0e-6);
        assert_eq!(sample.direction, [0.0, 0.0, -1.0]);
    }
}
