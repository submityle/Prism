//! Directional weighting that maps a source position to left/right intensity.
//!
//! When a device has a left and a right actuator, a haptic event should be felt
//! on the side it comes from and should fade as the source moves away.
//! [`SpatialWeighting`] turns a listener-relative direction and distance into a
//! pair of actuator gains: a source to the left biases the left actuator, a
//! source to the right biases the right actuator, and distance scales both via
//! an inverse-distance rolloff.
//!
//! The left/right split reuses the engine's constant-power pan law so the total
//! felt energy is preserved as the source pans across the body, matching how
//! stereo audio is panned.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the spatial weighting of design section 36 and consumes the
//! distance/direction shaping of design section 15. The pan split reuses
//! `prism_audio_core::math::equal_power_pan` rather than re-deriving a pan law.

use bevy_math::ops;
use bevy_math::Vec3;
use prism_audio_core::math::{equal_power_pan, Sample};

/// Left/right actuator intensities produced by the weighting.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ActuatorGains {
    /// Intensity for the left actuator in `[0, 1]`.
    pub left: Sample,
    /// Intensity for the right actuator in `[0, 1]`.
    pub right: Sample,
}

/// Maps source direction and distance to left/right actuator gains.
///
/// Distance attenuation uses the classic inverse-distance model: full strength
/// within `ref_distance`, then a `rolloff`-weighted falloff out to
/// `max_distance`, beyond which the gain holds at its `max_distance` value.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialWeighting {
    ref_distance: Sample,
    max_distance: Sample,
    rolloff: Sample,
}

impl Default for SpatialWeighting {
    fn default() -> Self {
        Self {
            ref_distance: 1.0,
            max_distance: 50.0,
            rolloff: 1.0,
        }
    }
}

impl SpatialWeighting {
    /// Builds a weighting from the reference distance, maximum distance, and
    /// rolloff factor.
    ///
    /// All inputs are forced non-negative and `max_distance` is kept at or
    /// above `ref_distance`.
    #[must_use]
    pub fn new(ref_distance: Sample, max_distance: Sample, rolloff: Sample) -> Self {
        let ref_distance = ref_distance.max(0.0);
        Self {
            ref_distance,
            max_distance: max_distance.max(ref_distance),
            rolloff: rolloff.max(0.0),
        }
    }

    /// Returns the distance attenuation factor in `[0, 1]`.
    #[must_use]
    pub fn attenuation(&self, distance: Sample) -> Sample {
        let d = distance.max(0.0).min(self.max_distance);
        if d <= self.ref_distance {
            return 1.0;
        }
        let denom = self.ref_distance + self.rolloff * (d - self.ref_distance);
        if denom <= 0.0 {
            1.0
        } else {
            (self.ref_distance / denom).clamp(0.0, 1.0)
        }
    }

    /// Weights by a pan position in `[-1, 1]` (negative = left) and distance.
    #[must_use]
    pub fn weight_pan(&self, pan: Sample, distance: Sample) -> ActuatorGains {
        let atten = self.attenuation(distance);
        let (left, right) = equal_power_pan(pan.clamp(-1.0, 1.0));
        ActuatorGains {
            left: left * atten,
            right: right * atten,
        }
    }

    /// Weights by an azimuth in radians (0 = front, positive = right) and
    /// distance.
    #[must_use]
    pub fn weight_azimuth(&self, azimuth_rad: Sample, distance: Sample) -> ActuatorGains {
        self.weight_pan(ops::sin(azimuth_rad), distance)
    }

    /// Weights by a listener-relative direction (x = right, z = forward).
    ///
    /// The vertical (y) component is ignored; distance is taken from the
    /// horizontal projection of `direction` unless the vector is degenerate.
    #[must_use]
    pub fn weight_direction(&self, direction: Vec3) -> ActuatorGains {
        let horizontal = ops::sqrt(direction.x * direction.x + direction.z * direction.z);
        let distance = ops::sqrt(
            direction.x * direction.x + direction.y * direction.y + direction.z * direction.z,
        );
        let pan = if horizontal > 0.0 {
            (direction.x / horizontal).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        self.weight_pan(pan, distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_is_balanced() {
        let w = SpatialWeighting::default();
        let g = w.weight_pan(0.0, 0.0);
        assert!((g.left - g.right).abs() < 1e-6);
    }

    #[test]
    fn left_pan_biases_left() {
        let w = SpatialWeighting::default();
        let g = w.weight_pan(-1.0, 0.0);
        assert!(g.left > 0.99);
        assert!(g.right < 1e-3);
    }

    #[test]
    fn right_pan_biases_right() {
        let w = SpatialWeighting::default();
        let g = w.weight_pan(1.0, 0.0);
        assert!(g.right > 0.99);
        assert!(g.left < 1e-3);
    }

    #[test]
    fn attenuation_is_unity_within_reference() {
        let w = SpatialWeighting::new(2.0, 50.0, 1.0);
        assert!((w.attenuation(0.0) - 1.0).abs() < 1e-6);
        assert!((w.attenuation(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn attenuation_falls_with_distance() {
        let w = SpatialWeighting::new(1.0, 100.0, 1.0);
        let near = w.attenuation(2.0);
        let far = w.attenuation(20.0);
        assert!(far < near);
        assert!(far > 0.0);
    }

    #[test]
    fn azimuth_right_biases_right() {
        let w = SpatialWeighting::default();
        let g = w.weight_azimuth(core::f32::consts::FRAC_PI_2, 0.0);
        assert!(g.right > g.left);
    }

    #[test]
    fn direction_left_biases_left() {
        let w = SpatialWeighting::default();
        let g = w.weight_direction(Vec3::new(-1.0, 0.0, 0.0));
        assert!(g.left > g.right);
    }

    #[test]
    fn direction_distance_attenuates() {
        let w = SpatialWeighting::new(1.0, 100.0, 1.0);
        let near = w.weight_direction(Vec3::new(0.0, 0.0, 1.0));
        let far = w.weight_direction(Vec3::new(0.0, 0.0, 40.0));
        assert!(far.left < near.left);
    }

    #[test]
    fn degenerate_direction_is_centred() {
        let w = SpatialWeighting::default();
        let g = w.weight_direction(Vec3::ZERO);
        assert!((g.left - g.right).abs() < 1e-6);
    }
}
