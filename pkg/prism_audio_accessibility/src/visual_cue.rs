//! Visual sound cues describing key audio events for on-screen indicators.
//!
//! A [`VisualCue`] is a one-shot descriptor emitted for an important sound
//! (an explosion, an alert, a line of dialogue) so the UI can draw a directional
//! indicator for players who cannot rely on hearing alone. The descriptor
//! carries a direction (azimuth and elevation, in radians, relative to the
//! listener facing), a [`CueKind`] category, and a normalized `intensity`.
//!
//! Directions follow the engine convention: the listener faces `-Z`, `+X` is to
//! the right, and `+Y` is up. Azimuth is measured clockwise from the forward
//! axis when viewed from above; elevation is positive upward.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the visual-sound-cue item of design section 23. Cues are produced
//! at event-trigger time and travel to the UI through the same telemetry ring
//! (design section 26) that carries captions; the direction convention matches
//! the spatial listener frame used elsewhere in the engine.

use bevy_math::{ops, Vec3};
use prism_audio_core::math::Sample;

/// Category of a visual sound cue, used by the UI to pick an icon or color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum CueKind {
    /// A sharp, high-energy event such as an explosion or impact.
    Impact,
    /// A warning or notification the player should act on.
    Alert,
    /// A continuous or background sound providing context.
    Ambient,
    /// Spoken dialogue.
    Voice,
}

/// A one-shot visual indicator describing a key sound event.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VisualCue {
    /// Event category.
    kind: CueKind,
    /// Azimuth in radians, clockwise from the forward axis (viewed from above).
    azimuth_rad: Sample,
    /// Elevation in radians, positive upward.
    elevation_rad: Sample,
    /// Normalized intensity in the closed range `[0, 1]`.
    intensity: Sample,
}

impl VisualCue {
    /// Creates a cue, clamping `intensity` into `[0, 1]`.
    #[must_use]
    pub fn new(kind: CueKind, azimuth_rad: Sample, elevation_rad: Sample, intensity: Sample) -> Self {
        Self {
            kind,
            azimuth_rad,
            elevation_rad,
            intensity: intensity.clamp(0.0, 1.0),
        }
    }

    /// Returns the event category.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> CueKind {
        self.kind
    }

    /// Returns the azimuth in radians.
    #[inline]
    #[must_use]
    pub fn azimuth_rad(&self) -> Sample {
        self.azimuth_rad
    }

    /// Returns the elevation in radians.
    #[inline]
    #[must_use]
    pub fn elevation_rad(&self) -> Sample {
        self.elevation_rad
    }

    /// Returns the normalized intensity in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn intensity(&self) -> Sample {
        self.intensity
    }

    /// Sets the intensity, clamping into `[0, 1]`, and returns the updated cue.
    #[must_use]
    pub fn with_intensity(mut self, intensity: Sample) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Returns the unit direction vector in the listener frame.
    ///
    /// The vector is `(cos(elev) * sin(azim), sin(elev), -cos(elev) * cos(azim))`,
    /// so a zero azimuth and elevation points straight ahead (`-Z`).
    #[must_use]
    pub fn direction_vector(&self) -> Vec3 {
        let (sin_az, cos_az) = ops::sin_cos(self.azimuth_rad);
        let (sin_el, cos_el) = ops::sin_cos(self.elevation_rad);
        Vec3::new(cos_el * sin_az, sin_el, -cos_el * cos_az)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn intensity_is_clamped() {
        let cue = VisualCue::new(CueKind::Impact, 0.0, 0.0, 1.5);
        assert!((cue.intensity() - 1.0).abs() < EPS);
        let cue = cue.with_intensity(-0.5);
        assert!(cue.intensity().abs() < EPS);
    }

    #[test]
    fn forward_points_negative_z() {
        let cue = VisualCue::new(CueKind::Voice, 0.0, 0.0, 1.0);
        let dir = cue.direction_vector();
        assert!(dir.x.abs() < EPS);
        assert!(dir.y.abs() < EPS);
        assert!((dir.z + 1.0).abs() < EPS);
    }

    #[test]
    fn right_azimuth_points_positive_x() {
        let cue = VisualCue::new(CueKind::Alert, core::f32::consts::FRAC_PI_2, 0.0, 1.0);
        let dir = cue.direction_vector();
        assert!((dir.x - 1.0).abs() < EPS);
        assert!(dir.z.abs() < EPS);
    }

    #[test]
    fn direction_is_unit_length() {
        let cue = VisualCue::new(CueKind::Ambient, 0.7, 0.3, 0.5);
        let dir = cue.direction_vector();
        let len_sq = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
        assert!((len_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn kind_is_preserved() {
        let cue = VisualCue::new(CueKind::Impact, 0.1, 0.2, 0.9);
        assert_eq!(cue.kind(), CueKind::Impact);
    }
}
