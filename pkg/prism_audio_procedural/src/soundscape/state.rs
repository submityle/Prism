//! High-level soundscape state: the environmental context that selects ambience.
//!
//! Gameplay and the spatial-acoustics layer describe where the listener is
//! (biome, weather, time of day, indoor or outdoor) and how busy the ambience
//! should be. [`SoundscapeState`] captures that context as a small, `Copy`
//! record. It carries no palette data itself; instead it is the input the
//! scheduler reads each block to decide which elements are eligible and how
//! strongly to weight them. Its one piece of derived behaviour is the daylight
//! factor: a smooth day/night curve computed from the normalised time of day
//! that elements use to cross-fade their dawn/dusk activity.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the soundscape state of design section 37; consumed by
//! [`crate::soundscape::scheduler::SoundscapeScheduler`] together with a
//! [`crate::soundscape::palette::SoundscapePalette`]. The indoor flag mirrors
//! the room ownership of the reverb-zone layer so outdoor elements are muted
//! indoors.

use bevy_math::ops;

use crate::dsp::TWO_PI;
use prism_audio_core::math::Sample;

/// The environmental context that selects and weights an ambience palette.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoundscapeState {
    time_of_day: Sample,
    density: Sample,
    indoor: bool,
}

impl SoundscapeState {
    /// Builds a state.
    ///
    /// `time_of_day` is a normalised clock in `[0, 1)` where `0.0` is midnight
    /// and `0.5` is noon (values are wrapped into that range). `density` is an
    /// overall activity multiplier in `[0, 1]` driven by gameplay (a calm vs a
    /// lively scene); it is clamped. `indoor` mutes elements that are only
    /// allowed outdoors.
    #[inline]
    #[must_use]
    pub fn new(time_of_day: Sample, density: Sample, indoor: bool) -> Self {
        Self {
            time_of_day: Self::wrap_unit(time_of_day),
            density: if density.is_finite() {
                density.clamp(0.0, 1.0)
            } else {
                0.0
            },
            indoor,
        }
    }

    /// A fully lit, outdoor, maximally active default (noon, density `1`).
    #[inline]
    #[must_use]
    pub fn outdoor_noon() -> Self {
        Self::new(0.5, 1.0, false)
    }

    /// Returns the normalised time of day in `[0, 1)`.
    #[inline]
    #[must_use]
    pub fn time_of_day(self) -> Sample {
        self.time_of_day
    }

    /// Returns the overall activity multiplier in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn density(self) -> Sample {
        self.density
    }

    /// Returns `true` when the listener is indoors.
    #[inline]
    #[must_use]
    pub fn is_indoor(self) -> bool {
        self.indoor
    }

    /// Returns the daylight factor in `[0, 1]`: `0` at midnight, `1` at noon,
    /// with a smooth cosine twilight either side.
    ///
    /// `daylight = 0.5 - 0.5 * cos(2*pi * time_of_day)`.
    #[inline]
    #[must_use]
    pub fn daylight(self) -> Sample {
        0.5 - 0.5 * ops::cos(TWO_PI * self.time_of_day)
    }

    #[inline]
    fn wrap_unit(x: Sample) -> Sample {
        if !x.is_finite() {
            return 0.0;
        }
        let f = x - ops::floor(x);
        // `floor` of a tiny negative can round to exactly `1.0`; keep it below.
        if f >= 1.0 { 0.0 } else { f }
    }
}

impl Default for SoundscapeState {
    #[inline]
    fn default() -> Self {
        Self::outdoor_noon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daylight_is_dark_at_midnight_bright_at_noon() {
        let midnight = SoundscapeState::new(0.0, 1.0, false);
        let noon = SoundscapeState::new(0.5, 1.0, false);
        assert!(midnight.daylight() < 1e-3, "{}", midnight.daylight());
        assert!(noon.daylight() > 1.0 - 1e-3, "{}", noon.daylight());
    }

    #[test]
    fn daylight_is_half_at_dawn_and_dusk() {
        let dawn = SoundscapeState::new(0.25, 1.0, false);
        let dusk = SoundscapeState::new(0.75, 1.0, false);
        assert!((dawn.daylight() - 0.5).abs() < 1e-3);
        assert!((dusk.daylight() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn time_of_day_wraps_and_density_clamps() {
        let s = SoundscapeState::new(1.25, 2.0, true);
        assert!((s.time_of_day() - 0.25).abs() < 1e-6);
        assert_eq!(s.density(), 1.0);
        assert!(s.is_indoor());
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let s = SoundscapeState::new(Sample::NAN, Sample::INFINITY, false);
        assert_eq!(s.time_of_day(), 0.0);
        assert_eq!(s.density(), 0.0);
    }
}
