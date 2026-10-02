//! **Stingers**: short musical phrases overlaid on top of the running music.
//!
//! A stinger is a one-shot flourish -- a sting, an accent, a short motif --
//! that plays *over* the current segment without replacing it (an achievement
//! chime, a danger sting). Like a transition it aligns to a quantization grid
//! so it lands musically, but unlike a transition it never changes the playing
//! segment, the playlist cursor, or the clock: it is a transient overlay.
//!
//! A [`Stinger`] is pure data: a leaf sound to play, the quantization grid it
//! snaps to (reusing [`crate::transition::TransitionType`]), and a playback
//! gain in decibels. The live [`crate::system::MusicSystem`] resolves the grid
//! to a sample-accurate point and emits a single
//! [`crate::action::MusicAction::PlayStinger`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The quantized
//! overlay phrase is reconstructed from first principles as plain data. No
//! AI/ML.
//!
//! # Relationship
//!
//! A [`Stinger`] shares the [`crate::transition::TransitionType`] grid with
//! [`crate::transition::Transition`] but is resolved independently by
//! [`crate::system::MusicSystem::trigger_stinger`], which leaves the current
//! segment and clock untouched.

use prism_audio_core::math::Sample;

use crate::id::{SoundId, StingerId};
use crate::transition::TransitionType;

/// A quantization-aligned one-shot overlay phrase.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Stinger {
    /// Stable id this stinger is referenced by.
    pub id: StingerId,
    /// Leaf audio asset the stinger plays.
    pub sound: SoundId,
    /// Quantization grid the overlay snaps to (relative to the active clock).
    pub quantize: TransitionType,
    /// Playback gain in decibels applied to the overlay voice.
    pub gain_db: Sample,
}

impl Stinger {
    /// Builds a stinger that snaps to `quantize` and plays at unity gain
    /// (`0 dB`).
    #[must_use]
    pub fn new(id: StingerId, sound: SoundId, quantize: TransitionType) -> Self {
        Self {
            id,
            sound,
            quantize,
            gain_db: 0.0,
        }
    }

    /// Sets the playback gain in decibels, returning `self` for builder-style
    /// chaining.
    #[must_use]
    pub fn with_gain_db(mut self, gain_db: Sample) -> Self {
        self.gain_db = gain_db;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-5;

    #[test]
    fn new_defaults_to_unity_gain() {
        let s = Stinger::new(StingerId::new(1), SoundId::new(7), TransitionType::NextBar);
        assert_eq!(s.quantize, TransitionType::NextBar);
        assert!((s.gain_db - 0.0).abs() < EPS);
    }

    #[test]
    fn with_gain_db_overrides() {
        let s = Stinger::new(StingerId::new(2), SoundId::new(8), TransitionType::NextBeat)
            .with_gain_db(-6.0);
        assert!((s.gain_db - (-6.0)).abs() < EPS);
        assert_eq!(s.sound, SoundId::new(8));
    }
}
