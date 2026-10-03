//! Sub-step time fraction to in-block sample offset mapping.
//!
//! Physics steps and audio blocks run at different rates, but a collision that
//! happens part-way through a physics step should land on the matching sample
//! inside the audio block rather than being quantised to the block boundary
//! (the cause of frame-rate "machine gun" artefacts). [`BlockClock`] knows the
//! block length, and [`BlockClock::sample_offset`] maps a sub-step fraction in
//! `[0, 1)` to a sample index in `[0, block_frames)`. The physics step is
//! assumed aligned to the current audio block: fraction `0.0` is the first
//! sample of the block and fraction approaching `1.0` is the last.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the sample-accurate offset of design section 47.1 consumed by
//! [`prism_audio_procedural::contact::ImpactEvent::sample_offset`] and the
//! sample-accurate dispatch at the integration layer.

/// Block timing used to place a contact on the right sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockClock {
    /// Audio sample rate in hertz.
    pub sample_rate: u32,
    /// Number of frames in the current audio block.
    pub block_frames: u32,
}

impl BlockClock {
    /// Builds a block clock from a sample rate and block length.
    #[inline]
    #[must_use]
    pub fn new(sample_rate: u32, block_frames: u32) -> Self {
        Self {
            sample_rate,
            block_frames,
        }
    }

    /// Maps a sub-step fraction in `[0, 1)` to a sample index in the block.
    ///
    /// The fraction is clamped into `[0, 1)` first (non-finite becomes `0`),
    /// scaled by `block_frames`, floored with [`bevy_math::ops::floor`], and
    /// clamped to `block_frames - 1` so the result is always a valid in-block
    /// index. A zero-length block maps everything to `0`.
    #[inline]
    #[must_use]
    pub fn sample_offset(&self, substep_fraction: f32) -> u32 {
        if self.block_frames == 0 {
            return 0;
        }
        let frac = if substep_fraction.is_finite() {
            substep_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let scaled = bevy_math::ops::floor(frac * self.block_frames as f32);
        let last = self.block_frames - 1;
        if scaled >= last as f32 {
            last
        } else if scaled <= 0.0 {
            0
        } else {
            scaled as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_zero_is_first_sample() {
        let clock = BlockClock::new(48_000, 128);
        assert_eq!(clock.sample_offset(0.0), 0);
    }

    #[test]
    fn fraction_half_is_middle() {
        let clock = BlockClock::new(48_000, 128);
        assert_eq!(clock.sample_offset(0.5), 64);
    }

    #[test]
    fn fraction_near_one_is_last_sample() {
        let clock = BlockClock::new(48_000, 128);
        assert_eq!(clock.sample_offset(0.999), 127);
    }

    #[test]
    fn fraction_one_clamps_to_last() {
        let clock = BlockClock::new(48_000, 128);
        assert_eq!(clock.sample_offset(1.0), 127);
    }

    #[test]
    fn non_finite_fraction_is_first_sample() {
        let clock = BlockClock::new(48_000, 128);
        assert_eq!(clock.sample_offset(f32::NAN), 0);
    }

    #[test]
    fn zero_length_block_is_zero() {
        let clock = BlockClock::new(48_000, 0);
        assert_eq!(clock.sample_offset(0.5), 0);
    }
}
