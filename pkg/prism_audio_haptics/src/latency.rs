//! Playhead- and PDC-aware latency compensation for haptic output.
//!
//! Audio and haptics travel through different hardware with different fixed
//! delays, so a felt vibration can lead or lag the sound it belongs to. The
//! [`LatencyAligner`] converts a backend's intrinsic latency (reported in its
//! own haptic-rate samples) into audio-rate samples, derives the compensation
//! needed to line a backend up with the slowest device in the group, and
//! retards a playhead by that compensation so the haptic path reads the audio
//! that will be heard at the aligned moment.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the latency alignment of design section 36. It shares the sample
//! playhead of design section 8 and enforces the plugin-delay-compensation
//! contract of design section 29 against the intrinsic latency reported by
//! `crate::backend::HapticBackend`.

/// Aligns haptic output against audio using sample-domain delays.
///
/// The aligner is a pure unit converter and arithmetic helper; it holds the
/// audio and haptic rates and performs no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyAligner {
    audio_rate: u32,
    haptic_rate: u32,
}

impl LatencyAligner {
    /// Builds an aligner relating the audio and haptic sample rates.
    ///
    /// Both rates are clamped to at least one hertz.
    #[must_use]
    pub fn new(audio_rate: u32, haptic_rate: u32) -> Self {
        Self {
            audio_rate: audio_rate.max(1),
            haptic_rate: haptic_rate.max(1),
        }
    }

    /// Returns the audio sample rate in hertz.
    #[inline]
    #[must_use]
    pub fn audio_rate(&self) -> u32 {
        self.audio_rate
    }

    /// Returns the haptic sample rate in hertz.
    #[inline]
    #[must_use]
    pub fn haptic_rate(&self) -> u32 {
        self.haptic_rate
    }

    /// Converts a count of haptic-rate samples into audio-rate samples.
    #[must_use]
    pub fn haptic_to_audio(&self, haptic_samples: u32) -> u32 {
        let scaled =
            u64::from(haptic_samples) * u64::from(self.audio_rate) / u64::from(self.haptic_rate);
        scaled.min(u64::from(u32::MAX)) as u32
    }

    /// Converts a count of audio-rate samples into haptic-rate samples.
    #[must_use]
    pub fn audio_to_haptic(&self, audio_samples: u32) -> u32 {
        let scaled =
            u64::from(audio_samples) * u64::from(self.haptic_rate) / u64::from(self.audio_rate);
        scaled.min(u64::from(u32::MAX)) as u32
    }

    /// Returns the compensation (in audio samples) needed to delay a backend
    /// with `backend_latency` up to the group's `target_latency`.
    ///
    /// Both arguments are in audio samples; the result saturates at zero when
    /// the backend is already the slowest.
    #[must_use]
    pub fn compensation_samples(&self, target_latency: u32, backend_latency: u32) -> u32 {
        target_latency.saturating_sub(backend_latency)
    }

    /// Retards `playhead` by `compensation` audio samples, saturating at zero.
    ///
    /// The haptic path reads from this earlier playhead so its output, after
    /// the device's own latency, lands in step with the heard audio.
    #[must_use]
    pub fn compensated_playhead(&self, playhead: u64, compensation: u32) -> u64 {
        playhead.saturating_sub(u64::from(compensation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_are_clamped() {
        let a = LatencyAligner::new(0, 0);
        assert_eq!(a.audio_rate(), 1);
        assert_eq!(a.haptic_rate(), 1);
    }

    #[test]
    fn haptic_to_audio_scales_up() {
        let a = LatencyAligner::new(48_000, 1_000);
        assert_eq!(a.haptic_to_audio(10), 480);
    }

    #[test]
    fn audio_to_haptic_scales_down() {
        let a = LatencyAligner::new(48_000, 1_000);
        assert_eq!(a.audio_to_haptic(480), 10);
    }

    #[test]
    fn compensation_is_difference() {
        let a = LatencyAligner::new(48_000, 1_000);
        assert_eq!(a.compensation_samples(500, 200), 300);
    }

    #[test]
    fn compensation_saturates_for_slowest() {
        let a = LatencyAligner::new(48_000, 1_000);
        assert_eq!(a.compensation_samples(200, 500), 0);
    }

    #[test]
    fn playhead_retards_and_saturates() {
        let a = LatencyAligner::new(48_000, 1_000);
        assert_eq!(a.compensated_playhead(1_000, 300), 700);
        assert_eq!(a.compensated_playhead(100, 300), 0);
    }

    #[test]
    fn conversion_round_trips_cleanly() {
        let a = LatencyAligner::new(44_100, 1_050);
        let audio = a.haptic_to_audio(21);
        assert_eq!(a.audio_to_haptic(audio), 21);
    }
}
