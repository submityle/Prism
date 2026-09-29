//! Sample-accurate transport and musical time.
//!
//! The [`Transport`] is the single source of truth for "where are we?" on the
//! audio thread. Because it is driven by a monotonically increasing sample
//! counter rather than the frame-rate-jittered game clock, it enables
//! sample-accurate scheduling of one-shots, seamless loops, and beat-quantized
//! musical transitions (the role UE's Quartz clock plays).

use crate::math::Sample;

/// A musical time signature (e.g. 4/4, 3/4, 6/8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeSignature {
    /// Beats per bar (the numerator).
    pub beats_per_bar: u16,
    /// Note value that represents one beat (the denominator, e.g. 4 or 8).
    pub beat_unit: u16,
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            beats_per_bar: 4,
            beat_unit: 4,
        }
    }
}

/// The global audio transport.
///
/// Holds the sample rate, a monotonic sample position, and the current musical
/// tempo so higher layers can quantize events to beats and bars.
#[derive(Debug, Clone, Copy)]
pub struct Transport {
    sample_rate: u32,
    playhead: u64,
    tempo_bpm: f32,
    signature: TimeSignature,
}

impl Transport {
    /// Creates a transport at sample position 0 with a default 120 BPM 4/4.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        Self {
            sample_rate,
            playhead: 0,
            tempo_bpm: 120.0,
            signature: TimeSignature::default(),
        }
    }

    /// Returns the sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the current sample position since transport start.
    #[inline]
    #[must_use]
    pub fn playhead(&self) -> u64 {
        self.playhead
    }

    /// Returns the tempo in beats per minute.
    #[inline]
    #[must_use]
    pub fn tempo_bpm(&self) -> f32 {
        self.tempo_bpm
    }

    /// Sets the tempo in beats per minute (clamped to a sane musical range).
    #[inline]
    pub fn set_tempo_bpm(&mut self, bpm: f32) {
        self.tempo_bpm = bpm.clamp(1.0, 1000.0);
    }

    /// Returns the current time signature.
    #[inline]
    #[must_use]
    pub fn signature(&self) -> TimeSignature {
        self.signature
    }

    /// Sets the time signature.
    #[inline]
    pub fn set_signature(&mut self, signature: TimeSignature) {
        self.signature = signature;
    }

    /// Advances the playhead by `frames` samples (called once per block).
    #[inline]
    pub fn advance(&mut self, frames: usize) {
        self.playhead += frames as u64;
    }

    /// Number of samples in one beat at the current tempo.
    #[inline]
    #[must_use]
    pub fn samples_per_beat(&self) -> f64 {
        (self.sample_rate as f64 * 60.0) / self.tempo_bpm as f64
    }

    /// Number of samples in one bar at the current tempo and signature.
    #[inline]
    #[must_use]
    pub fn samples_per_bar(&self) -> f64 {
        self.samples_per_beat() * self.signature.beats_per_bar as f64
    }

    /// Returns the sample position of the next bar boundary at or after
    /// `from_sample`. Used to quantize musical transitions.
    #[must_use]
    pub fn next_bar_boundary(&self, from_sample: u64) -> u64 {
        let per_bar = self.samples_per_bar();
        if per_bar <= 0.0 {
            return from_sample;
        }
        let bar_index = libm::ceil(from_sample as f64 / per_bar);
        libm::round(bar_index * per_bar) as u64
    }

    /// Converts a duration in seconds into a sample count at this sample rate.
    #[inline]
    #[must_use]
    pub fn seconds_to_samples(&self, seconds: Sample) -> u64 {
        libm::round(seconds.max(0.0) as f64 * self.sample_rate as f64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_moves_playhead() {
        let mut t = Transport::new(48_000);
        t.advance(256);
        assert_eq!(t.playhead(), 256);
    }

    #[test]
    fn samples_per_beat_at_120bpm() {
        let t = Transport::new(48_000);
        // 120 BPM => 0.5 s/beat => 24000 samples/beat.
        assert!((t.samples_per_beat() - 24_000.0).abs() < 1e-6);
    }

    #[test]
    fn next_bar_boundary_rounds_up() {
        let t = Transport::new(48_000);
        let per_bar = t.samples_per_bar() as u64; // 96000
        assert_eq!(t.next_bar_boundary(0), 0);
        assert_eq!(t.next_bar_boundary(1), per_bar);
        assert_eq!(t.next_bar_boundary(per_bar), per_bar);
    }
}
