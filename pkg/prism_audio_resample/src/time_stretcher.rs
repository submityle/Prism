//! Streaming time-stretch / pitch-shift trait with decoupled factors.
//!
//! A [`TimeStretcher`] changes a signal's duration and its pitch
//! *independently*. The time-stretch factor is `output_duration /
//! input_duration` (a factor of `2.0` makes the sound last twice as long), and
//! the pitch-shift ratio is `output_frequency / input_frequency` (a ratio of
//! `2.0` raises everything one octave). Setting one does not disturb the other:
//! stretching to half speed keeps the pitch, and shifting up an octave keeps
//! the duration. This is the content-side counterpart to the physical
//! sample-rate conversion in [`crate::resampler`] -- Doppler and play-rate live
//! in the resampler, musical pitch and tempo live here.
//!
//! The streaming contract mirrors [`crate::resampler::Resampler`]: a call to
//! [`TimeStretcher::process`] consumes some input, produces some output, and
//! reports both counts in a [`StretchProgress`]. Implementations buffer only a
//! bounded, preallocated amount of audio internally; the caller re-presents any
//! unconsumed input tail on the next call.
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`]. Implemented by
//! [`crate::wsola::WsolaStretcher`] (time-domain, voice/SFX grade) and
//! [`crate::phase_vocoder::PhaseVocoderStretcher`] (frequency-domain, music
//! grade). Both reuse [`crate::fractional_delay`] and the resamplers for the
//! decoupled pitch stage.

use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;

/// Smallest time-stretch factor honored (fastest playback).
pub const MIN_STRETCH: Sample = 0.25;

/// Largest time-stretch factor honored (slowest playback).
pub const MAX_STRETCH: Sample = 4.0;

/// Smallest pitch-shift ratio honored (two octaves down).
pub const MIN_PITCH: Sample = 0.25;

/// Largest pitch-shift ratio honored (two octaves up).
pub const MAX_PITCH: Sample = 4.0;

/// Clamps a time-stretch factor into `[MIN_STRETCH, MAX_STRETCH]`, mapping a
/// non-finite request to unity.
#[inline]
#[must_use]
pub fn clamp_stretch(factor: Sample) -> Sample {
    if factor.is_finite() {
        factor.clamp(MIN_STRETCH, MAX_STRETCH)
    } else {
        1.0
    }
}

/// Clamps a pitch-shift ratio into `[MIN_PITCH, MAX_PITCH]`, mapping a
/// non-finite request to unity.
#[inline]
#[must_use]
pub fn clamp_pitch(ratio: Sample) -> Sample {
    if ratio.is_finite() {
        ratio.clamp(MIN_PITCH, MAX_PITCH)
    } else {
        1.0
    }
}

/// Converts a pitch interval in semitones to a frequency ratio
/// (`2^(semitones / 12)`), computed with [`bevy_math::ops`].
#[inline]
#[must_use]
pub fn semitones_to_ratio(semitones: Sample) -> Sample {
    ops::exp2(semitones / 12.0)
}

/// Outcome of a single [`TimeStretcher::process`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StretchProgress {
    /// Number of leading `input` samples that were consumed.
    pub consumed: usize,
    /// Number of leading `output` samples that were written.
    pub produced: usize,
}

/// Pitch-preserving time scaling and time-preserving pitch scaling.
///
/// Implementations preallocate every buffer at construction so
/// [`TimeStretcher::process`] performs no allocation, takes no locks, and never
/// panics. Non-finite input is treated as silence.
pub trait TimeStretcher {
    /// Returns the current time-stretch factor (`output / input` duration).
    fn time_stretch(&self) -> Sample;

    /// Returns the current pitch-shift ratio (`output / input` frequency).
    fn pitch_shift(&self) -> Sample;

    /// Sets the time-stretch factor, clamped to `[MIN_STRETCH, MAX_STRETCH]`.
    fn set_time_stretch(&mut self, factor: Sample);

    /// Sets the pitch-shift ratio, clamped to `[MIN_PITCH, MAX_PITCH]`.
    fn set_pitch_shift(&mut self, ratio: Sample);

    /// Clears all internal state to the stream origin. Two reset instances with
    /// identical settings produce bit-identical output for identical input.
    fn reset(&mut self);

    /// Processes one block, returning how much input was consumed and how much
    /// output was produced.
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> StretchProgress;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractional_delay::close;

    #[test]
    fn semitone_octave_is_double() {
        assert!(close(semitones_to_ratio(12.0), 2.0));
        assert!(close(semitones_to_ratio(-12.0), 0.5));
        assert!(close(semitones_to_ratio(0.0), 1.0));
    }

    #[test]
    fn stretch_clamps() {
        assert!(close(clamp_stretch(100.0), MAX_STRETCH));
        assert!(close(clamp_stretch(0.0), MIN_STRETCH));
        assert!(close(clamp_stretch(Sample::NAN), 1.0));
    }

    #[test]
    fn pitch_clamps() {
        assert!(close(clamp_pitch(100.0), MAX_PITCH));
        assert!(close(clamp_pitch(0.0), MIN_PITCH));
        assert!(close(clamp_pitch(Sample::INFINITY), 1.0));
    }

    #[test]
    fn progress_defaults_to_zero() {
        let p = StretchProgress::default();
        assert_eq!(p.consumed, 0);
        assert_eq!(p.produced, 0);
    }
}

/// Fixed-capacity circular sample FIFO shared by the stretcher implementations.
///
/// Allocated once at construction; all operations are allocation free and panic
/// free. It additionally tracks the absolute stream index of its oldest sample
/// so WSOLA can address input by absolute position while the backing storage
/// slides.
#[derive(Clone, Debug)]
pub(crate) struct SampleFifo {
    /// Backing storage of fixed length `cap`.
    buf: Vec<Sample>,
    /// Index of the oldest valid sample within `buf`.
    head: usize,
    /// Number of valid samples currently stored.
    len: usize,
    /// Capacity (equal to `buf.len()`).
    cap: usize,
    /// Absolute stream index of the oldest valid sample.
    base: usize,
}

impl SampleFifo {
    /// Builds an empty FIFO with the given capacity (at least `1`).
    pub(crate) fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            buf: alloc::vec![0.0; cap],
            head: 0,
            len: 0,
            cap,
            base: 0,
        }
    }

    /// Number of valid samples stored.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Number of additional samples that can be pushed.
    pub(crate) fn free(&self) -> usize {
        self.cap - self.len
    }

    /// Absolute stream index of the oldest valid sample.
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    /// Absolute stream index one past the newest valid sample.
    pub(crate) fn end(&self) -> usize {
        self.base + self.len
    }

    /// Clears all samples and resets the absolute index to zero.
    pub(crate) fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.base = 0;
    }

    /// Pushes one sample, returning `false` if the FIFO is full.
    pub(crate) fn push(&mut self, x: Sample) -> bool {
        if self.len == self.cap {
            return false;
        }
        let idx = (self.head + self.len) % self.cap;
        self.buf[idx] = x;
        self.len += 1;
        true
    }

    /// Pops the oldest sample, returning `None` if empty.
    pub(crate) fn pop(&mut self) -> Option<Sample> {
        if self.len == 0 {
            return None;
        }
        let x = self.buf[self.head];
        self.head = (self.head + 1) % self.cap;
        self.len -= 1;
        self.base += 1;
        Some(x)
    }

    /// Reads the sample at absolute stream index `abs`, or `0.0` if it is not
    /// currently buffered.
    pub(crate) fn at(&self, abs: usize) -> Sample {
        if abs < self.base || abs >= self.base + self.len {
            return 0.0;
        }
        let rel = abs - self.base;
        self.buf[(self.head + rel) % self.cap]
    }

    /// Drops samples with absolute index below `keep`, advancing the base.
    pub(crate) fn drop_until(&mut self, keep: usize) {
        while self.base < keep && self.len > 0 {
            let _ = self.pop();
        }
    }
}
