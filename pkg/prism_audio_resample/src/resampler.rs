//! Streaming sample-rate conversion trait and quality grades.
//!
//! A [`Resampler`] converts a stream of input samples into a stream of output
//! samples at a (possibly continuously varying) ratio. The ratio is defined as
//! `output_rate / input_rate`: a ratio above `1.0` interpolates (up-samples)
//! and a ratio below `1.0` decimates (down-samples). The ratio may be changed
//! at any block boundary with [`Resampler::set_ratio`]; because every
//! implementation carries its fractional read position across calls the output
//! stays phase continuous (no clicks) when the ratio sweeps, which is exactly
//! what Doppler and play-rate automation require.
//!
//! The streaming contract is deliberately simple and allocation free. A call to
//! [`Resampler::process`] reads from `input`, writes to `output`, and returns a
//! [`ResampleProgress`] reporting how many input samples were consumed and how
//! many output samples were produced. The caller re-presents any unconsumed
//! tail of `input` on the next call; implementations retain only a bounded,
//! preallocated history of past samples (never the whole stream).
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`]. The trait is implemented by
//! [`crate::linear::LinearResampler`],
//! [`crate::polyphase_sinc::PolyphaseSincResampler`], and
//! [`crate::high_order_sinc::HighOrderSincResampler`], which all build on the
//! shared kernel in [`crate::fractional_delay`]. The grades mirror the
//! performance-governed quality ladder described in the engine design doc.

use prism_audio_core::math::Sample;

/// Lowest ratio (`output_rate / input_rate`) any resampler will honor.
///
/// Ratios are clamped into `[MIN_RATIO, MAX_RATIO]` so the anti-alias kernel
/// support (which grows as the ratio shrinks) stays bounded and the hot path
/// stays allocation free.
pub const MIN_RATIO: Sample = 1.0 / 16.0;

/// Highest ratio (`output_rate / input_rate`) any resampler will honor.
pub const MAX_RATIO: Sample = 16.0;

/// Clamps `ratio` into the supported `[MIN_RATIO, MAX_RATIO]` range, mapping a
/// non-finite request to unity so a poisoned parameter cannot stall the stream.
#[inline]
#[must_use]
pub fn clamp_ratio(ratio: Sample) -> Sample {
    if ratio.is_finite() {
        ratio.clamp(MIN_RATIO, MAX_RATIO)
    } else {
        1.0
    }
}

/// Resampling quality grade, ordered from cheapest to highest fidelity.
///
/// The grade is a performance-governance knob (see the engine design doc's
/// audio-LOD section): far-off or low-priority voices run [`Linear`], the
/// default voice bus runs [`Sinc`], and mastering / near-field paths run
/// [`HighOrderSinc`].
///
/// [`Linear`]: ResampleQuality::Linear
/// [`Sinc`]: ResampleQuality::Sinc
/// [`HighOrderSinc`]: ResampleQuality::HighOrderSinc
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ResampleQuality {
    /// Two-point linear interpolation. Cheapest; audible high-frequency roll-off
    /// and aliasing. Suitable for distant / low-priority sources.
    Linear,
    /// Polyphase Blackman-windowed sinc FIR. The default production grade with a
    /// flat pass band and strong image rejection.
    Sinc,
    /// Longer windowed-sinc kernel with more sub-sample phases for the lowest
    /// aliasing floor. Mastering / near-field grade.
    HighOrderSinc,
}

/// Outcome of a single [`Resampler::process`] call.
///
/// `consumed` input samples may be dropped by the caller; any remaining
/// `input[consumed..]` must be re-presented on the next call. `produced`
/// samples were written to the front of `output`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ResampleProgress {
    /// Number of leading `input` samples that were consumed.
    pub consumed: usize,
    /// Number of leading `output` samples that were written.
    pub produced: usize,
}

/// Streaming, phase-continuous sample-rate converter.
///
/// Implementations precompute every coefficient table at construction so that
/// [`Resampler::process`] performs no allocation, takes no locks, and never
/// panics. Non-finite input is treated as silence.
pub trait Resampler {
    /// Returns the resampler's quality grade.
    fn quality(&self) -> ResampleQuality;

    /// Returns the current ratio (`output_rate / input_rate`).
    fn ratio(&self) -> Sample;

    /// Sets the ratio (`output_rate / input_rate`), clamped to
    /// `[MIN_RATIO, MAX_RATIO]`.
    ///
    /// The fractional read position is preserved, so changing the ratio between
    /// blocks keeps the output phase continuous.
    fn set_ratio(&mut self, ratio: Sample);

    /// Clears all history and resets the read position to the stream origin.
    ///
    /// Two freshly reset instances with identical ratios produce bit-identical
    /// output for identical input.
    fn reset(&mut self);

    /// Processes one block, returning how much input was consumed and how much
    /// output was produced.
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> ResampleProgress;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    #[test]
    fn ratio_clamps_into_range() {
        assert!(close_ratio(clamp_ratio(0.0), MIN_RATIO));
        assert!(close_ratio(clamp_ratio(1000.0), MAX_RATIO));
        assert!(close_ratio(clamp_ratio(2.0), 2.0));
    }

    #[test]
    fn non_finite_ratio_maps_to_unity() {
        assert!(close_ratio(clamp_ratio(Sample::NAN), 1.0));
        assert!(close_ratio(clamp_ratio(Sample::INFINITY), 1.0));
    }

    #[test]
    fn progress_defaults_to_zero() {
        let p = ResampleProgress::default();
        assert_eq!(p.consumed, 0);
        assert_eq!(p.produced, 0);
    }

    #[test]
    fn quality_is_ordered() {
        assert!(ResampleQuality::Linear < ResampleQuality::Sinc);
        assert!(ResampleQuality::Sinc < ResampleQuality::HighOrderSinc);
    }

    fn close_ratio(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= 1.0e-6
    }
}

impl PartialOrd for ResampleQuality {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ResampleQuality {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl ResampleQuality {
    /// Integer rank used to order the grades from cheapest to highest fidelity.
    #[inline]
    #[must_use]
    fn rank(self) -> u8 {
        match self {
            ResampleQuality::Linear => 0,
            ResampleQuality::Sinc => 1,
            ResampleQuality::HighOrderSinc => 2,
        }
    }
}
