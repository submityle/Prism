//! Canonical deinterleaved `f32` PCM container for conditioned assets.
//!
//! Every stage in this crate consumes and produces audio through
//! [`ConditionedPcm`], a planar (one `Vec` per channel) `f32` buffer paired
//! with its sample rate and channel layout. Keeping one canonical shape means
//! the decode matrix, the resampler, the analyzers, and the authoring layer all
//! agree on sample ordering and scaling, which is what makes the pipeline
//! deterministic and golden-reproducible.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The offline counterpart of [`prism_audio_core::buffer::AudioBuffer`]
//! (section 51). Where `AudioBuffer` is a fixed-capacity real-time block, this
//! container owns a whole asset and may grow, so it lives strictly off the
//! audio thread.

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::math::Sample;

/// Errors returned when building or validating a [`ConditionedPcm`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PcmError {
    /// The sample rate was zero.
    ZeroSampleRate,
    /// The number of channel buffers did not match the layout's channel count.
    ChannelCountMismatch {
        /// Channels implied by the declared [`ChannelLayout`].
        expected: usize,
        /// Channel buffers actually supplied.
        actual: usize,
    },
    /// The channel buffers did not all have the same length.
    RaggedChannels,
    /// The interleaved sample count was not a whole multiple of the channel
    /// count.
    PartialFrame,
    /// The channel count has no supported [`ChannelLayout`] mapping.
    UnsupportedChannelCount(usize),
}

/// Returns the canonical [`ChannelLayout`] for an interleaved channel count.
///
/// Mono, stereo, quad, `5.1`, and `7.1` map to their obvious layouts; a
/// four-channel buffer is reported as [`ChannelLayout::Quad`] rather than
/// first-order ambisonics because a raw decode cannot distinguish them. Counts
/// without a canonical layout (for example three or five) return [`None`].
#[must_use]
pub fn layout_for_channel_count(channels: usize) -> Option<ChannelLayout> {
    match channels {
        1 => Some(ChannelLayout::Mono),
        2 => Some(ChannelLayout::Stereo),
        4 => Some(ChannelLayout::Quad),
        6 => Some(ChannelLayout::Surround5_1),
        8 => Some(ChannelLayout::Surround7_1),
        _ => None,
    }
}

/// Encoder-introduced leading/trailing silence recorded at decode time.
///
/// Lossy encoders (and some lossless container writers) prepend a decoder
/// warm-up region (`preroll_frames`) and pad the final block out to a codec
/// granule (`padding_frames`). Carrying the counts lets later stages trim to
/// the true program boundaries without guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderDelay {
    /// Leading frames the encoder added before the first real sample.
    pub preroll_frames: u32,
    /// Trailing frames the encoder appended after the last real sample.
    pub padding_frames: u32,
}

impl EncoderDelay {
    /// Creates a delay record.
    #[must_use]
    pub const fn new(preroll_frames: u32, padding_frames: u32) -> Self {
        Self {
            preroll_frames,
            padding_frames,
        }
    }

    /// Returns `true` when neither pre-roll nor padding is present.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.preroll_frames == 0 && self.padding_frames == 0
    }
}

/// A planar, deinterleaved `f32` PCM asset in `[-1.0, 1.0]` nominal range.
///
/// Invariants (enforced by every constructor): the sample rate is non-zero,
/// the number of channel buffers equals the layout's channel count, and all
/// channel buffers share the same length (the frame count).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditionedPcm {
    /// Sample rate in Hz (always non-zero).
    sample_rate: u32,
    /// Channel layout; its channel count matches `channels.len()`.
    layout: ChannelLayout,
    /// One sample buffer per channel; all buffers share one length.
    channels: Vec<Vec<Sample>>,
}

impl ConditionedPcm {
    /// Builds a container from per-channel buffers, validating the invariants.
    ///
    /// # Errors
    ///
    /// Returns [`PcmError::ZeroSampleRate`] for a zero rate,
    /// [`PcmError::ChannelCountMismatch`] when the buffer count disagrees with
    /// the layout, and [`PcmError::RaggedChannels`] when the buffers differ in
    /// length.
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        channels: Vec<Vec<Sample>>,
    ) -> Result<Self, PcmError> {
        if sample_rate == 0 {
            return Err(PcmError::ZeroSampleRate);
        }
        let expected = layout.channel_count();
        if channels.len() != expected {
            return Err(PcmError::ChannelCountMismatch {
                expected,
                actual: channels.len(),
            });
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|c| c.len() != frames) {
            return Err(PcmError::RaggedChannels);
        }
        Ok(Self {
            sample_rate,
            layout,
            channels,
        })
    }

    /// Builds a silent container of `frames` zero-valued frames.
    ///
    /// # Errors
    ///
    /// Returns [`PcmError::ZeroSampleRate`] when `sample_rate` is zero.
    pub fn silence(
        sample_rate: u32,
        layout: ChannelLayout,
        frames: usize,
    ) -> Result<Self, PcmError> {
        if sample_rate == 0 {
            return Err(PcmError::ZeroSampleRate);
        }
        let channels = (0..layout.channel_count())
            .map(|_| vec![0.0 as Sample; frames])
            .collect();
        Ok(Self {
            sample_rate,
            layout,
            channels,
        })
    }

    /// Builds a container from an interleaved frame buffer.
    ///
    /// `interleaved` holds `frames * channel_count` samples in frame-major
    /// order (`L0, R0, L1, R1, ...`). The channel count is taken from `layout`.
    ///
    /// # Errors
    ///
    /// Returns [`PcmError::ZeroSampleRate`] for a zero rate and
    /// [`PcmError::PartialFrame`] when the sample count is not a whole multiple
    /// of the channel count.
    pub fn from_interleaved(
        sample_rate: u32,
        layout: ChannelLayout,
        interleaved: &[Sample],
    ) -> Result<Self, PcmError> {
        if sample_rate == 0 {
            return Err(PcmError::ZeroSampleRate);
        }
        let channel_count = layout.channel_count();
        if channel_count == 0 || !interleaved.len().is_multiple_of(channel_count) {
            return Err(PcmError::PartialFrame);
        }
        let frames = interleaved.len() / channel_count;
        let mut channels: Vec<Vec<Sample>> =
            (0..channel_count).map(|_| Vec::with_capacity(frames)).collect();
        for frame in interleaved.chunks_exact(channel_count) {
            for (ch, &sample) in frame.iter().enumerate() {
                channels[ch].push(sample);
            }
        }
        Ok(Self {
            sample_rate,
            layout,
            channels,
        })
    }

    /// The sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The channel layout.
    #[must_use]
    #[inline]
    pub const fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// The number of channels.
    #[must_use]
    #[inline]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// The number of frames (shared by every channel).
    #[must_use]
    #[inline]
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// Returns `true` when there are no frames.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames() == 0
    }

    /// Borrows channel `ch`, or returns [`None`] when the index is out of
    /// range.
    #[must_use]
    #[inline]
    pub fn channel(&self, ch: usize) -> Option<&[Sample]> {
        self.channels.get(ch).map(Vec::as_slice)
    }

    /// Mutably borrows channel `ch`, or returns [`None`] when the index is out
    /// of range.
    #[inline]
    pub fn channel_mut(&mut self, ch: usize) -> Option<&mut [Sample]> {
        self.channels.get_mut(ch).map(Vec::as_mut_slice)
    }

    /// Borrows every channel buffer.
    #[must_use]
    #[inline]
    pub fn channels(&self) -> &[Vec<Sample>] {
        &self.channels
    }

    /// The duration in seconds (`frames / sample_rate`).
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_rate() {
        let err = ConditionedPcm::new(0, ChannelLayout::Mono, vec![vec![0.0]]);
        assert_eq!(err.err(), Some(PcmError::ZeroSampleRate));
    }

    #[test]
    fn new_rejects_channel_mismatch() {
        let err = ConditionedPcm::new(48_000, ChannelLayout::Stereo, vec![vec![0.0, 0.0]]);
        assert!(matches!(
            err,
            Err(PcmError::ChannelCountMismatch { expected: 2, actual: 1 })
        ));
    }

    #[test]
    fn new_rejects_ragged_channels() {
        let err = ConditionedPcm::new(
            48_000,
            ChannelLayout::Stereo,
            vec![vec![0.0, 0.0], vec![0.0]],
        );
        assert_eq!(err.err(), Some(PcmError::RaggedChannels));
    }

    #[test]
    fn from_interleaved_deinterleaves() {
        let pcm = ConditionedPcm::from_interleaved(
            48_000,
            ChannelLayout::Stereo,
            &[1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        )
        .unwrap();
        assert_eq!(pcm.frames(), 3);
        assert_eq!(pcm.channel(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(pcm.channel(1), Some(&[-1.0, -2.0, -3.0][..]));
    }

    #[test]
    fn from_interleaved_rejects_partial_frame() {
        let err = ConditionedPcm::from_interleaved(48_000, ChannelLayout::Stereo, &[1.0, 2.0, 3.0]);
        assert_eq!(err.err(), Some(PcmError::PartialFrame));
    }

    #[test]
    fn silence_is_zeroed() {
        let pcm = ConditionedPcm::silence(44_100, ChannelLayout::Mono, 8).unwrap();
        assert_eq!(pcm.frames(), 8);
        assert!(pcm.channel(0).unwrap().iter().all(|&s| s == 0.0));
    }

    #[test]
    fn layout_mapping_covers_canonical_counts() {
        assert_eq!(layout_for_channel_count(1), Some(ChannelLayout::Mono));
        assert_eq!(layout_for_channel_count(2), Some(ChannelLayout::Stereo));
        assert_eq!(layout_for_channel_count(6), Some(ChannelLayout::Surround5_1));
        assert_eq!(layout_for_channel_count(3), None);
    }

    #[test]
    fn encoder_delay_zero_flag() {
        assert!(EncoderDelay::default().is_zero());
        assert!(!EncoderDelay::new(1, 0).is_zero());
    }
}
