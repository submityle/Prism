//! Encoder pre-roll/padding trim calibration for gapless playback, loops, and
//! sample-accurate seeks.
//!
//! Lossy codecs (Opus, AAC, MP3, Vorbis) and some container writers prepend a
//! decoder warm-up region and pad the final packet out to a codec granule, so a
//! raw decode is longer than the authored program and is offset by the
//! pre-roll. [`crate::pcm::EncoderDelay`] *records* those two counts at decode
//! time; this module turns them into the concrete timeline math every later
//! stage needs:
//!
//! - [`trim_encoder_delay`] removes the leading pre-roll and trailing padding
//!   from a decoded [`ConditionedPcm`], yielding the exact program.
//! - [`program_to_raw_frame`] / [`raw_to_program_frame`] convert frame indices
//!   between the raw decoded timeline (what the decoder counts) and the program
//!   timeline (what gameplay and loop metadata count).
//! - [`adjust_loop_points_to_program`] rebases loop points detected on the raw
//!   decode into the trimmed program, so a seamless loop seam stays sample
//!   accurate after trimming.
//! - [`plan_seek`] turns a program-time seek target into a raw decode start plus
//!   a discard count, decoding a codec warm-up window ahead of the target so the
//!   first delivered frame is sample accurate rather than a filter transient.
//!
//! All math is exact integer arithmetic with saturating/clamping edge handling;
//! trimming more than the asset holds collapses to an empty program rather than
//! panicking.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. Encoder
//! delay/padding trimming for gapless playback is a public, widely documented
//! container/codec practice (for example Ogg/Opus pre-skip and iTunes/MP4 gapless
//! metadata); only the idea is borrowed.
//!
//! # Relationship
//!
//! Implements the design-section-42 open item "codec pre-roll/padding trim
//! calibration at loop points and seek". Consumes [`crate::pcm::EncoderDelay`]
//! (recorded by [`crate::codec_tier`] / [`crate::decode`]) and
//! [`crate::loop_point::LoopPoints`], and feeds the gapless/seamless-loop and
//! streaming-seek contracts of design sections 10, 20, and 44.1.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::loop_point::LoopPoints;
use crate::pcm::{ConditionedPcm, EncoderDelay, PcmError};

/// Error returned when applying an encoder-delay trim.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DelayTrimError {
    /// Rebuilding the trimmed container failed its invariants. This is not
    /// expected for a trim of a valid source and is surfaced for completeness.
    Rebuild(PcmError),
}

impl core::fmt::Display for DelayTrimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DelayTrimError::Rebuild(e) => write!(f, "failed to rebuild trimmed PCM: {e:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DelayTrimError {}

/// The resolved number of frames to drop from each end of a decoded asset,
/// already clamped so `leading + trailing` never exceeds the asset length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TrimPlan {
    /// Leading frames to drop (the encoder pre-roll), clamped to the length.
    pub leading: usize,
    /// Trailing frames to drop (the encoder padding), clamped to what remains
    /// after the leading trim so the program start is always preserved.
    pub trailing: usize,
}

impl TrimPlan {
    /// Program frame count that remains after applying this plan to an asset of
    /// `frames` frames.
    #[must_use]
    #[inline]
    pub const fn program_frames(&self, frames: usize) -> usize {
        frames - self.leading - self.trailing
    }
}

/// Resolves an [`EncoderDelay`] against an asset of `frames` frames into a
/// clamped [`TrimPlan`].
///
/// The pre-roll is clamped to the asset length first (so the program never
/// starts past the end), then the padding is clamped to whatever remains. When
/// the recorded delays overlap a short asset, the trailing trim shrinks before
/// the leading one, preserving the authored program onset.
#[must_use]
pub fn resolve_trim(delay: EncoderDelay, frames: usize) -> TrimPlan {
    let leading = (delay.preroll_frames as usize).min(frames);
    let remaining = frames - leading;
    let trailing = (delay.padding_frames as usize).min(remaining);
    TrimPlan { leading, trailing }
}

/// Program frame count remaining after trimming `delay` from an asset of
/// `frames` frames.
#[must_use]
#[inline]
pub fn program_frames(delay: EncoderDelay, frames: usize) -> usize {
    resolve_trim(delay, frames).program_frames(frames)
}

/// Removes the encoder pre-roll and padding from `pcm`, returning the exact
/// program as a new [`ConditionedPcm`].
///
/// Trimming beyond the asset collapses to an empty (zero-frame) program rather
/// than erroring. Sample rate and channel layout are preserved.
///
/// # Errors
///
/// Returns [`DelayTrimError::Rebuild`] only if the trimmed buffers somehow fail
/// the [`ConditionedPcm`] invariants, which cannot happen for a trim of a valid
/// source and is surfaced purely for forward compatibility.
pub fn trim_encoder_delay(
    pcm: &ConditionedPcm,
    delay: EncoderDelay,
) -> Result<ConditionedPcm, DelayTrimError> {
    let frames = pcm.frames();
    let plan = resolve_trim(delay, frames);
    let keep = plan.program_frames(frames);
    let start = plan.leading;
    let end = start + keep;
    let mut channels: Vec<Vec<f32>> = Vec::with_capacity(pcm.channel_count());
    for ch in pcm.channels() {
        channels.push(ch[start..end].to_vec());
    }
    ConditionedPcm::new(pcm.sample_rate(), pcm.layout(), channels).map_err(DelayTrimError::Rebuild)
}

/// Maps a frame index in the program timeline to the raw decoded timeline by
/// adding the encoder pre-roll. Saturates at [`usize::MAX`].
#[must_use]
#[inline]
pub fn program_to_raw_frame(delay: EncoderDelay, program_frame: usize) -> usize {
    program_frame.saturating_add(delay.preroll_frames as usize)
}

/// Maps a frame index in the raw decoded timeline to the program timeline by
/// removing the encoder pre-roll. Returns `None` when the raw frame lies inside
/// the pre-roll region (before the program starts).
#[must_use]
#[inline]
pub fn raw_to_program_frame(delay: EncoderDelay, raw_frame: usize) -> Option<usize> {
    raw_frame.checked_sub(delay.preroll_frames as usize)
}

/// Rebases loop points detected on the raw decoded timeline into the trimmed
/// program timeline.
///
/// Both endpoints shift left by the pre-roll and are clamped into
/// `[0, program_frames]`; the end is kept at or after the start, and the
/// crossfade is clamped to the resulting loop length so it can never read past
/// the seam. `program_frames` is the trimmed length (see [`program_frames`]).
#[must_use]
pub fn adjust_loop_points_to_program(
    points: LoopPoints,
    delay: EncoderDelay,
    program_frames: usize,
) -> LoopPoints {
    let pre = delay.preroll_frames as usize;
    let start = points.start.saturating_sub(pre).min(program_frames);
    let end = points.end.saturating_sub(pre).min(program_frames).max(start);
    let length = end - start;
    let crossfade_frames = points.crossfade_frames.min(length as u32);
    LoopPoints {
        start,
        end,
        crossfade_frames,
        mode: points.mode,
    }
}

/// A sample-accurate seek plan for a streaming decoder: where to start decoding
/// in the raw stream and how many leading frames to discard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SeekPlan {
    /// Frame in the raw decoded timeline at which to begin decoding. This is
    /// `warmup` frames before the target (clamped to the stream start) so the
    /// codec can rebuild its filter state.
    pub decode_start_raw: usize,
    /// Leading frames to discard after decoding resumes, so the first delivered
    /// frame is exactly the requested program target.
    pub discard_frames: usize,
}

/// Plans a sample-accurate seek to `program_target` (program timeline) through a
/// decoder that needs `warmup` frames of look-back to rebuild filter state.
///
/// The raw target is `program_target + pre-roll`; decoding starts `warmup`
/// frames earlier (clamped to the stream start) and the frames between the
/// decode start and the raw target are discarded. The discarded count is the
/// warm-up actually available (`min(warmup, raw_target)`).
#[must_use]
pub fn plan_seek(delay: EncoderDelay, program_target: usize, warmup: usize) -> SeekPlan {
    let raw_target = program_to_raw_frame(delay, program_target);
    let decode_start_raw = raw_target.saturating_sub(warmup);
    let discard_frames = raw_target - decode_start_raw;
    SeekPlan {
        decode_start_raw,
        discard_frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_point::LoopMode;
    use prism_audio_core::buffer::ChannelLayout;

    fn ramp_pcm(frames: usize) -> ConditionedPcm {
        // Two channels where sample k holds (k, -k) so trims are easy to verify.
        let left: Vec<f32> = (0..frames).map(|k| k as f32).collect();
        let right: Vec<f32> = (0..frames).map(|k| -(k as f32)).collect();
        ConditionedPcm::new(48_000, ChannelLayout::Stereo, alloc_vec(left, right)).unwrap()
    }

    #[cfg(feature = "std")]
    fn alloc_vec(l: Vec<f32>, r: Vec<f32>) -> Vec<Vec<f32>> {
        vec![l, r]
    }

    #[cfg(not(feature = "std"))]
    fn alloc_vec(l: Vec<f32>, r: Vec<f32>) -> Vec<Vec<f32>> {
        alloc::vec![l, r]
    }

    #[test]
    fn resolve_trim_clamps_to_length() {
        let plan = resolve_trim(EncoderDelay::new(10, 10), 100);
        assert_eq!(plan.leading, 10);
        assert_eq!(plan.trailing, 10);
        assert_eq!(plan.program_frames(100), 80);
    }

    #[test]
    fn resolve_trim_preserves_onset_when_overlapping() {
        // Pre-roll alone exceeds the asset: leading clamps, trailing goes zero.
        let plan = resolve_trim(EncoderDelay::new(200, 50), 100);
        assert_eq!(plan.leading, 100);
        assert_eq!(plan.trailing, 0);
        assert_eq!(plan.program_frames(100), 0);
    }

    #[test]
    fn resolve_trim_trailing_clamps_to_remaining() {
        let plan = resolve_trim(EncoderDelay::new(60, 60), 100);
        assert_eq!(plan.leading, 60);
        assert_eq!(plan.trailing, 40);
        assert_eq!(plan.program_frames(100), 0);
    }

    #[test]
    fn program_frames_free_fn_matches_plan() {
        assert_eq!(program_frames(EncoderDelay::new(5, 7), 100), 88);
        assert_eq!(program_frames(EncoderDelay::default(), 100), 100);
    }

    #[test]
    fn trim_removes_leading_and_trailing() {
        let pcm = ramp_pcm(100);
        let out = trim_encoder_delay(&pcm, EncoderDelay::new(10, 20)).unwrap();
        assert_eq!(out.frames(), 70);
        assert_eq!(out.sample_rate(), 48_000);
        assert_eq!(out.channel_count(), 2);
        // First kept frame is index 10 of the source.
        assert_eq!(out.channel(0).unwrap()[0], 10.0);
        assert_eq!(out.channel(1).unwrap()[0], -10.0);
        // Last kept frame is index 79 of the source (100 - 20 - 1).
        assert_eq!(out.channel(0).unwrap()[69], 79.0);
    }

    #[test]
    fn trim_zero_delay_is_identity() {
        let pcm = ramp_pcm(32);
        let out = trim_encoder_delay(&pcm, EncoderDelay::default()).unwrap();
        assert_eq!(out.frames(), 32);
        assert_eq!(out.channel(0).unwrap()[0], 0.0);
        assert_eq!(out.channel(0).unwrap()[31], 31.0);
    }

    #[test]
    fn trim_beyond_asset_collapses_to_empty() {
        let pcm = ramp_pcm(16);
        let out = trim_encoder_delay(&pcm, EncoderDelay::new(100, 100)).unwrap();
        assert_eq!(out.frames(), 0);
        assert_eq!(out.channel_count(), 2);
    }

    #[test]
    fn frame_mapping_round_trips_outside_preroll() {
        let d = EncoderDelay::new(312, 0);
        assert_eq!(program_to_raw_frame(d, 0), 312);
        assert_eq!(program_to_raw_frame(d, 1000), 1312);
        assert_eq!(raw_to_program_frame(d, 1312), Some(1000));
        // Inside the pre-roll there is no program frame.
        assert_eq!(raw_to_program_frame(d, 100), None);
        assert_eq!(raw_to_program_frame(d, 312), Some(0));
    }

    #[test]
    fn loop_points_rebase_into_program() {
        let d = EncoderDelay::new(312, 0);
        let raw = LoopPoints {
            start: 1312,
            end: 5312,
            crossfade_frames: 64,
            mode: LoopMode::Forward,
        };
        let prog = adjust_loop_points_to_program(raw, d, 100_000);
        assert_eq!(prog.start, 1000);
        assert_eq!(prog.end, 5000);
        assert_eq!(prog.crossfade_frames, 64);
        assert_eq!(prog.mode, LoopMode::Forward);
    }

    #[test]
    fn loop_points_clamp_and_order() {
        let d = EncoderDelay::new(10, 0);
        let raw = LoopPoints {
            start: 5,
            end: 500,
            crossfade_frames: 1000,
            mode: LoopMode::PingPong,
        };
        // start-10 saturates to 0; end-10=490 clamps to program length 100.
        let prog = adjust_loop_points_to_program(raw, d, 100);
        assert_eq!(prog.start, 0);
        assert_eq!(prog.end, 100);
        // Crossfade clamped to the loop length.
        assert_eq!(prog.crossfade_frames, 100);
        assert_eq!(prog.mode, LoopMode::PingPong);
    }

    #[test]
    fn seek_plan_warms_up_and_discards() {
        let d = EncoderDelay::new(312, 0);
        let plan = plan_seek(d, 10_000, 128);
        // Raw target = 10000 + 312 = 10312; start 128 earlier.
        assert_eq!(plan.decode_start_raw, 10_184);
        assert_eq!(plan.discard_frames, 128);
    }

    #[test]
    fn seek_plan_clamps_warmup_at_stream_start() {
        let d = EncoderDelay::new(0, 0);
        // Target 50 with 128 warm-up: only 50 frames of look-back available.
        let plan = plan_seek(d, 50, 128);
        assert_eq!(plan.decode_start_raw, 0);
        assert_eq!(plan.discard_frames, 50);
    }

    #[test]
    fn seek_to_program_zero_through_preroll() {
        let d = EncoderDelay::new(312, 0);
        let plan = plan_seek(d, 0, 0);
        // No warm-up: start exactly at the raw program onset, discard nothing.
        assert_eq!(plan.decode_start_raw, 312);
        assert_eq!(plan.discard_frames, 0);
    }

    #[test]
    fn error_display_is_non_empty() {
        let e = DelayTrimError::Rebuild(PcmError::ZeroSampleRate);
        #[cfg(feature = "std")]
        {
            assert!(!ToString::to_string(&e).is_empty());
        }
        #[cfg(not(feature = "std"))]
        {
            use alloc::string::ToString;
            assert!(!e.to_string().is_empty());
        }
    }
}
