//! End-to-end conditioning job tying every stage into one pure function.
//!
//! [`run`] threads a single [`ConditioningConfig`] through decode, resample,
//! loudness analysis, loop/transient/tempo/marker extraction, codec-tier
//! recommendation, and content hashing, producing a [`ConditionedArtifact`].
//! The job is a deterministic pure function of its bytes and configuration:
//! the same input yields a byte-identical artifact (and an identical content
//! hash) on every run.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The orchestration leg of design section 51; it composes the sibling modules
//! of this crate and reuses `prism_audio_assets`, `prism_audio_resample`, and
//! `prism_audio_core` through them.

use alloc::vec::Vec;

use crate::codec_tier::{self, CodecRecommendation};
use crate::config::ConditioningConfig;
use crate::content_hash::{ContentHash, Hasher};
use crate::dc_block;
use crate::decode::{self, DecodeError, SourceFormat};
use crate::finalize::{self, FinalizeError};
use crate::loop_point::{self, LoopPoints};
use crate::loudness_offline::{self, LoudnessStats};
use crate::loudness_normalize;
use crate::marker::MarkerTimeline;
use crate::pcm::{ConditionedPcm, EncoderDelay};
use prism_audio_core::math::Sample;
use crate::resample_offline;
use crate::tempo::{self, TempoEstimate};
use crate::transient;

/// Errors returned by [`run`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PipelineError {
    /// The decode/import stage failed.
    Decode(DecodeError),
    /// The program-finalization stage failed.
    Finalize(FinalizeError),
}

impl From<DecodeError> for PipelineError {
    fn from(error: DecodeError) -> Self {
        PipelineError::Decode(error)
    }
}

impl From<FinalizeError> for PipelineError {
    fn from(error: FinalizeError) -> Self {
        PipelineError::Finalize(error)
    }
}

/// The read-only, content-addressed result of a conditioning job.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditionedArtifact {
    /// Decoded, resampled canonical `PCM` at the project rate.
    pub pcm: ConditionedPcm,
    /// Encoder pre-roll/padding recorded at decode time.
    pub encoder_delay: EncoderDelay,
    /// DC offset removed per channel by the source-hygiene stage (`0`
    /// per channel when DC removal is disabled).
    pub removed_dc: Vec<Sample>,
    /// Loudness statistics (`BS.1770`).
    pub loudness: LoudnessStats,
    /// Detected seamless loop, when one was found.
    pub loop_points: Option<LoopPoints>,
    /// Estimated tempo and beat period.
    pub tempo: TempoEstimate,
    /// Combined transient / beat / bar marker timeline.
    pub markers: MarkerTimeline,
    /// Recommended codec tier and bitrate.
    pub codec: CodecRecommendation,
    /// Deterministic content hash of the conditioned audio and configuration.
    pub hash: ContentHash,
    /// Scalar gain applied by loudness normalization, in dB (`0` when the
    /// stage is disabled or the program is silent).
    pub applied_loudness_gain_db: Sample,
}

/// A reusable description of one conditioning job.
///
/// The job holds no mutable state; it exists so callers can name a configured
/// pipeline and run it over many inputs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConditioningJob {
    /// Configuration threaded through every stage.
    pub config: ConditioningConfig,
}

impl ConditioningJob {
    /// Creates a job with the given configuration.
    #[must_use]
    pub fn new(config: ConditioningConfig) -> Self {
        Self { config }
    }

    /// Runs the job over one source byte buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Decode`] when the import/decode stage fails.
    pub fn run(
        &self,
        bytes: &[u8],
        format: SourceFormat,
    ) -> Result<ConditionedArtifact, PipelineError> {
        run(bytes, format, &self.config)
    }
}

/// Folds the conditioned audio and the salient configuration into a hash.
fn hash_artifact(pcm: &ConditionedPcm, config: &ConditioningConfig) -> ContentHash {
    let mut hasher = Hasher::new();
    hasher.write_u32(pcm.sample_rate());
    hasher.write_u32(pcm.channel_count() as u32);
    hasher.write_u32(pcm.frames() as u32);
    for ch in 0..pcm.channel_count() {
        if let Some(samples) = pcm.channel(ch) {
            for &sample in samples {
                hasher.write_f32(sample);
            }
        }
    }
    hasher.write_u32(config.project_sample_rate);
    hasher.finish()
}

/// Runs the full conditioning pipeline over `bytes`.
///
/// Stages: decode -> resample to the project rate -> loudness analysis ->
/// loop/transient/tempo/marker extraction -> codec-tier recommendation ->
/// optional program finalization -> content hash. Analysis stages describe
/// the decoded program; the config-gated finalization stage (see
/// [`crate::finalize`], default off) then produces the delivered `PCM` and
/// rebased loop points that the content hash keys. The result is
/// deterministic for a fixed `bytes` and `config`.
///
/// # Errors
///
/// Returns [`PipelineError::Decode`] when the import/decode stage fails.
pub fn run(
    bytes: &[u8],
    format: SourceFormat,
    config: &ConditioningConfig,
) -> Result<ConditionedArtifact, PipelineError> {
    let (decoded, encoder_delay) = decode::decode(bytes, format, &config.decode)?;
    let pcm = resample_offline::resample_to(&decoded, config.project_sample_rate, &config.resample);

    // Source hygiene runs before analysis so loudness, loop, transient, and
    // tempo all describe the DC-scrubbed program the engine will play.
    let dc_blocked = dc_block::apply(&pcm, &config.dc_block);
    let pcm = dc_blocked.pcm;
    let removed_dc = dc_blocked.removed_dc;

    let loudness = loudness_offline::analyze(&pcm);

    let analysis_channel = pcm.channel(0).unwrap_or(&[]);
    let loop_points = loop_point::detect(analysis_channel, &config.loop_points);

    let onsets = transient::detect_onsets(analysis_channel, pcm.sample_rate(), &config.transient);
    let envelope = transient::onset_envelope(analysis_channel, &config.transient);
    let hop = config.transient.hop.max(1);
    let tempo = tempo::estimate(&envelope, hop, pcm.sample_rate(), &config.tempo);

    let beat_frames = tempo::beat_grid(envelope.len(), tempo.beat_period_frames, 0.0);
    let beat_samples: Vec<usize> = beat_frames.iter().map(|&f| f * hop).collect();
    let markers = MarkerTimeline::build(&onsets, &beat_samples, config.tempo.beats_per_bar);

    let profile = codec_tier::profile(&pcm);
    let codec = codec_tier::recommend(&profile, config.usage);

    let (pcm, loop_points) = if config.finalize.is_enabled() {
        let finalized =
            finalize::finalize(&pcm, encoder_delay, loop_points, &config.finalize)?;
        (finalized.pcm, finalized.loop_points)
    } else {
        (pcm, loop_points)
    };

    // Loudness normalization scales the delivered program toward a target under
    // a true-peak ceiling. It uses the loudness measured on the pre-finalize
    // program (encoder-delay trim and loop-seam crossfade are near-neutral in
    // integrated loudness), while enforcing the ceiling against the finalized
    // program's own peak so the no-clip guarantee always holds.
    let (pcm, applied_loudness_gain_db) = if config.loudness_normalize.is_enabled() {
        let normalized = loudness_normalize::normalize(&pcm, loudness, &config.loudness_normalize);
        (normalized.pcm, normalized.applied_gain_db)
    } else {
        (pcm, 0.0)
    };

    let hash = hash_artifact(&pcm, config);

    Ok(ConditionedArtifact {
        pcm,
        encoder_delay,
        removed_dc,
        loudness,
        loop_points,
        tempo,
        markers,
        codec,
        hash,
        applied_loudness_gain_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use bevy_math::ops;
    use core::f32::consts::TAU;

    /// Builds a minimal canonical WAV file holding 16-bit stereo `PCM`.
    fn build_wav(sample_rate: u32, data: &[u8]) -> Vec<u8> {
        let channels: u16 = 2;
        let bits: u16 = 16;
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    fn tone_wav(sample_rate: u32, freq: Sample, frames: usize) -> Vec<u8> {
        let mut data = Vec::new();
        for n in 0..frames {
            let x = 0.5 * ops::sin(TAU * freq * n as Sample / sample_rate as Sample);
            let q = (x * 32_767.0) as i16;
            data.extend_from_slice(&q.to_le_bytes());
            data.extend_from_slice(&q.to_le_bytes());
        }
        build_wav(sample_rate, &data)
    }

    use prism_audio_core::math::Sample;

    #[test]
    fn runs_end_to_end_and_is_deterministic() {
        let wav = tone_wav(48_000, 440.0, 48_000);
        let config = ConditioningConfig::default();

        let a = run(&wav, SourceFormat::Wav, &config).unwrap();
        let b = run(&wav, SourceFormat::Wav, &config).unwrap();

        assert_eq!(a.hash, b.hash);
        assert_eq!(a.pcm.frames(), b.pcm.frames());
        assert_eq!(a.pcm.sample_rate(), 48_000);
        assert_eq!(a.codec.tier, b.codec.tier);
        assert_eq!(a.markers.len(), b.markers.len());
        assert!((a.loudness.integrated_lufs - b.loudness.integrated_lufs).abs() < 1.0e-6);
        assert!((a.tempo.bpm - b.tempo.bpm).abs() < 1.0e-6);
    }

    #[test]
    fn different_bytes_change_the_hash() {
        let config = ConditioningConfig::default();
        let a = run(&tone_wav(48_000, 440.0, 4_000), SourceFormat::Wav, &config).unwrap();
        let b = run(&tone_wav(48_000, 660.0, 4_000), SourceFormat::Wav, &config).unwrap();
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn finalize_trims_program_when_enabled() {
        let frames = 48_000;
        let wav = tone_wav(48_000, 440.0, frames);
        let mut config = ConditioningConfig::default();
        config.decode.preroll_frames = 100;
        config.decode.padding_frames = 50;

        let raw = run(&wav, SourceFormat::Wav, &config).unwrap();
        assert_eq!(raw.pcm.frames(), frames);
        assert_eq!(raw.encoder_delay.preroll_frames, 100);

        config.finalize.trim_encoder_delay = true;
        let trimmed = run(&wav, SourceFormat::Wav, &config).unwrap();
        assert_eq!(trimmed.pcm.frames(), frames - 150);
        // Trimming changes the delivered program, so the content hash moves.
        assert_ne!(raw.hash, trimmed.hash);
        // Finalization is deterministic.
        let again = run(&wav, SourceFormat::Wav, &config).unwrap();
        assert_eq!(trimmed.hash, again.hash);
        assert_eq!(trimmed.pcm.frames(), again.pcm.frames());
    }

    #[test]
    fn finalize_disabled_matches_raw_program() {
        let wav = tone_wav(48_000, 440.0, 8_000);
        let mut config = ConditioningConfig::default();
        config.decode.preroll_frames = 32;
        config.decode.padding_frames = 16;

        let a = run(&wav, SourceFormat::Wav, &config).unwrap();
        // Default finalize is off, so the delivered program keeps every frame.
        assert_eq!(a.pcm.frames(), 8_000);
    }

    #[test]
    fn decode_failure_is_reported() {
        let config = ConditioningConfig::default();
        let err = run(&[0u8; 8], SourceFormat::Wav, &config);
        assert!(matches!(err, Err(PipelineError::Decode(_))));
    }

    #[test]
    fn dc_block_disabled_reports_zero_offset() {
        let wav = tone_wav(48_000, 440.0, 8_000);
        let out = run(&wav, SourceFormat::Wav, &ConditioningConfig::default()).unwrap();
        // Default source hygiene is off: one zero per channel, program intact.
        assert_eq!(out.removed_dc, alloc::vec![0.0, 0.0]);
    }

    #[test]
    fn dc_block_removal_moves_hash_and_reports_offset() {
        // Bias every sample by +0.1 so the stage has real DC to subtract.
        let frames = 48_000usize;
        let mut data = Vec::new();
        for n in 0..frames {
            let x = 0.1 + 0.4 * ops::sin(TAU * 440.0 * n as Sample / 48_000.0);
            let q = (x * 32_767.0) as i16;
            data.extend_from_slice(&q.to_le_bytes());
            data.extend_from_slice(&q.to_le_bytes());
        }
        let wav = build_wav(48_000, &data);

        let raw = run(&wav, SourceFormat::Wav, &ConditioningConfig::default()).unwrap();
        assert_eq!(raw.removed_dc, alloc::vec![0.0, 0.0]);

        let mut config = ConditioningConfig::default();
        config.dc_block.remove_dc_offset = true;
        let scrubbed = run(&wav, SourceFormat::Wav, &config).unwrap();

        // A positive DC offset near +0.1 was detected and removed per channel.
        assert!((scrubbed.removed_dc[0] - 0.1).abs() < 5.0e-3, "{}", scrubbed.removed_dc[0]);
        // Scrubbing the delivered program moves the content hash.
        assert_ne!(raw.hash, scrubbed.hash);
        assert_eq!(raw.pcm.frames(), scrubbed.pcm.frames());

        // Deterministic.
        let again = run(&wav, SourceFormat::Wav, &config).unwrap();
        assert_eq!(scrubbed.hash, again.hash);
    }

    #[test]
    fn loudness_normalize_disabled_keeps_gain_zero() {
        let wav = tone_wav(48_000, 440.0, 8_000);
        let config = ConditioningConfig::default();
        let out = run(&wav, SourceFormat::Wav, &config).unwrap();
        // Default normalization is off: no gain and the raw program survives.
        assert_eq!(out.applied_loudness_gain_db, 0.0);
    }

    #[test]
    fn loudness_normalize_scales_program_when_enabled() {
        let wav = tone_wav(48_000, 440.0, 48_000);
        let raw = run(&wav, SourceFormat::Wav, &ConditioningConfig::default()).unwrap();

        let mut config = ConditioningConfig::default();
        config.loudness_normalize.enabled = true;
        // A very low target forces a clear attenuation (ceiling never binds).
        config.loudness_normalize.target_lufs = -40.0;
        let normalized = run(&wav, SourceFormat::Wav, &config).unwrap();

        // A gain was applied and the delivered bytes (and hash) moved.
        assert!(normalized.applied_loudness_gain_db < 0.0);
        assert_ne!(raw.hash, normalized.hash);
        assert_eq!(raw.pcm.frames(), normalized.pcm.frames());

        // Normalization is deterministic.
        let again = run(&wav, SourceFormat::Wav, &config).unwrap();
        assert_eq!(normalized.hash, again.hash);
        assert_eq!(normalized.applied_loudness_gain_db, again.applied_loudness_gain_db);
    }
}
