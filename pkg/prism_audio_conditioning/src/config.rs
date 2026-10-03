//! Per-stage configuration records and the bundled pipeline configuration.
//!
//! Every stage is a pure function of its input plus a configuration value, so
//! the configuration is part of the content-addressed key: two runs with the
//! same bytes and the same [`ConditioningConfig`] must produce byte-identical
//! artifacts. Each record carries a sensible [`Default`] tuned for 48 kHz game
//! audio.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Collects the stage parameters of design section 51 so [`crate::pipeline`]
//! can thread one configuration through decode, resample, analysis, and
//! authoring.

use prism_audio_assets::codec::PcmSampleFormat;
use prism_audio_core::math::Sample;

use crate::codec_tier::UsageClass;
use crate::loop_point::LoopMode;

/// Hints that disambiguate headerless or ambiguous source bytes.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodeHint {
    /// Sample rate to assume for raw PCM input (ignored for self-describing
    /// containers such as WAV).
    pub sample_rate: Option<u32>,
    /// Channel count to assume for raw PCM input.
    pub channels: Option<u16>,
    /// Sample encoding to assume for raw PCM input.
    pub pcm_format: Option<PcmSampleFormat>,
    /// Known encoder pre-roll (leading warm-up frames) to record on the
    /// decoded asset.
    pub preroll_frames: u32,
    /// Known encoder padding (trailing frames) to record on the decoded asset.
    pub padding_frames: u32,
}

/// Parameters for seamless loop-point detection.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LoopConfig {
    /// Shortest loop length to consider, in frames.
    pub min_period_frames: usize,
    /// Longest loop length to consider, in frames.
    pub max_period_frames: usize,
    /// Half-width of the zero-crossing search around each candidate boundary,
    /// in frames.
    pub search_radius_frames: usize,
    /// Length of the correlation comparison window, in frames.
    pub window_frames: usize,
    /// Equal-power crossfade length written into the detected loop, in frames.
    pub crossfade_frames: u32,
    /// Playback mode recorded on a detected loop.
    pub mode: LoopMode,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            min_period_frames: 256,
            max_period_frames: 48_000,
            search_radius_frames: 128,
            window_frames: 512,
            crossfade_frames: 64,
            mode: LoopMode::Forward,
        }
    }
}

/// Parameters for spectral-flux transient detection.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TransientConfig {
    /// `STFT` analysis length, in samples (rounded up to a power of two).
    pub fft_size: usize,
    /// Hop between successive `STFT` frames, in samples.
    pub hop: usize,
    /// Threshold multiplier applied to the running standard deviation when
    /// forming the adaptive peak-picking threshold.
    pub threshold_k: Sample,
    /// Half-width of the moving-average window used for the adaptive
    /// threshold, in frames.
    pub mean_window_frames: usize,
    /// Minimum spacing enforced between accepted onsets, in frames.
    pub min_separation_frames: usize,
}

impl Default for TransientConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop: 256,
            threshold_k: 1.5,
            mean_window_frames: 8,
            min_separation_frames: 3,
        }
    }
}

/// Parameters for autocorrelation tempo estimation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TempoConfig {
    /// Slowest tempo to consider, in beats per minute.
    pub min_bpm: Sample,
    /// Fastest tempo to consider, in beats per minute.
    pub max_bpm: Sample,
    /// Number of beats per bar used when building a bar grid.
    pub beats_per_bar: u32,
}

impl Default for TempoConfig {
    fn default() -> Self {
        Self {
            min_bpm: 60.0,
            max_bpm: 200.0,
            beats_per_bar: 4,
        }
    }
}

/// Parameters for high-quality offline resampling.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ResampleConfig {
    /// Output block size pumped through the streaming resampler, in frames.
    pub chunk_frames: usize,
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self { chunk_frames: 1024 }
    }
}

/// Parameters for `HRIR` dataset conditioning.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HrtfConditionConfig {
    /// Target sample rate for the conditioned `HRIR` set, in Hz.
    pub target_sample_rate: u32,
    /// Whether to apply diffuse-field equalization.
    pub diffuse_field_eq: bool,
    /// Whether to convert each `HRIR` to its minimum-phase equivalent.
    pub minimum_phase: bool,
}

impl Default for HrtfConditionConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 48_000,
            diffuse_field_eq: true,
            minimum_phase: true,
        }
    }
}

/// Parameters for bank authoring (streaming vs resident partitioning).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BankConfig {
    /// Assets longer than this many frames are placed in the streaming tier.
    pub streaming_threshold_frames: u64,
    /// Prefetch length assigned to streaming entries, in frames.
    pub prefetch_frames: u32,
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            // Roughly five seconds at 48 kHz.
            streaming_threshold_frames: 240_000,
            prefetch_frames: 24_000,
        }
    }
}

/// Parameters for offline lip-sync / viseme analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LipsyncConfig {
    /// `STFT` analysis length, in samples (rounded up to a power of two).
    pub fft_size: usize,
    /// Hop between successive `STFT` frames, in samples.
    pub hop: usize,
    /// Upper bound of the low band, in Hz.
    pub low_band_hz: Sample,
    /// Upper bound of the mid band, in Hz (the high band runs above this).
    pub mid_band_hz: Sample,
    /// One-pole smoothing coefficient for the openness envelope, in `0..=1`.
    pub envelope_smoothing: Sample,
}

impl Default for LipsyncConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop: 256,
            low_band_hz: 500.0,
            mid_band_hz: 2_000.0,
            envelope_smoothing: 0.5,
        }
    }
}

/// The bundled configuration threaded through the whole pipeline.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditioningConfig {
    /// Project sample rate every asset is resampled to, in Hz.
    pub project_sample_rate: u32,
    /// Decode-matrix hints.
    pub decode: DecodeHint,
    /// Loop-detection parameters.
    pub loop_points: LoopConfig,
    /// Transient-detection parameters.
    pub transient: TransientConfig,
    /// Tempo-estimation parameters.
    pub tempo: TempoConfig,
    /// Resampling parameters.
    pub resample: ResampleConfig,
    /// `HRIR` conditioning parameters.
    pub hrtf: HrtfConditionConfig,
    /// Bank-authoring parameters.
    pub bank: BankConfig,
    /// Lip-sync parameters.
    pub lipsync: LipsyncConfig,
    /// Intended usage class, used to bias codec-tier recommendation.
    pub usage: UsageClass,
}

impl Default for ConditioningConfig {
    fn default() -> Self {
        Self {
            project_sample_rate: 48_000,
            decode: DecodeHint::default(),
            loop_points: LoopConfig::default(),
            transient: TransientConfig::default(),
            tempo: TempoConfig::default(),
            resample: ResampleConfig::default(),
            hrtf: HrtfConditionConfig::default(),
            bank: BankConfig::default(),
            lipsync: LipsyncConfig::default(),
            usage: UsageClass::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_defaults_are_ordered() {
        let c = LoopConfig::default();
        assert!(c.min_period_frames < c.max_period_frames);
        assert!(c.window_frames > 0);
    }

    #[test]
    fn transient_defaults_are_sane() {
        let c = TransientConfig::default();
        assert!(c.hop > 0 && c.hop <= c.fft_size);
        assert!(c.threshold_k > 0.0);
    }

    #[test]
    fn tempo_range_is_ordered() {
        let c = TempoConfig::default();
        assert!(c.min_bpm < c.max_bpm);
        assert!(c.beats_per_bar >= 1);
    }

    #[test]
    fn lipsync_bands_are_ordered() {
        let c = LipsyncConfig::default();
        assert!(c.low_band_hz < c.mid_band_hz);
        assert!((0.0..=1.0).contains(&c.envelope_smoothing));
    }

    #[test]
    fn bundle_default_rate_is_project_rate() {
        let c = ConditioningConfig::default();
        assert_eq!(c.project_sample_rate, 48_000);
        assert_eq!(c.hrtf.target_sample_rate, 48_000);
    }
}
