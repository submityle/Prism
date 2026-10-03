//! Content-based codec-tier classification.
//!
//! The actual encoders live elsewhere; this stage only *recommends* a codec
//! tier from measurable, classic signal descriptors - occupied bandwidth
//! (spectral rolloff), dynamic range (peak-to-`RMS`), and inter-channel
//! correlation - combined with the asset's intended usage. The mapping is a
//! fixed, documented threshold classifier so it is fully deterministic.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the codec-tier recommendation stage of design section 51. The
//! tiers name encoder *families* (lossless `PCM`, `ADPCM`, Vorbis-like,
//! Opus-like, `FLAC`-like lossless); the concrete encoder is selected by the
//! packaging layer, not here.

use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::analysis::spectral_features::SpectralFeatures;
use prism_audio_core::nodes::analysis::spectrum::{SpectrumAnalyzer, Window};

use crate::pcm::ConditionedPcm;

/// `STFT` length used by profiling.
const PROFILE_FFT: usize = 1024;
/// `STFT` hop used by profiling.
const PROFILE_HOP: usize = 512;
/// Rolloff fraction used to estimate occupied bandwidth.
const ROLLOFF_FRACTION: Sample = 0.85;

/// A recommended encoder family (tier), ordered loosely from most to least
/// transparent at a fixed bitrate.
///
/// These are recommendations only; the encoder that realizes the tier lives in
/// the packaging layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum CodecTier {
    /// Uncompressed linear `PCM` (short, latency-critical, or mastering).
    PcmLossless,
    /// Low-cost fixed-rate `ADPCM` (short memory-resident effects).
    Adpcm,
    /// A Vorbis-like transform codec (general music / ambience).
    VorbisLike,
    /// An Opus-like low-latency transform codec (voice / interactive).
    OpusLike,
    /// A `FLAC`-like lossless codec (high-dynamic-range music masters).
    FlacLossless,
}

/// How the asset is used, which biases the tier choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum UsageClass {
    /// Short, latency-sensitive sound effect (the default).
    #[default]
    Sfx,
    /// Full-length musical content.
    Music,
    /// Spoken dialogue.
    Voice,
    /// Long background ambience.
    Ambience,
}

/// Measurable descriptors of an asset's content.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentProfile {
    /// Mean occupied bandwidth (spectral rolloff), in Hz.
    pub bandwidth_hz: Sample,
    /// Dynamic range (sample-peak minus `RMS`), in dB.
    pub dynamic_range_db: Sample,
    /// Inter-channel correlation in `[-1, 1]` (1 for mono).
    pub channel_correlation: Sample,
    /// Program length, in frames.
    pub duration_frames: u64,
}

/// A codec-tier recommendation plus a nominal target bitrate and encoder
/// delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CodecRecommendation {
    /// Recommended encoder family.
    pub tier: CodecTier,
    /// Nominal target bitrate in kbps; `0` means variable / lossless.
    pub target_bitrate_kbps: u32,
    /// Nominal encoder pre-roll (leading warm-up frames) the tier introduces.
    pub preroll_frames: u32,
    /// Nominal encoder padding (trailing frames) the tier introduces.
    pub padding_frames: u32,
}

/// Mixes every channel down to a single mono buffer by averaging.
fn mono_mixdown(pcm: &ConditionedPcm) -> Vec<Sample> {
    let frames = pcm.frames();
    let channels = pcm.channel_count().max(1);
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut acc = 0.0 as Sample;
        for ch in 0..pcm.channel_count() {
            acc += pcm.channel(ch).map_or(0.0, |c| c[frame]);
        }
        mono.push(acc / channels as Sample);
    }
    mono
}

/// Estimates mean occupied bandwidth from the averaged spectral rolloff.
fn mean_bandwidth(mono: &[Sample], sample_rate: u32) -> Sample {
    let mut analyzer = SpectrumAnalyzer::new(PROFILE_FFT, PROFILE_HOP, Window::Hann);
    let mut features = SpectralFeatures::new(ROLLOFF_FRACTION);
    let mut sum = 0.0 as Sample;
    let mut count = 0u32;
    let mut last_frame = 0u64;
    for &x in mono {
        analyzer.feed_sample(x);
        let frames = analyzer.frames_computed();
        if frames != last_frame {
            last_frame = frames;
            let set = features.analyze(analyzer.magnitudes(), sample_rate);
            sum += set.rolloff;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as Sample
    }
}

/// Computes the sample-peak-minus-`RMS` dynamic range of a buffer, in dB.
fn dynamic_range_db(mono: &[Sample]) -> Sample {
    let mut peak = 0.0 as Sample;
    let mut sum_sq = 0.0 as Sample;
    for &x in mono {
        let a = ops::abs(x);
        if a > peak {
            peak = a;
        }
        sum_sq += x * x;
    }
    if mono.is_empty() || peak <= 0.0 {
        return 0.0;
    }
    let rms = ops::sqrt(sum_sq / mono.len() as Sample);
    if rms <= 0.0 {
        return 0.0;
    }
    20.0 * ops::log10(peak / rms)
}

/// Computes the Pearson correlation between the first two channels (1 for
/// mono).
fn inter_channel_correlation(pcm: &ConditionedPcm) -> Sample {
    if pcm.channel_count() < 2 {
        return 1.0;
    }
    let (Some(a), Some(b)) = (pcm.channel(0), pcm.channel(1)) else {
        return 1.0;
    };
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let inv = 1.0 / n as Sample;
    let mean_a: Sample = a[..n].iter().sum::<Sample>() * inv;
    let mean_b: Sample = b[..n].iter().sum::<Sample>() * inv;
    let mut cov = 0.0 as Sample;
    let mut var_a = 0.0 as Sample;
    let mut var_b = 0.0 as Sample;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = ops::sqrt(var_a * var_b);
    if denom <= 1.0e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Builds a [`ContentProfile`] from conditioned `PCM`.
#[must_use]
pub fn profile(pcm: &ConditionedPcm) -> ContentProfile {
    let mono = mono_mixdown(pcm);
    ContentProfile {
        bandwidth_hz: mean_bandwidth(&mono, pcm.sample_rate()),
        dynamic_range_db: dynamic_range_db(&mono),
        channel_correlation: inter_channel_correlation(pcm),
        duration_frames: pcm.frames() as u64,
    }
}

/// Frames below which an asset counts as "short" for tier selection.
const SHORT_FRAMES: u64 = 48_000;
/// Dynamic range (dB) above which lossless masters are preferred for music.
const HIGH_DYNAMIC_RANGE_DB: Sample = 18.0;

/// Recommends a codec tier, bitrate, and nominal encoder delay.
///
/// The mapping is a fixed classifier: short effects favor memory-resident
/// `ADPCM` or lossless `PCM`, voice favors a low-latency Opus-like tier, and
/// music favors a Vorbis-like tier (or a lossless master when its dynamic
/// range is wide).
#[must_use]
pub fn recommend(profile: &ContentProfile, usage: UsageClass) -> CodecRecommendation {
    let short = profile.duration_frames < SHORT_FRAMES;
    let wide_dynamics = profile.dynamic_range_db >= HIGH_DYNAMIC_RANGE_DB;

    let tier = match usage {
        UsageClass::Voice => CodecTier::OpusLike,
        UsageClass::Sfx => {
            if short {
                if wide_dynamics {
                    CodecTier::PcmLossless
                } else {
                    CodecTier::Adpcm
                }
            } else {
                CodecTier::VorbisLike
            }
        }
        UsageClass::Ambience => CodecTier::VorbisLike,
        UsageClass::Music => {
            if wide_dynamics {
                CodecTier::FlacLossless
            } else {
                CodecTier::VorbisLike
            }
        }
    };

    let bandwidth = if profile.bandwidth_hz.is_finite() {
        profile.bandwidth_hz.max(0.0)
    } else {
        0.0
    };
    let effective_rate = bandwidth * 2.0;
    let target_bitrate_kbps = match tier {
        // Lossless tiers carry no fixed target; the encoder is variable.
        CodecTier::PcmLossless | CodecTier::FlacLossless => 0,
        // ADPCM is a fixed 4 bits per sample over the occupied band.
        CodecTier::Adpcm => clamp_kbps(effective_rate * 4.0 / 1000.0, 32, 384),
        CodecTier::VorbisLike => clamp_kbps(bandwidth / 100.0, 96, 320),
        CodecTier::OpusLike => clamp_kbps(bandwidth / 160.0, 32, 256),
    };

    let (preroll_frames, padding_frames) = match tier {
        CodecTier::OpusLike => (312, 0),
        CodecTier::VorbisLike => (PROFILE_FFT as u32 / 2, 0),
        CodecTier::PcmLossless | CodecTier::Adpcm | CodecTier::FlacLossless => (0, 0),
    };

    CodecRecommendation {
        tier,
        target_bitrate_kbps,
        preroll_frames,
        padding_frames,
    }
}

/// Rounds and clamps a kbps estimate into an inclusive integer range.
fn clamp_kbps(value: Sample, lo: u32, hi: u32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return lo;
    }
    let rounded = ops::round(value) as i64;
    let clamped = rounded.clamp(i64::from(lo), i64::from(hi));
    clamped as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::TAU;
    use prism_audio_core::buffer::ChannelLayout;

    fn tone(freq: Sample, rate: u32, frames: usize) -> Vec<Sample> {
        (0..frames)
            .map(|i| 0.5 * ops::sin(TAU * freq * i as Sample / rate as Sample))
            .collect()
    }

    #[test]
    fn short_bright_sfx_and_long_music_differ() {
        let rate = 48_000;
        // Short bright SFX: a brief high-frequency burst.
        let sfx = ConditionedPcm::new(rate, ChannelLayout::Mono, vec![tone(9_000.0, rate, 4_000)])
            .unwrap();
        let sfx_profile = profile(&sfx);
        let sfx_rec = recommend(&sfx_profile, UsageClass::Sfx);

        // Long full-range music clip.
        let music =
            ConditionedPcm::new(rate, ChannelLayout::Mono, vec![tone(2_000.0, rate, 96_000)])
                .unwrap();
        let music_profile = profile(&music);
        let music_rec = recommend(&music_profile, UsageClass::Music);

        assert_ne!(sfx_rec.tier, music_rec.tier);
        assert!(matches!(
            sfx_rec.tier,
            CodecTier::Adpcm | CodecTier::PcmLossless
        ));
        assert!(matches!(
            music_rec.tier,
            CodecTier::VorbisLike | CodecTier::FlacLossless
        ));
    }

    #[test]
    fn mono_correlation_is_unity() {
        let rate = 48_000;
        let pcm =
            ConditionedPcm::new(rate, ChannelLayout::Mono, vec![tone(1_000.0, rate, 2_048)]).unwrap();
        let p = profile(&pcm);
        assert!((p.channel_correlation - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn voice_maps_to_opus_like() {
        let p = ContentProfile {
            bandwidth_hz: 4_000.0,
            dynamic_range_db: 10.0,
            channel_correlation: 1.0,
            duration_frames: 96_000,
        };
        let rec = recommend(&p, UsageClass::Voice);
        assert_eq!(rec.tier, CodecTier::OpusLike);
        assert!(rec.target_bitrate_kbps >= 32);
    }

    #[test]
    fn silent_bandwidth_is_zero() {
        let rate = 48_000;
        let pcm = ConditionedPcm::silence(rate, ChannelLayout::Mono, 4_096).unwrap();
        let p = profile(&pcm);
        assert!(p.bandwidth_hz.abs() < 1.0e-3);
        assert!((p.dynamic_range_db).abs() < 1.0e-6);
    }
}
