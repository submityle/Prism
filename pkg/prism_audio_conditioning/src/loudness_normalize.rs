//! Offline loudness normalization: scales a conditioned program to a target
//! integrated loudness under a true-peak ceiling.
//!
//! AAA delivery pipelines measure a program's integrated loudness once at bake
//! time and then apply a single, program-wide gain so every asset arrives at a
//! consistent perceived level (dialogue anchoring, music beds, one-shots). This
//! module is the *apply* dual of [`crate::loudness_offline`], which only
//! *measures*: given the measured [`LoudnessStats`] and a
//! [`LoudnessNormalizeConfig`], it computes one scalar gain and bakes it into
//! the delivered [`ConditionedPcm`].
//!
//! The gain is **peak-limited**: the loudness-driven gain is first clamped to a
//! configurable maximum boost (so near-silent material is not amplified without
//! bound), then reduced if necessary so the program's post-gain peak stays at
//! or below a true-peak ceiling (so normalization can never introduce
//! clipping). The ceiling is enforced against the larger of the program's own
//! sample peak (recomputed deterministically from the exact buffer being
//! scaled) and the measured true-peak estimate, so the guarantee holds even
//! when earlier delivery stages (encoder-delay trim, loop-seam crossfade) have
//! slightly reshaped the program since loudness was measured.
//!
//! The stage is a pure, deterministic function and defaults to disabled, so the
//! delivered program is byte-identical to its input unless a caller opts in.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! "measure integrated loudness, then apply one peak-limited gain to a target"
//! idea is the public ITU-R BS.1770 / EBU R128 / ATSC A/85 delivery practice;
//! only the idea is borrowed, not any implementation.
//!
//! # Relationship
//! Part of the delivery leg of design section 51 and the offline dual of the
//! runtime loudness treatment in section 13 and the platform loudness-delivery
//! targets of section 48.4. It consumes [`LoudnessStats`] produced by
//! [`crate::loudness_offline`] and is applied by [`crate::pipeline`] after the
//! [`crate::finalize`] stage, before content hashing.

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::loudness_offline::LoudnessStats;
use crate::pcm::ConditionedPcm;

/// Default target integrated loudness, in `LUFS`.
///
/// `-16 LUFS` is a common interactive-game delivery target; callers targeting
/// broadcast (`EBU R128` `-23`, `ATSC A/85` `-24`) should override it.
pub const DEFAULT_TARGET_LUFS: Sample = -16.0;

/// Default true-peak ceiling the post-gain program must not exceed, in `dBFS`.
///
/// `-1 dBFS` leaves headroom for inter-sample peaks and downstream lossy
/// encoding, matching the common `-1 dBTP` delivery ceiling.
pub const DEFAULT_TRUE_PEAK_CEILING_DBFS: Sample = -1.0;

/// Default maximum positive (boost) gain applied by normalization, in `dB`.
///
/// Caps how far quiet material is amplified toward the target, so a nearly
/// silent asset is not boosted without bound.
pub const DEFAULT_MAX_GAIN_DB: Sample = 12.0;

/// Measured integrated-loudness floor, in `LUFS`, below which the program is
/// treated as silent and left untouched.
///
/// Matches the `BS.1770` absolute gate at `-70 LUFS`: a program this quiet (or
/// `f32::NEG_INFINITY` for digital silence) carries no meaningful loudness to
/// normalize toward.
pub const SILENCE_FLOOR_LUFS: Sample = -70.0;

/// Configuration for the loudness-normalization stage.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LoudnessNormalizeConfig {
    /// Whether to apply loudness normalization. When `false` the program is
    /// returned unchanged with a reported gain of `0 dB`.
    pub enabled: bool,
    /// Target integrated loudness, in `LUFS`.
    pub target_lufs: Sample,
    /// True-peak ceiling the post-gain program must not exceed, in `dBFS`.
    pub true_peak_ceiling_dbfs: Sample,
    /// Maximum positive (boost) gain applied toward the target, in `dB`.
    pub max_gain_db: Sample,
}

impl Default for LoudnessNormalizeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_lufs: DEFAULT_TARGET_LUFS,
            true_peak_ceiling_dbfs: DEFAULT_TRUE_PEAK_CEILING_DBFS,
            max_gain_db: DEFAULT_MAX_GAIN_DB,
        }
    }
}

impl LoudnessNormalizeConfig {
    /// Returns whether the stage is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// A loudness-normalized program plus the gain that was applied.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalizedProgram {
    /// The delivered program after the normalization gain was baked in.
    pub pcm: ConditionedPcm,
    /// The scalar gain actually applied, in `dB` (`0 dB` means untouched).
    pub applied_gain_db: Sample,
}

/// Converts a decibel value to a linear amplitude ratio.
fn db_to_linear(db: Sample) -> Sample {
    ops::powf(10.0, db / 20.0)
}

/// Converts a positive linear amplitude ratio to decibels.
///
/// A non-positive ratio maps to `0 dB`, which only arises for the untouched
/// (unity-gain) paths that short-circuit before this helper is reached.
fn linear_to_db(linear: Sample) -> Sample {
    if linear > 0.0 {
        20.0 * ops::log10(linear)
    } else {
        0.0
    }
}

/// Returns the maximum absolute sample value across every channel of `pcm`.
fn program_peak_linear(pcm: &ConditionedPcm) -> Sample {
    let mut peak = 0.0;
    for ch in 0..pcm.channel_count() {
        if let Some(samples) = pcm.channel(ch) {
            for &sample in samples {
                let magnitude = ops::abs(sample);
                if magnitude > peak {
                    peak = magnitude;
                }
            }
        }
    }
    peak
}

/// Scales every sample of `pcm` by `gain`, returning the modified program.
fn apply_gain(mut pcm: ConditionedPcm, gain: Sample) -> ConditionedPcm {
    for ch in 0..pcm.channel_count() {
        if let Some(samples) = pcm.channel_mut(ch) {
            for sample in samples.iter_mut() {
                *sample *= gain;
            }
        }
    }
    pcm
}


/// Normalizes `pcm` toward `config.target_lufs` under the configured true-peak
/// ceiling, using `measured` loudness statistics.
///
/// The returned program is byte-identical to `pcm` (with a reported gain of
/// `0 dB`) when the stage is disabled, when the measured integrated loudness is
/// non-finite, or when it sits at or below [`SILENCE_FLOOR_LUFS`]. Otherwise a
/// single scalar gain is computed as `target - measured`, clamped to
/// `max_gain_db` on the boost side, then reduced if necessary so the post-gain
/// peak stays at or below the ceiling, and finally baked into every sample.
#[must_use]
pub fn normalize(
    pcm: &ConditionedPcm,
    measured: LoudnessStats,
    config: &LoudnessNormalizeConfig,
) -> NormalizedProgram {
    // Disabled, non-finite, or silent programs pass through untouched so the
    // delivered bytes (and the content hash keyed off them) are unchanged.
    if !config.is_enabled()
        || !measured.integrated_lufs.is_finite()
        || measured.integrated_lufs <= SILENCE_FLOOR_LUFS
    {
        return NormalizedProgram {
            pcm: pcm.clone(),
            applied_gain_db: 0.0,
        };
    }

    // Loudness-driven gain, clamped on the boost side so quiet material is not
    // amplified without bound. Attenuation is left unbounded; cutting level can
    // never clip.
    let mut gain_db = config.target_lufs - measured.integrated_lufs;
    if gain_db > config.max_gain_db {
        gain_db = config.max_gain_db;
    }
    let mut gain_linear = db_to_linear(gain_db);

    // Enforce the ceiling against the larger of the program's own sample peak
    // and the measured true-peak estimate, so the no-clip guarantee holds even
    // if earlier delivery stages reshaped the program since measurement.
    let mut effective_peak = program_peak_linear(pcm);
    if measured.true_peak_dbtp.is_finite() {
        let measured_peak = db_to_linear(measured.true_peak_dbtp);
        if measured_peak > effective_peak {
            effective_peak = measured_peak;
        }
    }
    if effective_peak > 0.0 {
        let ceiling_linear = db_to_linear(config.true_peak_ceiling_dbfs);
        let max_gain_linear = ceiling_linear / effective_peak;
        if gain_linear > max_gain_linear {
            gain_linear = max_gain_linear;
        }
    }

    NormalizedProgram {
        pcm: apply_gain(pcm.clone(), gain_linear),
        applied_gain_db: linear_to_db(gain_linear),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use prism_audio_core::buffer::ChannelLayout;

    /// Builds a mono program whose single sample value is `amplitude`, repeated
    /// `frames` times, at 48 kHz.
    fn mono_program(amplitude: Sample, frames: usize) -> ConditionedPcm {
        let channel: Vec<Sample> = (0..frames).map(|_| amplitude).collect();
        ConditionedPcm::new(48_000, ChannelLayout::Mono, vec![channel]).unwrap()
    }

    /// Builds `LoudnessStats` with the given integrated loudness and true peak,
    /// leaving the unused summary fields at neutral values.
    fn stats(integrated_lufs: Sample, true_peak_dbtp: Sample) -> LoudnessStats {
        LoudnessStats {
            integrated_lufs,
            true_peak_dbtp,
            loudness_range_lu: 0.0,
            sample_peak_dbfs: true_peak_dbtp,
        }
    }

    fn peak_of(pcm: &ConditionedPcm) -> Sample {
        program_peak_linear(pcm)
    }

    #[test]
    fn disabled_is_identity() {
        let pcm = mono_program(0.5, 16);
        let config = LoudnessNormalizeConfig::default();
        let out = normalize(&pcm, stats(-20.0, -6.0), &config);
        assert_eq!(out.applied_gain_db, 0.0);
        assert_eq!(out.pcm, pcm);
    }

    #[test]
    fn silent_program_is_untouched() {
        let pcm = mono_program(0.0, 16);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(Sample::NEG_INFINITY, Sample::NEG_INFINITY), &config);
        assert_eq!(out.applied_gain_db, 0.0);
        assert_eq!(out.pcm, pcm);
    }

    #[test]
    fn non_finite_measured_is_identity() {
        let pcm = mono_program(0.3, 8);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(Sample::NAN, -6.0), &config);
        assert_eq!(out.applied_gain_db, 0.0);
        assert_eq!(out.pcm, pcm);
    }

    #[test]
    fn below_silence_floor_is_untouched() {
        let pcm = mono_program(0.001, 8);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-80.0, -60.0), &config);
        assert_eq!(out.applied_gain_db, 0.0);
        assert_eq!(out.pcm, pcm);
    }

    #[test]
    fn boost_toward_target() {
        // Quiet program with plenty of headroom; +6 dB stays under the ceiling.
        let pcm = mono_program(0.1, 16);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -24.0,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-30.0, -20.0), &config);
        assert!((out.applied_gain_db - 6.0).abs() < 1.0e-3, "{}", out.applied_gain_db);
        assert!(peak_of(&out.pcm) > peak_of(&pcm));
    }

    #[test]
    fn attenuate_toward_target() {
        let pcm = mono_program(0.5, 16);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -24.0,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-18.0, -6.0), &config);
        assert!((out.applied_gain_db - (-6.0)).abs() < 1.0e-3, "{}", out.applied_gain_db);
        assert!(peak_of(&out.pcm) < peak_of(&pcm));
    }

    #[test]
    fn peak_ceiling_limits_gain() {
        // Loud program: the loudness target would boost it, but the ceiling
        // forces a net attenuation so the post-gain peak stays at the ceiling.
        let pcm = mono_program(0.9, 32);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -16.0,
            true_peak_ceiling_dbfs: -1.0,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-30.0, linear_to_db(0.9)), &config);
        let ceiling = db_to_linear(-1.0);
        assert!(peak_of(&out.pcm) <= ceiling + 1.0e-4, "{}", peak_of(&out.pcm));
        assert!(out.applied_gain_db < 0.0, "{}", out.applied_gain_db);
    }

    #[test]
    fn max_gain_caps_boost() {
        // Extremely quiet (but above the silence floor); the +44 dB the target
        // demands is capped to the 12 dB maximum, and headroom keeps it there.
        let pcm = mono_program(0.001, 16);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -16.0,
            max_gain_db: 12.0,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-60.0, -60.0), &config);
        assert!((out.applied_gain_db - 12.0).abs() < 1.0e-3, "{}", out.applied_gain_db);
    }

    #[test]
    fn applied_gain_matches_peak_scaling() {
        let pcm = mono_program(0.2, 16);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -20.0,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-26.0, -14.0), &config);
        let expected = 0.2 * db_to_linear(out.applied_gain_db);
        assert!((peak_of(&out.pcm) - expected).abs() < 1.0e-5, "{}", peak_of(&out.pcm));
    }

    #[test]
    fn is_deterministic() {
        let pcm = mono_program(0.3, 64);
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -18.0,
            ..LoudnessNormalizeConfig::default()
        };
        let a = normalize(&pcm, stats(-24.0, -10.0), &config);
        let b = normalize(&pcm, stats(-24.0, -10.0), &config);
        assert_eq!(a, b);
    }

    #[test]
    fn stereo_channels_scaled_equally() {
        let left: Vec<Sample> = (0..8).map(|_| 0.2).collect();
        let right: Vec<Sample> = (0..8).map(|_| 0.4).collect();
        let pcm = ConditionedPcm::new(48_000, ChannelLayout::Stereo, vec![left, right]).unwrap();
        let config = LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -20.0,
            ..LoudnessNormalizeConfig::default()
        };
        let out = normalize(&pcm, stats(-26.0, -8.0), &config);
        let gain = db_to_linear(out.applied_gain_db);
        assert!((out.pcm.channel(0).unwrap()[0] - 0.2 * gain).abs() < 1.0e-6);
        assert!((out.pcm.channel(1).unwrap()[0] - 0.4 * gain).abs() < 1.0e-6);
    }
}
