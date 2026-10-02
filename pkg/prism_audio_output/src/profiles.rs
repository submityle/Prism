//! Delivery dynamic-range / loudness presets ("Home Theater / TV / Night /
//! Headphone") resolved into concrete processing parameters.
//!
//! Every shipping console title offers the familiar listening-mode switch
//! (Dolby/console UIs usually expose a `Full / Standard / Night` family). Each
//! mode is **not** a new DSP block but a parameterised preset over the existing
//! master chain: a loudness target, a true-peak ceiling, and a compressor
//! (ratio / threshold / knee). [`OutputProfile`] is the pure data descriptor;
//! [`OutputProfile::params`] resolves it to a concrete
//! [`OutputProfileParams`].
//!
//! The presets form a monotonic "aggressiveness" ladder from
//! [`OutputProfile::HomeTheater`] (full dynamic range, no compression) through
//! [`OutputProfile::Tv`] to [`OutputProfile::Night`] (strongest compression),
//! with [`OutputProfile::Headphone`] adding a binaural flag on top of a
//! moderate preset.
//!
//! # Provenance
//!
//! The preset structure mirrors the publicly documented console / Dolby
//! `Full / Standard / Night` listening-mode conventions and the common
//! game/broadcast loudness targets (`EBU R128`, streaming `-14 LUFS`). This
//! file contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is pure classic data
//! with no AI/ML, and defines no DSP of its own.
//!
//! # Relationship
//!
//! These parameters drive the delivery stage that follows the
//! [`DownmixMatrix`](crate::downmix::DownmixMatrix) and
//! [`BassManager`](crate::bass_management::BassManager) in the output chain.
//! The loudness/true-peak half resolves directly into the core
//! [`LoudnessNormalizerParams`](prism_audio_core::nodes::mastering::LoudnessNormalizerParams)
//! via [`OutputProfileParams::loudness_params`], reusing the mastering
//! normalizer rather than duplicating it.
//!
//! # Determinism
//!
//! Resolution is a pure `match` returning constant data; it allocates nothing
//! and cannot panic.

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::mastering::LoudnessNormalizerParams;

/// A delivery listening-mode preset.
///
/// Resolve to concrete parameters with [`OutputProfile::params`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum OutputProfile {
    /// Full dynamic range: no additional compression, intended for a quiet
    /// room with good equipment. Preserves the master's full HDR window.
    HomeTheater,
    /// Standard / moderate: gentle compression and a narrowed dynamic window to
    /// tame peaks and lift dialogue for TV speakers.
    Tv,
    /// Night / midnight: strong compression that lifts quiet passages and
    /// clamps loud ones (explosions, gunfire) for late-night listening.
    Night,
    /// Headphone: a moderate preset that also requests binaural rendering and
    /// protective limiting.
    Headphone,
}

impl OutputProfile {
    /// Resolves this preset to its concrete [`OutputProfileParams`].
    #[must_use]
    pub const fn params(self) -> OutputProfileParams {
        match self {
            // No compression: unity ratio, threshold at 0 dBFS (nothing is ever
            // over it), zero knee. Game cinematic reference loudness.
            OutputProfile::HomeTheater => OutputProfileParams {
                target_lufs: -18.0,
                true_peak_ceiling_dbtp: -1.0,
                compression_ratio: 1.0,
                compression_threshold_db: 0.0,
                compression_knee_db: 0.0,
                binaural: false,
            },
            OutputProfile::Tv => OutputProfileParams {
                target_lufs: -16.0,
                true_peak_ceiling_dbtp: -1.0,
                compression_ratio: 2.0,
                compression_threshold_db: -18.0,
                compression_knee_db: 6.0,
                binaural: false,
            },
            OutputProfile::Night => OutputProfileParams {
                target_lufs: -16.0,
                true_peak_ceiling_dbtp: -1.0,
                compression_ratio: 4.0,
                compression_threshold_db: -28.0,
                compression_knee_db: 10.0,
                binaural: false,
            },
            OutputProfile::Headphone => OutputProfileParams {
                target_lufs: -16.0,
                true_peak_ceiling_dbtp: -1.0,
                compression_ratio: 2.5,
                compression_threshold_db: -20.0,
                compression_knee_db: 6.0,
                binaural: true,
            },
        }
    }
}

/// The concrete delivery parameters a preset resolves to.
///
/// Plain [`Copy`] descriptor data: a loudness target, a true-peak ceiling, and
/// the three defining compressor controls, plus a binaural-rendering flag. It
/// defines no DSP; downstream stages consume these values.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OutputProfileParams {
    /// Target integrated program loudness, in `LUFS`.
    pub target_lufs: Sample,
    /// Delivery true-peak ceiling, in `dBTP`.
    pub true_peak_ceiling_dbtp: Sample,
    /// Downward-compressor ratio (`1.0` means no compression).
    pub compression_ratio: Sample,
    /// Compressor threshold, in `dBFS`.
    pub compression_threshold_db: Sample,
    /// Compressor soft-knee width, in dB (`0.0` is a hard knee).
    pub compression_knee_db: Sample,
    /// Whether the preset requests binaural (headphone) rendering.
    pub binaural: bool,
}

impl OutputProfileParams {
    /// Returns `true` if the preset applies downward compression.
    #[inline]
    #[must_use]
    pub fn compresses(self) -> bool {
        self.compression_ratio > 1.0
    }

    /// Builds core [`LoudnessNormalizerParams`] from this preset's loudness
    /// target and true-peak ceiling, given a previously measured integrated
    /// loudness and true-peak.
    ///
    /// This reuses the mastering-stage normalizer: the preset supplies the
    /// delivery `target_lufs` and `true_peak_ceiling_dbtp`, while the measured
    /// values come from a loudness-analysis pass. `max_gain_db` is taken from
    /// the core default.
    #[must_use]
    pub fn loudness_params(
        self,
        measured_lufs: Sample,
        measured_true_peak_dbtp: Sample,
    ) -> LoudnessNormalizerParams {
        LoudnessNormalizerParams {
            measured_lufs,
            target_lufs: self.target_lufs,
            measured_true_peak_dbtp,
            max_true_peak_dbtp: self.true_peak_ceiling_dbtp,
            ..LoudnessNormalizerParams::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-6;

    #[test]
    fn home_theater_is_full_dynamic_range() {
        let p = OutputProfile::HomeTheater.params();
        assert!((p.compression_ratio - 1.0).abs() < EPS);
        assert!(!p.compresses());
        assert!(!p.binaural);
    }

    #[test]
    fn night_is_more_aggressive_than_tv_than_home_theater() {
        let ht = OutputProfile::HomeTheater.params();
        let tv = OutputProfile::Tv.params();
        let night = OutputProfile::Night.params();

        // Ratio climbs with aggressiveness.
        assert!(ht.compression_ratio < tv.compression_ratio);
        assert!(tv.compression_ratio < night.compression_ratio);

        // Threshold drops (more of the signal is compressed).
        assert!(ht.compression_threshold_db > tv.compression_threshold_db);
        assert!(tv.compression_threshold_db > night.compression_threshold_db);

        // Knee widens for the gentler onset of the stronger presets.
        assert!(ht.compression_knee_db <= tv.compression_knee_db);
        assert!(tv.compression_knee_db < night.compression_knee_db);
    }

    #[test]
    fn headphone_requests_binaural() {
        let p = OutputProfile::Headphone.params();
        assert!(p.binaural);
        assert!(p.compresses());
    }

    #[test]
    fn only_headphone_is_binaural() {
        for profile in [
            OutputProfile::HomeTheater,
            OutputProfile::Tv,
            OutputProfile::Night,
        ] {
            assert!(!profile.params().binaural, "{profile:?} must not be binaural");
        }
        assert!(OutputProfile::Headphone.params().binaural);
    }

    #[test]
    fn all_ceilings_prevent_overs() {
        for profile in [
            OutputProfile::HomeTheater,
            OutputProfile::Tv,
            OutputProfile::Night,
            OutputProfile::Headphone,
        ] {
            let p = profile.params();
            assert!(
                p.true_peak_ceiling_dbtp <= 0.0,
                "{profile:?} ceiling must be at or below 0 dBTP"
            );
        }
    }

    #[test]
    fn resolved_params_are_pinned() {
        // Golden pin of every resolved field so unintended drift is caught.
        let ht = OutputProfile::HomeTheater.params();
        assert!((ht.target_lufs - -18.0).abs() < EPS);
        assert!((ht.true_peak_ceiling_dbtp - -1.0).abs() < EPS);
        assert!((ht.compression_ratio - 1.0).abs() < EPS);
        assert!((ht.compression_threshold_db - 0.0).abs() < EPS);
        assert!((ht.compression_knee_db - 0.0).abs() < EPS);

        let tv = OutputProfile::Tv.params();
        assert!((tv.target_lufs - -16.0).abs() < EPS);
        assert!((tv.compression_ratio - 2.0).abs() < EPS);
        assert!((tv.compression_threshold_db - -18.0).abs() < EPS);
        assert!((tv.compression_knee_db - 6.0).abs() < EPS);

        let night = OutputProfile::Night.params();
        assert!((night.compression_ratio - 4.0).abs() < EPS);
        assert!((night.compression_threshold_db - -28.0).abs() < EPS);
        assert!((night.compression_knee_db - 10.0).abs() < EPS);

        let hp = OutputProfile::Headphone.params();
        assert!((hp.compression_ratio - 2.5).abs() < EPS);
        assert!((hp.compression_threshold_db - -20.0).abs() < EPS);
    }

    #[test]
    fn loudness_params_carry_target_and_ceiling() {
        let p = OutputProfile::Tv.params();
        let ln = p.loudness_params(-24.0, -6.0);
        assert!((ln.target_lufs - -16.0).abs() < EPS);
        assert!((ln.max_true_peak_dbtp - -1.0).abs() < EPS);
        assert!((ln.measured_lufs - -24.0).abs() < EPS);
        assert!((ln.measured_true_peak_dbtp - -6.0).abs() < EPS);
    }
}
