//! Offline `ITU-R` `BS.1770` / `EBU` `R128` loudness measurement.
//!
//! The runtime [`LoudnessMeter`] is a sample-by-sample meter; this stage drives
//! it over a whole [`ConditionedPcm`] asset and returns a compact summary. A
//! raw sample peak (independent of the meter's oversampled true peak) is also
//! computed so authoring can distinguish inter-sample overshoot from digital
//! full scale.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. `BS.1770`
//! loudness is a public broadcast standard; only its published algorithm is
//! reused through `prism_audio_core`.
//!
//! # Relationship
//! Implements the loudness-analysis leg of design section 51 by reusing the
//! `prism_audio_core` analysis meter rather than reimplementing `K`-weighting.

use bevy_math::ops;

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::analysis::loudness::LoudnessMeter;

use crate::pcm::ConditionedPcm;

/// A compact loudness summary of a conditioned asset.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LoudnessStats {
    /// Gated programme loudness in `LUFS` (`f32::NEG_INFINITY` when silent).
    pub integrated_lufs: Sample,
    /// Maximum true-peak level in `dBTP`.
    pub true_peak_dbtp: Sample,
    /// Loudness range in `LU`.
    pub loudness_range_lu: Sample,
    /// Raw sample-peak level in `dBFS` (`f32::NEG_INFINITY` when silent).
    pub sample_peak_dbfs: Sample,
}

/// Converts a linear amplitude to `dBFS` using `20 * log10(x)`.
///
/// A non-positive amplitude maps to negative infinity (digital silence).
fn amplitude_to_dbfs(amplitude: Sample) -> Sample {
    if amplitude > 0.0 {
        20.0 * ops::log10(amplitude)
    } else {
        Sample::NEG_INFINITY
    }
}

/// Measures integrated loudness, true peak, loudness range, and sample peak.
///
/// The meter is driven frame by frame: every channel of a frame is fed, then
/// the frame is advanced, exactly as the real-time path would.
#[must_use]
pub fn analyze(pcm: &ConditionedPcm) -> LoudnessStats {
    let mut meter = LoudnessMeter::new(pcm.sample_rate(), pcm.layout());
    let frames = pcm.frames();
    let channel_count = pcm.channel_count();
    let mut sample_peak = 0.0 as Sample;

    for frame in 0..frames {
        for ch in 0..channel_count {
            let x = pcm.channel(ch).map_or(0.0, |c| c[frame]);
            let magnitude = ops::abs(x);
            if magnitude > sample_peak {
                sample_peak = magnitude;
            }
            meter.feed_sample(ch, x);
        }
        meter.advance_frame();
    }

    let measurement = meter.measurement();
    LoudnessStats {
        integrated_lufs: measurement.integrated_lufs,
        true_peak_dbtp: measurement.true_peak_dbtp,
        loudness_range_lu: measurement.loudness_range_lu,
        sample_peak_dbfs: amplitude_to_dbfs(sample_peak),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::TAU;
    use prism_audio_core::buffer::ChannelLayout;

    fn tone(amp: Sample, freq: Sample, rate: u32, frames: usize) -> Vec<Sample> {
        (0..frames)
            .map(|i| amp * ops::sin(TAU * freq * i as Sample / rate as Sample))
            .collect()
    }

    #[test]
    fn silence_is_very_quiet() {
        let rate = 48_000;
        let pcm = ConditionedPcm::silence(rate, ChannelLayout::Mono, rate as usize).unwrap();
        let stats = analyze(&pcm);
        assert!(stats.integrated_lufs < -60.0 || stats.integrated_lufs == Sample::NEG_INFINITY);
        assert_eq!(stats.sample_peak_dbfs, Sample::NEG_INFINITY);
    }

    #[test]
    fn calibrated_tone_is_plausible() {
        let rate = 48_000;
        // -23 dBFS 1 kHz tone, two seconds, stereo.
        let amp = ops::powf(10.0, -23.0 / 20.0);
        let left = tone(amp, 1_000.0, rate, 2 * rate as usize);
        let right = left.clone();
        let pcm = ConditionedPcm::new(rate, ChannelLayout::Stereo, vec![left, right]).unwrap();
        let stats = analyze(&pcm);
        assert!(stats.integrated_lufs.is_finite());
        // A -23 dBFS tone sits near -23 LUFS within a loose tolerance.
        assert!((stats.integrated_lufs - (-23.0)).abs() < 2.0);
        // Sample peak is the tone amplitude in dBFS (~ -23 dB).
        assert!((stats.sample_peak_dbfs - (-23.0)).abs() < 1.0);
    }
}
