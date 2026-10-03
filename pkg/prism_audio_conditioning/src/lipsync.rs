//! Offline lip-sync / viseme energy-envelope analysis.
//!
//! A short-time Fourier transform drives a coarse amplitude and three-band
//! (low / mid / high) energy analysis. The per-frame magnitude energy is
//! turned into a smoothed "openness" envelope while the band split gives a
//! rough timbral fingerprint that a facial-animation layer can map onto mouth
//! shapes. This is classic amplitude/band analysis: there is no phoneme
//! recognition and no machine learning anywhere in this module.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the lip-sync export leg of design section 51. The `STFT` is
//! reused from `prism_audio_core`; the output is a plain timeline consumed by
//! an animation authoring tool, never by the audio thread.

use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::analysis::spectrum::{SpectrumAnalyzer, Window};

use crate::config::LipsyncConfig;

/// A single analysed `STFT` frame of lip-sync data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VisemeFrame {
    /// Sample position of this frame's analysis window start.
    pub time_frame: usize,
    /// Smoothed, peak-normalised mouth openness in `0..=1`.
    pub openness: Sample,
    /// Raw energy (sum of squared magnitudes) in the low, mid, and high bands.
    pub band_energy: [Sample; 3],
}

/// An ordered sequence of [`VisemeFrame`]s spanning one channel.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VisemeTimeline {
    /// Analysed frames in increasing time order.
    pub frames: Vec<VisemeFrame>,
}

impl VisemeTimeline {
    /// The number of analysed frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` when no frames were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// Analyses one channel into a viseme/openness timeline.
///
/// Each `STFT` frame contributes one [`VisemeFrame`]: magnitudes are split into
/// low (`< low_band_hz`), mid (`< mid_band_hz`), and high bands by
/// [`SpectrumAnalyzer::bin_frequency`]; the per-frame total magnitude energy is
/// smoothed with a one-pole filter (coefficient `envelope_smoothing`) and the
/// resulting envelope is peak-normalised into `openness`.
#[must_use]
pub fn analyze(
    channel: &[Sample],
    sample_rate: u32,
    config: &LipsyncConfig,
) -> VisemeTimeline {
    let hop = config.hop.max(1);
    let mut analyzer = SpectrumAnalyzer::new(config.fft_size, hop, Window::Hann);
    let mut last_frame = 0u64;
    let mut raw_env: Vec<Sample> = Vec::new();
    let mut bands: Vec<[Sample; 3]> = Vec::new();

    for &x in channel {
        analyzer.feed_sample(x);
        let frames = analyzer.frames_computed();
        if frames == last_frame {
            continue;
        }
        last_frame = frames;
        let magnitudes = analyzer.magnitudes();
        let mut band_energy = [0.0 as Sample; 3];
        let mut total = 0.0 as Sample;
        for (bin, &mag) in magnitudes.iter().enumerate() {
            let freq = analyzer.bin_frequency(bin, sample_rate);
            let band = if freq < config.low_band_hz {
                0
            } else if freq < config.mid_band_hz {
                1
            } else {
                2
            };
            let power = mag * mag;
            band_energy[band] += power;
            total += power;
        }
        raw_env.push(ops::sqrt(total));
        bands.push(band_energy);
    }

    // One-pole smoothing of the raw amplitude envelope.
    let alpha = config.envelope_smoothing.clamp(0.0, 1.0);
    let mut smoothed: Vec<Sample> = Vec::with_capacity(raw_env.len());
    let mut state = 0.0 as Sample;
    for &raw in &raw_env {
        state = alpha * state + (1.0 - alpha) * raw;
        smoothed.push(state);
    }

    // Peak-normalise into openness so the timeline is scale-independent.
    let peak = smoothed.iter().copied().fold(0.0 as Sample, Sample::max);
    let mut frames = Vec::with_capacity(smoothed.len());
    for (i, &env) in smoothed.iter().enumerate() {
        let openness = if peak > 0.0 { env / peak } else { 0.0 };
        frames.push(VisemeFrame {
            time_frame: i * hop,
            openness,
            band_energy: bands[i],
        });
    }

    VisemeTimeline { frames }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use bevy_math::ops;
    use core::f32::consts::TAU;

    fn tone(amp: Sample, freq: Sample, rate: u32, frames: usize) -> Vec<Sample> {
        (0..frames)
            .map(|i| amp * ops::sin(TAU * freq * i as Sample / rate as Sample))
            .collect()
    }

    #[test]
    fn louder_region_has_higher_openness() {
        let rate = 48_000;
        let mut signal = tone(0.05, 440.0, rate, 8_000);
        signal.extend(tone(0.8, 440.0, rate, 8_000));
        let timeline = analyze(&signal, rate, &LipsyncConfig::default());
        assert!(timeline.len() > 8);

        let n = timeline.len();
        let third = n / 3;
        let quiet: Sample = timeline.frames[..third]
            .iter()
            .map(|f| f.openness)
            .sum::<Sample>()
            / third as Sample;
        let loud: Sample = timeline.frames[n - third..]
            .iter()
            .map(|f| f.openness)
            .sum::<Sample>()
            / third as Sample;
        assert!(loud > quiet + 0.1, "loud {loud} vs quiet {quiet}");
    }

    #[test]
    fn silence_yields_zero_openness() {
        let timeline = analyze(&[0.0; 4_000], 48_000, &LipsyncConfig::default());
        assert!(timeline.frames.iter().all(|f| f.openness.abs() < 1.0e-6));
    }
}
