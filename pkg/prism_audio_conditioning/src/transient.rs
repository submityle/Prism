//! Spectral-flux transient / onset detection.
//!
//! A short-time Fourier transform is run over the signal; the per-frame
//! half-wave-rectified magnitude increase (spectral flux) forms an onset
//! detection function (`ODF`). Peaks of the `ODF` that clear an adaptive
//! threshold (a moving mean plus a multiple of the moving standard deviation)
//! and that are local maxima are reported as onsets, converted back to sample
//! positions. The `ODF` itself is exposed so [`crate::tempo`] can reuse it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//! Spectral-flux onset detection is a classic `MIR` technique; only the idea is
//! reused.
//!
//! # Relationship
//! Implements the transient-detection leg of design section 51 and feeds the
//! marker timeline and tempo estimator. The `STFT` is reused from
//! `prism_audio_core`.

use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::analysis::spectrum::{SpectrumAnalyzer, Window};

use crate::config::TransientConfig;

/// Computes the spectral-flux onset detection function, one value per `STFT`
/// frame.
///
/// Each value is the sum over bins of the half-wave-rectified increase in
/// magnitude relative to the previous frame, so sustained tones contribute
/// little while sharp attacks produce peaks.
#[must_use]
pub fn onset_envelope(channel: &[Sample], config: &TransientConfig) -> Vec<Sample> {
    let hop = config.hop.max(1);
    let mut analyzer = SpectrumAnalyzer::new(config.fft_size, hop, Window::Hann);
    let mut previous: Vec<Sample> = Vec::new();
    let mut odf: Vec<Sample> = Vec::new();
    let mut last_frame = 0u64;

    for &x in channel {
        analyzer.feed_sample(x);
        let frames = analyzer.frames_computed();
        if frames == last_frame {
            continue;
        }
        last_frame = frames;
        let magnitudes = analyzer.magnitudes();
        if previous.len() != magnitudes.len() {
            previous = alloc::vec![0.0 as Sample; magnitudes.len()];
        }
        let mut flux = 0.0 as Sample;
        for (bin, &mag) in magnitudes.iter().enumerate() {
            let diff = mag - previous[bin];
            if diff > 0.0 {
                flux += diff;
            }
            previous[bin] = mag;
        }
        odf.push(flux);
    }
    odf
}

/// Picks onset frames from an onset detection function.
///
/// A causal/centered moving window of `mean_window_frames` forms a local mean
/// and standard deviation; a frame is accepted when it exceeds `mean + k*std`,
/// is a strict local maximum against its neighbours, and lies at least
/// `min_separation_frames` after the previous accepted onset.
fn pick_peaks(odf: &[Sample], config: &TransientConfig) -> Vec<usize> {
    let window = config.mean_window_frames.max(1);
    let min_sep = config.min_separation_frames;
    let mut onsets: Vec<usize> = Vec::new();
    let mut last_onset: Option<usize> = None;

    for i in 0..odf.len() {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(odf.len());
        let span = &odf[lo..hi];
        let count = span.len() as Sample;
        let mean = span.iter().copied().sum::<Sample>() / count;
        let variance =
            span.iter().map(|&v| (v - mean) * (v - mean)).sum::<Sample>() / count;
        let std = ops::sqrt(variance);
        let threshold = mean + config.threshold_k * std;

        let value = odf[i];
        if value <= threshold || value <= 0.0 {
            continue;
        }
        let left_ok = i == 0 || odf[i - 1] <= value;
        let right_ok = i + 1 >= odf.len() || odf[i + 1] < value;
        if !(left_ok && right_ok) {
            continue;
        }
        if let Some(prev) = last_onset.filter(|&prev| i < prev + min_sep) {
            // Keep the stronger of two near-coincident peaks.
            if value > odf[prev] {
                onsets.pop();
            } else {
                continue;
            }
        }
        onsets.push(i);
        last_onset = Some(i);
    }
    onsets
}

/// Detects transient onsets and returns their sample positions.
///
/// The returned positions are frame-quantised to the configured `STFT` hop
/// (`frame_index * hop`); the `sample_rate` is accepted for API symmetry and
/// future rate-aware heuristics and does not change the quantisation.
#[must_use]
pub fn detect_onsets(
    channel: &[Sample],
    sample_rate: u32,
    config: &TransientConfig,
) -> Vec<usize> {
    let _ = sample_rate;
    let hop = config.hop.max(1);
    let odf = onset_envelope(channel, config);
    pick_peaks(&odf, config)
        .into_iter()
        .map(|frame| frame * hop)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn clicks(positions: &[usize], len: usize) -> Vec<Sample> {
        let mut signal = vec![0.0 as Sample; len];
        for &p in positions {
            if p < len {
                // A short decaying click.
                for (k, s) in signal.iter_mut().enumerate().skip(p).take(16) {
                    let d = (k - p) as Sample;
                    *s += ops::powf(0.6, d);
                }
            }
        }
        signal
    }

    #[test]
    fn detects_clicks_near_known_positions() {
        let config = TransientConfig {
            fft_size: 512,
            hop: 128,
            threshold_k: 1.0,
            mean_window_frames: 6,
            min_separation_frames: 2,
        };
        let positions = [4_096usize, 12_288, 20_480];
        let signal = clicks(&positions, 28_000);
        let onsets = detect_onsets(&signal, 48_000, &config);
        assert!(!onsets.is_empty());
        // Every known click has a detected onset within one hop-and-a-frame.
        let tolerance = config.hop + config.fft_size;
        for &p in &positions {
            let hit = onsets
                .iter()
                .any(|&o| (o as isize - p as isize).unsigned_abs() <= tolerance);
            assert!(hit, "no onset near {p}: {onsets:?}");
        }
    }

    #[test]
    fn silence_has_no_onsets() {
        let config = TransientConfig::default();
        let signal = vec![0.0 as Sample; 8_000];
        assert!(detect_onsets(&signal, 48_000, &config).is_empty());
    }
}
