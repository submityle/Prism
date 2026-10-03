//! Autocorrelation tempo and beat-grid estimation.
//!
//! The onset detection function from [`crate::transient`] is autocorrelated;
//! the lag with the strongest correlation inside a plausible tempo band is the
//! beat period. The lag (in `STFT` frames) is converted to beats per minute
//! and to an audio-frame beat period, and a confidence is taken from the
//! normalised correlation height.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//! Autocorrelation tempo estimation is a classic `MIR` technique; only the
//! idea is reused.
//!
//! # Relationship
//! Implements the tempo-estimation leg of design section 51; it consumes the
//! onset envelope produced by [`crate::transient`] and feeds the beat/bar
//! markers of [`crate::marker`].

use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::config::TempoConfig;

/// A tempo estimate derived from an onset envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TempoEstimate {
    /// Estimated tempo in beats per minute.
    pub bpm: Sample,
    /// Beat period in audio frames (`lag * hop`).
    pub beat_period_frames: Sample,
    /// Normalised autocorrelation height at the chosen lag, in `[0, 1]`.
    pub confidence: Sample,
}

/// Estimates tempo by autocorrelating an onset envelope.
///
/// `hop` is the `STFT` hop (in audio frames) that produced `onset_envelope`;
/// `sample_rate` converts the lag to beats per minute. When the envelope is
/// too short or carries no energy, a zero-confidence estimate at the slowest
/// configured tempo is returned.
#[must_use]
pub fn estimate(
    onset_envelope: &[Sample],
    hop: usize,
    sample_rate: u32,
    config: &TempoConfig,
) -> TempoEstimate {
    let hop = hop.max(1);
    let frame_rate = sample_rate as Sample / hop as Sample;

    // Convert the BPM band to a lag band (in frames). A faster tempo means a
    // shorter lag, so the min/max swap.
    let min_lag = frame_lag_for_bpm(config.max_bpm, frame_rate);
    let max_lag = frame_lag_for_bpm(config.min_bpm, frame_rate);
    let max_lag = max_lag.min(onset_envelope.len().saturating_sub(1));

    let zero_lag = autocorrelate(onset_envelope, 0);
    if min_lag < 1 || max_lag < min_lag || zero_lag <= 0.0 {
        return TempoEstimate {
            bpm: config.min_bpm,
            beat_period_frames: 0.0,
            confidence: 0.0,
        };
    }

    let mut best_lag = min_lag;
    let mut best_corr = Sample::NEG_INFINITY;
    for lag in min_lag..=max_lag {
        let corr = autocorrelate(onset_envelope, lag);
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    let beat_period_frames = best_lag as Sample * hop as Sample;
    let bpm = if beat_period_frames > 0.0 {
        60.0 * sample_rate as Sample / beat_period_frames
    } else {
        config.min_bpm
    };
    let confidence = (best_corr / zero_lag).clamp(0.0, 1.0);

    TempoEstimate {
        bpm,
        beat_period_frames,
        confidence,
    }
}

/// Builds a beat grid: sample positions at `phase`, `phase + period`, ...
///
/// `period` is the beat period in audio frames; positions are rounded to the
/// nearest frame and only those strictly inside `len_frames` are returned.
#[must_use]
pub fn beat_grid(len_frames: usize, period: Sample, phase: Sample) -> Vec<usize> {
    let mut grid = Vec::new();
    if !period.is_finite() || period <= 0.0 {
        return grid;
    }
    let mut position = phase;
    // Guard against a runaway loop for tiny periods.
    let max_beats = len_frames + 1;
    for _ in 0..max_beats {
        if position < 0.0 {
            position += period;
            continue;
        }
        let frame = ops::round(position) as usize;
        if frame >= len_frames {
            break;
        }
        grid.push(frame);
        position += period;
    }
    grid
}

/// Returns the autocorrelation of `signal` at `lag` (an unnormalised dot
/// product of the signal with its lagged self).
fn autocorrelate(signal: &[Sample], lag: usize) -> Sample {
    if lag >= signal.len() {
        return 0.0;
    }
    let mut sum = 0.0 as Sample;
    for i in lag..signal.len() {
        sum += signal[i] * signal[i - lag];
    }
    sum
}

/// Converts a tempo in beats per minute to a lag in `STFT` frames.
fn frame_lag_for_bpm(bpm: Sample, frame_rate: Sample) -> usize {
    if bpm <= 0.0 {
        return 0;
    }
    let beats_per_second = bpm / 60.0;
    let lag = frame_rate / beats_per_second;
    ops::round(lag.max(0.0)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn recovers_pulse_train_tempo() {
        // A pulse every 24 ODF frames. With hop=256 at 48 kHz the frame rate is
        // 187.5 fps, so 24 frames is 0.128 s -> ~468.75 ... keep tempo in band:
        // choose period so BPM lands near 120.
        let hop = 256usize;
        let sample_rate = 48_000u32;
        let frame_rate = sample_rate as Sample / hop as Sample;
        // 120 BPM -> 2 beats/s -> frame_rate/2 frames per beat.
        let period = ops::round(frame_rate / 2.0) as usize;
        let mut odf: Vec<Sample> = vec![0.0; period * 32];
        let mut i = 0;
        while i < odf.len() {
            odf[i] = 1.0;
            i += period;
        }
        let config = TempoConfig::default();
        let estimate = estimate(&odf, hop, sample_rate, &config);
        assert!(
            (estimate.bpm - 120.0).abs() < 8.0,
            "bpm {} not near 120",
            estimate.bpm
        );
        assert!(estimate.confidence > 0.0);
    }

    #[test]
    fn beat_grid_is_evenly_spaced() {
        let grid = beat_grid(1_000, 100.0, 50.0);
        assert_eq!(grid.first().copied(), Some(50));
        assert_eq!(grid[1], 150);
        assert!(grid.iter().all(|&f| f < 1_000));
    }

    #[test]
    fn flat_envelope_has_zero_confidence() {
        let odf = vec![0.0 as Sample; 512];
        let estimate = estimate(&odf, 256, 48_000, &TempoConfig::default());
        assert_eq!(estimate.confidence, 0.0);
    }
}
