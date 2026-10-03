//! Seamless loop-point detection via zero-crossing alignment and normalized
//! cross-correlation.
//!
//! A seamless sustain loop needs a start and an end such that jumping from the
//! end back to the start is inaudible. Two classic, deterministic constraints
//! do the job: (1) both boundaries sit on rising zero crossings so the
//! instantaneous waveform phase matches at the seam, and (2) the loop length is
//! chosen to maximize the normalized cross-correlation between the window at
//! the start and the window at the end, which locks onto the signal's natural
//! period. No AI/ML is involved.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//! Zero-crossing alignment and cross-correlation period estimation are
//! textbook classic DSP; only the ideas are used.
//!
//! # Relationship
//! Implements the loop-authoring stage of design section 51; its output feeds
//! the marker timeline and the runtime sampler's loop region.

use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::config::LoopConfig;

/// Smallest energy a correlation window may hold before it is treated as
/// silent and skipped.
const ENERGY_FLOOR: Sample = 1.0e-9;

/// How a detected loop region is played back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum LoopMode {
    /// Play `start..end`, then jump back to `start` and repeat.
    #[default]
    Forward,
    /// Play `start..end`, then play `end..start` in reverse, and repeat.
    PingPong,
}

/// A detected loop region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LoopPoints {
    /// First frame of the loop (inclusive).
    pub start: usize,
    /// Loop return frame (exclusive); the loop length is `end - start`.
    pub end: usize,
    /// Equal-power crossfade length to apply at the seam, in frames.
    pub crossfade_frames: u32,
    /// Playback mode.
    pub mode: LoopMode,
}

impl LoopPoints {
    /// The loop length in frames (`end - start`).
    #[must_use]
    pub const fn length(&self) -> usize {
        self.end - self.start
    }
}

/// Collects the indices of rising zero crossings (a non-positive sample
/// followed by a positive one); the crossing is reported at the first positive
/// sample.
fn rising_zero_crossings(channel: &[Sample]) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 1..channel.len() {
        if channel[i - 1] <= 0.0 && channel[i] > 0.0 {
            out.push(i);
        }
    }
    out
}

/// Returns the dot product of two equal-length windows.
fn dot(a: &[Sample], b: &[Sample]) -> Sample {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Returns the Euclidean norm of a window.
fn norm(a: &[Sample]) -> Sample {
    ops::sqrt(dot(a, a))
}

/// Detects a seamless loop region in a single channel.
///
/// Returns [`None`] when the signal is too short for the configured window and
/// period range, when it is effectively silent, or when no candidate exceeds a
/// basic correlation floor. On a tie the shortest qualifying period wins, so a
/// strongly periodic signal reports a single-period loop.
#[must_use]
pub fn detect(channel: &[Sample], config: &LoopConfig) -> Option<LoopPoints> {
    let window = config.window_frames.max(1);
    let n = channel.len();
    let min_period = config.min_period_frames.max(1);
    if n <= window + min_period {
        return None;
    }

    let crossings = rising_zero_crossings(channel);
    if crossings.len() < 2 {
        return None;
    }

    // The loop start is the first rising zero crossing that leaves room for a
    // full comparison window.
    let start = *crossings.iter().find(|&&c| c + window <= n)?;

    let max_period = config.max_period_frames.min(n - window - start);
    if max_period < min_period {
        return None;
    }

    let anchor = &channel[start..start + window];
    let anchor_norm = norm(anchor);
    if anchor_norm * anchor_norm <= ENERGY_FLOOR {
        return None;
    }

    let mut best: Option<(usize, Sample)> = None;
    for &end in &crossings {
        if end <= start {
            continue;
        }
        let period = end - start;
        if period < min_period || period > max_period {
            continue;
        }
        if end + window > n {
            continue;
        }
        let candidate = &channel[end..end + window];
        let candidate_norm = norm(candidate);
        if candidate_norm * candidate_norm <= ENERGY_FLOOR {
            continue;
        }
        let ncc = dot(anchor, candidate) / (anchor_norm * candidate_norm);
        let improved = match best {
            Some((_, best_ncc)) => ncc > best_ncc,
            None => true,
        };
        if improved {
            best = Some((end, ncc));
        }
    }

    let (end, ncc) = best?;
    // Require a positive, meaningful correlation to call it a seamless loop.
    if ncc < 0.5 {
        return None;
    }

    Some(LoopPoints {
        start,
        end,
        crossfade_frames: config.crossfade_frames,
        mode: config.mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::f32::consts::TAU;

    fn sine(period: usize, cycles: usize) -> Vec<Sample> {
        let n = period * cycles;
        (0..n)
            .map(|i| ops::sin(TAU * i as Sample / period as Sample))
            .collect()
    }

    #[test]
    fn detects_period_of_pure_sine() {
        let period = 100;
        let signal = sine(period, 40);
        let config = LoopConfig {
            min_period_frames: 50,
            max_period_frames: 1_000,
            search_radius_frames: 8,
            window_frames: 200,
            crossfade_frames: 16,
            mode: LoopMode::Forward,
        };
        let loop_points = detect(&signal, &config).expect("loop detected");
        let length = loop_points.length();
        // The loop length should be an integer number of periods, and the
        // shortest qualifying one is a single period.
        let remainder = length % period;
        let near = remainder.min(period - remainder);
        assert!(near <= 2, "length {length} not aligned to period {period}");
        assert!(length >= 50);
    }

    #[test]
    fn silence_has_no_loop() {
        let signal = alloc::vec![0.0 as Sample; 4_000];
        let config = LoopConfig::default();
        assert!(detect(&signal, &config).is_none());
    }

    #[test]
    fn too_short_returns_none() {
        let signal = sine(100, 1);
        let config = LoopConfig {
            min_period_frames: 50,
            max_period_frames: 1_000,
            search_radius_frames: 8,
            window_frames: 200,
            crossfade_frames: 16,
            mode: LoopMode::Forward,
        };
        assert!(detect(&signal, &config).is_none());
    }
}
