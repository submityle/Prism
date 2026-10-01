//! Initial time delay gap `ITDG` from a measured room impulse response.
//!
//! The initial time delay gap is the interval, in milliseconds, between the
//! arrival of the direct sound and the arrival of the first significant
//! reflection at the listener. It is Beranek's objective measure of acoustic
//! intimacy in concert halls: small gaps (empirically below `20` ms) make a
//! hall feel intimate, while large gaps make it feel distant.
//!
//! # Model
//!
//! Let `p[n]` be the measured room impulse response. The direct-sound index is
//! the first sample index that attains the global peak magnitude `peak`. The
//! scan then advances from the sample after the direct index and takes the
//! first local maximum (a sample whose magnitude is greater than or equal to
//! both neighbours, with one-sided comparison at the ends) whose magnitude is
//! at least `peak * db_to_linear(reflection_threshold_db)`. The gap is
//! `(reflection_index - direct_index) / sample_rate * 1000` milliseconds.
//!
//! # Relationship
//!
//! This module complements [`crate::early_reflections`] (which locates all of
//! the early reflections via the image-source method; this module takes only
//! the direct-to-first-reflection gap from a measured response),
//! [`crate::reflection_clustering`] (which buckets reflection taps by
//! direction), [`crate::room_clarity`] (clarity and reverberation ratios), and
//! [`crate::center_time`] (the energy centre of gravity). Each expresses a
//! distinct objective parameter and none reimplements another. All share the
//! [`Sample`] scalar from [`prism_audio_core`].
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator: it accepts a whole impulse
//! response and performs no heap allocation. It is not a per-sample callback
//! and must not run on an audio thread. It never panics: empty, all-zero,
//! non-finite, or non-positive sample-rate inputs, and responses with no
//! reflection above the threshold, return the safe default `0`.
//!
//! # Provenance
//!
//! This is a textbook implementation of Beranek's initial-time-delay-gap
//! intimacy criterion. It is pure classic DSP with no AI or ML. It is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustic criterion.

use prism_audio_core::math::{Sample, db_to_linear};

/// Default detection threshold, in decibels, for the first reflection peak
/// relative to the direct-sound peak.
pub const DEFAULT_REFLECTION_THRESHOLD_DB: Sample = -10.0;

/// Returns the magnitude of a sample, mapping non-finite values to `0`.
#[inline]
fn finite_abs(x: Sample) -> Sample {
    if x.is_finite() { x.abs() } else { 0.0 }
}

/// Computes the initial time delay gap, in milliseconds, between the direct
/// sound and the first significant reflection.
///
/// `reflection_threshold_db` sets the detection level of the first reflection
/// relative to the direct peak (for example `-10.0` dB). Empty responses, a
/// sample rate of `0`, a non-positive peak, or no reflection above the
/// threshold all return `0`.
#[must_use]
pub fn initial_time_delay_gap_ms(
    rir: &[Sample],
    sample_rate: u32,
    reflection_threshold_db: Sample,
) -> Sample {
    if rir.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    // Direct sound: the first index attaining the global peak magnitude.
    let mut peak = 0.0;
    let mut direct_index = 0usize;
    for (i, &x) in rir.iter().enumerate() {
        let mag = finite_abs(x);
        if mag > peak {
            peak = mag;
            direct_index = i;
        }
    }
    if peak <= 0.0 {
        return 0.0;
    }

    let threshold = peak * db_to_linear(reflection_threshold_db);
    let Some(reflection_index) = find_first_reflection(rir, direct_index, threshold) else {
        return 0.0;
    };

    let samples = (reflection_index - direct_index) as Sample;
    samples / sample_rate as Sample * 1000.0
}

/// Finds the index of the first local-maximum sample after `direct_index` whose
/// magnitude is at least `threshold`.
fn find_first_reflection(rir: &[Sample], direct_index: usize, threshold: Sample) -> Option<usize> {
    let len = rir.len();
    let start = direct_index + 1;
    for i in start..len {
        let mag = finite_abs(rir[i]);
        if mag < threshold {
            continue;
        }
        let prev = finite_abs(rir[i - 1]);
        let is_local_max = if i + 1 < len {
            let next = finite_abs(rir[i + 1]);
            mag >= prev && mag >= next
        } else {
            // One-sided comparison at the final sample.
            mag >= prev
        };
        if is_local_max {
            return Some(i);
        }
    }
    None
}

/// The initial time delay gap of a measured room, with the detected direct and
/// first-reflection sample indices.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct InitialTimeDelayGap {
    /// The gap between the direct sound and first reflection, in milliseconds.
    pub gap_ms: Sample,
    /// The sample index of the detected direct sound.
    pub direct_index: usize,
    /// The sample index of the detected first reflection.
    pub reflection_index: usize,
}

impl Default for InitialTimeDelayGap {
    fn default() -> Self {
        Self {
            gap_ms: 0.0,
            direct_index: 0,
            reflection_index: 0,
        }
    }
}

impl InitialTimeDelayGap {
    /// Computes the initial time delay gap from a measured impulse response.
    ///
    /// When no reflection above the threshold is found (or for any degenerate
    /// input) the gap is `0` and both indices are `0`.
    ///
    /// ```
    /// use prism_audio_spatial::initial_time_delay_gap::{
    ///     InitialTimeDelayGap, DEFAULT_REFLECTION_THRESHOLD_DB,
    /// };
    ///
    /// // Direct sound at n=0, a strong reflection 480 samples (10 ms) later.
    /// let mut rir = vec![0.0f32; 48_000];
    /// rir[0] = 1.0;
    /// rir[480] = 0.5;
    /// let itdg = InitialTimeDelayGap::from_impulse_response(
    ///     &rir,
    ///     48_000,
    ///     DEFAULT_REFLECTION_THRESHOLD_DB,
    /// );
    /// assert!((itdg.gap_ms - 10.0).abs() < 1e-3);
    /// assert_eq!(itdg.direct_index, 0);
    /// assert_eq!(itdg.reflection_index, 480);
    /// ```
    #[must_use]
    pub fn from_impulse_response(
        rir: &[Sample],
        sample_rate: u32,
        reflection_threshold_db: Sample,
    ) -> Self {
        if rir.is_empty() || sample_rate == 0 {
            return Self::default();
        }

        let mut peak = 0.0;
        let mut direct_index = 0usize;
        for (i, &x) in rir.iter().enumerate() {
            let mag = finite_abs(x);
            if mag > peak {
                peak = mag;
                direct_index = i;
            }
        }
        if peak <= 0.0 {
            return Self::default();
        }

        let threshold = peak * db_to_linear(reflection_threshold_db);
        let Some(reflection_index) = find_first_reflection(rir, direct_index, threshold) else {
            return Self::default();
        };

        let samples = (reflection_index - direct_index) as Sample;
        let gap_ms = samples / sample_rate as Sample * 1000.0;
        Self {
            gap_ms,
            direct_index,
            reflection_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    const SR: u32 = 48_000;
    const THR: Sample = DEFAULT_REFLECTION_THRESHOLD_DB;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn basic_single_reflection_gap() {
        let mut rir = vec![0.0; 48_000];
        rir[0] = 1.0;
        rir[480] = 0.5; // 10 ms later, -6 dB (above -10 dB threshold)
        let gap = initial_time_delay_gap_ms(&rir, SR, THR);
        assert!(approx(gap, 10.0, 1e-3), "gap {gap}");
    }

    #[test]
    fn gap_equals_sample_difference_over_rate() {
        let mut rir = vec![0.0; 48_000];
        rir[100] = 1.0;
        rir[1_300] = 0.8;
        let gap = initial_time_delay_gap_ms(&rir, SR, THR);
        let expected = (1_300 - 100) as Sample / SR as Sample * 1000.0;
        assert!(approx(gap, expected, 1e-4), "gap {gap} expected {expected}");
    }

    #[test]
    fn threshold_too_high_yields_no_reflection() {
        let mut rir = vec![0.0; 48_000];
        rir[0] = 1.0;
        rir[480] = 0.1; // -20 dB, below a strict -3 dB threshold
        let gap = initial_time_delay_gap_ms(&rir, SR, -3.0);
        assert_eq!(gap, 0.0);
    }

    #[test]
    fn all_zero_returns_zero() {
        let rir = vec![0.0; 1_000];
        assert_eq!(initial_time_delay_gap_ms(&rir, SR, THR), 0.0);
        let itdg = InitialTimeDelayGap::from_impulse_response(&rir, SR, THR);
        assert_eq!(itdg, InitialTimeDelayGap::default());
    }

    #[test]
    fn empty_returns_zero() {
        let empty: [Sample; 0] = [];
        assert_eq!(initial_time_delay_gap_ms(&empty, SR, THR), 0.0);
    }

    #[test]
    fn zero_sample_rate_returns_zero() {
        let mut rir = vec![0.0; 1_000];
        rir[0] = 1.0;
        rir[480] = 0.5;
        assert_eq!(initial_time_delay_gap_ms(&rir, 0, THR), 0.0);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut rir = vec![0.0; 2_000];
        rir[0] = 1.0;
        rir[200] = Sample::NAN;
        rir[300] = Sample::INFINITY;
        rir[480] = 0.5; // the first finite significant reflection
        let gap = initial_time_delay_gap_ms(&rir, SR, THR);
        assert!(gap.is_finite());
        assert!(approx(gap, 10.0, 1e-3), "gap {gap}");
    }

    #[test]
    fn direct_sound_is_global_peak() {
        let mut rir = vec![0.0; 48_000];
        rir[500] = 1.0; // global peak not at index 0
        rir[1_460] = 0.6; // 20 ms after the direct sound
        let itdg = InitialTimeDelayGap::from_impulse_response(&rir, SR, THR);
        assert_eq!(itdg.direct_index, 500);
        assert_eq!(itdg.reflection_index, 1_460);
        assert!(approx(itdg.gap_ms, 20.0, 1e-3), "gap {}", itdg.gap_ms);
    }

    #[test]
    fn reflection_must_follow_direct() {
        // A large sample before the peak must not count as a reflection.
        let mut rir = vec![0.0; 48_000];
        rir[200] = 0.9; // earlier but smaller than the peak
        rir[800] = 1.0; // the direct sound (global peak)
        rir[1_280] = 0.7; // first reflection after the direct sound
        let itdg = InitialTimeDelayGap::from_impulse_response(&rir, SR, THR);
        assert_eq!(itdg.direct_index, 800);
        assert_eq!(itdg.reflection_index, 1_280);
    }

    #[test]
    fn from_impulse_response_matches_free_function() {
        let mut rir = vec![0.0; 48_000];
        rir[0] = 1.0;
        rir[960] = 0.4;
        let itdg = InitialTimeDelayGap::from_impulse_response(&rir, SR, THR);
        let gap = initial_time_delay_gap_ms(&rir, SR, THR);
        assert!(approx(itdg.gap_ms, gap, 1e-6));
    }

    #[test]
    fn default_is_all_zero() {
        let d = InitialTimeDelayGap::default();
        assert_eq!(d.gap_ms, 0.0);
        assert_eq!(d.direct_index, 0);
        assert_eq!(d.reflection_index, 0);
    }

    #[test]
    fn later_reflection_gives_larger_gap() {
        let mut near = vec![0.0; 48_000];
        near[0] = 1.0;
        near[480] = 0.5; // 10 ms
        let mut far = vec![0.0; 48_000];
        far[0] = 1.0;
        far[1_440] = 0.5; // 30 ms
        let gap_near = initial_time_delay_gap_ms(&near, SR, THR);
        let gap_far = initial_time_delay_gap_ms(&far, SR, THR);
        assert!(gap_far > gap_near, "near {gap_near} far {gap_far}");
    }

    #[test]
    fn picks_first_local_max_above_threshold() {
        // Two reflections above threshold: the earlier one wins.
        let mut rir = vec![0.0; 48_000];
        rir[0] = 1.0;
        rir[480] = 0.4; // first reflection, 10 ms
        rir[2_000] = 0.9; // stronger but later
        let itdg = InitialTimeDelayGap::from_impulse_response(&rir, SR, THR);
        assert_eq!(itdg.reflection_index, 480);
    }

    #[test]
    fn ramp_without_local_peak_handles_end() {
        // Monotone increasing tail: the final sample is a one-sided local max.
        let rir: Vec<Sample> = (0..8).map(|n| 0.1 + 0.1 * n as Sample).collect();
        // The global peak is the last sample, so there is no reflection after
        // it: the gap is zero.
        let gap = initial_time_delay_gap_ms(&rir, SR, THR);
        assert_eq!(gap, 0.0);
    }
}
