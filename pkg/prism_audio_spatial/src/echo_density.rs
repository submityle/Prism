//! Abel-Huang normalized echo density profile (`NEDP`) and the measured mixing
//! time derived from a recorded room impulse response.
//!
//! The echo density of a reverberant tail is the number of reflections that
//! arrive per unit of time. Early in an impulse response the reflections are
//! sparse and strongly non-Gaussian; as the tail develops, reflections pile up
//! until the signal becomes a dense, zero-mean Gaussian process. The point at
//! which the response becomes statistically indistinguishable from Gaussian
//! noise is the mixing time, a key boundary between the early-reflection and
//! late-reverberation regimes.
//!
//! # Model
//!
//! Let `h[n]` be the broadband room impulse response sampled at `sample_rate`
//! hertz. Around each sample index `n` a symmetric window of half-width
//! `w = round(ECHO_DENSITY_WINDOW_MS / 1000 * sample_rate)` samples (clamped to
//! at least one) is examined. Assuming a zero-mean tail, the window standard
//! deviation is `sigma = sqrt((1 / (2w + 1)) * sum_{m in window} h[m]^2)`. The
//! normalized echo density at `n` is the fraction of window samples whose
//! magnitude exceeds `sigma`, divided by the Gaussian reference proportion:
//!
//! `eta[n] = (1 / (C * (2w + 1))) * sum_{m in window} 1{ |h[m]| > sigma }`,
//!
//! where `C = erfc(1 / sqrt(2)) = 0.317_310_5` is the probability that a
//! standard normal sample lies outside plus or minus one standard deviation.
//! For a sparse early tail `eta` is near zero; for a fully diffuse Gaussian
//! tail it approaches one. The measured mixing time is the first instant at
//! which `eta` reaches [`MIXING_THRESHOLD`], converted to milliseconds.
//!
//! # Relationship
//!
//! This module measures echo density directly from a recorded impulse
//! response. It complements [`crate::diffusion_field`], which predicts a
//! theoretical echo density `N(t) = (4 / 3) * PI * c^3 * t^3 / V` and a
//! theoretical mixing time from the room volume alone. The relationship mirrors
//! that between [`crate::direct_to_reverberant_ratio`] (measured) and the
//! statistical model on [`crate::reverberant_field`]: one is a geometry-based
//! prediction, the other an empirical measurement of a specific response. The
//! two never duplicate each other, and both share the [`Sample`] scalar from
//! [`prism_audio_core`].
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator. It accepts a whole impulse
//! response and performs no heap allocation (the profile variant writes into a
//! caller-provided buffer). It is not a per-sample callback and must not run on
//! an audio thread. It never panics: empty, all-zero, non-finite, or
//! non-positive sample-rate inputs yield the safe sentinel
//! [`NO_MIXING_TIME_MS`] and an all-zero profile. All deviation and length math
//! routes through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published Abel-Huang
//! normalized echo density profile. It is pure classic DSP with no AI or ML. It
//! is engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics literature.

use bevy_math::ops;

use prism_audio_core::math::Sample;

/// Half-width, in milliseconds, of the sliding window used to estimate the
/// local echo density. A `10` ms half-width gives a roughly `20` ms window, in
/// line with the Abel-Huang analysis.
pub const ECHO_DENSITY_WINDOW_MS: Sample = 10.0;

/// Gaussian exceedance proportion `erfc(1 / sqrt(2))`: the probability that a
/// standard normal sample lies outside plus or minus one standard deviation.
/// Supplied as a literal so the module needs no runtime error function.
pub const GAUSSIAN_EXCEEDANCE: Sample = 0.317_310_5;

/// Normalized echo density value that marks the mixing time. A profile value of
/// one indicates a fully diffuse, Gaussian tail.
pub const MIXING_THRESHOLD: Sample = 1.0;

/// Safe sentinel in milliseconds returned when no mixing time can be measured
/// (degenerate input, a silent response, or a tail that never reaches
/// [`MIXING_THRESHOLD`]).
pub const NO_MIXING_TIME_MS: Sample = -1.0;

/// Returns a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Returns the magnitude of a sample, mapping non-finite values to `0`.
#[inline]
fn finite_abs(x: Sample) -> Sample {
    if x.is_finite() { x.abs() } else { 0.0 }
}

/// Reports whether the inputs are degenerate (empty response or a non-finite or
/// non-positive sample rate).
#[inline]
fn is_degenerate(ir: &[Sample], sample_rate: Sample) -> bool {
    ir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0
}

/// Converts the window half-width from milliseconds to a sample count, clamped
/// to at least one sample. Non-finite rates fall back to one.
fn window_half_samples(sample_rate: Sample) -> usize {
    let x = ECHO_DENSITY_WINDOW_MS / 1000.0 * sample_rate;
    if !x.is_finite() || x < 1.0 {
        return 1;
    }
    (ops::round(x) as usize).max(1)
}

/// Computes the normalized echo density at a single centre index using a window
/// of half-width `w`. Returns `0` for an empty window or a silent window.
fn eta_at(ir: &[Sample], center: usize, w: usize) -> Sample {
    let len = ir.len();
    let lo = center.saturating_sub(w);
    let hi = center.saturating_add(w).saturating_add(1).min(len);
    if hi <= lo {
        return 0.0;
    }
    let window = &ir[lo..hi];
    let count = window.len();

    let mut sum_sq = 0.0_f64;
    for &x in window {
        let v = f64::from(finite(x));
        sum_sq += v * v;
    }
    // sum_sq is a finite non-negative energy and count is positive, so this is
    // a clean guard rather than a sign comparison.
    let variance = (sum_sq / count as f64) as Sample;
    let sigma = ops::sqrt(variance);
    if sigma <= 0.0 {
        return 0.0;
    }

    let mut exceed = 0usize;
    for &x in window {
        if finite_abs(x) > sigma {
            exceed += 1;
        }
    }

    let eta = (exceed as Sample) / (GAUSSIAN_EXCEEDANCE * (count as Sample));
    if eta.is_finite() { eta } else { 0.0 }
}

/// Writes the normalized echo density profile into `out`.
///
/// Entry `out[n]` receives the echo density `eta[n]` for every index covered by
/// both `ir` and `out`; any trailing slots of `out` beyond `ir.len()` are
/// zero-filled. Degenerate inputs leave `out` entirely zero. No heap is
/// allocated: the caller owns the buffer.
pub fn normalized_echo_density(ir: &[Sample], sample_rate: Sample, out: &mut [Sample]) {
    if is_degenerate(ir, sample_rate) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let len = ir.len();
    let w = window_half_samples(sample_rate);
    for (n, v) in out.iter_mut().enumerate() {
        *v = if n < len { eta_at(ir, n, w) } else { 0.0 };
    }
}

/// Computes the mixing time in milliseconds: the first instant at which the
/// normalized echo density reaches [`MIXING_THRESHOLD`].
///
/// The scan stops as soon as the threshold is crossed, so no profile buffer is
/// required and no heap is allocated. Degenerate inputs, a silent response, or
/// a tail that never becomes diffuse return [`NO_MIXING_TIME_MS`].
#[must_use]
pub fn mixing_time_ms(ir: &[Sample], sample_rate: Sample) -> Sample {
    if is_degenerate(ir, sample_rate) {
        return NO_MIXING_TIME_MS;
    }
    let len = ir.len();
    let w = window_half_samples(sample_rate);
    for n in 0..len {
        let eta = eta_at(ir, n, w);
        if eta >= MIXING_THRESHOLD {
            return (n as Sample) / sample_rate * 1000.0;
        }
    }
    NO_MIXING_TIME_MS
}

/// Computes the mixing time, final profile value, and convergence flag in a
/// single scan. Degenerate inputs return `(NO_MIXING_TIME_MS, 0.0, false)`.
fn compute(ir: &[Sample], sample_rate: Sample) -> (Sample, Sample, bool) {
    if is_degenerate(ir, sample_rate) {
        return (NO_MIXING_TIME_MS, 0.0, false);
    }
    let len = ir.len();
    let w = window_half_samples(sample_rate);
    let mut mixing = NO_MIXING_TIME_MS;
    let mut converged = false;
    for n in 0..len {
        let eta = eta_at(ir, n, w);
        if !converged && eta >= MIXING_THRESHOLD {
            converged = true;
            mixing = (n as Sample) / sample_rate * 1000.0;
        }
    }
    let final_density = eta_at(ir, len - 1, w);
    (mixing, final_density, converged)
}

/// The measured echo-density summary of a room impulse response.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EchoDensityProfile {
    /// Measured mixing time in milliseconds, or [`NO_MIXING_TIME_MS`] when the
    /// tail never reaches [`MIXING_THRESHOLD`].
    pub mixing_time_ms: Sample,
    /// Normalized echo density at the final sample of the response.
    pub final_density: Sample,
    /// Whether the response reached [`MIXING_THRESHOLD`] (became diffuse).
    pub converged: bool,
}

impl EchoDensityProfile {
    /// Computes the echo-density summary from a broadband room impulse response
    /// sampled at `sample_rate` hertz.
    ///
    /// Degenerate inputs report [`NO_MIXING_TIME_MS`], a final density of `0`,
    /// and `converged = false`.
    ///
    /// ```
    /// use prism_audio_spatial::echo_density::EchoDensityProfile;
    ///
    /// // A dense pseudo-random tail becomes diffuse and reports a mixing time.
    /// let sr = 48_000.0f32;
    /// let mut ir = vec![0.0f32; 8_000];
    /// let mut state = 0x1234_5678u32;
    /// for (i, s) in ir.iter_mut().enumerate() {
    ///     if i == 0 {
    ///         *s = 1.0;
    ///     } else {
    ///         state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ///         *s = (state >> 9) as f32 / 8_388_608.0 - 1.0;
    ///     }
    /// }
    /// let profile = EchoDensityProfile::from_impulse_response(&ir, sr);
    /// assert!(profile.converged);
    /// assert!(profile.final_density > 0.0);
    /// ```
    #[must_use]
    pub fn from_impulse_response(ir: &[Sample], sample_rate: Sample) -> Self {
        let (mixing_time_ms, final_density, converged) = compute(ir, sample_rate);
        Self {
            mixing_time_ms,
            final_density,
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const SR: Sample = 48_000.0;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    /// A uniform pseudo-random tail with a direct hit at index `0`.
    fn noisy_ir(len: usize) -> Vec<Sample> {
        let mut ir = vec![0.0; len];
        let mut state = 0x9E37_79B9u32;
        for (i, s) in ir.iter_mut().enumerate() {
            if i == 0 {
                *s = 1.0;
            } else {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *s = (state >> 9) as Sample / 8_388_608.0 - 1.0;
            }
        }
        ir
    }

    /// An approximately Gaussian tail built from the central-limit sum of
    /// twelve uniform draws (mean zero, unit variance).
    fn gaussian_ir(len: usize) -> Vec<Sample> {
        let mut ir = vec![0.0; len];
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 9) as Sample / 8_388_608.0
        };
        for s in ir.iter_mut() {
            let mut acc = 0.0;
            for _ in 0..12 {
                acc += next();
            }
            *s = acc - 6.0;
        }
        ir
    }

    #[test]
    fn dense_tail_converges() {
        let ir = noisy_ir(8_000);
        let profile = EchoDensityProfile::from_impulse_response(&ir, SR);
        assert!(profile.converged);
        assert!(profile.mixing_time_ms >= 0.0);
    }

    #[test]
    fn gaussian_tail_density_near_one() {
        let ir = gaussian_ir(8_000);
        let w = window_half_samples(SR);
        let eta = eta_at(&ir, 4_000, w);
        assert!(approx(eta, 1.0, 0.2), "eta {eta}");
    }

    #[test]
    fn sparse_reflections_have_low_density() {
        // A few isolated reflections are far from a diffuse Gaussian tail.
        let mut ir = vec![0.0; 8_000];
        ir[0] = 1.0;
        ir[1_000] = 0.5;
        ir[2_000] = 0.3;
        let w = window_half_samples(SR);
        let eta = eta_at(&ir, 1_000, w);
        assert!(eta < MIXING_THRESHOLD, "eta {eta}");
    }

    #[test]
    fn empty_response_is_sentinel() {
        let empty: [Sample; 0] = [];
        assert_eq!(mixing_time_ms(&empty, SR), NO_MIXING_TIME_MS);
        let profile = EchoDensityProfile::from_impulse_response(&empty, SR);
        assert_eq!(profile.mixing_time_ms, NO_MIXING_TIME_MS);
        assert!(!profile.converged);
    }

    #[test]
    fn all_zero_response_is_sentinel() {
        let ir = vec![0.0; 4_000];
        assert_eq!(mixing_time_ms(&ir, SR), NO_MIXING_TIME_MS);
        let profile = EchoDensityProfile::from_impulse_response(&ir, SR);
        assert_eq!(profile.mixing_time_ms, NO_MIXING_TIME_MS);
        assert_eq!(profile.final_density, 0.0);
        assert!(!profile.converged);
    }

    #[test]
    fn zero_sample_rate_is_sentinel() {
        let ir = noisy_ir(4_000);
        assert_eq!(mixing_time_ms(&ir, 0.0), NO_MIXING_TIME_MS);
    }

    #[test]
    fn nan_sample_rate_is_sentinel() {
        let ir = noisy_ir(4_000);
        assert_eq!(mixing_time_ms(&ir, Sample::NAN), NO_MIXING_TIME_MS);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut ir = noisy_ir(4_000);
        ir[5] = Sample::NAN;
        ir[100] = Sample::INFINITY;
        let profile = EchoDensityProfile::from_impulse_response(&ir, SR);
        assert!(profile.mixing_time_ms.is_finite() || profile.mixing_time_ms == NO_MIXING_TIME_MS);
        assert!(profile.final_density.is_finite());
    }

    #[test]
    fn from_impulse_response_matches_free_function() {
        let ir = noisy_ir(6_000);
        let profile = EchoDensityProfile::from_impulse_response(&ir, SR);
        assert!(approx(profile.mixing_time_ms, mixing_time_ms(&ir, SR), 1e-6));
    }

    #[test]
    fn profile_values_finite_non_negative() {
        let ir = noisy_ir(2_000);
        let mut out = vec![0.0; ir.len()];
        normalized_echo_density(&ir, SR, &mut out);
        for &v in &out {
            assert!(v.is_finite() && v >= 0.0, "v {v}");
        }
    }

    #[test]
    fn profile_zeroes_slots_beyond_response() {
        let ir = noisy_ir(1_000);
        let mut out = vec![9.0; ir.len() + 50];
        normalized_echo_density(&ir, SR, &mut out);
        for &v in &out[ir.len()..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn degenerate_profile_all_zero() {
        let ir = noisy_ir(1_000);
        let mut out = vec![3.0; ir.len()];
        normalized_echo_density(&ir, 0.0, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn earlier_tail_mixes_sooner() {
        // A tail that is dense from the start mixes no later than one whose
        // dense region is delayed by a long sparse gap.
        let dense = noisy_ir(8_000);
        let mut delayed = vec![0.0; 8_000];
        delayed[0] = 1.0;
        let noisy = noisy_ir(4_000);
        delayed[4_000..8_000].copy_from_slice(&noisy[..4_000]);
        let a = EchoDensityProfile::from_impulse_response(&dense, SR);
        let b = EchoDensityProfile::from_impulse_response(&delayed, SR);
        assert!(a.converged && b.converged);
        assert!(a.mixing_time_ms <= b.mixing_time_ms + 1e-3);
    }

    #[test]
    fn default_is_zero() {
        let profile = EchoDensityProfile::default();
        assert_eq!(profile.mixing_time_ms, 0.0);
        assert_eq!(profile.final_density, 0.0);
        assert!(!profile.converged);
    }

    #[test]
    fn constants_are_stable() {
        assert_eq!(ECHO_DENSITY_WINDOW_MS, 10.0);
        assert_eq!(GAUSSIAN_EXCEEDANCE, 0.317_310_5);
        assert_eq!(MIXING_THRESHOLD, 1.0);
        assert_eq!(NO_MIXING_TIME_MS, -1.0);
    }
}
