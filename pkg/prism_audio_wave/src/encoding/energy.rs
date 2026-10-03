//! Energy measures of an impulse response: the direct, early, and late energy
//! windows and the derived occlusion gain and spectral low-pass corner.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "direct/early energy -> occlusion gain + low-pass cutoff"
//! half of the perceptual encoding in design section 43. The direct energy
//! ratio against a free-field reference yields an occlusion gain; the spectral
//! tilt of the direct arrival yields a diffraction low-pass corner. Both feed
//! [`crate::encoding::PerceptualParams`].

use alloc::vec;
use alloc::vec::Vec;
use bevy_math::ops;
use prism_audio_core::fft::Fft;

use crate::solver::ImpulseResponse;

/// Energy inside the direct-arrival window: a short span of `window_samples`
/// starting at `onset`.
#[must_use]
pub fn direct_energy(ir: &ImpulseResponse, onset: usize, window_samples: usize) -> f32 {
    ir.window_energy(onset, onset.saturating_add(window_samples.max(1)))
}

/// Energy inside the early window (direct plus early reflections): a span of
/// `window_samples` starting at `onset`.
#[must_use]
pub fn early_energy(ir: &ImpulseResponse, onset: usize, window_samples: usize) -> f32 {
    ir.window_energy(onset, onset.saturating_add(window_samples.max(1)))
}

/// Reverberant energy: everything after the early window ends.
#[must_use]
pub fn reverberant_energy(ir: &ImpulseResponse, onset: usize, early_samples: usize) -> f32 {
    let start = onset.saturating_add(early_samples.max(1));
    ir.window_energy(start, ir.len())
}

/// Occlusion direct gain in `[0, 1]`: the square root of the direct energy
/// ratio against a free-field `reference` energy.
///
/// A non-positive reference means "unknown free field", in which case the
/// path is treated as fully open and the gain is `1`.
#[must_use]
pub fn occlusion_gain(direct: f32, reference: f32) -> f32 {
    if reference <= 0.0 {
        return 1.0;
    }
    let ratio = (direct / reference).max(0.0);
    ops::sqrt(ratio).min(1.0)
}

/// Spectral tilt of a short signal in `[0, 1]`: the fraction of (non-`DC`)
/// spectral energy that lies in the upper half of the represented band.
///
/// A bright, open arrival keeps high-frequency energy and tilts toward `1`; a
/// diffracted arrival loses its highs and tilts toward `0`. The transform is
/// an in-crate [`Fft`] over the zero-padded window and allocates, so this is
/// an offline-only measure. The sum runs over bins `1..=n/2`, so the Nyquist
/// bin at the top of the band is counted and a fully alternating signal reads
/// a tilt of `1`.
#[must_use]
pub fn spectral_tilt(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let fft = Fft::new(samples.len());
    let n = fft.size();
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for (dst, src) in re.iter_mut().zip(samples.iter()) {
        *dst = *src;
    }
    fft.forward(&mut re, &mut im);
    let half = (n / 2).max(1);
    let mut total = 0.0;
    let mut high = 0.0;
    // Skip bin 0 (DC): tilt is a shape measure, not an offset measure. The
    // loop is inclusive of the Nyquist bin at index `half` so the brightest
    // band edge contributes.
    for k in 1..=half {
        let mag2 = re[k] * re[k] + im[k] * im[k];
        total += mag2;
        if k >= half / 2 {
            high += mag2;
        }
    }
    if total <= 0.0 {
        0.0
    } else {
        (high / total).clamp(0.0, 1.0)
    }
}

/// Maps a spectral `tilt` in `[0, 1]` to a low-pass corner, log-interpolating
/// between `min_hz` (fully diffracted) and `max_hz` (fully open).
#[must_use]
pub fn cutoff_from_tilt(tilt: f32, min_hz: f32, max_hz: f32) -> f32 {
    let lo = min_hz.max(1.0);
    let hi = max_hz.max(lo);
    let t = tilt.clamp(0.0, 1.0);
    lo * ops::exp(t * ops::ln(hi / lo))
}

/// Convenience wrapper computing the direct low-pass corner directly from the
/// direct-arrival window of `ir`.
#[must_use]
pub fn direct_cutoff(
    ir: &ImpulseResponse,
    onset: usize,
    window_samples: usize,
    min_hz: f32,
    max_hz: f32,
) -> f32 {
    let end = onset.saturating_add(window_samples.max(1)).min(ir.len());
    if onset >= end {
        return max_hz.max(min_hz.max(1.0));
    }
    let window: Vec<f32> = ir.samples()[onset..end].to_vec();
    cutoff_from_tilt(spectral_tilt(&window), min_hz, max_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn occlusion_gain_is_bounded_and_monotonic() {
        assert!(approx(occlusion_gain(1.0, 1.0), 1.0, 1e-6));
        assert!(approx(occlusion_gain(0.0, 1.0), 0.0, 1e-6));
        // Half the energy -> gain sqrt(0.5).
        assert!(approx(occlusion_gain(0.5, 1.0), ops::sqrt(0.5), 1e-6));
        // Over-unity energy clamps to 1.
        assert!(approx(occlusion_gain(4.0, 1.0), 1.0, 1e-6));
        // Unknown reference -> open.
        assert!(approx(occlusion_gain(0.3, 0.0), 1.0, 1e-6));
    }

    #[test]
    fn bright_signal_tilts_higher_than_smooth_signal() {
        // Alternating sign puts all energy at the band edge (Nyquist).
        let bright = [1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        // A slow half-cosine concentrates energy low in the band.
        let smooth = [0.0_f32, 0.38, 0.71, 0.92, 1.0, 0.92, 0.71, 0.38];
        let tb = spectral_tilt(&bright);
        let ts = spectral_tilt(&smooth);
        assert!(tb > ts, "bright tilt {tb} should exceed smooth tilt {ts}");
    }

    #[test]
    fn cutoff_is_monotonic_in_tilt() {
        let lo = cutoff_from_tilt(0.0, 250.0, 20_000.0);
        let mid = cutoff_from_tilt(0.5, 250.0, 20_000.0);
        let hi = cutoff_from_tilt(1.0, 250.0, 20_000.0);
        assert!(approx(lo, 250.0, 1e-2));
        assert!(approx(hi, 20_000.0, 1e-1));
        assert!(mid > lo && mid < hi);
    }

    #[test]
    fn energy_windows_partition_the_response() {
        let ir = ImpulseResponse::new(1000.0, vec![0.0, 1.0, 1.0, 0.5, 0.25, 0.1]);
        let direct = direct_energy(&ir, 1, 2);
        let reverb = reverberant_energy(&ir, 1, 2);
        let total = ir.total_energy();
        // Direct window [1,3), reverb window [3, end): they omit sample 0 only.
        assert!(approx(direct + reverb, total, 1e-6));
    }
}
