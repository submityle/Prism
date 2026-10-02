//! Shared fractional-delay interpolation kernel.
//!
//! This module is the common DSP core that every primitive in the crate reuses:
//! the graded resamplers, the WSOLA/SOLA stretcher, and the phase vocoder all
//! read samples at non-integer positions and therefore all need the same
//! band-limited interpolation. Two interpolators are provided: a cheap linear
//! blend (far-field / LOD grade) and a windowed-sinc fractional interpolator
//! whose coefficients are precomputed at construction so the real-time hot path
//! performs no allocation.
//!
//! The windowed-sinc design is the textbook fractional-delay filter: a shifted
//! cardinal sine (`sinc`) truncated to a finite support and tapered by a
//! Blackman window to suppress the truncation ripple. A dense polyphase table
//! stores one coefficient row per sub-sample phase; the hot path selects the
//! nearest row and evaluates a fixed-length dot product.
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only. The windowed-sinc fractional-delay filter and the Blackman
//! window are standard, publicly documented signal-processing constructions
//! (for example Oppenheim and Schafer, and the Smith "Digital Audio Resampling"
//! notes). All transcendental math routes through [`bevy_math::ops`] so the
//! tables are bit-reproducible across platforms.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`] (reuses its [`Sample`] scalar). This is
//! the shared interpolation core consumed by
//! [`crate::linear`], [`crate::polyphase_sinc`], [`crate::high_order_sinc`],
//! [`crate::wsola`], and [`crate::phase_vocoder`]. It is the time-domain
//! counterpart that lets the three families (sample-rate conversion, time
//! stretching, and a continuously varying delay line) share one kernel.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::PI;

use prism_audio_core::math::Sample;

/// Floating-point tolerance for the never-compare-floats-for-equality rule.
pub const EPS: Sample = 1.0e-6;

/// Returns `true` when `a` and `b` differ by at most [`EPS`].
///
/// The crate never asserts exact floating-point equality; this helper is used
/// everywhere a near-equality check is required (including in tests).
#[inline]
#[must_use]
pub fn close(a: Sample, b: Sample) -> bool {
    ops::abs(a - b) <= EPS
}

/// Replaces a non-finite sample (`NaN`/`+/-inf`) with silence.
///
/// Every public entry point sanitizes its input through this helper so a
/// poisoned upstream value can never propagate a `NaN` through the filter state
/// or trigger a panic in the hot path.
#[inline]
#[must_use]
pub fn sanitize(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Linearly interpolates between `a` (at fraction `0.0`) and `b` (at `1.0`).
///
/// This is the lowest-quality, lowest-cost interpolation grade. It is exact at
/// the endpoints and introduces a mild low-pass tilt in between.
#[inline]
#[must_use]
pub fn linear_interp(a: Sample, b: Sample, frac: Sample) -> Sample {
    a + (b - a) * frac
}

/// Normalized cardinal sine `sin(pi * x) / (pi * x)`, with `sinc(0) = 1`.
#[inline]
#[must_use]
fn sinc(x: Sample) -> Sample {
    if close(x, 0.0) {
        1.0
    } else {
        let px = PI * x;
        ops::sin(px) / px
    }
}

/// Blackman window sampled at `t` over the symmetric support `[-half, half]`.
///
/// Returns `0.0` outside the support. The Blackman window trades a slightly
/// wider main lobe for roughly `-58 dB` side lobes, which keeps resampling
/// images and interpolation ripple well below the noise floor.
#[inline]
#[must_use]
fn blackman(t: Sample, half: Sample) -> Sample {
    if ops::abs(t) > half {
        return 0.0;
    }
    // Map t in [-half, half] to the standard window phase [0, 2*pi].
    let phase = PI * (t + half) / half;
    0.42 - 0.5 * ops::cos(phase) + 0.08 * ops::cos(2.0 * phase)
}

/// Evaluates a Blackman-windowed sinc low-pass tap at continuous offset `t`.
///
/// `half` is the one-sided support in samples and `cutoff` is the normalized
/// cutoff frequency (`1.0` == input Nyquist). Values of `cutoff` below `1.0`
/// lower the pass band, which is how the resamplers reject aliasing when they
/// decimate. The returned value is a raw (un-normalized) coefficient; callers
/// that need unity DC gain divide by the running coefficient sum.
#[inline]
#[must_use]
pub fn windowed_sinc(t: Sample, half: Sample, cutoff: Sample) -> Sample {
    cutoff * sinc(cutoff * t) * blackman(t, half)
}

/// Precomputed polyphase windowed-sinc fractional-delay interpolator.
///
/// The interpolator stores `phases + 1` coefficient rows, each `taps` long,
/// covering sub-sample delays from `0.0` to `1.0`. Each row is normalized to
/// unity DC gain. The hot path ([`SincInterpolator::interpolate`]) selects the
/// nearest row and evaluates one fixed-length dot product with no allocation.
#[derive(Clone, Debug)]
pub struct SincInterpolator {
    /// One-sided support in whole samples; the filter spans `2 * half` taps.
    half: usize,
    /// Number of taps per phase row (`2 * half`).
    taps: usize,
    /// Number of sub-sample phase subdivisions.
    phases: usize,
    /// Flattened coefficient table of length `(phases + 1) * taps`.
    coeffs: Vec<Sample>,
}

impl SincInterpolator {
    /// Builds an interpolator with `half` taps per side and `phases` sub-sample
    /// rows.
    ///
    /// `half` is clamped to at least `1` and `phases` to at least `1`. The
    /// unity cutoff (`1.0`) is used because a fractional-delay interpolator must
    /// pass the entire input band unchanged.
    #[must_use]
    pub fn new(half: usize, phases: usize) -> Self {
        let half = half.max(1);
        let phases = phases.max(1);
        let taps = 2 * half;
        let mut coeffs = Vec::with_capacity((phases + 1) * taps);
        let center = (half - 1) as Sample;
        for p in 0..=phases {
            let frac = p as Sample / phases as Sample;
            // Accumulate the row, then normalize to unity DC gain.
            let start = coeffs.len();
            let mut sum = 0.0;
            for j in 0..taps {
                let t = (j as Sample - center) - frac;
                let c = windowed_sinc(t, half as Sample, 1.0);
                coeffs.push(c);
                sum += c;
            }
            if ops::abs(sum) > EPS {
                let inv = 1.0 / sum;
                for value in &mut coeffs[start..start + taps] {
                    *value *= inv;
                }
            }
        }
        Self {
            half,
            taps,
            phases,
            coeffs,
        }
    }

    /// One-sided support in whole samples.
    #[inline]
    #[must_use]
    pub fn half(&self) -> usize {
        self.half
    }

    /// Number of taps the [`SincInterpolator::interpolate`] window must supply.
    #[inline]
    #[must_use]
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Number of sub-sample phase subdivisions.
    #[inline]
    #[must_use]
    pub fn phases(&self) -> usize {
        self.phases
    }

    /// Interpolates a value at sub-sample position `frac` in `[0, 1]`.
    ///
    /// `window` must contain exactly [`SincInterpolator::taps`] samples ordered
    /// oldest to newest; the interpolated point lies between index `half - 1`
    /// and index `half`. `frac` is clamped to `[0, 1]`. Non-finite window
    /// samples are treated as silence. Allocation free and panic free.
    #[inline]
    #[must_use]
    pub fn interpolate(&self, window: &[Sample], frac: Sample) -> Sample {
        if window.len() < self.taps {
            return 0.0;
        }
        let frac = frac.clamp(0.0, 1.0);
        let row = ((frac * self.phases as Sample) + 0.5) as usize;
        let row = row.min(self.phases);
        let base = row * self.taps;
        let mut acc = 0.0;
        let mut j = 0;
        while j < self.taps {
            acc += self.coeffs[base + j] * sanitize(window[j]);
            j += 1;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn linear_interp_hits_endpoints() {
        assert!(close(linear_interp(2.0, 6.0, 0.0), 2.0));
        assert!(close(linear_interp(2.0, 6.0, 1.0), 6.0));
        assert!(close(linear_interp(2.0, 6.0, 0.5), 4.0));
    }

    #[test]
    fn sinc_is_unity_at_origin() {
        assert!(close(sinc(0.0), 1.0));
        // sinc is zero at non-zero integers.
        assert!(close(sinc(1.0), 0.0));
        assert!(close(sinc(2.0), 0.0));
    }

    #[test]
    fn windowed_sinc_vanishes_outside_support() {
        assert!(close(windowed_sinc(5.0, 4.0, 1.0), 0.0));
        assert!(close(windowed_sinc(-5.0, 4.0, 1.0), 0.0));
    }

    #[test]
    fn rows_have_unity_dc_gain() {
        let interp = SincInterpolator::new(8, 64);
        // A constant window must pass through unchanged at every phase.
        let window = vec![1.0 as Sample; interp.taps()];
        for p in 0..=interp.phases() {
            let frac = p as Sample / interp.phases() as Sample;
            assert!(close(interp.interpolate(&window, frac), 1.0));
        }
    }

    #[test]
    fn interpolation_passes_dc_offset() {
        let interp = SincInterpolator::new(4, 32);
        let window = vec![-3.0 as Sample; interp.taps()];
        assert!(close(interp.interpolate(&window, 0.37), -3.0));
    }

    #[test]
    fn non_finite_window_is_finite_out() {
        let interp = SincInterpolator::new(4, 16);
        let mut window = vec![0.0 as Sample; interp.taps()];
        window[2] = Sample::NAN;
        window[3] = Sample::INFINITY;
        let y = interp.interpolate(&window, 0.5);
        assert!(y.is_finite());
    }

    #[test]
    fn frac_endpoints_match_sample_values() {
        // With frac == 0 the interpolated point sits exactly on index half-1.
        let interp = SincInterpolator::new(4, 128);
        let mut window = vec![0.0 as Sample; interp.taps()];
        let center = interp.half() - 1;
        window[center] = 1.0;
        let y = interp.interpolate(&window, 0.0);
        assert!(close(y, 1.0));
    }
}
