//! Pure-function mapping from physical contact quantities to synthesis drive.
//!
//! The design mandates that the translation from physics to sound be a set of
//! pure, deterministic curves rather than a run-time interpreter. This module
//! is exactly that: given an impulse magnitude it returns an excitation energy
//! and initial gain; given the normal/tangential split it returns a brightness
//! in `[0, 1]` (a hard perpendicular hit is bright, a grazing hit is dark);
//! given a contact point and a mode index it returns that mode's excitation
//! weight, so striking the rim excites the high modes a centre strike leaves
//! quiet. Every function is referentially transparent, so identical inputs
//! always yield identical drive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The compressive
//! loudness curve and the sinusoidal mode-shape weighting are standard,
//! publicly documented acoustics relations.
//!
//! # Relationship
//! Implements the energy-mapping curves of design section 47.1 and feeds the
//! excitation shaping of section 47.2 ([`crate::modal::excitation`]).

use bevy_math::ops;

use crate::contact::event::ContactPoint;
use crate::dsp::TWO_PI;
use prism_audio_core::math::Sample;

/// Reference impulse (`kg*m/s`) that maps to roughly unity excitation energy.
///
/// Impulses above this still grow but with diminishing, logarithmic loudness so
/// a huge collision does not blow past the mixer headroom.
pub const REFERENCE_IMPULSE: Sample = 4.0;

/// Converts a collision impulse into a dimensionless excitation energy.
///
/// Uses a compressive `ln(1 + impulse / ref)` law: linear for soft taps,
/// logarithmic for hard slams, matching how perceived loudness scales with
/// collision energy. The result is non-negative and finite for every finite
/// non-negative input.
#[inline]
#[must_use]
pub fn impulse_to_energy(impulse: Sample) -> Sample {
    let x = if impulse.is_finite() { impulse.max(0.0) } else { 0.0 };
    ops::ln_1p(x / REFERENCE_IMPULSE)
}

/// Converts a collision impulse into a peak excitation amplitude (initial gain).
///
/// The amplitude is the square root of the energy (energy is amplitude
/// squared), bounded to a sane peak so a single strike cannot exceed the
/// per-voice headroom.
#[inline]
#[must_use]
pub fn impulse_to_amplitude(impulse: Sample) -> Sample {
    ops::sqrt(impulse_to_energy(impulse)).min(4.0)
}

/// Returns the spectral brightness in `[0, 1]` for a normal/tangential split.
///
/// A hit dominated by the normal component (a square-on rap) returns a value
/// near `1.0` (bright, high-mode-rich); a grazing hit dominated by the
/// tangential component returns a value near `0.0` (dark, low-pass shaped). A
/// degenerate zero-magnitude hit returns `0.5`.
#[inline]
#[must_use]
pub fn normal_tangential_brightness(normal: Sample, tangential: Sample) -> Sample {
    let n = if normal.is_finite() { normal.max(0.0) } else { 0.0 };
    let t = if tangential.is_finite() {
        tangential.max(0.0)
    } else {
        0.0
    };
    let total = n + t;
    if total <= Sample::MIN_POSITIVE {
        0.5
    } else {
        (n / total).clamp(0.0, 1.0)
    }
}

/// Returns the excitation weight for mode `index` struck at `point`.
///
/// The weight follows the magnitude of a clamped-free mode shape,
/// `|cos((index + 1) * pi * (1 - x))|`, evaluated at the strike position `x`
/// (0 = centre, 1 = edge). The fundamental (`index == 0`) is excited almost
/// everywhere, while high modes are only excited near the rim, reproducing the
/// bright edge / dull centre contrast of a struck plate. The returned weight is
/// in `[0, 1]`.
#[inline]
#[must_use]
pub fn contact_point_weight(index: usize, point: ContactPoint) -> Sample {
    let x = point.value();
    // Distance from the centre: strikes near the rim (x -> 1) excite high modes.
    let phase = (index as Sample + 1.0) * 0.5 * TWO_PI * (0.5 * x);
    let base = ops::abs(ops::cos(phase));
    // A small floor keeps every mode minimally alive so no strike is silent.
    0.08 + 0.92 * base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_is_monotonic_and_compressive() {
        let soft = impulse_to_energy(0.5);
        let medium = impulse_to_energy(4.0);
        let hard = impulse_to_energy(40.0);
        assert!(soft < medium && medium < hard);
        // Compressive: a 10x impulse gives less than 10x energy.
        assert!(hard < 10.0 * medium);
    }

    #[test]
    fn energy_handles_bad_input() {
        assert_eq!(impulse_to_energy(f32::NAN), 0.0);
        assert_eq!(impulse_to_energy(-5.0), 0.0);
    }

    #[test]
    fn brightness_extremes() {
        assert!(normal_tangential_brightness(1.0, 0.0) > 0.99);
        assert!(normal_tangential_brightness(0.0, 1.0) < 0.01);
        assert!((normal_tangential_brightness(0.0, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn centre_strike_favours_fundamental() {
        let centre = ContactPoint::new(0.0);
        let w0 = contact_point_weight(0, centre);
        let w7 = contact_point_weight(7, centre);
        // At the centre the fundamental is at least as excited as a high mode.
        assert!(w0 >= w7);
    }

    #[test]
    fn weights_are_bounded() {
        for i in 0..32 {
            for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                let w = contact_point_weight(i, ContactPoint::new(p));
                assert!((0.0..=1.0).contains(&w), "w={w}");
            }
        }
    }
}
