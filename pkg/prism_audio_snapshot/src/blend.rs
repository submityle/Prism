//! Domain-aware interpolation and weighted combination of parameter values.
//!
//! This is a pure, stateless layer. Every function receives the
//! [`ParameterKind`] of the value(s) it operates on and performs the blend in
//! that kind's domain, so callers never have to special-case decibels or
//! hertz. All deterministic transcendental math routes through
//! `bevy_math::ops` so results are reproducible across platforms.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Wraps the domain logic of [`crate::parameter::ParameterKind`]. Used by
//! [`crate::transition`] (pairwise interpolation) and [`crate::stack`]
//! (weighted multi-snapshot combination).

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::parameter::ParameterKind;

/// Interpolates from `a` to `b` by `t` in the domain of `kind`.
///
/// `t` is clamped to `[0, 1]`. See [`ParameterKind::interpolate`] for the
/// per-domain rules.
#[must_use]
pub fn interpolate(kind: ParameterKind, a: Sample, b: Sample, t: Sample) -> Sample {
    kind.interpolate(a, b, t)
}

/// Combines weighted values in the domain of `kind`.
///
/// Weights are normalized by their sum. Linear, [`Decibel`](ParameterKind::Decibel),
/// and [`Ratio`](ParameterKind::Ratio) take the weighted arithmetic mean of the
/// stored numbers. [`Hertz`](ParameterKind::Hertz) takes the weighted
/// geometric mean (arithmetic mean of the logarithms, then `exp`) when every
/// value is strictly positive, falling back to the arithmetic mean otherwise.
///
/// Returns `None` when the slice is empty or the weights sum to zero or less.
#[must_use]
pub fn blend_weighted(kind: ParameterKind, values: &[(Sample, Sample)]) -> Option<Sample> {
    let total: Sample = values.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return None;
    }
    let combined = match kind {
        ParameterKind::Linear | ParameterKind::Decibel | ParameterKind::Ratio => {
            let acc: Sample = values.iter().map(|&(v, w)| v * w).sum();
            acc / total
        }
        ParameterKind::Hertz => {
            if values.iter().all(|&(v, _)| v > 0.0) {
                let acc: Sample = values.iter().map(|&(v, w)| ops::ln(v) * w).sum();
                ops::exp(acc / total)
            } else {
                let acc: Sample = values.iter().map(|&(v, w)| v * w).sum();
                acc / total
            }
        }
    };
    Some(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn interpolate_delegates_to_kind() {
        let v = interpolate(ParameterKind::Hertz, 100.0, 400.0, 0.5);
        assert!((v - 200.0).abs() < 1e-3);
    }

    #[test]
    fn blend_weighted_normalizes_weights() {
        // Weights 1 and 3 over values 0 and 8 -> 0.75 of the way to 8 = 6.0.
        let v = blend_weighted(ParameterKind::Linear, &[(0.0, 1.0), (8.0, 3.0)])
            .expect("non-zero weights");
        assert!((v - 6.0).abs() < EPS);
    }

    #[test]
    fn blend_weighted_decibel_is_arithmetic() {
        let v = blend_weighted(ParameterKind::Decibel, &[(-12.0, 1.0), (0.0, 1.0)])
            .expect("non-zero weights");
        assert!((v - (-6.0)).abs() < EPS);
    }

    #[test]
    fn blend_weighted_hertz_is_geometric() {
        // Geometric mean of 100 and 400 is 200.
        let v = blend_weighted(ParameterKind::Hertz, &[(100.0, 1.0), (400.0, 1.0)])
            .expect("non-zero weights");
        assert!((v - 200.0).abs() < 1e-2);
    }

    #[test]
    fn blend_weighted_hertz_falls_back_when_non_positive() {
        let v = blend_weighted(ParameterKind::Hertz, &[(0.0, 1.0), (400.0, 1.0)])
            .expect("non-zero weights");
        assert!((v - 200.0).abs() < EPS);
    }

    #[test]
    fn blend_weighted_zero_weights_returns_none() {
        assert!(blend_weighted(ParameterKind::Linear, &[(1.0, 0.0), (2.0, 0.0)]).is_none());
        assert!(blend_weighted(ParameterKind::Linear, &[]).is_none());
    }

    #[test]
    fn blend_weighted_single_value_is_identity() {
        let v = blend_weighted(ParameterKind::Linear, &[(3.3, 0.2)]).expect("non-zero weight");
        assert!((v - 3.3).abs() < EPS);
    }
}
