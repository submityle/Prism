//! Parameter identity and the blending domain of a parameter's value.
//!
//! A snapshot targets individual mixer, bus, and effect parameters identified
//! by a stable [`ParameterId`]. Every parameter also carries a
//! [`ParameterKind`] that declares the mathematical domain in which its value
//! must be interpolated and weight-combined: a raw linear gain blends
//! arithmetically, a decibel value blends in the logarithmic amplitude domain
//! it is already expressed in (so arithmetic on the stored number is correct),
//! a frequency in hertz blends geometrically so that an octave sweep is
//! perceptually even, and a dimensionless ratio blends arithmetically.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supplies the identity and domain tags consumed by [`crate::target`],
//! [`crate::blend`], and [`crate::transition`]. The hertz geometric path and
//! the decibel arithmetic path reuse `bevy_math::ops` for deterministic
//! `ln`/`exp` and reuse the decibel convention of `prism_audio_core::math`.

use bevy_math::ops;
use prism_audio_core::math::Sample;

/// Opaque, stable identifier for one mixer, bus, or effect parameter.
///
/// The numeric value is minted by the authoring layer and is interpreted by
/// the runtime only for equality, ordering, and table lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct ParameterId(pub u32);

impl ParameterId {
    /// Wraps a raw numeric handle.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric handle.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for ParameterId {
    fn from(raw: u32) -> Self {
        Self(raw)
    }
}

impl From<ParameterId> for u32 {
    fn from(id: ParameterId) -> Self {
        id.0
    }
}

/// The mathematical domain in which a parameter's value is blended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterKind {
    /// Raw linear amplitude or an arbitrary linear quantity; blends
    /// arithmetically.
    Linear,
    /// A value already expressed in decibels; blends arithmetically on the
    /// stored decibel number, which is the logarithmic amplitude domain.
    Decibel,
    /// A frequency in hertz; blends geometrically (interpolating the natural
    /// logarithm) so equal fractional steps sound even.
    Hertz,
    /// A dimensionless ratio such as a send fraction; blends arithmetically.
    Ratio,
}

impl ParameterKind {
    /// Interpolates from `a` to `b` by the shaped position `t`, in this kind's
    /// domain. `t` is clamped to `[0, 1]`.
    ///
    /// Linear, [`Decibel`](Self::Decibel), and [`Ratio`](Self::Ratio) blend
    /// arithmetically. [`Hertz`](Self::Hertz) blends geometrically when both
    /// endpoints are strictly positive, falling back to arithmetic blending
    /// otherwise (a non-positive frequency has no logarithm).
    #[must_use]
    pub fn interpolate(self, a: Sample, b: Sample, t: Sample) -> Sample {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear | Self::Decibel | Self::Ratio => a + (b - a) * t,
            Self::Hertz => {
                if a > 0.0 && b > 0.0 {
                    let la = ops::ln(a);
                    let lb = ops::ln(b);
                    ops::exp(la + (lb - la) * t)
                } else {
                    a + (b - a) * t
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn id_round_trips_through_u32() {
        let id = ParameterId::new(7);
        assert_eq!(id.get(), 7);
        assert_eq!(ParameterId::from(9u32), ParameterId(9));
        let raw: u32 = ParameterId::new(11).into();
        assert_eq!(raw, 11);
    }

    #[test]
    fn linear_interpolates_arithmetically() {
        let v = ParameterKind::Linear.interpolate(0.0, 10.0, 0.25);
        assert!((v - 2.5).abs() < EPS);
    }

    #[test]
    fn decibel_interpolates_arithmetically_on_stored_db() {
        let v = ParameterKind::Decibel.interpolate(-12.0, 0.0, 0.5);
        assert!((v - (-6.0)).abs() < EPS);
    }

    #[test]
    fn ratio_interpolates_arithmetically() {
        let v = ParameterKind::Ratio.interpolate(0.2, 0.8, 0.5);
        assert!((v - 0.5).abs() < EPS);
    }

    #[test]
    fn hertz_interpolates_geometrically() {
        // Midpoint of 100 Hz and 400 Hz in log space is 200 Hz.
        let v = ParameterKind::Hertz.interpolate(100.0, 400.0, 0.5);
        assert!((v - 200.0).abs() < 1e-3);
    }

    #[test]
    fn hertz_falls_back_to_linear_for_non_positive() {
        let v = ParameterKind::Hertz.interpolate(0.0, 400.0, 0.5);
        assert!((v - 200.0).abs() < EPS);
    }

    #[test]
    fn interpolate_clamps_t() {
        let lo = ParameterKind::Linear.interpolate(1.0, 5.0, -2.0);
        let hi = ParameterKind::Linear.interpolate(1.0, 5.0, 4.0);
        assert!((lo - 1.0).abs() < EPS);
        assert!((hi - 5.0).abs() < EPS);
    }
}
