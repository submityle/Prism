//! Hybrid propagation: overlaying the wave backend's perceptual contribution
//! with a geometric backend on the shared spatial parameter bus.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "hybrid propagation" of design section 43. The wave backend
//! owns the low-frequency, diffraction, soft-occlusion, room-coupling, and
//! reverb-tail behaviour; the geometric backend owns high-frequency specular
//! paths and Doppler. [`blend_spatial`] crossfades the two
//! [`prism_audio_spatial::SpatialParams`] contributions field by field under a
//! [`HybridWeights`] policy. It is a pure function: allocation free, lock
//! free, and panic free.

use prism_audio_spatial::SpatialParams;

use crate::encoding::lerp_angle;

/// Per-aspect blend weights controlling how much of each parameter comes from
/// the wave backend versus the geometric backend.
///
/// Each weight is in `[0, 1]`, where `0` keeps the geometric value and `1`
/// takes the wave value fully.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HybridWeights {
    /// Blend weight for the direct-path gain and low-pass corner (soft
    /// occlusion and diffraction).
    pub direct: f32,
    /// Blend weight for the arrival azimuth and elevation (early-reflection
    /// direction).
    pub spatial: f32,
    /// Blend weight for the reverb wet send (room coupling and tail).
    pub wet: f32,
}

impl HybridWeights {
    /// A balanced policy: an even split on every aspect.
    pub const BALANCED: Self = Self {
        direct: 0.5,
        spatial: 0.5,
        wet: 0.5,
    };

    /// A wave-dominant policy: the wave backend wins every aspect.
    pub const WAVE_DOMINANT: Self = Self {
        direct: 1.0,
        spatial: 1.0,
        wet: 1.0,
    };

    /// A geometric-dominant policy: the geometric backend wins every aspect.
    pub const GEOMETRIC_DOMINANT: Self = Self {
        direct: 0.0,
        spatial: 0.0,
        wet: 0.0,
    };

    /// Builds a weight set, clamping each component to `[0, 1]`.
    #[must_use]
    pub fn new(direct: f32, spatial: f32, wet: f32) -> Self {
        Self {
            direct: direct.clamp(0.0, 1.0),
            spatial: spatial.clamp(0.0, 1.0),
            wet: wet.clamp(0.0, 1.0),
        }
    }
}

impl Default for HybridWeights {
    #[inline]
    fn default() -> Self {
        Self::BALANCED
    }
}

/// Linearly interpolates two scalars.
#[must_use]
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Blends a geometric and a wave [`SpatialParams`] contribution under
/// `weights`.
///
/// The policy mirrors the physics each backend models best:
///
/// - `direct_gain` crossfades by `weights.direct`.
/// - `direct_cutoff_hz` crossfades toward the *tighter* of the two corners by
///   `weights.direct`, so the wave backend can only add diffraction damping.
/// - `azimuth` / `elevation` crossfade on the unit circle by `weights.spatial`.
/// - `wet_gain` crossfades by `weights.wet`.
/// - `pitch_ratio` stays geometric (Doppler is a geometric concern) and
///   `spread` stays geometric (image width is the geometric backend's role).
///
/// # Examples
///
/// ```
/// # use prism_audio_spatial::{SpatialParams, SpreadParams};
/// # use prism_audio_wave::hybrid::{blend_spatial, HybridWeights};
/// let geo = SpatialParams {
///     direct_gain: 1.0,
///     pitch_ratio: 1.0,
///     azimuth: 0.0,
///     elevation: 0.0,
///     direct_cutoff_hz: 20_000.0,
///     wet_gain: 0.0,
///     spread: SpreadParams::POINT,
/// };
/// let mut wave = geo;
/// wave.direct_gain = 0.0;
/// wave.direct_cutoff_hz = 500.0;
/// let mixed = blend_spatial(geo, wave, HybridWeights::WAVE_DOMINANT);
/// assert!(mixed.direct_gain < 0.01);
/// assert!(mixed.direct_cutoff_hz < 600.0);
/// ```
#[must_use]
pub fn blend_spatial(geo: SpatialParams, wave: SpatialParams, weights: HybridWeights) -> SpatialParams {
    let tighter = geo.direct_cutoff_hz.min(wave.direct_cutoff_hz);
    SpatialParams {
        direct_gain: lerp(geo.direct_gain, wave.direct_gain, weights.direct),
        pitch_ratio: geo.pitch_ratio,
        azimuth: lerp_angle(geo.azimuth, wave.azimuth, weights.spatial),
        elevation: lerp_angle(geo.elevation, wave.elevation, weights.spatial),
        direct_cutoff_hz: lerp(geo.direct_cutoff_hz, tighter, weights.direct),
        wet_gain: lerp(geo.wet_gain, wave.wet_gain, weights.wet),
        spread: geo.spread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_spatial::SpreadParams;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn open() -> SpatialParams {
        SpatialParams {
            direct_gain: 1.0,
            pitch_ratio: 1.2,
            azimuth: 0.0,
            elevation: 0.0,
            direct_cutoff_hz: 20_000.0,
            wet_gain: 0.0,
            spread: SpreadParams::POINT,
        }
    }

    #[test]
    fn weights_clamp() {
        let w = HybridWeights::new(2.0, -1.0, 0.5);
        assert!(approx(w.direct, 1.0, 1e-6));
        assert!(approx(w.spatial, 0.0, 1e-6));
        assert!(approx(w.wet, 0.5, 1e-6));
    }

    #[test]
    fn geometric_dominant_returns_geometric_values() {
        let geo = open();
        let mut wave = open();
        wave.direct_gain = 0.0;
        wave.wet_gain = 1.0;
        let out = blend_spatial(geo, wave, HybridWeights::GEOMETRIC_DOMINANT);
        assert!(approx(out.direct_gain, geo.direct_gain, 1e-6));
        assert!(approx(out.wet_gain, geo.wet_gain, 1e-6));
    }

    #[test]
    fn wave_dominant_takes_wave_values() {
        let geo = open();
        let mut wave = open();
        wave.direct_gain = 0.2;
        wave.wet_gain = 0.8;
        let out = blend_spatial(geo, wave, HybridWeights::WAVE_DOMINANT);
        assert!(approx(out.direct_gain, 0.2, 1e-6));
        assert!(approx(out.wet_gain, 0.8, 1e-6));
    }

    #[test]
    fn pitch_and_spread_stay_geometric() {
        let geo = open();
        let mut wave = open();
        wave.pitch_ratio = 2.0;
        wave.spread = SpreadParams {
            spread: 1.0,
            focus: 1.0,
            half_width: 1.0,
        };
        let out = blend_spatial(geo, wave, HybridWeights::WAVE_DOMINANT);
        assert!(approx(out.pitch_ratio, geo.pitch_ratio, 1e-6));
        assert!(approx(out.spread.half_width, geo.spread.half_width, 1e-6));
    }

    #[test]
    fn cutoff_only_tightens() {
        let geo = open();
        let mut wave = open();
        // A wave backend reporting a wider corner must not open the direct path
        // beyond the geometric value.
        wave.direct_cutoff_hz = 40_000.0;
        let out = blend_spatial(geo, wave, HybridWeights::WAVE_DOMINANT);
        assert!(out.direct_cutoff_hz <= geo.direct_cutoff_hz + 1e-3);
    }

    #[test]
    fn balanced_is_the_midpoint() {
        let geo = open();
        let mut wave = open();
        wave.direct_gain = 0.0;
        let out = blend_spatial(geo, wave, HybridWeights::BALANCED);
        assert!(approx(out.direct_gain, 0.5, 1e-6));
    }
}
