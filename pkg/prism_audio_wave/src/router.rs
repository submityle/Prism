//! Runtime propagation-tier routing: the consumer side of the audio-LOD
//! propagation decision.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Closes the loop of design sections 32 and 43. The quality governor
//! ([`prism_audio_governor`]) selects a [`PropagationTier`] per source from the
//! CPU budget; this module turns that selection into an actual blend of a
//! source's geometric and wave-field [`SpatialParams`] contributions on the
//! shared spatial bus.
//!
//! The governor only *produces* a tier decision and never links against the
//! wave or spatial crates. This router is the single authoritative consumer
//! that maps a tier onto [`HybridWeights`] and runs
//! [`blend_spatial`](crate::hybrid::blend_spatial). The dependency direction is
//! one-way (wave depends on governor, not the reverse), so there is no cycle.
//!
//! Routing is uniform: every tier flows through the same blend with
//! tier-selected weights. This keeps kinematic parameters that the geometric
//! backend owns (Doppler `pitch_ratio` and image-width `spread`) intact in
//! every tier, because a baked wave field is static and carries no Doppler.
//! The wave tier still crossfades through the balanced blend's low-pass
//! tightening rule so soft occlusion is never lost. The function is pure:
//! allocation free, lock free, and panic free.

use prism_audio_governor::governor::lod::PropagationTier;
use prism_audio_spatial::SpatialParams;

use crate::hybrid::{blend_spatial, HybridWeights};

/// Maps a governor-selected [`PropagationTier`] onto the [`HybridWeights`] that
/// drive the geometric/wave crossfade.
///
/// - [`PropagationTier::Geometric`] routes entirely to the geometric backend
///   ([`HybridWeights::GEOMETRIC_DOMINANT`]): no baked data participates.
/// - [`PropagationTier::Wave`] routes entirely to the wave backend
///   ([`HybridWeights::WAVE_DOMINANT`]) for the propagation aspects, while the
///   blend keeps kinematic parameters geometric.
/// - [`PropagationTier::Hybrid`] runs both and crossfades evenly
///   ([`HybridWeights::BALANCED`]): the wave field supplies the baseline and
///   the geometric rays add the dynamic high-frequency increment.
#[must_use]
pub fn weights_for_tier(tier: PropagationTier) -> HybridWeights {
    match tier {
        PropagationTier::Geometric => HybridWeights::GEOMETRIC_DOMINANT,
        PropagationTier::Wave => HybridWeights::WAVE_DOMINANT,
        PropagationTier::Hybrid => HybridWeights::BALANCED,
    }
}

/// Resolves a source's geometric and wave-field spatial contributions into a
/// single [`SpatialParams`] on the shared bus, according to the governor's
/// propagation tier.
///
/// `geometric` is the fully dynamic geometric-backend result (always valid, it
/// carries Doppler and image width). `wave` is the baked wave-field lookup
/// result. For the geometric tier the wave contribution is ignored; for the
/// wave tier the geometric kinematics are still preserved; for the hybrid tier
/// the two are crossfaded.
///
/// # Examples
///
/// ```
/// # use prism_audio_spatial::{SpatialParams, SpreadParams};
/// # use prism_audio_governor::governor::lod::PropagationTier;
/// # use prism_audio_wave::router::route;
/// let geo = SpatialParams {
///     direct_gain: 1.0,
///     pitch_ratio: 1.5,
///     azimuth: 0.0,
///     elevation: 0.0,
///     direct_cutoff_hz: 20_000.0,
///     wet_gain: 0.0,
///     spread: SpreadParams::POINT,
/// };
/// let mut wave = geo;
/// wave.pitch_ratio = 1.0;
/// wave.direct_gain = 0.2;
/// wave.direct_cutoff_hz = 500.0;
/// wave.wet_gain = 0.6;
///
/// // Geometric tier ignores the wave field entirely.
/// let g = route(PropagationTier::Geometric, geo, wave);
/// assert!((g.direct_gain - 1.0).abs() < 1e-6);
///
/// // Wave tier takes the wave propagation but keeps the geometric Doppler.
/// let w = route(PropagationTier::Wave, geo, wave);
/// assert!((w.direct_gain - 0.2).abs() < 1e-6);
/// assert!((w.pitch_ratio - 1.5).abs() < 1e-6);
/// ```
#[must_use]
pub fn route(tier: PropagationTier, geometric: SpatialParams, wave: SpatialParams) -> SpatialParams {
    blend_spatial(geometric, wave, weights_for_tier(tier))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_spatial::SpreadParams;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    fn geo_sample() -> SpatialParams {
        SpatialParams {
            direct_gain: 1.0,
            pitch_ratio: 1.5,
            azimuth: 0.3,
            elevation: -0.1,
            direct_cutoff_hz: 20_000.0,
            wet_gain: 0.05,
            spread: SpreadParams::POINT,
        }
    }

    fn wave_sample() -> SpatialParams {
        SpatialParams {
            direct_gain: 0.2,
            pitch_ratio: 1.0,
            azimuth: -0.4,
            elevation: 0.2,
            direct_cutoff_hz: 500.0,
            wet_gain: 0.7,
            spread: SpreadParams {
                spread: 0.9,
                focus: 0.3,
                half_width: 0.8,
            },
        }
    }

    #[test]
    fn tier_maps_to_expected_weights() {
        assert_eq!(
            weights_for_tier(PropagationTier::Geometric),
            HybridWeights::GEOMETRIC_DOMINANT
        );
        assert_eq!(
            weights_for_tier(PropagationTier::Wave),
            HybridWeights::WAVE_DOMINANT
        );
        assert_eq!(
            weights_for_tier(PropagationTier::Hybrid),
            HybridWeights::BALANCED
        );
    }

    #[test]
    fn geometric_tier_returns_geometric_contribution() {
        let geo = geo_sample();
        let out = route(PropagationTier::Geometric, geo, wave_sample());
        assert!(approx(out.direct_gain, geo.direct_gain));
        assert!(approx(out.azimuth, geo.azimuth));
        assert!(approx(out.elevation, geo.elevation));
        assert!(approx(out.direct_cutoff_hz, geo.direct_cutoff_hz));
        assert!(approx(out.wet_gain, geo.wet_gain));
    }

    #[test]
    fn wave_tier_takes_wave_propagation_but_keeps_geometric_kinematics() {
        let geo = geo_sample();
        let wave = wave_sample();
        let out = route(PropagationTier::Wave, geo, wave);
        // Propagation aspects follow the wave field.
        assert!(approx(out.direct_gain, wave.direct_gain));
        assert!(approx(out.azimuth, wave.azimuth));
        assert!(approx(out.wet_gain, wave.wet_gain));
        // Low-pass takes the tighter corner (the wave field only ever damps).
        assert!(approx(out.direct_cutoff_hz, wave.direct_cutoff_hz));
        // Kinematics stay geometric: a static bake carries no Doppler or width.
        assert!(approx(out.pitch_ratio, geo.pitch_ratio));
        assert!(approx(out.spread.half_width, geo.spread.half_width));
    }

    #[test]
    fn hybrid_tier_equals_balanced_blend() {
        let geo = geo_sample();
        let wave = wave_sample();
        let out = route(PropagationTier::Hybrid, geo, wave);
        let expected = blend_spatial(geo, wave, HybridWeights::BALANCED);
        assert!(approx(out.direct_gain, expected.direct_gain));
        assert!(approx(out.azimuth, expected.azimuth));
        assert!(approx(out.elevation, expected.elevation));
        assert!(approx(out.direct_cutoff_hz, expected.direct_cutoff_hz));
        assert!(approx(out.wet_gain, expected.wet_gain));
    }

    #[test]
    fn hybrid_blend_lies_between_the_two_backends() {
        let geo = geo_sample();
        let wave = wave_sample();
        let out = route(PropagationTier::Hybrid, geo, wave);
        let lo = geo.direct_gain.min(wave.direct_gain);
        let hi = geo.direct_gain.max(wave.direct_gain);
        assert!(out.direct_gain >= lo && out.direct_gain <= hi);
    }

    #[test]
    fn routing_is_deterministic() {
        let geo = geo_sample();
        let wave = wave_sample();
        for tier in [
            PropagationTier::Geometric,
            PropagationTier::Wave,
            PropagationTier::Hybrid,
        ] {
            let a = route(tier, geo, wave);
            let b = route(tier, geo, wave);
            assert!(approx(a.direct_gain, b.direct_gain));
            assert!(approx(a.direct_cutoff_hz, b.direct_cutoff_hz));
        }
    }
}
