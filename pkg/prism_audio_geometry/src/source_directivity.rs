//! Per-path source radiation directivity for geometric arrivals.
//!
//! A real sound source does not radiate equally in every direction: a voice, a
//! loudspeaker, or a horn beams its highs forward and grows more
//! omnidirectional toward the lows. Every arrival this crate resolves leaves
//! the emitter along a specific departure direction — the direct arrival heads
//! straight at the listener, a reflection sets off toward its first bounce
//! point, a diffraction toward its first silhouette corner — so each one should
//! be weighted by how strongly the source actually radiates along *that*
//! departure, not treated as if the emitter were a bare point. This module
//! supplies that missing term: given the emitter's facing axis and a path's
//! departure direction, it turns the source's radiation pattern into a
//! three-band attenuation spectrum and folds it into the arrival's existing
//! colour.
//!
//! The radiation pattern itself is the spatial crate's engine-agnostic
//! [`SourceDirectivity`] (the textbook weighted omni-to-cardioid model, sampled
//! here at the three [`PROPAGATION_BAND_CENTERS`]); this module only decides the
//! off-axis angle from the propagation geometry and applies the result.
//!
//! # Gain convention
//!
//! Source directivity is a purely *spectral and directional* loss: an off-axis
//! arrival is darker and quieter than an on-axis one, but the geometric
//! spreading law a path already carries is independent of which way the source
//! happens to face. Accordingly this module only ever enters the per-band
//! [`BandGains`] through [`BandGains::combine`] (which multiplies band by band
//! and keeps the product in `[0, 1]`); it never touches the scalar
//! [`gain`](prism_audio_spatial::propagation::PropagationPath::gain). Leaving the
//! scalar gain alone means the backend's loudest-first ordering and audibility
//! floor rank arrivals identically whether or not this module runs, exactly as
//! the sibling [`crate::air_absorption`] term does.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! radiation pattern is the published electro-acoustics
//! [`SourceDirectivity`] model from the spatial crate; this module adds only the
//! geometry-to-angle mapping and the per-band fold.
//!
//! # Relationship
//!
//! Reuses [`prism_audio_spatial::source_directivity`] for the radiation pattern
//! and [`prism_audio_spatial::band_spectrum`] for the three propagation bands it
//! attenuates. It is applied by the path builders ([`crate::direct_path`],
//! [`crate::reflection_path`], [`crate::diffraction_path`]) at the point where
//! each arrival's departure direction is known, and is enabled through
//! [`crate::config::GeometricConfig::with_source_directivity`]. It complements,
//! and never duplicates, the geometric spreading, surface/edge colour, and
//! atmospheric roll-off the other stages compute.

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;
use prism_audio_spatial::band_spectrum::{
    BandGains, PROPAGATION_BAND_CENTERS, PROPAGATION_BAND_COUNT,
};
use prism_audio_spatial::geometry::Emitter;
use prism_audio_spatial::propagation::PropagationPath;
use prism_audio_spatial::source_directivity::SourceDirectivity;

use crate::config::GeometricConfig;

/// Smallest squared length a vector may have and still define a direction.
///
/// A departure or facing vector shorter than this is treated as degenerate and
/// mapped to the on-axis case (unity radiation), so a coincident emitter and
/// first-hop point can never silence or colour an arrival.
const COINCIDENT_EPSILON_SQ: Sample = 1.0e-12;

/// The three-band radiation spectrum a source with pattern `directivity` and
/// facing `emitter_forward` emits along `departure_dir`.
///
/// The off-axis angle is taken between the (normalised) facing axis and the
/// (normalised) departure direction; its cosine drives the weighted radiation
/// pattern, sampled at each of the three [`PROPAGATION_BAND_CENTERS`]. The
/// result is a [`BandGains`] in `[0, 1]`: unity on axis, progressively darker
/// and quieter off axis, with the highs (sharper directivity) cut first. A
/// degenerate `departure_dir` or `emitter_forward` falls back to the on-axis
/// unity spectrum.
#[must_use]
pub fn radiation_band_gains(
    directivity: &SourceDirectivity,
    emitter_forward: Vec3,
    departure_dir: Vec3,
) -> BandGains {
    let cos_theta = departure_cosine(emitter_forward, departure_dir);
    let mut bands = [0.0; PROPAGATION_BAND_COUNT];
    for (gain, &centre) in bands.iter_mut().zip(PROPAGATION_BAND_CENTERS.iter()) {
        *gain = directivity
            .broadband_gain(cos_theta, centre)
            .clamp(0.0, 1.0);
    }
    BandGains::new(bands)
}

/// Folds `emitter`'s source directivity into `path` for a wave that leaves the
/// emitter along `departure_dir` (any non-unit world vector pointing from the
/// emitter toward the arrival's first hop).
///
/// Does nothing when [`GeometricConfig::source_directivity_enabled`] is unset,
/// so the default configuration leaves every arrival untouched. When enabled,
/// it multiplies the radiation spectrum into the path's per-band
/// [`BandGains`] and leaves the scalar
/// [`gain`](prism_audio_spatial::propagation::PropagationPath::gain) unchanged.
pub fn weight_path(
    path: &mut PropagationPath,
    config: &GeometricConfig,
    emitter: &Emitter,
    departure_dir: Vec3,
) {
    if !config.source_directivity_enabled {
        return;
    }
    let radiation =
        radiation_band_gains(&config.source_directivity, emitter.forward, departure_dir);
    path.bands = path.bands.combine(radiation);
}

/// Cosine of the angle between `emitter_forward` and `departure_dir`, both
/// normalised deterministically through [`bevy_math::ops`].
///
/// Returns `1.0` (on axis) when either vector is too short to have a
/// well-defined direction, and clamps the result into `[-1, 1]` for robustness.
fn departure_cosine(emitter_forward: Vec3, departure_dir: Vec3) -> Sample {
    let Some(forward) = normalize(emitter_forward) else {
        return 1.0;
    };
    let Some(departure) = normalize(departure_dir) else {
        return 1.0;
    };
    forward.dot(departure).clamp(-1.0, 1.0)
}

/// Normalises `v`, returning `None` when it is non-finite or shorter than
/// [`COINCIDENT_EPSILON_SQ`]. The length routes through [`bevy_math::ops`] so
/// the result is bit-reproducible across targets.
fn normalize(v: Vec3) -> Option<Vec3> {
    let len_sq = v.dot(v);
    if !len_sq.is_finite() || len_sq <= COINCIDENT_EPSILON_SQ {
        return None;
    }
    Some(v / ops::sqrt(len_sq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_spatial::propagation::{PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ};
    use prism_audio_spatial::source_directivity::DirectivityPreset;

    use crate::config::GeometricConfig;

    fn omni_path() -> PropagationPath {
        PropagationPath {
            kind: PathKind::Direct,
            delay_seconds: 0.01,
            gain: 1.0,
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            bands: BandGains::UNITY,
            direction: Vec3::NEG_Z,
        }
    }

    #[test]
    fn omni_radiates_unity_in_every_direction() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Omni);
        for dir in [Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::Y] {
            let bands = radiation_band_gains(&d, Vec3::NEG_Z, dir);
            for band in bands.bands() {
                assert!((band - 1.0).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn cardioid_is_unity_on_axis_and_silent_to_the_rear() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        // Facing -Z, a departure straight ahead (-Z) is on axis.
        let front = radiation_band_gains(&d, Vec3::NEG_Z, Vec3::NEG_Z);
        for band in front.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
        // A departure straight behind (+Z) is the rear null.
        let rear = radiation_band_gains(&d, Vec3::NEG_Z, Vec3::Z);
        for band in rear.bands() {
            assert!(band.abs() < 1.0e-6);
        }
    }

    #[test]
    fn frequency_dependent_source_darkens_the_highs_off_axis() {
        // A voice is nearly omni at the lows and directional at the highs, so a
        // 90-degree departure keeps the lows but cuts the highs.
        let d = SourceDirectivity::from_preset(DirectivityPreset::Voice);
        let bands = radiation_band_gains(&d, Vec3::NEG_Z, Vec3::X);
        assert!(bands.high() <= bands.low() + 1.0e-6);
        for band in bands.bands() {
            assert!(band.is_finite() && (0.0..=1.0).contains(&band));
        }
    }

    #[test]
    fn degenerate_departure_is_treated_as_on_axis() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        let bands = radiation_band_gains(&d, Vec3::NEG_Z, Vec3::ZERO);
        for band in bands.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn disabled_config_leaves_the_path_untouched() {
        let cfg = GeometricConfig::new(48_000);
        assert!(!cfg.source_directivity_enabled);
        let emitter = Emitter::new(Vec3::ZERO, Vec3::ZERO, Vec3::NEG_Z);
        let mut path = omni_path();
        // Even a departure that a cardioid would null leaves the path alone when
        // directivity is disabled.
        weight_path(&mut path, &cfg, &emitter, Vec3::Z);
        for band in path.bands.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn enabled_config_folds_radiation_into_the_bands() {
        let cfg = GeometricConfig::new(48_000)
            .with_source_directivity_preset(DirectivityPreset::Cardioid);
        assert!(cfg.source_directivity_enabled);
        let emitter = Emitter::new(Vec3::ZERO, Vec3::ZERO, Vec3::NEG_Z);
        // Departing straight behind the cardioid: the rear null silences it.
        let mut path = omni_path();
        weight_path(&mut path, &cfg, &emitter, Vec3::Z);
        for band in path.bands.bands() {
            assert!(band.abs() < 1.0e-6);
        }
        // Departing straight ahead: unity, untouched.
        let mut on_axis = omni_path();
        weight_path(&mut on_axis, &cfg, &emitter, Vec3::NEG_Z);
        for band in on_axis.bands.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
    }
}
