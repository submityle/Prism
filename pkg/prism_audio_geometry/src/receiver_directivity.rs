//! Per-path receiver (listener) directivity for geometric arrivals.
//!
//! A real receiver is no more omnidirectional than a real source: a shotgun or
//! cardioid microphone, a creature's forward-facing hearing, or a steerable
//! virtual pickup captures sound arriving from in front of it more strongly
//! than sound sneaking in from the side or rear, and often grows more
//! directional toward the highs. Every arrival this crate resolves reaches the
//! listener along a specific *arrival direction* — the direct wave straight
//! from the source, a reflection from its bounce point, a diffraction from its
//! silhouette corner — so each one should be weighted by how sensitively the
//! receiver actually listens along *that* arrival, not treated as if the
//! listener were a bare omnidirectional point. This module supplies that
//! missing term: given a path's listener-local arrival direction, it turns the
//! receiver's pickup pattern into a three-band attenuation spectrum and folds it
//! into the arrival's existing colour.
//!
//! The pickup pattern itself is the spatial crate's engine-agnostic
//! [`SourceDirectivity`] (the textbook weighted omni-to-cardioid model, sampled
//! here at the three [`PROPAGATION_BAND_CENTERS`]); the very same polar maths
//! that describes a cardioid *loudspeaker* also describes a cardioid
//! *microphone*, so this module reuses it rather than inventing a parallel type.
//! It only decides the off-axis angle from the propagation geometry and applies
//! the result.
//!
//! # Gain convention
//!
//! Receiver directivity is a purely *spectral and directional* loss: an
//! off-axis arrival is darker and quieter than an on-axis one, but the geometric
//! spreading law a path already carries is independent of which way the listener
//! happens to face. Accordingly this module only ever enters the per-band
//! [`BandGains`] through [`BandGains::combine`] (which multiplies band by band
//! and keeps the product in `[0, 1]`); it never touches the scalar
//! [`gain`](prism_audio_spatial::propagation::PropagationPath::gain). Leaving the
//! scalar gain alone means the backend's loudest-first ordering and audibility
//! floor rank arrivals identically whether or not this module runs, exactly as
//! the sibling [`crate::source_directivity`] and [`crate::air_absorption`] terms
//! do.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The pickup
//! pattern is the published electro-acoustics [`SourceDirectivity`] model from
//! the spatial crate; this module adds only the geometry-to-angle mapping and
//! the per-band fold.
//!
//! # Relationship
//!
//! Reuses [`prism_audio_spatial::source_directivity`] for the pickup pattern and
//! [`prism_audio_spatial::band_spectrum`] for the three propagation bands it
//! attenuates. It is applied by the path builders ([`crate::direct_path`],
//! [`crate::reflection_path`], [`crate::diffraction_path`]) once each arrival's
//! listener-local direction is known, and is enabled through
//! [`crate::config::GeometricConfig::with_receiver_directivity`]. It is the
//! receiver-side mirror of [`crate::source_directivity`]: that module weights an
//! arrival by how the *source* radiates along its departure, this one by how the
//! *listener* hears along its arrival; the two compose cleanly when both run.

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;
use prism_audio_spatial::band_spectrum::{
    BandGains, PROPAGATION_BAND_CENTERS, PROPAGATION_BAND_COUNT,
};
use prism_audio_spatial::propagation::PropagationPath;
use prism_audio_spatial::source_directivity::SourceDirectivity;

use crate::config::GeometricConfig;

/// The listener's on-axis (maximum-sensitivity) direction in its own local
/// frame.
///
/// A [`LocalSource`](prism_audio_spatial::geometry::LocalSource) direction is
/// expressed in listener-local space where `-Z` is forward, matching Bevy's
/// right-handed convention, so a receiver aimed straight ahead listens best
/// along `-Z`.
const LISTENER_FORWARD_LOCAL: Vec3 = Vec3::NEG_Z;

/// Smallest squared length an arrival direction may have and still define a
/// bearing.
///
/// A direction shorter than this is treated as degenerate and mapped to the
/// on-axis case (unity pickup), so a listener coincident with an arrival's last
/// hop can never silence or colour it.
const COINCIDENT_EPSILON_SQ: Sample = 1.0e-12;

/// The three-band pickup spectrum a receiver with pattern `directivity`
/// captures for a wave arriving along `arrival_dir` (a listener-local
/// direction, `-Z` forward).
///
/// The off-axis angle is taken between the receiver's forward axis
/// ([`LISTENER_FORWARD_LOCAL`]) and the (normalised) arrival direction; its
/// cosine drives the weighted pickup pattern, sampled at each of the three
/// [`PROPAGATION_BAND_CENTERS`]. The result is a [`BandGains`] in `[0, 1]`:
/// unity on axis, progressively darker and quieter off axis, with the highs
/// (sharper directivity) cut first. A degenerate `arrival_dir` falls back to the
/// on-axis unity spectrum.
#[must_use]
pub fn pickup_band_gains(directivity: &SourceDirectivity, arrival_dir: Vec3) -> BandGains {
    let cos_theta = arrival_cosine(arrival_dir);
    let mut bands = [0.0; PROPAGATION_BAND_COUNT];
    for (gain, &centre) in bands.iter_mut().zip(PROPAGATION_BAND_CENTERS.iter()) {
        *gain = directivity
            .broadband_gain(cos_theta, centre)
            .clamp(0.0, 1.0);
    }
    BandGains::new(bands)
}

/// Folds the listener's receiver directivity into `path`, using the path's own
/// listener-local [`direction`](PropagationPath::direction) as the arrival
/// bearing.
///
/// Does nothing when [`GeometricConfig::receiver_directivity_enabled`] is unset,
/// so default configurations render exactly as before. When enabled, the pickup
/// spectrum is multiplied band by band into the arrival's colour, leaving its
/// scalar gain untouched (see the module-level gain convention).
pub fn weight_path(path: &mut PropagationPath, config: &GeometricConfig) {
    if !config.receiver_directivity_enabled {
        return;
    }
    let pickup = pickup_band_gains(&config.receiver_directivity, path.direction);
    path.bands = path.bands.combine(pickup);
}

/// Cosine of the angle between the receiver's forward axis and `arrival_dir`,
/// the latter normalised deterministically through [`bevy_math::ops`].
///
/// Returns `1.0` (on axis) when `arrival_dir` is too short to have a
/// well-defined bearing, and clamps the result into `[-1, 1]` for robustness.
fn arrival_cosine(arrival_dir: Vec3) -> Sample {
    let Some(arrival) = normalize(arrival_dir) else {
        return 1.0;
    };
    LISTENER_FORWARD_LOCAL.dot(arrival).clamp(-1.0, 1.0)
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

    fn path_from(direction: Vec3) -> PropagationPath {
        PropagationPath {
            kind: PathKind::Direct,
            delay_seconds: 0.01,
            gain: 1.0,
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            bands: BandGains::UNITY,
            direction,
        }
    }

    #[test]
    fn omni_captures_unity_from_every_bearing() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Omni);
        for dir in [Vec3::NEG_Z, Vec3::Z, Vec3::X, Vec3::Y] {
            let bands = pickup_band_gains(&d, dir);
            for band in bands.bands() {
                assert!((band - 1.0).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn cardioid_is_unity_ahead_and_silent_behind() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        // A source straight ahead arrives along local forward (-Z): on axis.
        let front = pickup_band_gains(&d, Vec3::NEG_Z);
        for band in front.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
        // A source directly behind arrives along +Z: the rear null.
        let rear = pickup_band_gains(&d, Vec3::Z);
        for band in rear.bands() {
            assert!(band.abs() < 1.0e-6);
        }
    }

    #[test]
    fn frequency_dependent_receiver_darkens_the_highs_off_axis() {
        // A voice-like pickup is nearly omni at the lows and directional at the
        // highs, so a 90-degree arrival keeps the lows but cuts the highs.
        let d = SourceDirectivity::from_preset(DirectivityPreset::Voice);
        let bands = pickup_band_gains(&d, Vec3::X);
        assert!(bands.high() <= bands.low() + 1.0e-6);
        for band in bands.bands() {
            assert!(band.is_finite() && (0.0..=1.0).contains(&band));
        }
    }

    #[test]
    fn degenerate_arrival_is_treated_as_on_axis() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        let bands = pickup_band_gains(&d, Vec3::ZERO);
        for band in bands.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn disabled_config_leaves_the_path_untouched() {
        let cfg = GeometricConfig::new(48_000);
        assert!(!cfg.receiver_directivity_enabled);
        // Even an arrival a cardioid would null leaves the path alone when
        // receiver directivity is disabled.
        let mut path = path_from(Vec3::Z);
        weight_path(&mut path, &cfg);
        for band in path.bands.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn enabled_config_folds_pickup_into_the_bands() {
        let cfg = GeometricConfig::new(48_000)
            .with_receiver_directivity_preset(DirectivityPreset::Cardioid);
        assert!(cfg.receiver_directivity_enabled);
        // Arriving from straight behind the cardioid: the rear null silences it.
        let mut rear = path_from(Vec3::Z);
        weight_path(&mut rear, &cfg);
        for band in rear.bands.bands() {
            assert!(band.abs() < 1.0e-6);
        }
        // Arriving from straight ahead: unity, untouched.
        let mut front = path_from(Vec3::NEG_Z);
        weight_path(&mut front, &cfg);
        for band in front.bands.bands() {
            assert!((band - 1.0).abs() < 1.0e-6);
        }
    }
}
