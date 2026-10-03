//! Edge diffraction resolved with the Maekawa barrier model.
//!
//! When the straight line of sight is blocked, a wave still bends around the
//! obstacle's edges into the geometric shadow, attenuated by how far it must
//! detour and how high its frequency is. This module first distils the mesh
//! into its *diffracting* edges (boundary edges and non-coplanar creases, never
//! the internal edges that merely triangulate a flat face), then for each such
//! edge finds the point that minimises the over-the-edge detour, confirms both
//! legs of that bent route are clear, and applies the spatial crate's
//! Fresnel/Maekawa helpers to size the attenuation and the low-pass colour.
//!
//! # Gain convention
//!
//! A diffracted arrival carries the Maekawa barrier gain for its detour scaled
//! by the extra spherical spreading of the longer bent route relative to the
//! direct arrival's base distance (`base_distance / path_length`), matching the
//! convention documented in [`crate::direct_path`]. The result is clamped to
//! `[0, 1]`; arrivals quieter than the configured floor are discarded.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumes [`crate::scene::AcousticScene`] and the spatial crate's
//! [`propagation`](prism_audio_spatial::propagation) diffraction helpers,
//! emitting [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)s
//! of kind [`Diffraction`](prism_audio_spatial::propagation::PathKind::Diffraction);
//! merged with the direct and reflected arrivals by
//! [`crate::backend::GeometricBackend`].

use alloc::vec::Vec;
use core::cmp::Ordering;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{
    diffraction_cutoff_hz, diffraction_gain, edge_path_difference, PathKind, PropagationPath,
};
use prism_audio_spatial::{BandGains, PROPAGATION_BAND_COUNT};

use crate::config::{DiffractionModel, GeometricConfig};
use crate::diffraction_edges::{diffracting_edges, distance, least_detour_point, utd_wedge};
use crate::scene::AcousticScene;

/// Resolves the edge-diffracted arrivals of `emitter` at `listener`.
///
/// Returns the audible diffractions sorted loudest-first (equivalently,
/// shortest-detour-first), at most [`GeometricConfig::max_diffractions`]. An
/// empty scene, a disabled diffraction stage, or a zero diffraction budget
/// yields an empty list. Callers invoke this only when the direct line of sight
/// is shadowed; each surviving edge is one whose corner both endpoints can see.
#[must_use]
pub fn resolve_diffraction(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> Vec<PropagationPath> {
    let mut paths = Vec::new();
    if !config.diffraction_enabled || config.max_diffractions == 0 || scene.is_empty() {
        return paths;
    }

    let eps = config.surface_epsilon_m.max(0.0);
    for edge in diffracting_edges(scene) {
        let corner = least_detour_point(edge.start, edge.end, listener.position, emitter.position);

        // The bent route is physical only when both legs reach the corner
        // unobstructed; otherwise some other surface swallows it.
        if scene.segment_blocked(listener.position, corner, eps)
            || scene.segment_blocked(corner, emitter.position, eps)
        {
            continue;
        }

        let path_length = distance(listener.position, corner) + distance(corner, emitter.position);
        if path_length <= 0.0 {
            continue;
        }

        let delta = edge_path_difference(listener.position, corner, emitter.position);
        let spreading = (base_distance / path_length).clamp(0.0, 1.0);
        // Both models carry the detour-dependent single-pole corner for legacy
        // consumers that read only `cutoff_hz`; the authoritative spectral
        // shaping lives in `bands`.
        let cutoff_hz = diffraction_cutoff_hz(delta, config.sample_rate);

        let (gain, bands) = match config.diffraction_model {
            DiffractionModel::Maekawa => {
                let barrier = diffraction_gain(delta, config.diffraction_freq_hz);
                let gain = (barrier * spreading).clamp(0.0, 1.0);
                (gain, BandGains::from_lowpass_cutoff(cutoff_hz))
            }
            DiffractionModel::Utd => {
                let wedge = utd_wedge(&edge, corner, emitter.position, listener.position);
                let gain =
                    (wedge.relative_gain(config.diffraction_freq_hz) * spreading).clamp(0.0, 1.0);
                let raw = wedge.band_gains().bands();
                let mut shaped = [0.0; PROPAGATION_BAND_COUNT];
                for (out, &band_gain) in shaped.iter_mut().zip(raw.iter()) {
                    *out = (band_gain * spreading).clamp(0.0, 1.0);
                }
                (gain, BandGains::new(shaped))
            }
        };
        if gain <= config.min_gain {
            continue;
        }

        let local = listener.localize(&Emitter::point(corner, Vec3::ZERO));
        let candidate = PropagationPath {
            kind: PathKind::Diffraction,
            delay_seconds: path_length / SPEED_OF_SOUND_MPS,
            gain,
            cutoff_hz,
            bands,
            direction: local.direction,
        };
        if !is_duplicate(&paths, &candidate) {
            paths.push(candidate);
        }
    }

    paths.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));
    paths.truncate(config.max_diffractions);
    paths
}

/// Whether `candidate` duplicates an arrival already kept (same delay and
/// direction within a tight tolerance).
#[must_use]
fn is_duplicate(paths: &[PropagationPath], candidate: &PropagationPath) -> bool {
    paths.iter().any(|existing| {
        (existing.delay_seconds - candidate.delay_seconds).abs() < 1.0e-6
            && existing.direction.dot(candidate.direction) > 0.9999
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_diffraction;
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::{DiffractionModel, GeometricConfig};
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    // A finite barrier: a quad in the plane x = 0 spanning y in [-1, 1] and
    // z in [-5, 5]. Its top edge is at y = 1.
    fn barrier() -> AcousticScene {
        let vertices = vec![
            Vec3::new(0.0, -1.0, -5.0),
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::new(0.0, -1.0, 5.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        AcousticScene::new(
            vertices,
            indices,
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.0)),
        )
        .unwrap()
    }

    #[test]
    fn shadowed_source_bends_over_the_edge() {
        let scene = barrier();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000);
        let base = (emitter.position - listener.position).length();
        let paths = resolve_diffraction(&scene, &listener, &emitter, &cfg, base);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].kind, PathKind::Diffraction);
        // Bending over the edge is longer than the straight line: delayed.
        assert!(paths[0].delay_seconds > base / 343.0);
        assert!(paths[0].cutoff_hz > 0.0);
        assert!(paths[0].gain > 0.0 && paths[0].gain <= 1.0);
    }

    #[test]
    fn disabled_stage_yields_nothing() {
        let scene = barrier();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).without_diffraction();
        let base = (emitter.position - listener.position).length();
        assert!(resolve_diffraction(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn budget_caps_diffraction_count() {
        let scene = barrier();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).with_max_diffractions(1);
        let base = (emitter.position - listener.position).length();
        let paths = resolve_diffraction(&scene, &listener, &emitter, &cfg, base);
        assert!(paths.len() <= 1);
    }

    #[test]
    fn utd_model_shadows_highs_more_than_lows() {
        let scene = barrier();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).with_diffraction_model(DiffractionModel::Utd);
        let base = (emitter.position - listener.position).length();
        let paths = resolve_diffraction(&scene, &listener, &emitter, &cfg, base);
        assert!(!paths.is_empty());
        let path = paths[0];
        assert_eq!(path.kind, PathKind::Diffraction);
        // Bending over the edge is longer than the straight line: delayed.
        assert!(path.delay_seconds > base / 343.0);
        // Every band gain stays in the physical [0, 1] range.
        for band in path.bands.bands() {
            assert!(band.is_finite() && (0.0..=1.0).contains(&band));
        }
        assert!(path.gain.is_finite() && path.gain > 0.0 && path.gain <= 1.0);
        // The |D| ~ 1/sqrt(k) UTD roll-off attenuates highs at least as much as
        // lows inside the geometric shadow.
        assert!(path.bands.high() <= path.bands.low() + 1.0e-4);
    }

    #[test]
    fn utd_and_maekawa_share_geometry_but_differ_in_colour() {
        let scene = barrier();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let base = (emitter.position - listener.position).length();

        let maekawa = GeometricConfig::new(48_000);
        let utd = maekawa.with_diffraction_model(DiffractionModel::Utd);
        let m = resolve_diffraction(&scene, &listener, &emitter, &maekawa, base);
        let u = resolve_diffraction(&scene, &listener, &emitter, &utd, base);
        assert!(!m.is_empty() && !u.is_empty());
        // Same resolved detour geometry: identical delay and arrival direction.
        assert!((m[0].delay_seconds - u[0].delay_seconds).abs() < 1.0e-6);
        assert!(m[0].direction.dot(u[0].direction) > 0.9999);
        // Different models shape the three bands differently.
        let same = (m[0].bands.low() - u[0].bands.low()).abs() < 1.0e-6
            && (m[0].bands.mid() - u[0].bands.mid()).abs() < 1.0e-6
            && (m[0].bands.high() - u[0].bands.high()).abs() < 1.0e-6;
        assert!(!same);
    }
}
