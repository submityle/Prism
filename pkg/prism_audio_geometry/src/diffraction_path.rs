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

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::f32::consts::PI;

use bevy_math::ops;
use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::{BandGains, PROPAGATION_BAND_COUNT, UtdWedge};
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{
    diffraction_cutoff_hz, diffraction_gain, edge_path_difference, PathKind, PropagationPath,
};

use crate::config::{DiffractionModel, GeometricConfig};
use crate::scene::AcousticScene;

/// Number of golden-section-style narrowing steps used to locate the
/// minimum-detour point on an edge. The detour sum is convex along the edge, so
/// a fixed ternary narrowing converges deterministically without a tolerance
/// loop that could vary across targets.
const DETOUR_SEARCH_STEPS: u32 = 60;

/// How close two adjacent face normals must be (by absolute dot product) to be
/// treated as coplanar, so the edge between them is an internal triangulation
/// seam rather than a real diffracting crease.
const COPLANAR_DOT: Sample = 1.0 - 1.0e-4;

/// The two endpoints of a mesh edge together with the face normals of every
/// triangle that uses it. One adjacent face marks a boundary (silhouette) edge;
/// two or more non-coplanar faces mark a crease; coplanar faces mark an
/// internal seam that does not diffract.
struct EdgeData {
    start: Vec3,
    end: Vec3,
    normals: Vec<Vec3>,
}

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

/// Distils `scene` into the edges that can diffract: boundary edges (used by a
/// single face) and creases (shared by non-coplanar faces). Internal edges that
/// merely split a flat polygon into triangles are excluded.
#[must_use]
fn diffracting_edges(scene: &AcousticScene) -> Vec<EdgeData> {
    let mut edges: BTreeMap<[u32; 6], EdgeData> = BTreeMap::new();
    for triangle in 0..scene.triangle_count() {
        let Some([a, b, c]) = scene.triangle(triangle) else {
            continue;
        };
        let Some(normal) = scene.triangle_normal(triangle) else {
            continue;
        };
        for &(start, end) in &[(a, b), (b, c), (c, a)] {
            let key = edge_key(start, end);
            let entry = edges.entry(key).or_insert_with(|| EdgeData {
                start,
                end,
                normals: Vec::new(),
            });
            entry.normals.push(normal);
        }
    }

    edges
        .into_values()
        .filter(|edge| is_diffracting(&edge.normals))
        .collect()
}

/// Whether an edge with these adjacent face normals diffracts.
#[must_use]
fn is_diffracting(normals: &[Vec3]) -> bool {
    match normals.first() {
        None => false,
        Some(&first) => {
            if normals.len() == 1 {
                return true;
            }
            // A crease if any adjacent face tilts away from the first; coplanar
            // faces (parallel or anti-parallel normals) are an internal seam.
            normals.iter().any(|n| first.dot(*n).abs() < COPLANAR_DOT)
        }
    }
}

/// A canonical, order-independent key for the undirected edge `p`..`q`, built
/// from the raw bit patterns of the endpoint coordinates so the two triangles
/// sharing an edge map to the same key deterministically.
#[must_use]
fn edge_key(p: Vec3, q: Vec3) -> [u32; 6] {
    let kp = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let kq = [q.x.to_bits(), q.y.to_bits(), q.z.to_bits()];
    let (lo, hi) = if kp <= kq { (kp, kq) } else { (kq, kp) };
    [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
}

/// The point on segment `start`..`end` that minimises the detour
/// `|listener - p| + |p - emitter|`, found by a fixed ternary narrowing of the
/// convex detour along the edge parameter.
#[must_use]
fn least_detour_point(start: Vec3, end: Vec3, listener: Vec3, emitter: Vec3) -> Vec3 {
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;
    for _ in 0..DETOUR_SEARCH_STEPS {
        let third = (hi - lo) / 3.0;
        let m1 = lo + third;
        let m2 = hi - third;
        let p1 = start + (end - start) * m1;
        let p2 = start + (end - start) * m2;
        let f1 = distance(listener, p1) + distance(p1, emitter);
        let f2 = distance(listener, p2) + distance(p2, emitter);
        if f1 < f2 {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let t = 0.5 * (lo + hi);
    start + (end - start) * t
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

/// Deterministic Euclidean distance (routes through [`bevy_math::ops::sqrt`]).
#[inline]
#[must_use]
fn distance(a: Vec3, b: Vec3) -> Sample {
    let d = a - b;
    ops::sqrt(d.dot(d))
}

/// Builds a [`UtdWedge`] for the diffracting `edge` at the resolved `corner`.
///
/// The wedge opening is inferred from the edge's adjacent face normals (see
/// [`utd_wedge_index`]); the reference face axis is taken perpendicular to the
/// edge within the first face's plane as `edge_dir x n0`, which
/// [`UtdWedge::from_geometry`] re-orthogonalises, falling back to a safe
/// default when degenerate.
#[must_use]
fn utd_wedge(edge: &EdgeData, corner: Vec3, source: Vec3, receiver: Vec3) -> UtdWedge {
    let edge_dir = edge.end - edge.start;
    let face_ref = match edge.normals.first() {
        Some(&n0) => edge_dir.cross(n0),
        None => Vec3::ZERO,
    };
    UtdWedge::from_geometry(
        source,
        corner,
        edge_dir,
        receiver,
        face_ref,
        utd_wedge_index(&edge.normals),
        SPEED_OF_SOUND_MPS,
    )
}

/// The UTD wedge index `n` inferred from an edge's adjacent face normals.
///
/// Returns `2` (a thin screen) for a silhouette edge with a single face. For a
/// crease, `n = 1 + acos(n0 . n1) / pi`, so coplanar faces give `n = 1` (no
/// real wedge) and anti-parallel outward normals give `n = 2`, matching the
/// convex opening between the two half-planes. The result is further clamped to
/// `[1, 2]` by [`UtdWedge::new`].
#[must_use]
fn utd_wedge_index(normals: &[Vec3]) -> Sample {
    match (normals.first(), normals.get(1)) {
        (Some(&n0), Some(&n1)) => {
            let dot = n0
                .normalize_or_zero()
                .dot(n1.normalize_or_zero())
                .clamp(-1.0, 1.0);
            (1.0 + ops::acos(dot) / PI).clamp(1.0, 2.0)
        }
        _ => 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{diffracting_edges, least_detour_point, resolve_diffraction, utd_wedge_index};
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
    fn internal_seam_is_not_a_diffracting_edge() {
        // The quad has four boundary edges and one internal diagonal; only the
        // four boundaries should survive.
        let edges = diffracting_edges(&barrier());
        assert_eq!(edges.len(), 4);
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
    fn least_detour_sits_between_the_endpoints() {
        // Edge along z through the origin; listener and emitter symmetric about
        // the x axis, so the minimum detour is the midpoint z = 0.
        let start = Vec3::new(0.0, 0.0, -5.0);
        let end = Vec3::new(0.0, 0.0, 5.0);
        let listener = Vec3::new(-3.0, 0.0, 0.0);
        let emitter = Vec3::new(3.0, 0.0, 0.0);
        let p = least_detour_point(start, end, listener, emitter);
        assert!(p.z.abs() < 1.0e-2);
    }

    #[test]
    fn utd_wedge_index_matches_the_geometry() {
        // A single face is a thin screen (n = 2).
        assert!((utd_wedge_index(&[Vec3::X]) - 2.0).abs() < 1.0e-4);
        // Anti-parallel outward normals (both sides of a thin screen) also n = 2.
        assert!((utd_wedge_index(&[Vec3::X, -Vec3::X]) - 2.0).abs() < 1.0e-4);
        // Perpendicular faces (a right-angle convex corner) give n = 1.5.
        assert!((utd_wedge_index(&[Vec3::X, Vec3::Y]) - 1.5).abs() < 1.0e-4);
        // Coplanar faces are not a real wedge (n = 1).
        assert!((utd_wedge_index(&[Vec3::X, Vec3::X]) - 1.0).abs() < 1.0e-4);
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
