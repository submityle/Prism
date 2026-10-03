//! Shared diffracting-edge geometry for the first- and higher-order edge
//! diffraction resolvers.
//!
//! Both [`crate::diffraction_path`] (single edge) and
//! [`crate::higher_order_diffraction`] (a sequence of edges) need the same
//! primitives: distil a mesh into the edges that actually diffract, find the
//! least-detour point a bent route takes over one edge, and build the
//! [`UtdWedge`] that colours the shadow of one wedge. Factoring them here keeps
//! the two resolvers from maintaining divergent copies of the geometry and the
//! wedge-index convention.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumed by [`crate::diffraction_path`] and
//! [`crate::higher_order_diffraction`]; reads a [`crate::scene::AcousticScene`]
//! and emits [`UtdWedge`](prism_audio_spatial::UtdWedge)s for the spatial
//! crate's diffraction helpers.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::f32::consts::PI;

use bevy_math::ops;
use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::UtdWedge;

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
pub(crate) struct EdgeData {
    /// First endpoint of the undirected edge.
    pub(crate) start: Vec3,
    /// Second endpoint of the undirected edge.
    pub(crate) end: Vec3,
    /// Outward face normals of every triangle sharing this edge.
    pub(crate) normals: Vec<Vec3>,
}

/// Distils `scene` into the edges that can diffract: boundary edges (used by a
/// single face) and creases (shared by non-coplanar faces). Internal edges that
/// merely split a flat polygon into triangles are excluded. Edges are returned
/// in the deterministic order of their canonical bit-pattern key.
#[must_use]
pub(crate) fn diffracting_edges(scene: &AcousticScene) -> Vec<EdgeData> {
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
pub(crate) fn least_detour_point(start: Vec3, end: Vec3, listener: Vec3, emitter: Vec3) -> Vec3 {
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

/// Deterministic Euclidean distance (routes through [`bevy_math::ops::sqrt`]).
#[inline]
#[must_use]
pub(crate) fn distance(a: Vec3, b: Vec3) -> Sample {
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
pub(crate) fn utd_wedge(edge: &EdgeData, corner: Vec3, source: Vec3, receiver: Vec3) -> UtdWedge {
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
pub(crate) fn utd_wedge_index(normals: &[Vec3]) -> Sample {
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
    use super::{diffracting_edges, least_detour_point, utd_wedge_index};
    use alloc::vec;
    use bevy_math::Vec3;
    use prism_audio_spatial::propagation::AcousticMaterial;

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
}
