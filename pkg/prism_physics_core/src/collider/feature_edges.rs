//! Feature-edge extraction: boundary, crease and non-manifold edges of a
//! triangle mesh, classified by dihedral angle.
//!
//! Sharp edges are what collision and authoring pipelines care about long after
//! the smooth interior is uninteresting: edge-edge contact generation keys off
//! real geometric creases, convex decomposition likes to cut along them, and
//! SDF/mesh simplification must preserve them to avoid rounding corners. The
//! standard criterion (shared by DCC tools and physics cookers alike) is the
//! *dihedral angle* between the two faces sharing an edge: a fold sharper than a
//! threshold is a crease, an edge touched by a single face is a boundary, and an
//! edge touched by three or more faces is non-manifold.
//!
//! Near-coincident vertices are welded first (reusing
//! [`weld_mesh`](crate::collider::weld_mesh)) so hairline cracks do not masquerade
//! as boundaries. This is pure triangle-soup geometry with no coupling to the
//! collision pipeline, and nothing here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashMap;

use crate::collider::weld::{weld_mesh, WeldParams};

/// Shortest cross-product length below which a triangle is treated as
/// degenerate and contributes no face normal.
const DEGENERATE_EPSILON: f32 = 1.0e-12;

/// How an edge sits in the mesh surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// Used by exactly one triangle: an open boundary / hole rim.
    Boundary,
    /// Shared by two triangles folded at or beyond the crease threshold.
    Crease,
    /// Shared by two triangles below the crease threshold (a smooth interior
    /// edge).
    Smooth,
    /// Shared by three or more triangles: a non-manifold junction.
    NonManifold,
}

/// One classified mesh edge, keyed by welded vertex indices with `v0 < v1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureEdge {
    /// Lower welded vertex index of the edge.
    pub v0: u32,
    /// Higher welded vertex index of the edge.
    pub v1: u32,
    /// How the edge sits in the surface.
    pub kind: EdgeKind,
    /// Dihedral angle in radians between incident faces. Zero for a boundary
    /// edge; the maximum pairwise angle for a non-manifold edge.
    pub dihedral_angle: f32,
}

/// Tuning for [`extract_feature_edges`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureEdgeParams {
    /// Dihedral angle (radians) at or above which a two-face edge is a crease.
    pub crease_angle_radians: f32,
    /// Position tolerance used to weld near-coincident vertices before analysis.
    pub weld_epsilon: f32,
}

impl FeatureEdgeParams {
    /// Builds params from a crease threshold in degrees.
    #[must_use]
    pub fn with_crease_degrees(crease_degrees: f32, weld_epsilon: f32) -> Self {
        Self {
            crease_angle_radians: crease_degrees.to_radians(),
            weld_epsilon,
        }
    }
}

impl Default for FeatureEdgeParams {
    /// A 40-degree crease threshold (a common DCC default) and a `1e-5` weld
    /// tolerance.
    fn default() -> Self {
        Self::with_crease_degrees(40.0, 1.0e-5)
    }
}

/// The classified edge set of a mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureEdges {
    /// Welded vertex positions the edges index into.
    pub vertices: Vec<Vec3>,
    /// All distinct edges, sorted by `(v0, v1)` for determinism.
    pub edges: Vec<FeatureEdge>,
}

impl FeatureEdges {
    /// Number of boundary edges.
    #[must_use]
    pub fn boundary_count(&self) -> usize {
        self.count_kind(EdgeKind::Boundary)
    }

    /// Number of crease edges.
    #[must_use]
    pub fn crease_count(&self) -> usize {
        self.count_kind(EdgeKind::Crease)
    }

    /// Number of smooth interior edges.
    #[must_use]
    pub fn smooth_count(&self) -> usize {
        self.count_kind(EdgeKind::Smooth)
    }

    /// Number of non-manifold edges.
    #[must_use]
    pub fn non_manifold_count(&self) -> usize {
        self.count_kind(EdgeKind::NonManifold)
    }

    /// Total number of edges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether there are no edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Iterates the geometrically sharp edges: boundaries, creases and
    /// non-manifold junctions (everything except smooth interior edges).
    pub fn sharp_edges(&self) -> impl Iterator<Item = &FeatureEdge> {
        self.edges.iter().filter(|e| e.kind != EdgeKind::Smooth)
    }

    fn count_kind(&self, kind: EdgeKind) -> usize {
        self.edges.iter().filter(|e| e.kind == kind).count()
    }
}

/// Extracts and classifies the edges of a triangle mesh by dihedral angle.
///
/// Returns `None` when `vertices` or `indices` is empty, when
/// `crease_angle_radians` is not finite, or when welding removes every triangle.
/// Degenerate (zero-area) triangles and triangles referencing out-of-range
/// vertices are skipped. Output edges reference the returned welded
/// [`vertices`](FeatureEdges::vertices).
#[must_use]
pub fn extract_feature_edges(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: FeatureEdgeParams,
) -> Option<FeatureEdges> {
    if vertices.is_empty() || indices.is_empty() || !params.crease_angle_radians.is_finite() {
        return None;
    }

    let welded = weld_mesh(
        vertices,
        indices,
        WeldParams {
            position_epsilon: params.weld_epsilon,
            drop_duplicate_triangles: false,
        },
    )?;
    if welded.indices.is_empty() {
        return None;
    }

    // Accumulate the unit face normals incident on every unordered edge.
    let mut edge_faces: HashMap<(u32, u32), Vec<Vec3>> = HashMap::new();
    for tri in &welded.indices {
        let Some(normal) = face_normal(&welded.vertices, *tri) else {
            continue;
        };
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_faces.entry(key).or_default().push(normal);
        }
    }

    if edge_faces.is_empty() {
        return None;
    }

    let mut edges: Vec<FeatureEdge> = edge_faces
        .into_iter()
        .map(|((v0, v1), normals)| classify_edge(v0, v1, &normals, params.crease_angle_radians))
        .collect();
    edges.sort_by_key(|e| (e.v0, e.v1));

    Some(FeatureEdges {
        vertices: welded.vertices,
        edges,
    })
}

/// Classifies one edge from the unit normals of its incident faces.
fn classify_edge(v0: u32, v1: u32, normals: &[Vec3], crease_angle: f32) -> FeatureEdge {
    let (kind, dihedral_angle) = match normals.len() {
        1 => (EdgeKind::Boundary, 0.0),
        2 => {
            let angle = dihedral(normals[0], normals[1]);
            let kind = if angle >= crease_angle {
                EdgeKind::Crease
            } else {
                EdgeKind::Smooth
            };
            (kind, angle)
        }
        _ => {
            // Non-manifold: report the sharpest pairwise fold as a diagnostic.
            let mut max_angle = 0.0f32;
            for i in 0..normals.len() {
                for j in (i + 1)..normals.len() {
                    max_angle = max_angle.max(dihedral(normals[i], normals[j]));
                }
            }
            (EdgeKind::NonManifold, max_angle)
        }
    };
    FeatureEdge {
        v0,
        v1,
        kind,
        dihedral_angle,
    }
}

/// Dihedral angle (radians) between two unit face normals.
fn dihedral(n0: Vec3, n1: Vec3) -> f32 {
    let cos = n0.dot(n1).clamp(-1.0, 1.0);
    // f64 acos: the f32 trig ops are disallowed in this workspace for libm
    // determinism, and the extra precision is harmless here.
    f64::from(cos).acos() as f32
}

/// Right-handed unit normal of one triangle, or `None` when it references an
/// out-of-range vertex or is degenerate.
fn face_normal(vertices: &[Vec3], tri: [u32; 3]) -> Option<Vec3> {
    let n = vertices.len();
    let i0 = tri[0] as usize;
    let i1 = tri[1] as usize;
    let i2 = tri[2] as usize;
    if i0 >= n || i1 >= n || i2 >= n {
        return None;
    }
    let a = vertices[i0];
    let b = vertices[i1];
    let c = vertices[i2];
    let normal = (b - a).cross(c - a);
    if normal.length_squared() <= DEGENERATE_EPSILON {
        return None;
    }
    Some(normal.normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound, with shared vertices.
    /// Every one of its 12 silhouette edges is a 90-degree crease.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(extract_feature_edges(&[], &[[0, 1, 2]], FeatureEdgeParams::default()).is_none());
        assert!(extract_feature_edges(&[Vec3::ZERO], &[], FeatureEdgeParams::default()).is_none());
    }

    #[test]
    fn non_finite_crease_angle_is_rejected() {
        let (v, f) = unit_cube();
        let params = FeatureEdgeParams {
            crease_angle_radians: f32::NAN,
            weld_epsilon: 1.0e-5,
        };
        assert!(extract_feature_edges(&v, &f, params).is_none());
    }

    #[test]
    fn single_flat_quad_has_only_boundary_and_one_smooth_diagonal() {
        // Two coplanar triangles forming a square; the shared diagonal is a
        // smooth edge, the four outer edges are boundaries.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let fe = extract_feature_edges(&verts, &faces, FeatureEdgeParams::default()).unwrap();
        assert_eq!(fe.boundary_count(), 4);
        assert_eq!(fe.smooth_count(), 1);
        assert_eq!(fe.crease_count(), 0);
        assert_eq!(fe.non_manifold_count(), 0);
        // The one shared diagonal is coplanar: dihedral ~0.
        let diag = fe
            .edges
            .iter()
            .find(|e| e.kind == EdgeKind::Smooth)
            .unwrap();
        assert!(diag.dihedral_angle.abs() < 1.0e-4);
    }

    #[test]
    fn cube_edges_are_all_ninety_degree_creases() {
        let (v, f) = unit_cube();
        let fe = extract_feature_edges(&v, &f, FeatureEdgeParams::default()).unwrap();
        // A closed cube has 18 edges: 12 silhouette creases + 6 face diagonals.
        assert_eq!(fe.len(), 18);
        assert_eq!(fe.boundary_count(), 0);
        assert_eq!(fe.non_manifold_count(), 0);
        assert_eq!(fe.crease_count(), 12);
        assert_eq!(fe.smooth_count(), 6);
        for e in &fe.edges {
            if e.kind == EdgeKind::Crease {
                assert!(
                    (e.dihedral_angle - core::f32::consts::FRAC_PI_2).abs() < 1.0e-3,
                    "crease angle {} not ~90deg",
                    e.dihedral_angle
                );
            }
        }
    }

    #[test]
    fn high_threshold_demotes_cube_creases_to_smooth() {
        let (v, f) = unit_cube();
        // Threshold above 90 degrees: no fold qualifies as a crease.
        let params = FeatureEdgeParams::with_crease_degrees(120.0, 1.0e-5);
        let fe = extract_feature_edges(&v, &f, params).unwrap();
        assert_eq!(fe.crease_count(), 0);
        assert_eq!(fe.smooth_count(), 18);
    }

    #[test]
    fn third_face_on_an_edge_is_non_manifold() {
        // Two triangles share edge (0,1); a third triangle also uses it.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, -1.0, 0.0),
            Vec3::new(0.5, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]];
        let fe = extract_feature_edges(&verts, &faces, FeatureEdgeParams::default()).unwrap();
        let shared = fe.edges.iter().find(|e| e.v0 == 0 && e.v1 == 1).unwrap();
        assert_eq!(shared.kind, EdgeKind::NonManifold);
        assert_eq!(fe.non_manifold_count(), 1);
    }

    #[test]
    fn sharp_edges_excludes_smooth() {
        let (v, f) = unit_cube();
        let fe = extract_feature_edges(&v, &f, FeatureEdgeParams::default()).unwrap();
        let sharp = fe.sharp_edges().count();
        assert_eq!(sharp, fe.len() - fe.smooth_count());
        assert!(fe.sharp_edges().all(|e| e.kind != EdgeKind::Smooth));
    }

    #[test]
    fn edges_are_sorted_and_well_formed() {
        let (v, f) = unit_cube();
        let fe = extract_feature_edges(&v, &f, FeatureEdgeParams::default()).unwrap();
        for e in &fe.edges {
            assert!(e.v0 < e.v1);
            assert!((e.v1 as usize) < fe.vertices.len());
        }
        for pair in fe.edges.windows(2) {
            assert!((pair[0].v0, pair[0].v1) < (pair[1].v0, pair[1].v1));
        }
    }

    #[test]
    fn hairline_crack_is_welded_before_classification() {
        // Two triangles meeting along an edge, but the shared edge's vertices
        // are duplicated with a sub-epsilon offset (a hairline crack). Welding
        // must merge them so the edge is a single crease, not two boundaries.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            // Second triangle: shares edge (0,1) but with a tiny offset copy.
            Vec3::new(0.0, 1.0e-7, 0.0),
            Vec3::new(1.0, 1.0e-7, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let params = FeatureEdgeParams::with_crease_degrees(40.0, 1.0e-4);
        let fe = extract_feature_edges(&verts, &faces, params).unwrap();
        // After welding, the two triangles share one edge -> not 6 boundaries.
        assert!(fe.boundary_count() < 6);
    }
}
