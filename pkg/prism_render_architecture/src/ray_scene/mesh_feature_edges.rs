//! Feature-edge classification (crease, boundary, non-manifold) for the `CPU`
//! golden path.
//!
//! A *feature edge* is one that should visually or topologically break the
//! surface: a sharp crease between two faces that meet at a steep angle, an
//! open boundary at the mesh's rim, or a non-manifold edge where more than two
//! faces meet. Feature edges are the canonical input to several `AAA` passes —
//! splitting vertices so hard edges render crisply (smoothing-group / normal
//! seams), drawing silhouette and crease outlines, and seeding `UV` or
//! simplification boundaries that must not be smoothed across.
//!
//! [`detect_feature_edges`] walks every undirected edge once:
//!
//! * An edge shared by **one** triangle is a [`EdgeKind::Boundary`] rim edge.
//! * An edge shared by **more than two** triangles is [`EdgeKind::NonManifold`].
//! * An edge shared by exactly **two** triangles is a [`EdgeKind::Crease`] when
//!   the dot product of its two unit face normals is **below** the supplied
//!   `cos_threshold` (i.e. the dihedral angle is sharper than the angle whose
//!   cosine is that threshold); otherwise it is [`EdgeKind::Smooth`].
//!
//! Passing `cos_threshold` as a cosine keeps the classifier free of inverse
//! trigonometry: callers precompute `angle.cos()` once (or pick a constant such
//! as `0.5` for 60°) and the per-edge test is a single comparison. Face normals
//! come from a cross product normalized with `sqrt`; degenerate zero-area faces
//! yield a zero normal (dot `0`), so their edges fall on the crease side for
//! any positive threshold. All math is linear plus `sqrt`, honouring the
//! golden-path ban on `f32` transcendental functions.

use std::collections::HashMap;

use super::triangle_mesh::TriangleMesh;

/// Classification of a mesh edge by its local topology and dihedral sharpness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// Manifold edge whose two faces meet smoothly (face-normal dot at or above
    /// the threshold).
    Smooth,
    /// Manifold edge whose two faces meet at a sharp dihedral angle (face-normal
    /// dot below the threshold).
    Crease,
    /// Open rim edge touched by exactly one triangle.
    Boundary,
    /// Edge shared by more than two triangles.
    NonManifold,
}

/// Errors returned by [`detect_feature_edges`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureEdgeError {
    /// `cos_threshold` was not finite or lay outside the valid cosine range
    /// `[-1, 1]`.
    InvalidThreshold,
}

impl core::fmt::Display for FeatureEdgeError {
    /// Formats the error as a short human-readable diagnostic.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidThreshold => {
                write!(f, "cos_threshold must be a finite value in [-1, 1]")
            }
        }
    }
}

impl std::error::Error for FeatureEdgeError {}

/// Feature-edge classification of a [`TriangleMesh`], keyed by sorted endpoint
/// pair.
#[derive(Clone, Debug)]
pub struct FeatureEdges {
    /// Classification of every undirected edge in the mesh.
    kinds: HashMap<(u32, u32), EdgeKind>,
    /// Sorted crease edges.
    creases: Vec<(u32, u32)>,
    /// Sorted boundary edges.
    boundaries: Vec<(u32, u32)>,
    /// Sorted non-manifold edges.
    non_manifolds: Vec<(u32, u32)>,
    /// Sorted union of crease, boundary, and non-manifold edges — the full
    /// feature-edge set.
    features: Vec<(u32, u32)>,
}

impl FeatureEdges {
    /// Returns the classification of the undirected edge `{a, b}`, or `None`
    /// when no such edge exists in the mesh.
    pub fn kind(&self, a: u32, b: u32) -> Option<EdgeKind> {
        self.kinds.get(&sorted_pair(a, b)).copied()
    }

    /// Returns whether the undirected edge `{a, b}` is a feature edge (crease,
    /// boundary, or non-manifold).
    pub fn is_feature(&self, a: u32, b: u32) -> bool {
        matches!(
            self.kind(a, b),
            Some(EdgeKind::Crease | EdgeKind::Boundary | EdgeKind::NonManifold)
        )
    }

    /// Returns the sorted crease edges.
    pub fn crease_edges(&self) -> &[(u32, u32)] {
        &self.creases
    }

    /// Returns the sorted boundary (open rim) edges.
    pub fn boundary_edges(&self) -> &[(u32, u32)] {
        &self.boundaries
    }

    /// Returns the sorted non-manifold edges.
    pub fn non_manifold_edges(&self) -> &[(u32, u32)] {
        &self.non_manifolds
    }

    /// Returns the sorted union of all feature edges.
    pub fn feature_edges(&self) -> &[(u32, u32)] {
        &self.features
    }
}

/// Classifies every edge of `mesh` as smooth, crease, boundary, or
/// non-manifold, treating a manifold edge as a crease when its two unit face
/// normals have a dot product below `cos_threshold`.
///
/// # Errors
///
/// Returns [`FeatureEdgeError::InvalidThreshold`] when `cos_threshold` is not
/// finite or lies outside `[-1, 1]`.
pub fn detect_feature_edges(
    mesh: &TriangleMesh,
    cos_threshold: f32,
) -> Result<FeatureEdges, FeatureEdgeError> {
    if !cos_threshold.is_finite() || !(-1.0..=1.0).contains(&cos_threshold) {
        return Err(FeatureEdgeError::InvalidThreshold);
    }

    // Precompute a unit normal per triangle.
    let positions = mesh.positions();
    let face_normals: Vec<[f32; 3]> = mesh
        .indices()
        .iter()
        .map(|tri| face_normal(positions, tri))
        .collect();

    // Map each undirected edge to the triangles touching it.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (triangle, tri) in mesh.indices().iter().enumerate() {
        let [a, b, c] = *tri;
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            edge_faces.entry(sorted_pair(u, v)).or_default().push(triangle);
        }
    }

    let mut kinds = HashMap::with_capacity(edge_faces.len());
    let mut creases = Vec::new();
    let mut boundaries = Vec::new();
    let mut non_manifolds = Vec::new();
    for (&edge, faces) in &edge_faces {
        let kind = match faces.as_slice() {
            [_] => {
                boundaries.push(edge);
                EdgeKind::Boundary
            }
            [f0, f1] => {
                let n0 = face_normals[*f0];
                let n1 = face_normals[*f1];
                let dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
                if dot < cos_threshold {
                    creases.push(edge);
                    EdgeKind::Crease
                } else {
                    EdgeKind::Smooth
                }
            }
            _ => {
                non_manifolds.push(edge);
                EdgeKind::NonManifold
            }
        };
        kinds.insert(edge, kind);
    }

    creases.sort_unstable();
    boundaries.sort_unstable();
    non_manifolds.sort_unstable();
    let mut features = Vec::with_capacity(creases.len() + boundaries.len() + non_manifolds.len());
    features.extend_from_slice(&creases);
    features.extend_from_slice(&boundaries);
    features.extend_from_slice(&non_manifolds);
    features.sort_unstable();

    Ok(FeatureEdges {
        kinds,
        creases,
        boundaries,
        non_manifolds,
        features,
    })
}

/// Returns the sorted `(min, max)` endpoint pair keying a shared edge.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Returns the unit geometric normal of a triangle, or a zero vector for a
/// degenerate (zero-area) face.
fn face_normal(positions: &[[f32; 3]], tri: &[u32; 3]) -> [f32; 3] {
    let p0 = positions[tri[0] as usize];
    let p1 = positions[tri[1] as usize];
    let p2 = positions[tri[2] as usize];
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        [n[0] * inv, n[1] * inv, n[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat unit square (two coplanar triangles) sharing diagonal (1, 2).
    fn flat_quad() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [2, 1, 3]],
        )
        .unwrap()
    }

    /// Two triangles folded at a 90° dihedral along their shared edge (0, 1).
    fn folded_pair() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [0, 1, 3]],
        )
        .unwrap()
    }

    #[test]
    fn rejects_invalid_threshold() {
        let mesh = flat_quad();
        for bad in [f32::NAN, f32::INFINITY, 2.0, -2.0] {
            assert_eq!(
                detect_feature_edges(&mesh, bad).unwrap_err(),
                FeatureEdgeError::InvalidThreshold
            );
        }
    }

    #[test]
    fn coplanar_interior_edge_is_smooth() {
        let fe = detect_feature_edges(&flat_quad(), 0.5).unwrap();
        // Shared diagonal (1, 2) is interior and flat.
        assert_eq!(fe.kind(1, 2), Some(EdgeKind::Smooth));
        assert!(!fe.is_feature(1, 2));
    }

    #[test]
    fn boundary_edges_are_detected() {
        let fe = detect_feature_edges(&flat_quad(), 0.5).unwrap();
        // The quad's four rim edges are boundaries; its one interior diagonal
        // is not.
        assert_eq!(fe.boundary_edges().len(), 4);
        assert_eq!(fe.kind(0, 1), Some(EdgeKind::Boundary));
        assert!(fe.is_feature(0, 1));
    }

    #[test]
    fn sharp_fold_is_a_crease() {
        // 90° fold → face-normal dot 0, below cos(45°) ≈ 0.707.
        let fe = detect_feature_edges(&folded_pair(), 0.707).unwrap();
        assert_eq!(fe.kind(0, 1), Some(EdgeKind::Crease));
        assert_eq!(fe.crease_edges(), &[(0, 1)]);
    }

    #[test]
    fn gentle_fold_below_threshold_is_smooth() {
        // Same 90° fold, but a very permissive threshold (cos ≈ 0 at 90°):
        // dot 0 is not strictly below 0 → smooth.
        let fe = detect_feature_edges(&folded_pair(), 0.0).unwrap();
        assert_eq!(fe.kind(0, 1), Some(EdgeKind::Smooth));
        assert!(fe.crease_edges().is_empty());
    }

    #[test]
    fn non_manifold_edge_is_detected() {
        // Three triangles share edge (0, 1).
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        )
        .unwrap();
        let fe = detect_feature_edges(&mesh, 0.5).unwrap();
        assert_eq!(fe.kind(0, 1), Some(EdgeKind::NonManifold));
        assert_eq!(fe.non_manifold_edges(), &[(0, 1)]);
    }

    #[test]
    fn feature_set_is_sorted_union() {
        let fe = detect_feature_edges(&folded_pair(), 0.707).unwrap();
        // One crease (0,1) plus the four open rim edges = five features.
        assert_eq!(fe.feature_edges().len(), 5);
        let mut sorted = fe.feature_edges().to_vec();
        sorted.sort_unstable();
        assert_eq!(fe.feature_edges(), sorted.as_slice());
    }

    #[test]
    fn unknown_edge_has_no_kind() {
        let fe = detect_feature_edges(&flat_quad(), 0.5).unwrap();
        assert_eq!(fe.kind(0, 3), None);
    }

    #[test]
    fn empty_mesh_has_no_edges() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let fe = detect_feature_edges(&mesh, 0.5).unwrap();
        assert!(fe.feature_edges().is_empty());
        assert!(fe.crease_edges().is_empty());
        assert!(fe.boundary_edges().is_empty());
        assert!(fe.non_manifold_edges().is_empty());
    }

    #[test]
    fn threshold_extremes_classify_consistently() {
        // cos_threshold = 1.0 → every manifold edge whose dot < 1 is a crease.
        // The folded pair's interior edge has dot 0 < 1 → crease.
        let strict = detect_feature_edges(&folded_pair(), 1.0).unwrap();
        assert_eq!(strict.kind(0, 1), Some(EdgeKind::Crease));
        // cos_threshold = -1.0 → nothing is strictly below → smooth interior.
        let loose = detect_feature_edges(&folded_pair(), -1.0).unwrap();
        assert_eq!(loose.kind(0, 1), Some(EdgeKind::Smooth));
    }
}
