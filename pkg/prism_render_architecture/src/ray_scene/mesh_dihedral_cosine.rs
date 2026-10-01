//! Per-interior-edge dihedral geometry for the `CPU` golden path.
//!
//! The dihedral angle across an edge — how sharply its two incident triangles
//! fold — drives several `AAA` geometry stages: cloth / hair bending energy
//! (Bridson-style), feature-aware adaptive tessellation and simplification,
//! and crease-preserving level-of-detail. Rather than the angle itself (whose
//! recovery needs an inverse trigonometric function), this module exposes the
//! two transcendental-free components that fully determine it:
//!
//! * the **cosine** `dot(n0, n1)` of the two unit face normals, which is `1`
//!   for coplanar faces and decreases toward `-1` as the fold sharpens, and
//! * the **signed sine** `dot(cross(n0, n1), e)` along the unit edge direction
//!   `e`, whose sign distinguishes a ridge (fold one way) from a valley (fold
//!   the other), so downstream code can reconstruct the full signed angle with
//!   `atan2` only when it genuinely needs the angle.
//!
//! [`dihedral_cosines`] visits every undirected edge once, keeps only the
//! *interior* edges shared by exactly two triangles (boundary and non-manifold
//! edges have no well-defined single dihedral and are tallied separately), and
//! skips edges whose incident faces are degenerate (zero-area, hence no
//! normal). Each cosine and signed sine costs one `sqrt`-based normalization
//! and dot / cross products; aggregates accumulate in `f64` for stability. No
//! `f32` transcendental function is used.

use std::collections::HashMap;

use super::triangle_mesh::TriangleMesh;

/// Dihedral geometry of a single interior edge of a [`TriangleMesh`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DihedralEdge {
    /// The edge endpoints as `(min, max)` vertex indices.
    pub endpoints: (u32, u32),
    /// Cosine `dot(n0, n1)` of the two unit incident-face normals, clamped to
    /// `[-1, 1]`. `1` is coplanar; smaller values are sharper folds.
    pub cosine: f32,
    /// Signed sine `dot(cross(n0, n1), e)` along the unit edge direction `e`
    /// (from the lower to the higher endpoint), clamped to `[-1, 1]`. Its sign
    /// separates ridge from valley folds; the two faces are ordered by face
    /// index so the sign is deterministic.
    pub signed_sine: f32,
}

/// Per-interior-edge dihedral statistics for a [`TriangleMesh`].
#[derive(Clone, Debug)]
pub struct DihedralCosines {
    /// Each interior edge's dihedral geometry, sorted by endpoint pair.
    edges: Vec<DihedralEdge>,
    /// Number of boundary edges (exactly one incident face) that were skipped.
    boundary_edges: usize,
    /// Number of non-manifold edges (more than two incident faces) skipped.
    nonmanifold_edges: usize,
    /// Number of interior edges skipped because an incident face is degenerate
    /// (zero area, so it has no usable normal).
    degenerate_edges: usize,
}

impl DihedralCosines {
    /// Returns the number of interior edges with a valid dihedral.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns whether no interior edge yielded a valid dihedral.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Returns the interior edges, sorted by endpoint pair.
    pub fn edges(&self) -> &[DihedralEdge] {
        &self.edges
    }

    /// Returns the dihedral geometry of the undirected edge `{a, b}`, or
    /// `None` when it is not a valid interior edge.
    pub fn edge(&self, a: u32, b: u32) -> Option<&DihedralEdge> {
        let key = sorted_pair(a, b);
        self.edges
            .binary_search_by(|e| e.endpoints.cmp(&key))
            .ok()
            .map(|i| &self.edges[i])
    }

    /// Returns the dihedral cosine of the undirected edge `{a, b}`, or `None`
    /// when it is not a valid interior edge.
    pub fn cosine(&self, a: u32, b: u32) -> Option<f32> {
        self.edge(a, b).map(|e| e.cosine)
    }

    /// Returns the smallest (sharpest-fold) cosine, or `None` with no interior
    /// edge.
    pub fn min_cosine(&self) -> Option<f32> {
        self.edges.iter().map(|e| e.cosine).reduce(f32::min)
    }

    /// Returns the largest (flattest) cosine, or `None` with no interior edge.
    pub fn max_cosine(&self) -> Option<f32> {
        self.edges.iter().map(|e| e.cosine).reduce(f32::max)
    }

    /// Returns the mean dihedral cosine accumulated in `f64`, or `None` with no
    /// interior edge.
    pub fn mean_cosine(&self) -> Option<f32> {
        if self.edges.is_empty() {
            return None;
        }
        let sum: f64 = self.edges.iter().map(|e| f64::from(e.cosine)).sum();
        Some((sum / self.edges.len() as f64) as f32)
    }

    /// Returns how many interior edges fold sharper than `cos_threshold`, i.e.
    /// whose cosine is strictly less than it — the crease candidates for that
    /// threshold.
    pub fn count_sharper_than(&self, cos_threshold: f32) -> usize {
        self.edges.iter().filter(|e| e.cosine < cos_threshold).count()
    }

    /// Returns how many interior edges are flatter than `cos_threshold`, i.e.
    /// whose cosine is greater than or equal to it.
    pub fn count_flatter_than(&self, cos_threshold: f32) -> usize {
        self.edges.iter().filter(|e| e.cosine >= cos_threshold).count()
    }

    /// Returns the number of boundary edges (one incident face) skipped.
    pub fn boundary_edges(&self) -> usize {
        self.boundary_edges
    }

    /// Returns the number of non-manifold edges (more than two faces) skipped.
    pub fn nonmanifold_edges(&self) -> usize {
        self.nonmanifold_edges
    }

    /// Returns the number of interior edges skipped due to a degenerate face.
    pub fn degenerate_edges(&self) -> usize {
        self.degenerate_edges
    }
}

/// Orders two vertex indices into a canonical `(min, max)` undirected key.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Returns `lhs - rhs` component-wise.
fn sub(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

/// Returns the cross product `lhs x rhs`.
fn cross(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

/// Returns the dot product `lhs . rhs`.
fn dot(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

/// Returns the unit-length form of `v`, or `None` when `v` is (near) zero.
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = dot(v, v);
    if len_sq <= 0.0 {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// Returns the unit normal of the triangle whose vertex indices are `tri`, or
/// `None` when the triangle is degenerate (zero area).
fn face_unit_normal(mesh: &TriangleMesh, tri: [u32; 3]) -> Option<[f32; 3]> {
    let p = mesh.positions();
    let a = p[tri[0] as usize];
    let b = p[tri[1] as usize];
    let c = p[tri[2] as usize];
    normalize(cross(sub(b, a), sub(c, a)))
}

/// Computes the dihedral cosine and signed sine of every interior edge.
///
/// Interior edges (exactly two incident triangles) with non-degenerate faces
/// are measured; boundary, non-manifold, and degenerate-face edges are counted
/// but excluded from the per-edge list. The result is sorted by endpoint pair.
pub fn dihedral_cosines(mesh: &TriangleMesh) -> DihedralCosines {
    // Gather up to the first two incident faces per undirected edge, and the
    // total incident-face count to flag non-manifold edges.
    let mut incident: HashMap<(u32, u32), ([usize; 2], usize)> = HashMap::new();
    for (face_index, tri) in mesh.indices().iter().enumerate() {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = sorted_pair(a, b);
            let entry = incident.entry(key).or_insert(([usize::MAX; 2], 0));
            if entry.1 < 2 {
                entry.0[entry.1] = face_index;
            }
            entry.1 += 1;
        }
    }

    let positions = mesh.positions();
    let indices = mesh.indices();
    let mut edges = Vec::new();
    let mut boundary_edges = 0usize;
    let mut nonmanifold_edges = 0usize;
    let mut degenerate_edges = 0usize;

    for (&endpoints, &(faces, count)) in &incident {
        if count == 1 {
            boundary_edges += 1;
            continue;
        }
        if count > 2 {
            nonmanifold_edges += 1;
            continue;
        }
        // Order the two faces by index so the signed-sine sign is deterministic.
        let (f0, f1) = if faces[0] <= faces[1] {
            (faces[0], faces[1])
        } else {
            (faces[1], faces[0])
        };
        let (Some(n0), Some(n1)) = (
            face_unit_normal(mesh, indices[f0]),
            face_unit_normal(mesh, indices[f1]),
        ) else {
            degenerate_edges += 1;
            continue;
        };
        let edge_vec = sub(positions[endpoints.1 as usize], positions[endpoints.0 as usize]);
        let signed_sine = match normalize(edge_vec) {
            Some(e) => dot(cross(n0, n1), e).clamp(-1.0, 1.0),
            None => 0.0,
        };
        edges.push(DihedralEdge {
            endpoints,
            cosine: dot(n0, n1).clamp(-1.0, 1.0),
            signed_sine,
        });
    }

    edges.sort_by_key(|e| e.endpoints);

    DihedralCosines {
        edges,
        boundary_edges,
        nonmanifold_edges,
        degenerate_edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a two-triangle mesh folded about the shared edge `(0, 1)` by the
    /// given dihedral so face normals meet at that angle.
    ///
    /// The quad shares edge `0-1` on the x-axis; triangle `0-1-2` lies flat in
    /// the `+y` half-plane and triangle `1-0-3` is lifted by `height` in `z`,
    /// producing a controllable crease.
    fn folded_pair(height: f32) -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, height],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 0, 3]],
        )
        .expect("valid mesh")
    }

    #[test]
    fn flat_pair_has_unit_cosine() {
        let d = dihedral_cosines(&folded_pair(0.0));
        assert_eq!(d.edge_count(), 1);
        let cos = d.cosine(0, 1).expect("shared edge");
        assert!((cos - 1.0).abs() < 1e-5, "coplanar faces, got {cos}");
    }

    #[test]
    fn right_angle_fold_has_zero_cosine() {
        // Lift the second triangle straight up so the faces meet at 90 degrees.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 0, 3]],
        )
        .expect("valid mesh");
        let cos = dihedral_cosines(&mesh).cosine(0, 1).expect("shared edge");
        assert!(cos.abs() < 1e-5, "right-angle fold, got {cos}");
    }

    #[test]
    fn sharper_fold_lowers_cosine() {
        let shallow = dihedral_cosines(&folded_pair(0.2)).cosine(0, 1).unwrap();
        let steep = dihedral_cosines(&folded_pair(2.0)).cosine(0, 1).unwrap();
        assert!(steep < shallow, "steeper fold {steep} !< shallow {shallow}");
    }

    #[test]
    fn opposite_folds_flip_signed_sine() {
        let up = dihedral_cosines(&folded_pair(1.0)).edge(0, 1).unwrap().signed_sine;
        let down = dihedral_cosines(&folded_pair(-1.0)).edge(0, 1).unwrap().signed_sine;
        assert!(up.abs() > 1e-4 && down.abs() > 1e-4, "folds should bend");
        assert!(up * down < 0.0, "ridge/valley should flip sign: {up} vs {down}");
    }

    #[test]
    fn cosine_and_signed_sine_are_bounded() {
        let d = dihedral_cosines(&folded_pair(3.0));
        for e in d.edges() {
            assert!((-1.0..=1.0).contains(&e.cosine));
            assert!((-1.0..=1.0).contains(&e.signed_sine));
        }
    }

    #[test]
    fn boundary_edges_are_counted_not_measured() {
        // A single triangle has three boundary edges and no interior edge.
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .expect("valid mesh");
        let d = dihedral_cosines(&mesh);
        assert_eq!(d.edge_count(), 0);
        assert_eq!(d.boundary_edges(), 3);
        assert!(d.is_empty());
    }

    #[test]
    fn nonmanifold_edge_is_counted_not_measured() {
        // Three triangles sharing edge (0, 1): a non-manifold fan.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
        )
        .expect("valid mesh");
        let d = dihedral_cosines(&mesh);
        assert_eq!(d.nonmanifold_edges(), 1);
        assert!(d.cosine(0, 1).is_none());
    }

    #[test]
    fn degenerate_face_edge_is_skipped() {
        // Triangle 1-0-3 is degenerate (collinear), so the shared edge has no
        // valid second normal.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 0, 3]],
        )
        .expect("valid mesh");
        let d = dihedral_cosines(&mesh);
        assert_eq!(d.degenerate_edges(), 1);
        assert_eq!(d.edge_count(), 0);
    }

    #[test]
    fn aggregates_match_single_edge() {
        let d = dihedral_cosines(&folded_pair(0.5));
        let cos = d.cosine(0, 1).unwrap();
        assert_eq!(d.min_cosine(), Some(cos));
        assert_eq!(d.max_cosine(), Some(cos));
        assert_eq!(d.mean_cosine(), Some(cos));
    }

    #[test]
    fn sharp_and_flat_counts_partition_edges() {
        let d = dihedral_cosines(&folded_pair(0.3));
        let cos = d.cosine(0, 1).unwrap();
        // Threshold just above the single edge's cosine: it counts as sharper.
        assert_eq!(d.count_sharper_than(cos + 0.01), 1);
        assert_eq!(d.count_flatter_than(cos + 0.01), 0);
        // Threshold at the cosine: `>=` places it on the flat side.
        assert_eq!(d.count_flatter_than(cos), 1);
        assert_eq!(d.count_sharper_than(cos), 0);
    }

    #[test]
    fn empty_mesh_has_no_edges() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())
            .expect("valid empty mesh");
        let d = dihedral_cosines(&mesh);
        assert!(d.is_empty());
        assert_eq!(d.mean_cosine(), None);
        assert_eq!(d.min_cosine(), None);
    }
}
