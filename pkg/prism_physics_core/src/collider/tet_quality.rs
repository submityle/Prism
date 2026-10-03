//! Quality metrics for tetrahedra and tetrahedral meshes.
//!
//! Finite-element and finite-volume solvers are only as stable as their worst
//! element: inverted ("flipped") tets break the physics outright, and slivers
//! -- near-degenerate tets with a tiny inradius-to-circumradius ratio -- wreck
//! conditioning. This module measures the standard shape descriptors so a
//! meshing stage (for example [`crate::collider::tetrahedralize`]) can be
//! graded and bad elements flagged.
//!
//! The core metric is the normalised *radius ratio* `3 * r_in / r_circ`, which
//! is `1` for a regular tetrahedron and approaches `0` as the element
//! degenerates. Dihedral angles bound the sliver/needle/wedge failure modes
//! directly. These are textbook mesh-quality measures; nothing here is derived
//! from Unreal Engine source.

use glam::{Mat3, Vec3};

/// Below this `|determinant|` of the edge matrix the tetrahedron is treated as
/// flat (no finite circumsphere).
const DEGENERATE_DET: f32 = 1e-12;

/// Shape descriptors for a single tetrahedron.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetQuality {
    /// Signed volume. Negative means the tetrahedron is inverted relative to
    /// the positive `(v1-v0) . ((v2-v0) x (v3-v0)) > 0` convention.
    pub volume: f32,
    /// `true` when the signed volume is not strictly positive (inverted or
    /// flat).
    pub inverted: bool,
    /// `true` when the element is numerically flat (no finite circumsphere).
    pub degenerate: bool,
    /// Normalised radius ratio `3 * r_in / r_circ` in `(0, 1]`; `1` for a
    /// regular tetrahedron, `0` when degenerate.
    pub radius_ratio: f32,
    /// Smallest dihedral angle across the six edges, in degrees.
    pub min_dihedral_deg: f32,
    /// Largest dihedral angle across the six edges, in degrees.
    pub max_dihedral_deg: f32,
    /// Shortest edge length.
    pub min_edge: f32,
    /// Longest edge length.
    pub max_edge: f32,
}

impl TetQuality {
    /// Whether this tetrahedron is usable: positive volume and not flat.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.inverted && !self.degenerate
    }
}

/// Outward unit normal of the face `(a, b, c)` of a tetrahedron whose remaining
/// vertex is `apex`, i.e. the normal points away from `apex`.
fn outward_face_normal(a: Vec3, b: Vec3, c: Vec3, apex: Vec3) -> Vec3 {
    let n = (b - a).cross(c - a);
    let len = n.length();
    if len <= 1e-20 {
        return Vec3::ZERO;
    }
    let unit = n / len;
    if unit.dot(apex - a) > 0.0 {
        -unit
    } else {
        unit
    }
}

/// Dihedral angle (degrees) between two outward face normals meeting at an edge.
fn dihedral_deg(n0: Vec3, n1: Vec3) -> f32 {
    let cos = n0.dot(n1).clamp(-1.0, 1.0);
    // Dihedral angle = pi - angle(n0, n1) for outward normals. f32 trig is
    // disallowed in this workspace for libm determinism, so route through f64.
    let angle = core::f64::consts::PI - f64::from(cos).acos();
    (angle * 180.0 / core::f64::consts::PI) as f32
}

/// Circumradius of a tetrahedron, or `None` when it is numerically flat.
fn circumradius(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let e3 = v3 - v0;
    // Solve A c' = b with rows e1,e2,e3 and b_i = 0.5 * |e_i|^2 for the
    // circumcentre offset from v0.
    let a = Mat3::from_cols(e1, e2, e3).transpose();
    if a.determinant().abs() < DEGENERATE_DET {
        return None;
    }
    let b = Vec3::new(
        0.5 * e1.length_squared(),
        0.5 * e2.length_squared(),
        0.5 * e3.length_squared(),
    );
    let centre = a.inverse() * b;
    Some(centre.length())
}

/// Computes shape metrics for a single tetrahedron `(v0, v1, v2, v3)`.
#[must_use]
pub fn tet_quality(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> TetQuality {
    let vol6 = (v1 - v0).dot((v2 - v0).cross(v3 - v0));
    let volume = vol6 / 6.0;
    let inverted = volume.is_nan() || volume <= 0.0;

    // Edge lengths over the six edges.
    let edges = [
        (v1 - v0).length(),
        (v2 - v0).length(),
        (v3 - v0).length(),
        (v2 - v1).length(),
        (v3 - v1).length(),
        (v3 - v2).length(),
    ];
    let mut min_edge = f32::INFINITY;
    let mut max_edge = 0.0f32;
    for &e in &edges {
        min_edge = min_edge.min(e);
        max_edge = max_edge.max(e);
    }

    // Face areas (for the inradius) and the four outward normals.
    let areas = [
        0.5 * (v2 - v1).cross(v3 - v1).length(), // face opposite v0: (v1,v2,v3)
        0.5 * (v2 - v0).cross(v3 - v0).length(), // opposite v1: (v0,v2,v3)
        0.5 * (v1 - v0).cross(v3 - v0).length(), // opposite v2: (v0,v1,v3)
        0.5 * (v1 - v0).cross(v2 - v0).length(), // opposite v3: (v0,v1,v2)
    ];
    let area_total: f32 = areas.iter().sum();

    let normals = [
        outward_face_normal(v1, v2, v3, v0),
        outward_face_normal(v0, v2, v3, v1),
        outward_face_normal(v0, v1, v3, v2),
        outward_face_normal(v0, v1, v2, v3),
    ];
    // The six dihedral angles correspond to the six face pairs.
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut min_dihedral_deg = f32::INFINITY;
    let mut max_dihedral_deg = 0.0f32;
    for &(i, j) in &pairs {
        let d = dihedral_deg(normals[i], normals[j]);
        min_dihedral_deg = min_dihedral_deg.min(d);
        max_dihedral_deg = max_dihedral_deg.max(d);
    }

    let (degenerate, radius_ratio) = match circumradius(v0, v1, v2, v3) {
        Some(r_circ) if r_circ > 0.0 && area_total > 0.0 => {
            let r_in = 3.0 * volume.abs() / area_total;
            let ratio = (3.0 * r_in / r_circ).clamp(0.0, 1.0);
            (false, ratio)
        }
        _ => (true, 0.0),
    };

    TetQuality {
        volume,
        inverted,
        degenerate,
        radius_ratio,
        min_dihedral_deg,
        max_dihedral_deg,
        min_edge,
        max_edge,
    }
}

/// Thresholds controlling how a tetrahedral mesh is graded.
#[derive(Clone, Copy, Debug)]
pub struct TetQualityParams {
    /// Elements whose radius ratio falls below this bound are counted as
    /// slivers.
    pub sliver_radius_ratio: f32,
}

impl Default for TetQualityParams {
    fn default() -> Self {
        Self {
            sliver_radius_ratio: 0.1,
        }
    }
}

/// Aggregate quality report for a whole tetrahedral mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetMeshQualityReport {
    /// Number of tetrahedra inspected.
    pub tet_count: usize,
    /// Count with non-positive signed volume (inverted or flat).
    pub inverted_count: usize,
    /// Count that are numerically flat (no finite circumsphere).
    pub degenerate_count: usize,
    /// Count whose radius ratio is below the sliver threshold.
    pub sliver_count: usize,
    /// Smallest radius ratio observed.
    pub min_radius_ratio: f32,
    /// Mean radius ratio across all tetrahedra.
    pub mean_radius_ratio: f32,
    /// Smallest dihedral angle observed, in degrees.
    pub min_dihedral_deg: f32,
    /// Largest dihedral angle observed, in degrees.
    pub max_dihedral_deg: f32,
    /// Smallest signed volume observed.
    pub min_volume: f32,
}

impl TetMeshQualityReport {
    /// Whether the mesh has no inverted or degenerate elements.
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.inverted_count == 0 && self.degenerate_count == 0
    }
}

/// Grades every tetrahedron of a mesh and aggregates the result.
///
/// Returns `None` when `tets` is empty or any tetrahedron references a vertex
/// index outside `vertices`.
#[must_use]
pub fn analyze_tet_mesh_quality(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    params: &TetQualityParams,
) -> Option<TetMeshQualityReport> {
    if tets.is_empty() {
        return None;
    }
    let n = vertices.len();

    let mut inverted_count = 0usize;
    let mut degenerate_count = 0usize;
    let mut sliver_count = 0usize;
    let mut min_radius_ratio = f32::INFINITY;
    let mut ratio_sum = 0.0f64;
    let mut min_dihedral_deg = f32::INFINITY;
    let mut max_dihedral_deg = 0.0f32;
    let mut min_volume = f32::INFINITY;

    for t in tets {
        for &id in t {
            if (id as usize) >= n {
                return None;
            }
        }
        let q = tet_quality(
            vertices[t[0] as usize],
            vertices[t[1] as usize],
            vertices[t[2] as usize],
            vertices[t[3] as usize],
        );
        if q.inverted {
            inverted_count += 1;
        }
        if q.degenerate {
            degenerate_count += 1;
        }
        if q.radius_ratio < params.sliver_radius_ratio {
            sliver_count += 1;
        }
        min_radius_ratio = min_radius_ratio.min(q.radius_ratio);
        ratio_sum += f64::from(q.radius_ratio);
        min_dihedral_deg = min_dihedral_deg.min(q.min_dihedral_deg);
        max_dihedral_deg = max_dihedral_deg.max(q.max_dihedral_deg);
        min_volume = min_volume.min(q.volume);
    }

    let mean_radius_ratio = (ratio_sum / tets.len() as f64) as f32;

    Some(TetMeshQualityReport {
        tet_count: tets.len(),
        inverted_count,
        degenerate_count,
        sliver_count,
        min_radius_ratio,
        mean_radius_ratio,
        min_dihedral_deg,
        max_dihedral_deg,
        min_volume,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::{tetrahedralize, TetMeshParams};

    fn regular_tet() -> [Vec3; 4] {
        [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ]
    }

    #[test]
    fn regular_tet_has_unit_radius_ratio() {
        let v = regular_tet();
        let q = tet_quality(v[0], v[1], v[2], v[3]);
        assert!(
            (q.radius_ratio - 1.0).abs() < 0.02,
            "ratio {}",
            q.radius_ratio
        );
        // All dihedral angles equal arccos(1/3) ~ 70.5288 degrees.
        assert!(
            (q.min_dihedral_deg - 70.5288).abs() < 0.5,
            "min {}",
            q.min_dihedral_deg
        );
        assert!(
            (q.max_dihedral_deg - 70.5288).abs() < 0.5,
            "max {}",
            q.max_dihedral_deg
        );
        assert!(!q.degenerate);
        let expected_edge = (2.0f32 * 2.0).sqrt() * core::f32::consts::SQRT_2; // 2*sqrt(2)
        assert!((q.min_edge - expected_edge).abs() < 1e-3);
        assert!((q.max_edge - expected_edge).abs() < 1e-3);
    }

    #[test]
    fn inverted_tet_is_detected() {
        let a = Vec3::ZERO;
        let b = Vec3::X;
        let c = Vec3::Y;
        let d = Vec3::Z;
        let good = tet_quality(a, b, c, d);
        assert!(!good.inverted);
        assert!(good.volume > 0.0);
        // Swapping two vertices flips orientation.
        let bad = tet_quality(a, b, d, c);
        assert!(bad.inverted);
        assert!(bad.volume < 0.0);
    }

    #[test]
    fn flat_tet_is_degenerate() {
        // All four points coplanar (z = 0).
        let q = tet_quality(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0));
        assert!(q.degenerate);
        assert_eq!(q.radius_ratio, 0.0);
        assert!(q.inverted); // zero volume is not strictly positive
    }

    #[test]
    fn sliver_has_low_radius_ratio_and_wide_dihedral() {
        // A thin wedge: three points in the z=0 plane plus one barely above.
        let q = tet_quality(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.02),
        );
        assert!(!q.degenerate);
        assert!(q.radius_ratio < 0.2, "ratio {}", q.radius_ratio);
        assert!(
            q.max_dihedral_deg > 120.0,
            "max dihedral {}",
            q.max_dihedral_deg
        );
    }

    #[test]
    fn empty_mesh_returns_none() {
        assert!(analyze_tet_mesh_quality(&[], &[], &TetQualityParams::default()).is_none());
    }

    #[test]
    fn out_of_range_index_returns_none() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let tets = vec![[0u32, 1, 2, 9]];
        assert!(analyze_tet_mesh_quality(&verts, &tets, &TetQualityParams::default()).is_none());
    }

    #[test]
    fn lattice_cube_mesh_is_sound() {
        // Reuse the lattice tetrahedraliser and confirm every element is valid.
        let h = 1.0f32;
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        let idx = vec![
            [0u32, 2, 1],
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
        let mesh = tetrahedralize(&verts, &idx, &TetMeshParams::new(12)).unwrap();
        let report =
            analyze_tet_mesh_quality(&mesh.vertices, &mesh.tets, &TetQualityParams::default())
                .unwrap();
        assert_eq!(report.tet_count, mesh.tet_count());
        assert!(report.is_sound(), "report {report:?}");
        assert_eq!(report.inverted_count, 0);
        assert_eq!(report.degenerate_count, 0);
        assert!(report.min_radius_ratio > 0.0);
        assert!(report.min_volume > 0.0);
    }

    #[test]
    fn report_is_deterministic() {
        let v = regular_tet();
        let verts = vec![v[0], v[1], v[2], v[3]];
        let tets = vec![[0u32, 1, 2, 3]];
        let a = analyze_tet_mesh_quality(&verts, &tets, &TetQualityParams::default()).unwrap();
        let b = analyze_tet_mesh_quality(&verts, &tets, &TetQualityParams::default()).unwrap();
        assert_eq!(a, b);
    }
}
