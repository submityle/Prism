//! Rigid-body mass properties of a closed triangle mesh for the `CPU` golden
//! path.
//!
//! Dropping an authored mesh into a physics solver needs its *volume*,
//! *centre of mass*, and *inertia tensor* — the same quantities `AAA` content
//! pipelines bake when turning art assets into simulated rigid bodies. This
//! module computes them exactly (for a closed, consistently wound surface) with
//! the signed-tetrahedron decomposition of Blow & Binstock: every triangle is
//! joined to the origin to form a tetrahedron, each tetrahedron contributes a
//! signed volume and a closed-form second-moment (covariance) matrix, and the
//! contributions sum regardless of where the origin sits relative to the solid.
//!
//! [`mass_properties`] returns [`MeshMassProperties`] carrying the surface
//! area, signed volume, volume centroid, and the inertia tensor about that
//! centroid for unit density, plus a watertight flag (every directed edge
//! matched by exactly one opposite). All sums accumulate in `f64`; the inertia
//! derivation uses only polynomial arithmetic and no transcendental function.

use std::collections::HashMap;

use super::triangle_mesh::TriangleMesh;

/// Canonical covariance of the reference tetrahedron (origin plus the unit
/// basis), scaled by `1/120`, used by the Blow & Binstock transform.
const CANONICAL_COVARIANCE: [[f64; 3]; 3] = [
    [2.0 / 120.0, 1.0 / 120.0, 1.0 / 120.0],
    [1.0 / 120.0, 2.0 / 120.0, 1.0 / 120.0],
    [1.0 / 120.0, 1.0 / 120.0, 2.0 / 120.0],
];

/// Rigid-body mass properties of a [`TriangleMesh`], all for unit density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshMassProperties {
    /// Total surface area summed over all triangles.
    surface_area: f64,
    /// Signed enclosed volume; positive for outward (counter-clockwise) winding
    /// and negative for inward winding. Meaningful only for closed meshes.
    signed_volume: f64,
    /// Volume centroid (centre of mass for uniform density).
    centroid: [f64; 3],
    /// Inertia tensor about the centroid for unit density, as a symmetric
    /// 3x3 matrix in row-major order.
    inertia: [[f64; 3]; 3],
    /// Whether every directed edge is matched by exactly one opposite directed
    /// edge, i.e. the surface is closed and consistently wound.
    watertight: bool,
}

impl MeshMassProperties {
    /// Returns the total surface area.
    pub fn surface_area(&self) -> f64 {
        self.surface_area
    }

    /// Returns the signed enclosed volume (positive for outward winding).
    pub fn signed_volume(&self) -> f64 {
        self.signed_volume
    }

    /// Returns the unsigned enclosed volume.
    pub fn volume(&self) -> f64 {
        self.signed_volume.abs()
    }

    /// Returns the volume centroid (centre of mass for uniform density).
    pub fn centroid(&self) -> [f64; 3] {
        self.centroid
    }

    /// Returns the mass for the given uniform `density` (volume times density).
    pub fn mass(&self, density: f64) -> f64 {
        self.volume() * density
    }

    /// Returns the inertia tensor about the centroid for unit density, as a
    /// symmetric row-major 3x3 matrix.
    pub fn inertia_about_centroid(&self) -> [[f64; 3]; 3] {
        self.inertia
    }

    /// Returns the inertia tensor about the centroid scaled for uniform
    /// `density`.
    pub fn inertia_for_density(&self, density: f64) -> [[f64; 3]; 3] {
        let mut scaled = self.inertia;
        for row in &mut scaled {
            for value in row {
                *value *= density;
            }
        }
        scaled
    }

    /// Returns whether the surface is closed and consistently wound.
    pub fn is_watertight(&self) -> bool {
        self.watertight
    }
}

/// Computes the unit-density mass properties of `mesh`.
///
/// Surface area and the watertight flag are always valid. Volume, centroid, and
/// inertia are only physically meaningful when [`MeshMassProperties::is_watertight`]
/// is `true`; for open meshes they still reflect the signed-tetrahedron sum but
/// have no closed-solid interpretation.
pub fn mass_properties(mesh: &TriangleMesh) -> MeshMassProperties {
    let positions = mesh.positions();

    let mut surface_area = 0.0_f64;
    let mut volume = 0.0_f64;
    // First moment integral of position over the solid: integral of x dV.
    let mut moment1 = [0.0_f64; 3];
    // Second moment integral of position outer product: integral of x x^T dV.
    let mut moment2 = [[0.0_f64; 3]; 3];
    // Directed-edge tally for the watertight / consistent-winding test.
    let mut directed: HashMap<(u32, u32), i32> = HashMap::new();

    for tri in mesh.indices() {
        let a = to_f64(positions[tri[0] as usize]);
        let b = to_f64(positions[tri[1] as usize]);
        let c = to_f64(positions[tri[2] as usize]);

        // Surface area via half the cross-product magnitude.
        let ab = sub(b, a);
        let ac = sub(c, a);
        surface_area += 0.5 * length(cross(ab, ac));

        // Tetrahedron (origin, a, b, c): A has columns a, b, c.
        let det = determinant(a, b, c);
        let tet_volume = det / 6.0;
        volume += tet_volume;

        // Tetrahedron centroid is the average of its four vertices, one of
        // which is the origin, so (a + b + c) / 4.
        let tet_centroid = [
            (a[0] + b[0] + c[0]) / 4.0,
            (a[1] + b[1] + c[1]) / 4.0,
            (a[2] + b[2] + c[2]) / 4.0,
        ];
        for i in 0..3 {
            moment1[i] += tet_volume * tet_centroid[i];
        }

        // Second moment via det(A) * A * Ccanon * A^T (Blow & Binstock).
        let transformed = covariance_contribution(a, b, c, det);
        for i in 0..3 {
            for j in 0..3 {
                moment2[i][j] += transformed[i][j];
            }
        }

        for &(u, v) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            *directed.entry((u, v)).or_insert(0) += 1;
        }
    }

    let watertight = !mesh.indices().is_empty()
        && directed.iter().all(|(&(u, v), &count)| {
            count == 1 && directed.get(&(v, u)).copied() == Some(1)
        });

    let centroid = if volume.abs() > 0.0 {
        [moment1[0] / volume, moment1[1] / volume, moment1[2] / volume]
    } else {
        [0.0; 3]
    };

    // Translate the second-moment matrix to the centroid: M_c = M - V c c^T.
    let mut central = moment2;
    for i in 0..3 {
        for j in 0..3 {
            central[i][j] -= volume * centroid[i] * centroid[j];
        }
    }

    // Inertia tensor: I = trace(M_c) * Identity - M_c.
    let trace = central[0][0] + central[1][1] + central[2][2];
    let mut inertia = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inertia[i][j] = if i == j { trace } else { 0.0 } - central[i][j];
        }
    }

    MeshMassProperties {
        surface_area,
        signed_volume: volume,
        centroid,
        inertia,
        watertight,
    }
}

/// Widens a `[f32; 3]` position to `[f64; 3]` for stable accumulation.
fn to_f64(v: [f32; 3]) -> [f64; 3] {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

/// Returns `lhs - rhs` component-wise.
fn sub(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

/// Returns the cross product `lhs x rhs`.
fn cross(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

/// Returns the Euclidean length of `v`.
fn length(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Returns the determinant of the matrix whose columns are `a`, `b`, `c`.
fn determinant(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Returns `det * A * Ccanon * A^T`, the tetrahedron's second-moment
/// contribution, where `A` has columns `a`, `b`, `c`.
fn covariance_contribution(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    det: f64,
) -> [[f64; 3]; 3] {
    // Columns of A.
    let col = [a, b, c];
    // First compute Ccanon * A^T, a 3x3 matrix whose (k, j) entry is
    // sum_m Ccanon[k][m] * A[j][m] = sum_m Ccanon[k][m] * col[m][j].
    let mut cat = [[0.0_f64; 3]; 3];
    for k in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for m in 0..3 {
                acc += CANONICAL_COVARIANCE[k][m] * col[m][j];
            }
            cat[k][j] = acc;
        }
    }
    // Then A * (Ccanon * A^T): (i, j) entry is sum_k A[i][k] * cat[k][j]
    // where A[i][k] = col[k][i].
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += col[k][i] * cat[k][j];
            }
            out[i][j] = det * acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an axis-aligned box spanning `[0, size]` per axis with outward
    /// (counter-clockwise) winding: 8 corners, 12 triangles.
    fn axis_box(sx: f32, sy: f32, sz: f32) -> TriangleMesh {
        let p = vec![
            [0.0, 0.0, 0.0],
            [sx, 0.0, 0.0],
            [sx, sy, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, sz],
            [sx, 0.0, sz],
            [sx, sy, sz],
            [0.0, sy, sz],
        ];
        // Outward-facing triangles for each of the six faces.
        let idx = vec![
            [0, 2, 1],
            [0, 3, 2], // bottom (z = 0), normal -z
            [4, 5, 6],
            [4, 6, 7], // top (z = sz), normal +z
            [0, 1, 5],
            [0, 5, 4], // front (y = 0), normal -y
            [2, 3, 7],
            [2, 7, 6], // back (y = sy), normal +y
            [1, 2, 6],
            [1, 6, 5], // right (x = sx), normal +x
            [0, 4, 7],
            [0, 7, 3], // left (x = 0), normal -x
        ];
        TriangleMesh::new(p, Vec::new(), Vec::new(), idx).expect("valid box")
    }

    #[test]
    fn unit_cube_volume_area_centroid() {
        let m = mass_properties(&axis_box(1.0, 1.0, 1.0));
        assert!(m.is_watertight());
        assert!((m.volume() - 1.0).abs() < 1e-9, "vol {}", m.volume());
        assert!(m.signed_volume() > 0.0, "outward winding should be positive");
        assert!((m.surface_area() - 6.0).abs() < 1e-9, "area {}", m.surface_area());
        for (i, &axis) in m.centroid().iter().enumerate() {
            assert!((axis - 0.5).abs() < 1e-9, "centroid[{i}] = {axis}");
        }
    }

    #[test]
    fn unit_cube_inertia_is_isotropic_sixth() {
        let m = mass_properties(&axis_box(1.0, 1.0, 1.0));
        let i = m.inertia_about_centroid();
        for (axis, row) in i.iter().enumerate() {
            assert!((row[axis] - 1.0 / 6.0).abs() < 1e-9, "I[{axis}] = {}", row[axis]);
        }
        // Off-diagonal products of inertia vanish for a centred box.
        assert!(i[0][1].abs() < 1e-9 && i[0][2].abs() < 1e-9 && i[1][2].abs() < 1e-9);
    }

    #[test]
    fn stretched_box_matches_closed_form() {
        // Box dims (2, 1, 1), mass = volume = 2 at unit density.
        let m = mass_properties(&axis_box(2.0, 1.0, 1.0));
        assert!((m.volume() - 2.0).abs() < 1e-9);
        let i = m.inertia_about_centroid();
        // Closed form: Ixx = m/12 (ly^2+lz^2), etc. with m = 2.
        let mass = 2.0;
        let ixx = mass / 12.0 * (1.0 + 1.0);
        let iyy = mass / 12.0 * (4.0 + 1.0);
        let izz = mass / 12.0 * (4.0 + 1.0);
        assert!((i[0][0] - ixx).abs() < 1e-9, "Ixx {} vs {ixx}", i[0][0]);
        assert!((i[1][1] - iyy).abs() < 1e-9, "Iyy {} vs {iyy}", i[1][1]);
        assert!((i[2][2] - izz).abs() < 1e-9, "Izz {} vs {izz}", i[2][2]);
    }

    #[test]
    fn surface_area_scales_with_face_sizes() {
        let m = mass_properties(&axis_box(2.0, 1.0, 1.0));
        // Faces: 2*(2*1) + 2*(2*1) + 2*(1*1) = 4 + 4 + 2 = 10.
        assert!((m.surface_area() - 10.0).abs() < 1e-9, "area {}", m.surface_area());
    }

    #[test]
    fn translation_invariance_of_inertia() {
        let origin_box = mass_properties(&axis_box(1.5, 0.7, 2.3));
        let shifted = {
            let base = axis_box(1.5, 0.7, 2.3);
            let shifted_positions: Vec<[f32; 3]> = base
                .positions()
                .iter()
                .map(|p| [p[0] + 10.0, p[1] - 4.0, p[2] + 2.0])
                .collect();
            let idx: Vec<[u32; 3]> = base.indices().to_vec();
            mass_properties(
                &TriangleMesh::new(shifted_positions, Vec::new(), Vec::new(), idx)
                    .expect("valid shifted box"),
            )
        };
        let a = origin_box.inertia_about_centroid();
        let b = shifted.inertia_about_centroid();
        for i in 0..3 {
            for j in 0..3 {
                assert!((a[i][j] - b[i][j]).abs() < 1e-6, "I[{i}][{j}] differs");
            }
        }
    }

    #[test]
    fn mass_and_inertia_scale_with_density() {
        let m = mass_properties(&axis_box(1.0, 1.0, 1.0));
        assert!((m.mass(5.0) - 5.0).abs() < 1e-9);
        let scaled = m.inertia_for_density(5.0);
        let base = m.inertia_about_centroid();
        assert!((scaled[0][0] - base[0][0] * 5.0).abs() < 1e-9);
    }

    #[test]
    fn inverted_winding_flips_volume_sign() {
        let base = axis_box(1.0, 1.0, 1.0);
        let flipped: Vec<[u32; 3]> =
            base.indices().iter().map(|t| [t[0], t[2], t[1]]).collect();
        let m = mass_properties(
            &TriangleMesh::new(base.positions().to_vec(), Vec::new(), Vec::new(), flipped)
                .expect("valid flipped box"),
        );
        assert!(m.signed_volume() < 0.0, "inward winding should be negative");
        assert!((m.volume() - 1.0).abs() < 1e-9);
        // Still closed and consistently wound, just inward.
        assert!(m.is_watertight());
    }

    #[test]
    fn open_mesh_is_not_watertight() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .expect("valid triangle");
        let m = mass_properties(&mesh);
        assert!(!m.is_watertight());
        assert!((m.surface_area() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn inertia_tensor_is_symmetric() {
        let m = mass_properties(&axis_box(1.3, 2.1, 0.6));
        let i = m.inertia_about_centroid();
        for (a, row) in i.iter().enumerate() {
            for (b, &value) in row.iter().enumerate() {
                assert!((value - i[b][a]).abs() < 1e-9, "asymmetry at {a},{b}");
            }
        }
    }

    #[test]
    fn empty_mesh_has_zero_properties() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())
            .expect("valid empty mesh");
        let m = mass_properties(&mesh);
        assert_eq!(m.volume(), 0.0);
        assert_eq!(m.surface_area(), 0.0);
        assert!(!m.is_watertight());
        assert_eq!(m.centroid(), [0.0, 0.0, 0.0]);
    }
}
