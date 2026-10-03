//! Rest-pose basis of the linear (constant-strain) tetrahedral finite element.
//!
//! A finite-element or corotational soft-body solver advances a tetrahedral
//! mesh by comparing each element's current shape to its rest shape. The two
//! quantities every such solver precomputes once, in the undeformed
//! configuration, are:
//!
//! * the inverse of the rest **edge matrix** `Dm = [x1-x0 | x2-x0 | x3-x0]`,
//!   which maps a deformed element back to barycentric (material) coordinates,
//!   and
//! * the signed **rest volume** `det(Dm) / 6`.
//!
//! From the inverse edge matrix the four constant shape-function gradients
//! `grad N_i` follow directly: `grad N_1, grad N_2, grad N_3` are the rows of
//! `Dm^-1` and `grad N_0 = -(grad N_1 + grad N_2 + grad N_3)`. These gradients
//! are constant over the element (the defining property of the linear
//! "constant-strain tetrahedron"), so the discrete deformation gradient of a
//! deformed element is simply `F = Ds * Dm^-1`, where `Ds` is the current edge
//! matrix. `F` is the input to every hyperelastic stress model (co-rotational,
//! St. Venant-Kirchhoff, stable Neo-Hookean, and so on).
//!
//! This module computes only the rest-pose geometry; it holds no simulation
//! state and performs no time integration, so it is fully decoupled from any
//! solver. All of it is standard linear finite-element mathematics; nothing
//! here is derived from Unreal Engine source.

use glam::{Mat3, Vec3};

/// Parameters controlling [`build_tet_fem_basis`].
#[derive(Clone, Copy, Debug)]
pub struct TetFemBasisParams {
    /// Rest tetrahedra whose absolute rest volume is at or below this threshold
    /// are treated as degenerate: their edge matrix cannot be inverted, so the
    /// basis cannot be built and the builder reports failure. This is an
    /// absolute volume in the mesh's units.
    pub min_rest_volume: f32,
}

impl TetFemBasisParams {
    /// Creates parameters with the given degeneracy threshold.
    #[must_use]
    pub fn new(min_rest_volume: f32) -> Self {
        Self { min_rest_volume }
    }
}

impl Default for TetFemBasisParams {
    fn default() -> Self {
        Self {
            min_rest_volume: 1e-12,
        }
    }
}

/// The rest-pose basis of a single linear tetrahedral element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetFemElement {
    /// Inverse of the rest edge matrix `Dm = [x1-x0 | x2-x0 | x3-x0]`.
    /// Multiplying a deformed edge matrix on the right by this maps the element
    /// to its deformation gradient.
    pub dm_inverse: Mat3,
    /// Signed rest volume `det(Dm) / 6`. Negative when the rest tetrahedron is
    /// inverted (its vertices are wound with the opposite orientation).
    pub rest_volume: f32,
    /// The four constant shape-function gradients `grad N_0 .. grad N_3`, one
    /// per element vertex, in the mesh's coordinate frame.
    pub shape_gradients: [Vec3; 4],
}

impl TetFemElement {
    /// Builds the rest basis from the four rest-pose vertex positions.
    ///
    /// Returns `None` when the tetrahedron is degenerate, i.e. its absolute
    /// rest volume is at or below `min_rest_volume`, because the edge matrix is
    /// then non-invertible.
    #[must_use]
    pub fn from_rest(x0: Vec3, x1: Vec3, x2: Vec3, x3: Vec3, min_rest_volume: f32) -> Option<Self> {
        let dm = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
        let rest_volume = dm.determinant() / 6.0;
        if rest_volume.abs() <= min_rest_volume {
            return None;
        }
        let dm_inverse = dm.inverse();
        let g1 = dm_inverse.row(0);
        let g2 = dm_inverse.row(1);
        let g3 = dm_inverse.row(2);
        let g0 = -(g1 + g2 + g3);
        Some(Self {
            dm_inverse,
            rest_volume,
            shape_gradients: [g0, g1, g2, g3],
        })
    }

    /// The discrete deformation gradient `F = Ds * Dm^-1` for the given deformed
    /// vertex positions, where `Ds = [p1-p0 | p2-p0 | p3-p0]`.
    ///
    /// Supplying the element's own rest positions yields the identity matrix.
    #[must_use]
    pub fn deformation_gradient(&self, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Mat3 {
        let ds = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);
        ds * self.dm_inverse
    }

    /// Whether the rest tetrahedron is inverted (its signed rest volume is
    /// negative).
    #[must_use]
    pub fn is_inverted(&self) -> bool {
        self.rest_volume < 0.0
    }
}

/// The rest-pose finite-element basis of a whole tetrahedral mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct TetFemBasis {
    /// One [`TetFemElement`] per input tetrahedron, in input order.
    pub elements: Vec<TetFemElement>,
}

impl TetFemBasis {
    /// Number of elements (equal to the number of input tetrahedra).
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Number of elements whose rest tetrahedron is inverted.
    #[must_use]
    pub fn inverted_count(&self) -> usize {
        self.elements.iter().filter(|e| e.is_inverted()).count()
    }

    /// Total unsigned rest volume summed over every element.
    #[must_use]
    pub fn total_rest_volume(&self) -> f32 {
        self.elements.iter().map(|e| e.rest_volume.abs()).sum()
    }
}

/// Builds the rest-pose finite-element basis of a tetrahedral mesh.
///
/// `tets` indexes into `vertices`; the resulting basis has one element per
/// tetrahedron, in input order. Returns `None` when `tets` is empty, when any
/// tet index is out of range, or when any rest tetrahedron is degenerate (its
/// absolute rest volume is at or below `params.min_rest_volume`).
#[must_use]
pub fn build_tet_fem_basis(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    params: &TetFemBasisParams,
) -> Option<TetFemBasis> {
    if tets.is_empty() {
        return None;
    }
    let n = vertices.len();
    let mut elements = Vec::with_capacity(tets.len());
    for &t in tets {
        if t.iter().any(|&vi| vi as usize >= n) {
            return None;
        }
        let element = TetFemElement::from_rest(
            vertices[t[0] as usize],
            vertices[t[1] as usize],
            vertices[t[2] as usize],
            vertices[t[3] as usize],
            params.min_rest_volume,
        )?;
        elements.push(element);
    }
    Some(TetFemBasis { elements })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_mass::{compute_tet_mass_properties, TetMassParams};
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};

    fn cube_surface(h: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
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
        (verts, idx)
    }

    const UNIT: [Vec3; 4] = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    fn mat_close(a: Mat3, b: Mat3, eps: f32) -> bool {
        (a - b).to_cols_array().iter().all(|&x| x.abs() <= eps)
    }

    #[test]
    fn reference_unit_tet_basis() {
        let e = TetFemElement::from_rest(UNIT[0], UNIT[1], UNIT[2], UNIT[3], 1e-12).unwrap();
        assert!(mat_close(e.dm_inverse, Mat3::IDENTITY, 1e-6));
        assert!((e.rest_volume - 1.0 / 6.0).abs() < 1e-6);
        assert!((e.shape_gradients[0] - Vec3::new(-1.0, -1.0, -1.0)).length() < 1e-6);
        assert!((e.shape_gradients[1] - Vec3::X).length() < 1e-6);
        assert!((e.shape_gradients[2] - Vec3::Y).length() < 1e-6);
        assert!((e.shape_gradients[3] - Vec3::Z).length() < 1e-6);
    }

    #[test]
    fn shape_gradients_sum_to_zero() {
        // A skewed, non-axis-aligned tetrahedron.
        let x0 = Vec3::new(0.3, -0.2, 0.1);
        let x1 = Vec3::new(1.4, 0.1, -0.3);
        let x2 = Vec3::new(-0.2, 1.1, 0.4);
        let x3 = Vec3::new(0.1, 0.2, 1.7);
        let e = TetFemElement::from_rest(x0, x1, x2, x3, 1e-12).unwrap();
        let sum: Vec3 = e.shape_gradients.iter().copied().sum();
        assert!(sum.length() < 1e-5, "shape gradients summed to {sum:?}");
    }

    #[test]
    fn deformation_gradient_is_identity_at_rest() {
        let x0 = Vec3::new(0.3, -0.2, 0.1);
        let x1 = Vec3::new(1.4, 0.1, -0.3);
        let x2 = Vec3::new(-0.2, 1.1, 0.4);
        let x3 = Vec3::new(0.1, 0.2, 1.7);
        let e = TetFemElement::from_rest(x0, x1, x2, x3, 1e-12).unwrap();
        let f = e.deformation_gradient(x0, x1, x2, x3);
        assert!(mat_close(f, Mat3::IDENTITY, 1e-5));
    }

    #[test]
    fn deformation_gradient_recovers_affine_map() {
        let x0 = Vec3::new(0.3, -0.2, 0.1);
        let x1 = Vec3::new(1.4, 0.1, -0.3);
        let x2 = Vec3::new(-0.2, 1.1, 0.4);
        let x3 = Vec3::new(0.1, 0.2, 1.7);
        let e = TetFemElement::from_rest(x0, x1, x2, x3, 1e-12).unwrap();

        // Apply a rotation composed with a non-uniform scale to every vertex.
        let a = Mat3::from_rotation_z(0.6) * Mat3::from_diagonal(Vec3::new(2.0, 1.5, 0.5));
        let f = e.deformation_gradient(a * x0, a * x1, a * x2, a * x3);
        assert!(mat_close(f, a, 1e-4), "F = {f:?} expected {a:?}");
    }

    #[test]
    fn inverted_tet_is_flagged() {
        // Swapping two vertices flips the orientation and the sign of the volume.
        let e = TetFemElement::from_rest(UNIT[0], UNIT[2], UNIT[1], UNIT[3], 1e-12).unwrap();
        assert!(e.is_inverted());
        assert!(e.rest_volume < 0.0);
        assert!((e.rest_volume + 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_tet_is_rejected() {
        // Four coplanar points: zero volume, non-invertible edge matrix.
        let flat = TetFemElement::from_rest(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            Vec3::new(1.0, 1.0, 0.0),
            1e-12,
        );
        assert!(flat.is_none());
    }

    #[test]
    fn build_rejects_empty_and_out_of_range() {
        let params = TetFemBasisParams::default();
        assert!(build_tet_fem_basis(&UNIT, &[], &params).is_none());
        assert!(build_tet_fem_basis(&UNIT, &[[0u32, 1, 2, 9]], &params).is_none());
    }

    #[test]
    fn basis_over_cube_mesh_is_sound() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let params = TetFemBasisParams::default();
        let basis = build_tet_fem_basis(&mesh.vertices, &mesh.tets, &params).unwrap();
        assert_eq!(basis.element_count(), mesh.tets.len());

        // tetrahedralize produces positively-oriented, non-degenerate tets, so
        // none should be inverted and the gradients should always cancel.
        assert_eq!(basis.inverted_count(), 0);
        for e in &basis.elements {
            let sum: Vec3 = e.shape_gradients.iter().copied().sum();
            assert!(sum.length() < 1e-3);
            assert!(e.rest_volume > 0.0);
        }

        // The summed element volumes agree with the independent solid-volume
        // integral computed by the mass-properties module.
        let mass =
            compute_tet_mass_properties(&mesh.vertices, &mesh.tets, &TetMassParams::default())
                .unwrap();
        assert!(
            (basis.total_rest_volume() - mass.total_volume).abs()
                < 1e-4 * mass.total_volume.max(1.0),
            "basis volume {} vs mass volume {}",
            basis.total_rest_volume(),
            mass.total_volume
        );
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let params = TetFemBasisParams::default();
        let a = build_tet_fem_basis(&mesh.vertices, &mesh.tets, &params).unwrap();
        let b = build_tet_fem_basis(&mesh.vertices, &mesh.tets, &params).unwrap();
        assert_eq!(a, b);
    }
}
