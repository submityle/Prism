//! Global assembly of corotational (warped) stiffness and internal forces.
//!
//! This module lifts the per-element corotational operators from
//! [`tet_fem_corotational`](crate::collider::tet_fem_corotational) to a whole
//! tetrahedral mesh. At the current deformed state it scatters each element's
//! warped tangent `R Ke Rᵀ` into a global [`GlobalStiffness`] (`CSR`) and each
//! element's warped restoring force into a per-vertex force vector.
//!
//! The sparsity pattern is identical to the linear assembly, so the pattern is
//! obtained once from [`assemble_global_stiffness`] and the warped values are
//! re-scattered into it. This keeps the two assemblers structurally consistent
//! and lets the warped stiffness feed the same implicit solvers
//! (`solve_implicit_system` / `solve_implicit_system_jacobi`).
//!
//! Like the rest of the `FEM` layer this module is stateless and performs no
//! time integration. Nothing here is derived from Unreal Engine source.

use super::tet_fem_assembly::{assemble_global_stiffness, GlobalStiffness};
use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_corotational::{corotational_internal_force, corotational_stiffness};
use super::tet_fem_stiffness::{element_stiffness, IsotropicElasticity};
use glam::Vec3;

/// Gathers the four nodal positions of a tetrahedron from a global array.
fn gather(positions: &[Vec3], tet: [u32; 4]) -> [Vec3; 4] {
    [
        positions[tet[0] as usize],
        positions[tet[1] as usize],
        positions[tet[2] as usize],
        positions[tet[3] as usize],
    ]
}

/// Assembles the global corotational tangent stiffness at the deformed state.
///
/// `rest` is the reference configuration used to build `basis`; `current` holds
/// the deformed vertex positions. The returned [`GlobalStiffness`] uses the
/// same `CSR` pattern as [`assemble_global_stiffness`] and reduces to it when
/// `current == rest`. Returns [`None`] when the inputs are inconsistent
/// (mismatched element/tet counts, out-of-range indices, or position arrays
/// whose length differs from `n_vertices`).
#[must_use]
pub fn assemble_corotational_stiffness(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    current: &[Vec3],
    n_vertices: usize,
) -> Option<GlobalStiffness> {
    if current.len() != n_vertices {
        return None;
    }
    let pattern = assemble_global_stiffness(basis, tets, material, n_vertices)?;
    let mut accum = vec![0.0f64; pattern.col_indices.len()];
    for (element, &tet) in basis.elements.iter().zip(tets.iter()) {
        if tet.iter().any(|&v| v as usize >= n_vertices) {
            return None;
        }
        let ke = element_stiffness(element, material);
        let warped = corotational_stiffness(element, &ke, gather(current, tet));
        for a in 0..12 {
            let row = 3 * tet[a / 3] as usize + a % 3;
            let start = pattern.row_offsets[row];
            let end = pattern.row_offsets[row + 1];
            let cols = &pattern.col_indices[start..end];
            for b in 0..12 {
                let col = 3 * tet[b / 3] as usize + b % 3;
                let offset = cols
                    .binary_search(&col)
                    .expect("scattered DOF must lie inside the assembled pattern");
                accum[start + offset] += f64::from(warped.get(a, b));
            }
        }
    }
    let values: Vec<f32> = accum.iter().map(|&x| x as f32).collect();
    Some(GlobalStiffness {
        n_dofs: pattern.n_dofs,
        row_offsets: pattern.row_offsets,
        col_indices: pattern.col_indices,
        values,
    })
}

/// Assembles the global corotational restoring force at the deformed state.
///
/// Returns one force vector per vertex, `f = Σ_e f_e`, where each element
/// contributes `-R Ke (Rᵀ x - x0)` scattered to its four nodes. The total force
/// vanishes under rigid translation and rigid rotation of the whole mesh.
/// Returns [`None`] on the same inconsistencies as
/// [`assemble_corotational_stiffness`].
#[must_use]
pub fn assemble_corotational_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    rest: &[Vec3],
    current: &[Vec3],
    n_vertices: usize,
) -> Option<Vec<Vec3>> {
    if tets.is_empty() || basis.elements.len() != tets.len() {
        return None;
    }
    if rest.len() != n_vertices || current.len() != n_vertices {
        return None;
    }
    let mut forces = vec![Vec3::ZERO; n_vertices];
    for (element, &tet) in basis.elements.iter().zip(tets.iter()) {
        if tet.iter().any(|&v| v as usize >= n_vertices) {
            return None;
        }
        let ke = element_stiffness(element, material);
        let fe = corotational_internal_force(element, &ke, gather(rest, tet), gather(current, tet));
        for (local, &vi) in tet.iter().enumerate() {
            forces[vi as usize] += fe[local];
        }
    }
    Some(forces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasis, TetFemBasisParams};
    use glam::Mat3;

    fn two_tets() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::default()).unwrap()
    }

    fn material() -> IsotropicElasticity {
        IsotropicElasticity::new(3.0e3, 0.3).unwrap()
    }

    #[test]
    fn rest_state_matches_linear_assembly() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let linear = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let corot =
            assemble_corotational_stiffness(&basis, &tets, &mat, &verts, verts.len()).unwrap();
        assert_eq!(linear.nnz(), corot.nnz());
        for i in 0..linear.n_dofs() {
            for j in 0..linear.n_dofs() {
                let a = linear.get(i, j);
                let b = corot.get(i, j);
                assert!(
                    (a - b).abs() <= 1e-2 * (1.0 + a.abs()),
                    "{i},{j}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn rigid_translation_is_force_free() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let t = Vec3::new(-4.0, 2.5, 1.0);
        let current: Vec<Vec3> = verts.iter().map(|&p| p + t).collect();
        let f =
            assemble_corotational_forces(&basis, &tets, &material(), &verts, &current, verts.len())
                .unwrap();
        let max = f.iter().map(|v| v.length()).fold(0.0, f32::max);
        assert!(max <= 1e-2, "max force {max}");
    }

    #[test]
    fn rigid_rotation_is_force_free() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let q = Mat3::from_axis_angle(Vec3::new(0.2, -0.6, 0.4).normalize(), 1.1);
        let current: Vec<Vec3> = verts.iter().map(|&p| q * p).collect();
        let f =
            assemble_corotational_forces(&basis, &tets, &material(), &verts, &current, verts.len())
                .unwrap();
        let max = f.iter().map(|v| v.length()).fold(0.0, f32::max);
        assert!(max <= 1e-2, "max force {max}");
    }

    #[test]
    fn stretch_produces_restoring_force() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let current: Vec<Vec3> = verts.iter().map(|&p| p * 1.2).collect();
        let f =
            assemble_corotational_forces(&basis, &tets, &material(), &verts, &current, verts.len())
                .unwrap();
        let max = f.iter().map(|v| v.length()).fold(0.0, f32::max);
        assert!(
            max > 1.0,
            "expected a non-trivial restoring force, got {max}"
        );
    }

    #[test]
    fn global_stiffness_is_symmetric() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let q = Mat3::from_axis_angle(Vec3::Y, 0.7);
        let current: Vec<Vec3> = verts.iter().map(|&p| q * (p * 1.05)).collect();
        let k = assemble_corotational_stiffness(&basis, &tets, &material(), &current, verts.len())
            .unwrap();
        for i in 0..k.n_dofs() {
            for j in 0..k.n_dofs() {
                let d = (k.get(i, j) - k.get(j, i)).abs();
                assert!(d <= 1e-2 * (1.0 + k.get(i, j).abs()), "asym {i},{j}: {d}");
            }
        }
    }

    #[test]
    fn rejects_inconsistent_inputs() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        // wrong position-array length
        assert!(
            assemble_corotational_stiffness(&basis, &tets, &mat, &verts, verts.len() + 1).is_none()
        );
        assert!(
            assemble_corotational_forces(&basis, &tets, &mat, &verts, &verts, verts.len() + 1)
                .is_none()
        );
        // mismatched element/tet count
        let short = vec![tets[0]];
        assert!(
            assemble_corotational_forces(&basis, &short, &mat, &verts, &verts, verts.len())
                .is_none()
        );
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let q = Mat3::from_axis_angle(Vec3::Z, 0.5);
        let current: Vec<Vec3> = verts.iter().map(|&p| q * p).collect();
        let a = assemble_corotational_stiffness(&basis, &tets, &material(), &current, verts.len())
            .unwrap();
        let b = assemble_corotational_stiffness(&basis, &tets, &material(), &current, verts.len())
            .unwrap();
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(b.values.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}
