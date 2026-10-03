//! Global assembly of nonlinear elastic forces over a tetrahedral `FEM` mesh.
//!
//! Given a rest-pose [`TetFemBasis`], the tet connectivity, and the current
//! deformed vertex positions, these functions evaluate the total elastic
//! energy and the per-vertex internal force by scattering each element's
//! contribution (see
//! [`element_internal_force`](crate::collider::element_internal_force)) into a
//! global vector. They are pure and stateless: no solver state is retained and
//! no time integration is performed, so the assembled force vector is the
//! negative gradient of the assembled energy and feeds directly into an
//! implicit or explicit integrator supplied elsewhere.

use crate::collider::tet_fem_basis::TetFemBasis;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_internal_force::{element_energy, element_internal_force};
use glam::Vec3;

/// Gathers the four deformed node positions of tet `t` from `positions`.
///
/// Returns `None` when any index is out of range.
fn gather(positions: &[Vec3], t: [u32; 4]) -> Option<[Vec3; 4]> {
    let n = positions.len();
    if t.iter().any(|&vi| vi as usize >= n) {
        return None;
    }
    Some([
        positions[t[0] as usize],
        positions[t[1] as usize],
        positions[t[2] as usize],
        positions[t[3] as usize],
    ])
}

/// Validates that the basis, connectivity, and position arrays are mutually
/// consistent: one element per tet and every tet index addressable.
fn is_consistent(basis: &TetFemBasis, tets: &[[u32; 4]], positions: &[Vec3]) -> bool {
    if basis.element_count() != tets.len() {
        return false;
    }
    let n = positions.len();
    tets.iter().all(|t| t.iter().all(|&vi| (vi as usize) < n))
}

/// Total elastic potential energy `U = sum_e V_e * Psi(F_e)` of the mesh at
/// the supplied deformed positions.
///
/// Returns `None` when the basis element count does not match `tets` or when
/// any tet index is out of range for `positions`.
#[must_use]
pub fn total_elastic_energy(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
) -> Option<f32> {
    if !is_consistent(basis, tets, positions) {
        return None;
    }
    let mut energy = 0.0_f32;
    for (element, &t) in basis.elements.iter().zip(tets.iter()) {
        let nodes = gather(positions, t)?;
        energy += element_energy(element, nodes, model, lame);
    }
    Some(energy)
}

/// Per-vertex internal elastic force vector, one entry per position.
///
/// Each element scatters its four nodal forces into the global vector; because
/// every element's forces sum to zero, the assembled total force is zero to
/// within rounding. Returns `None` under the same consistency conditions as
/// [`total_elastic_energy`].
#[must_use]
pub fn assemble_internal_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
) -> Option<Vec<Vec3>> {
    if !is_consistent(basis, tets, positions) {
        return None;
    }
    let mut forces = vec![Vec3::ZERO; positions.len()];
    for (element, &t) in basis.elements.iter().zip(tets.iter()) {
        let nodes = gather(positions, t)?;
        let fe = element_internal_force(element, nodes, model, lame);
        for (local, &vi) in t.iter().enumerate() {
            forces[vi as usize] += fe[local];
        }
    }
    Some(forces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_constitutive::HyperelasticModel as HM;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;
    use glam::Mat3;

    const MODELS: [HM; 3] = [HM::Linear, HM::StVenantKirchhoff, HM::StableNeoHookean];

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.4).unwrap())
    }

    // Five vertices forming two tetrahedra sharing a face.
    fn two_tets() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::new(1e-12)).unwrap()
    }

    fn perturb(verts: &[Vec3], seed: u64) -> Vec<Vec3> {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.1 - 0.05
        };
        verts
            .iter()
            .map(|v| *v + Vec3::new(next(), next(), next()))
            .collect()
    }

    #[test]
    fn global_force_sums_to_zero() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        let deformed = perturb(&verts, 0x9e37_79b9_7f4a_7c15);
        for model in MODELS {
            let f = assemble_internal_forces(&basis, &tets, &deformed, model, &lame).unwrap();
            let net: Vec3 = f.iter().copied().sum();
            assert!(net.length() <= 1e-1, "{model:?} net force {net:?}");
        }
    }

    #[test]
    fn rest_pose_is_force_free() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        for model in MODELS {
            let f = assemble_internal_forces(&basis, &tets, &verts, model, &lame).unwrap();
            for (i, fi) in f.iter().enumerate() {
                assert!(fi.length() <= 1e-1, "{model:?} vertex {i} force {fi:?}");
            }
        }
    }

    #[test]
    fn rigid_rotation_is_force_free_for_objective_models() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        let r = Mat3::from_axis_angle(Vec3::new(0.1, -0.5, 0.8).normalize(), 0.7);
        let rotated: Vec<Vec3> = verts.iter().map(|v| r * *v).collect();
        for model in [HM::StVenantKirchhoff, HM::StableNeoHookean] {
            let f = assemble_internal_forces(&basis, &tets, &rotated, model, &lame).unwrap();
            for fi in &f {
                assert!(fi.length() <= 1.0, "{model:?} rotation force {fi:?}");
            }
        }
    }

    #[test]
    fn forces_match_global_energy_gradient() {
        // Independent cross-check: the assembled force is the negative
        // numerical gradient of the assembled total energy.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        let deformed = perturb(&verts, 0x1234_5678_9abc_def1);
        let eps = 1.0e-3_f32;
        for model in MODELS {
            let f = assemble_internal_forces(&basis, &tets, &deformed, model, &lame).unwrap();
            for vtx in 0..deformed.len() {
                for axis in 0..3 {
                    let mut up = deformed.clone();
                    let mut dn = deformed.clone();
                    up[vtx][axis] += eps;
                    dn[vtx][axis] -= eps;
                    let eu = total_elastic_energy(&basis, &tets, &up, model, &lame).unwrap();
                    let ed = total_elastic_energy(&basis, &tets, &dn, model, &lame).unwrap();
                    let num = -(eu - ed) / (2.0 * eps);
                    let ana = f[vtx][axis];
                    let scale = 1.0 + ana.abs();
                    assert!(
                        (num - ana).abs() <= 3.0 * scale,
                        "{model:?} grad v{vtx}[{axis}]: num={num} ana={ana}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_element_mesh_matches_element_force() {
        let verts = vec![
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(1.0, 0.2, 0.0),
            Vec3::new(0.0, 1.1, 0.1),
            Vec3::new(0.0, 0.0, 1.2),
        ];
        let tets = vec![[0u32, 1, 2, 3]];
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        let deformed = perturb(&verts, 0xdead_beef_cafe_1234);
        for model in MODELS {
            let global = assemble_internal_forces(&basis, &tets, &deformed, model, &lame).unwrap();
            let local = element_internal_force(
                &basis.elements[0],
                [deformed[0], deformed[1], deformed[2], deformed[3]],
                model,
                &lame,
            );
            for i in 0..4 {
                assert!((global[i] - local[i]).length() <= 1e-2);
            }
        }
    }

    #[test]
    fn rejects_inconsistent_input() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        // Position count too small to address every tet index.
        let short = vec![Vec3::ZERO; 3];
        assert!(assemble_internal_forces(&basis, &tets, &short, HM::Linear, &lame).is_none());
        // Connectivity length mismatched against the basis element count.
        let one_tet = vec![[0u32, 1, 2, 3]];
        assert!(assemble_internal_forces(&basis, &one_tet, &verts, HM::Linear, &lame).is_none());
        assert!(total_elastic_energy(&basis, &one_tet, &verts, HM::Linear, &lame).is_none());
    }

    #[test]
    fn assembly_is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let lame = lame();
        let deformed = perturb(&verts, 0x0f0f_0f0f_0f0f_0f0f);
        for model in MODELS {
            let a = assemble_internal_forces(&basis, &tets, &deformed, model, &lame).unwrap();
            let b = assemble_internal_forces(&basis, &tets, &deformed, model, &lame).unwrap();
            for (va, vb) in a.iter().zip(b.iter()) {
                assert_eq!(va.x.to_bits(), vb.x.to_bits());
                assert_eq!(va.y.to_bits(), vb.y.to_bits());
                assert_eq!(va.z.to_bits(), vb.z.to_bits());
            }
        }
    }
}
