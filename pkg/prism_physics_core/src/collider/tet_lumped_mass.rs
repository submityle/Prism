//! Lumped (diagonal) mass matrix of a tetrahedral finite-element mesh.
//!
//! The companion of the global stiffness matrix `K` in an implicit `FEM`
//! integrator is the mass matrix `M`: the backward-Euler / Newmark update
//! solves a system of the form `(M + h^2 K) dx = b`, and explicit integrators
//! need `M^{-1}` to turn nodal forces into accelerations. The *consistent* mass
//! matrix is sparse and couples neighbouring nodes, but the overwhelmingly
//! common choice in real-time solvers is the *lumped* (row-summed) mass matrix,
//! which is purely diagonal: every one of a vertex's three translational
//! degrees of freedom (`DOF`s) carries that vertex's lumped mass.
//!
//! The per-vertex lumped masses are exactly the `nodal_masses` already computed
//! by [`compute_tet_mass_properties`](super::tet_mass::compute_tet_mass_properties)
//! (each tetrahedron distributes a quarter of its mass to each of its four
//! vertices). This module only replicates them across the three Cartesian
//! components and exposes diagonal matrix-vector products; it holds no
//! simulation state and performs no time integration, so it is fully decoupled
//! from any solver. It is standard mass lumping; nothing here is derived from
//! Unreal Engine source.

use super::tet_mass::{compute_tet_mass_properties, TetMassParams};
use glam::Vec3;

/// The lumped (diagonal) mass matrix of a mesh over `3 N` scalar degrees of
/// freedom.
///
/// Degree of freedom `3 v + c` is the `c`-th Cartesian component of vertex `v`
/// and carries that vertex's lumped mass, so entries come in identical triples.
#[derive(Clone, Debug, PartialEq)]
pub struct LumpedMass {
    /// The diagonal entries, length `3 N`; `diagonal[3 v + c]` is the lumped
    /// mass of vertex `v` (independent of the component `c`).
    pub diagonal: Vec<f32>,
}

impl LumpedMass {
    /// Number of scalar degrees of freedom (`3 N`).
    #[must_use]
    pub fn n_dofs(&self) -> usize {
        self.diagonal.len()
    }

    /// The diagonal entry (lumped mass) of degree of freedom `i`.
    #[must_use]
    pub fn get(&self, i: usize) -> f32 {
        self.diagonal[i]
    }

    /// Total physical mass of the mesh (the sum of the per-vertex lumped
    /// masses, i.e. the diagonal sum divided by three).
    #[must_use]
    pub fn total_mass(&self) -> f32 {
        let sum: f64 = self.diagonal.iter().map(|&m| f64::from(m)).sum();
        (sum / 3.0) as f32
    }

    /// Whether every diagonal entry is strictly positive, which is required for
    /// the matrix to be invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        self.diagonal.iter().all(|&m| m > 0.0)
    }

    /// The product `M x`.
    ///
    /// Returns `None` when `x` does not have exactly [`LumpedMass::n_dofs`]
    /// entries.
    #[must_use]
    pub fn apply(&self, x: &[f32]) -> Option<Vec<f32>> {
        if x.len() != self.diagonal.len() {
            return None;
        }
        let out = self
            .diagonal
            .iter()
            .zip(x.iter())
            .map(|(&m, &xi)| (f64::from(m) * f64::from(xi)) as f32)
            .collect();
        Some(out)
    }

    /// The product `M^{-1} x`.
    ///
    /// Returns `None` when `x` has the wrong length or when any diagonal entry
    /// is not strictly positive (the matrix is then singular).
    #[must_use]
    pub fn apply_inverse(&self, x: &[f32]) -> Option<Vec<f32>> {
        if x.len() != self.diagonal.len() || !self.is_invertible() {
            return None;
        }
        let out = self
            .diagonal
            .iter()
            .zip(x.iter())
            .map(|(&m, &xi)| (f64::from(xi) / f64::from(m)) as f32)
            .collect();
        Some(out)
    }
}

/// Builds a lumped mass matrix directly from per-vertex lumped masses.
///
/// `nodal_masses` is typically
/// [`TetMassProperties::nodal_masses`](super::tet_mass::TetMassProperties).
/// Returns `None` when the slice is empty or contains a negative or non-finite
/// mass. Zero masses are permitted (an unreferenced vertex has zero lumped
/// mass), but they make the matrix singular, which
/// [`LumpedMass::apply_inverse`] reports.
#[must_use]
pub fn build_lumped_mass(nodal_masses: &[f32]) -> Option<LumpedMass> {
    if nodal_masses.is_empty() {
        return None;
    }
    for &m in nodal_masses {
        // Rejects negatives and NaN.
        if m.is_nan() || m < 0.0 {
            return None;
        }
    }
    let diagonal = nodal_masses.iter().flat_map(|&m| [m, m, m]).collect();
    Some(LumpedMass { diagonal })
}

/// Builds a lumped mass matrix from a tetrahedral mesh and a uniform density.
///
/// This is a convenience wrapper that computes the per-vertex lumped masses via
/// [`compute_tet_mass_properties`] and then calls [`build_lumped_mass`].
/// Returns `None` whenever the underlying mass computation fails (empty mesh,
/// out-of-range index, or non-positive density).
#[must_use]
pub fn build_lumped_mass_from_mesh(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    params: &TetMassParams,
) -> Option<LumpedMass> {
    let props = compute_tet_mass_properties(vertices, tets, params)?;
    build_lumped_mass(&props.nodal_masses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};

    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let tris = vec![
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
        (verts, tris)
    }

    #[test]
    fn diagonal_replicates_each_vertex_mass_three_times() {
        let masses = [2.0f32, 0.5, 1.5];
        let m = build_lumped_mass(&masses).unwrap();
        assert_eq!(m.n_dofs(), 9);
        for (v, &mv) in masses.iter().enumerate() {
            for c in 0..3 {
                assert!((m.get(3 * v + c) - mv).abs() <= 1e-7);
            }
        }
    }

    #[test]
    fn total_mass_sums_the_nodal_masses() {
        let masses = [2.0f32, 0.5, 1.5, 1.0];
        let m = build_lumped_mass(&masses).unwrap();
        let want: f32 = masses.iter().sum();
        assert!((m.total_mass() - want).abs() <= 1e-6, "{}", m.total_mass());
    }

    #[test]
    fn apply_then_inverse_round_trips() {
        let masses = [2.0f32, 0.5, 1.5];
        let m = build_lumped_mass(&masses).unwrap();
        assert!(m.is_invertible());
        let x = [0.3f32, -0.4, 0.7, 1.0, -1.0, 0.2, 0.5, -0.6, 0.9];
        let mx = m.apply(&x).unwrap();
        let back = m.apply_inverse(&mx).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() <= 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn kinetic_energy_of_a_uniform_velocity_matches_rigid_body() {
        // 0.5 v^T M v for a uniform translation v equals 0.5 * total_mass |v|^2.
        let masses = [2.0f32, 0.5, 1.5, 1.0];
        let m = build_lumped_mass(&masses).unwrap();
        let vel = Vec3::new(0.3, -0.7, 1.1);
        let mut v = vec![0.0f32; m.n_dofs()];
        for i in 0..masses.len() {
            v[3 * i] = vel.x;
            v[3 * i + 1] = vel.y;
            v[3 * i + 2] = vel.z;
        }
        let mv = m.apply(&v).unwrap();
        let mut ke = 0.0f64;
        for (&vi, &mvi) in v.iter().zip(mv.iter()) {
            ke += f64::from(vi) * f64::from(mvi);
        }
        ke *= 0.5;
        let want = 0.5 * f64::from(m.total_mass()) * f64::from(vel.length_squared());
        assert!((ke - want).abs() <= 1e-3 * want.max(1.0), "{ke} vs {want}");
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        // A zero-mass (unreferenced) vertex makes the matrix singular.
        let masses = [1.0f32, 0.0, 2.0];
        let m = build_lumped_mass(&masses).unwrap();
        assert!(!m.is_invertible());
        assert!(m.apply_inverse(&[1.0; 9]).is_none());
        // Forward product is still well defined.
        assert!(m.apply(&[1.0; 9]).is_some());
    }

    #[test]
    fn invalid_masses_are_rejected() {
        assert!(build_lumped_mass(&[]).is_none());
        assert!(build_lumped_mass(&[1.0, -0.5]).is_none());
        assert!(build_lumped_mass(&[1.0, f32::NAN]).is_none());
        assert!(build_lumped_mass(&[0.0, 1.0]).is_some());
    }

    #[test]
    fn mesh_total_mass_matches_density_times_volume() {
        let (verts, tris) = unit_cube();
        let mesh = tetrahedralize(&verts, &tris, &TetMeshParams::new(10)).unwrap();
        let params = TetMassParams { density: 2.0 };
        let m = build_lumped_mass_from_mesh(&mesh.vertices, &mesh.tets, &params).unwrap();
        let props = compute_tet_mass_properties(&mesh.vertices, &mesh.tets, &params).unwrap();
        assert!(
            (m.total_mass() - props.total_mass).abs() <= 1e-3 * props.total_mass,
            "{} vs {}",
            m.total_mass(),
            props.total_mass
        );
    }

    #[test]
    fn wrong_length_vectors_are_rejected() {
        let m = build_lumped_mass(&[1.0, 2.0]).unwrap();
        assert!(m.apply(&[1.0; 5]).is_none());
        assert!(m.apply_inverse(&[1.0; 5]).is_none());
    }

    #[test]
    fn build_is_deterministic() {
        let masses = [2.0f32, 0.5, 1.5, 1.0, 3.0];
        let a = build_lumped_mass(&masses).unwrap();
        let b = build_lumped_mass(&masses).unwrap();
        assert_eq!(a, b);
    }
}
