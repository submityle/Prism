//! Conservation diagnostics for tetrahedral FEM states.
//!
//! These are read-only observers used to validate an integrator over a
//! trajectory and to drive AAA-grade regression checks: kinetic energy, linear
//! and angular momentum, and total hyperelastic strain energy. They never
//! mutate simulation state and have no coupling to the stepping pipeline, so
//! they can be sampled before/after any step to measure drift.
//!
//! All accumulation is performed in `f64`. The crate's `glam` build is
//! single-precision only (no `DVec3`), so vector sums are accumulated
//! component-wise as `[f64; 3]` and narrowed back to [`glam::Vec3`] on return.
//!
//! # Attribution
//!
//! Clean-room implementation of standard rigid-body momentum/energy integrals
//! and reuse of the crate's own constitutive models. No Unreal Engine source
//! or derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

/// `a - b` for `[f64; 3]` triples.
#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `a × b` for `[f64; 3]` triples.
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Promotes a [`Vec3`] to an `[f64; 3]` triple.
#[inline]
fn to_f64(v: Vec3) -> [f64; 3] {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

/// Narrows an `[f64; 3]` triple back to a [`Vec3`].
#[inline]
fn to_vec3(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)
}

/// Total kinetic energy `½ Σ mᵢ |vᵢ|²` in joules.
///
/// Returns `None` when `velocities` does not have one entry per mass vertex.
#[must_use]
pub fn kinetic_energy(mass: &LumpedMass, velocities: &[Vec3]) -> Option<f64> {
    if mass.n_dofs() != 3 * velocities.len() {
        return None;
    }
    let mut e = 0.0_f64;
    for (i, v) in velocities.iter().enumerate() {
        let m = f64::from(mass.get(3 * i));
        e += m * f64::from(v.length_squared());
    }
    Some(0.5 * e)
}

/// Total linear momentum `Σ mᵢ vᵢ` in kg·m/s.
///
/// Returns `None` when `velocities` does not have one entry per mass vertex.
#[must_use]
pub fn linear_momentum(mass: &LumpedMass, velocities: &[Vec3]) -> Option<Vec3> {
    if mass.n_dofs() != 3 * velocities.len() {
        return None;
    }
    let mut p = [0.0_f64; 3];
    for (i, v) in velocities.iter().enumerate() {
        let m = f64::from(mass.get(3 * i));
        p[0] += m * f64::from(v.x);
        p[1] += m * f64::from(v.y);
        p[2] += m * f64::from(v.z);
    }
    Some(to_vec3(p))
}

/// Total angular momentum `Σ mᵢ (xᵢ − c) × vᵢ` about the point `about`, in
/// kg·m²/s.
///
/// Returns `None` on a dimension mismatch between `positions`, `velocities`
/// and the mass.
#[must_use]
pub fn angular_momentum(
    mass: &LumpedMass,
    positions: &[Vec3],
    velocities: &[Vec3],
    about: Vec3,
) -> Option<Vec3> {
    let n = positions.len();
    if velocities.len() != n || mass.n_dofs() != 3 * n {
        return None;
    }
    let c = to_f64(about);
    let mut l = [0.0_f64; 3];
    for i in 0..n {
        let m = f64::from(mass.get(3 * i));
        let r = sub3(to_f64(positions[i]), c);
        let rxv = cross3(r, to_f64(velocities[i]));
        l[0] += m * rxv[0];
        l[1] += m * rxv[1];
        l[2] += m * rxv[2];
    }
    Some(to_vec3(l))
}

/// Total hyperelastic strain energy `Σₑ Ψ(Fₑ)·Vₑ` in joules.
///
/// `tets` maps each basis element to its four vertex indices into `positions`;
/// `model`/`lame` select the constitutive law (`lame` can be built with
/// [`LameParameters::from_isotropic`]). The per-element deformation gradient is
/// evaluated from the current `positions` through the element's rest basis and
/// weighted by the absolute rest volume.
///
/// Returns `None` when `tets.len()` differs from the element count or any tet
/// index is out of range for `positions`.
#[must_use]
pub fn elastic_energy(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
) -> Option<f64> {
    if tets.len() != basis.elements.len() {
        return None;
    }
    let n = positions.len();
    let mut energy = 0.0_f64;
    for (element, tet) in basis.elements.iter().zip(tets.iter()) {
        let [i0, i1, i2, i3] = *tet;
        let (i0, i1, i2, i3) = (i0 as usize, i1 as usize, i2 as usize, i3 as usize);
        if i0 >= n || i1 >= n || i2 >= n || i3 >= n {
            return None;
        }
        let f = element.deformation_gradient(
            positions[i0],
            positions[i1],
            positions[i2],
            positions[i3],
        );
        let psi = model.strain_energy_density(f, lame);
        energy += f64::from(psi) * f64::from(element.rest_volume.abs());
    }
    Some(energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;
    use crate::collider::tet_lumped_mass::build_lumped_mass_from_mesh;
    use crate::collider::tet_mass::TetMassParams;

    fn mesh() -> (Vec<Vec3>, Vec<[u32; 4]>) {
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

    fn mass_of(verts: &[Vec3], tets: &[[u32; 4]]) -> LumpedMass {
        build_lumped_mass_from_mesh(verts, tets, &TetMassParams { density: 1200.0 }).unwrap()
    }

    #[test]
    fn kinetic_energy_of_uniform_velocity_is_half_m_v_squared() {
        let (verts, tets) = mesh();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = Vec3::new(2.0, -1.0, 0.5);
        let vel = vec![v; n];
        let e = kinetic_energy(&mass, &vel).unwrap();
        let expected = 0.5 * f64::from(mass.total_mass()) * f64::from(v.length_squared());
        assert!((e - expected).abs() < 1e-3 * expected.max(1.0));
    }

    #[test]
    fn linear_momentum_of_uniform_velocity_is_m_v() {
        let (verts, tets) = mesh();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = Vec3::new(2.0, -1.0, 0.5);
        let vel = vec![v; n];
        let p = linear_momentum(&mass, &vel).unwrap();
        let expected = mass.total_mass() * v;
        assert!((p - expected).length() < 1e-3 * expected.length().max(1.0));
    }

    #[test]
    fn angular_momentum_of_translation_matches_com_identity() {
        // For a uniform velocity field L(about) = M (x_com − about) × v.
        let (verts, tets) = mesh();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = Vec3::new(0.3, 0.9, -0.2);
        let vel = vec![v; n];
        let about = Vec3::new(0.1, -0.2, 0.4);

        let mut com = [0.0_f64; 3];
        let mut mtot = 0.0_f64;
        for i in 0..n {
            let m = f64::from(mass.get(3 * i));
            mtot += m;
            com[0] += m * f64::from(verts[i].x);
            com[1] += m * f64::from(verts[i].y);
            com[2] += m * f64::from(verts[i].z);
        }
        let com = Vec3::new(
            (com[0] / mtot) as f32,
            (com[1] / mtot) as f32,
            (com[2] / mtot) as f32,
        );
        let expected = mass.total_mass() * (com - about).cross(v);

        let l = angular_momentum(&mass, &verts, &vel, about).unwrap();
        assert!(
            (l - expected).length() < 1e-3 * expected.length().max(1.0),
            "l = {l:?}, expected {expected:?}"
        );
    }

    #[test]
    fn elastic_energy_is_zero_at_rest_for_objective_models() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(2.0e4, 0.3).unwrap();
        let lame = LameParameters::from_isotropic(&mat);
        for model in [
            HyperelasticModel::Linear,
            HyperelasticModel::StVenantKirchhoff,
        ] {
            let e = elastic_energy(&basis, &tets, &verts, model, &lame).unwrap();
            assert!(e.abs() < 1e-4, "{model:?} rest energy = {e}");
        }
    }

    #[test]
    fn elastic_energy_is_rotation_invariant_for_stvk() {
        use glam::{Mat3, Quat};
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(2.0e4, 0.3).unwrap();
        let lame = LameParameters::from_isotropic(&mat);

        let r = Mat3::from_quat(Quat::from_axis_angle(
            Vec3::new(0.3, 0.7, 0.1).normalize(),
            1.2,
        ));
        let rotated: Vec<Vec3> = verts.iter().map(|&p| r * p).collect();
        let e = elastic_energy(
            &basis,
            &tets,
            &rotated,
            HyperelasticModel::StVenantKirchhoff,
            &lame,
        )
        .unwrap();
        assert!(e.abs() < 1e-3, "StVK energy under rigid rotation = {e}");
    }

    #[test]
    fn elastic_energy_is_positive_under_stretch() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(2.0e4, 0.3).unwrap();
        let lame = LameParameters::from_isotropic(&mat);
        let stretched: Vec<Vec3> = verts
            .iter()
            .map(|&p| Vec3::new(p.x * 1.3, p.y, p.z))
            .collect();
        let e = elastic_energy(
            &basis,
            &tets,
            &stretched,
            HyperelasticModel::StVenantKirchhoff,
            &lame,
        )
        .unwrap();
        assert!(e > 1e-3, "stretched energy = {e}");
    }

    #[test]
    fn dimension_mismatch_returns_none() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mass = mass_of(&verts, &tets);
        let mat = IsotropicElasticity::new(2.0e4, 0.3).unwrap();
        let lame = LameParameters::from_isotropic(&mat);

        assert!(kinetic_energy(&mass, &verts[..verts.len() - 1]).is_none());
        assert!(linear_momentum(&mass, &verts[..verts.len() - 1]).is_none());
        assert!(angular_momentum(&mass, &verts, &verts[..verts.len() - 1], Vec3::ZERO).is_none());
        assert!(
            elastic_energy(&basis, &tets[..1], &verts, HyperelasticModel::Linear, &lame).is_none()
        );
    }
}
