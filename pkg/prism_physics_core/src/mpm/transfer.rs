//! Particle-to-grid (P2G) and grid-to-particle (G2P) transfers for MLS-MPM.
//!
//! The transfers use the APIC (affine particle-in-cell) scheme: each particle
//! carries an affine velocity field `v + C·(x − x_p)` that is scattered to and
//! gathered from the grid, which conserves both linear and angular momentum.
//! The internal-force contribution is folded into the same affine scatter as
//! the MLS-MPM stress term `−dt·V·D⁻¹·(P Fᵀ)` with `D⁻¹ = 4/dx²`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The APIC
//! transfers and the MLS-MPM affine stress scatter follow Jiang et al. 2015
//! and Hu et al. 2018.

use glam::{Mat3, Vec3};

use super::config::MpmConfig;
use super::config::MpmMaterial;
use super::constitutive::{corotated_pf, hardening_factor, snow_return_mapping};
use super::grid::Grid;
use super::particle::MaterialPoints;
use super::weights::QuadraticWeights;

/// Returns the outer product `a ⊗ b = a bᵀ` as a 3x3 matrix.
#[inline]
#[must_use]
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Scatters particle mass and (affine) momentum onto the background grid.
///
/// The grid must be cleared beforehand. After this call the grid holds
/// accumulated mass and momentum; call [`Grid::finalize_velocity`] to turn the
/// momentum into velocity.
pub fn particle_to_grid(
    mp: &MaterialPoints,
    grid: &mut Grid,
    cfg: &MpmConfig,
    material: &MpmMaterial,
) {
    let dx = grid.dx();
    let origin = grid.origin();
    let inv_dx2 = 1.0 / (dx * dx);
    let dinv = 4.0 * inv_dx2;
    let (lambda0, mu0) = material.lame();

    let positions = mp.positions();
    let velocities = mp.velocities();
    let affine = mp.affine();
    let deformation = mp.deformation();
    let masses = mp.masses();
    let volumes = mp.volumes();
    let plastic = mp.plastic_det();

    for p in 0..mp.len() {
        let x = positions[p];
        let v = velocities[p];
        let mass = masses[p];
        let f = deformation[p];

        let harden = if cfg.plastic {
            hardening_factor(cfg.plasticity.hardening, plastic[p])
        } else {
            1.0
        };
        let mu = mu0 * harden;
        let lambda = lambda0 * harden;

        let pf = corotated_pf(f, mu, lambda);
        let stress = pf * (-cfg.dt * volumes[p] * dinv);
        let affine_p = stress + affine[p] * mass;

        let w = QuadraticWeights::new(x, origin, dx);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let weight = w.value(i, j, k);
                    let dpos = w.dpos(i, j, k, dx);
                    let momentum = (v * mass) + affine_p * dpos;
                    grid.accumulate(
                        w.base[0] + i as i32,
                        w.base[1] + j as i32,
                        w.base[2] + k as i32,
                        weight * mass,
                        weight * momentum,
                    );
                }
            }
        }
    }
}

/// Gathers grid velocities back to the particles (APIC), advects positions,
/// updates the deformation gradients, and applies snow plasticity when enabled.
pub fn grid_to_particle(
    mp: &mut MaterialPoints,
    grid: &Grid,
    cfg: &MpmConfig,
    material: &MpmMaterial,
) {
    let dx = grid.dx();
    let origin = grid.origin();
    let dinv = 4.0 / (dx * dx);
    let _ = material; // material affects P2G only; kept for a symmetric API.

    let len = mp.len();
    // Snapshot immutable inputs we need while mutating columns.
    let mut new_positions = mp.positions().to_vec();
    let mut new_velocities = mp.velocities().to_vec();
    let mut new_affine = mp.affine().to_vec();
    let mut new_deformation = mp.deformation().to_vec();
    let mut new_plastic = mp.plastic_det().to_vec();

    for p in 0..len {
        let x = new_positions[p];
        let w = QuadraticWeights::new(x, origin, dx);
        let mut vel = Vec3::ZERO;
        let mut cmat = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let ni = w.base[0] + i as i32;
                    let nj = w.base[1] + j as i32;
                    let nk = w.base[2] + k as i32;
                    if !grid.in_bounds(ni, nj, nk) {
                        continue;
                    }
                    let gv = grid.velocity_at(ni as usize, nj as usize, nk as usize);
                    let weight = w.value(i, j, k);
                    let dpos = w.dpos(i, j, k, dx);
                    vel += weight * gv;
                    cmat += outer(gv, dpos) * (weight * dinv);
                }
            }
        }
        // Advect and update the deformation gradient with the affine gradient.
        new_positions[p] = x + cfg.dt * vel;
        let f_trial = (Mat3::IDENTITY + cmat * cfg.dt) * new_deformation[p];
        if cfg.plastic {
            let upd = snow_return_mapping(f_trial, new_plastic[p], &cfg.plasticity);
            new_deformation[p] = upd.deformation;
            new_plastic[p] = upd.plastic_det;
        } else {
            new_deformation[p] = f_trial;
        }
        new_velocities[p] = vel;
        new_affine[p] = cmat;
    }

    mp.positions_mut().copy_from_slice(&new_positions);
    mp.velocities_mut().copy_from_slice(&new_velocities);
    mp.affine_mut().copy_from_slice(&new_affine);
    mp.deformation_mut().copy_from_slice(&new_deformation);
    mp.plastic_det_mut().copy_from_slice(&new_plastic);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::Real;
    use crate::mpm::config::BoundaryCondition;

    fn tiny_config() -> MpmConfig {
        let mut c = MpmConfig::new(1.0e-3, Vec3::ZERO);
        c.boundary = BoundaryCondition::Slip;
        c
    }

    #[test]
    fn p2g_conserves_mass_and_momentum() {
        let mut mp = MaterialPoints::new();
        mp.spawn(
            Vec3::new(0.55, 0.55, 0.55),
            Vec3::new(1.0, -2.0, 0.5),
            1.5,
            1.0e-3,
        );
        mp.spawn(
            Vec3::new(0.62, 0.48, 0.51),
            Vec3::new(-0.5, 0.3, 0.0),
            0.8,
            1.0e-3,
        );
        let mut grid = Grid::new(16, 16, 16, 0.1, Vec3::ZERO);
        let cfg = tiny_config();
        let material = MpmMaterial::default();
        grid.clear();
        particle_to_grid(&mp, &mut grid, &cfg, &material);

        let mut grid_mass = 0.0;
        let mut grid_mom = Vec3::ZERO;
        for i in 0..grid.nx() {
            for j in 0..grid.ny() {
                for k in 0..grid.nz() {
                    let m = grid.mass_at(i, j, k);
                    grid_mass += m;
                    grid_mom += grid.velocity_at(i, j, k); // still momentum before finalize
                }
            }
        }
        assert!((grid_mass - mp.total_mass()).abs() < 1.0e-4);
        assert!((grid_mom - mp.total_momentum()).length() < 1.0e-4);
    }

    #[test]
    fn roundtrip_recovers_uniform_velocity() {
        // With a uniform velocity field and F = I, a P2G/G2P roundtrip should
        // return the same velocity (rigid translation).
        let mut mp = MaterialPoints::new();
        let vel = Vec3::new(0.3, -0.2, 0.1);
        for a in 0..3 {
            for b in 0..3 {
                let x = Vec3::new(0.5 + a as Real * 0.03, 0.5 + b as Real * 0.03, 0.5);
                mp.spawn(x, vel, 1.0, 1.0e-3);
            }
        }
        let mut grid = Grid::new(20, 20, 20, 0.1, Vec3::ZERO);
        let cfg = tiny_config();
        let material = MpmMaterial::default();
        grid.clear();
        particle_to_grid(&mp, &mut grid, &cfg, &material);
        grid.finalize_velocity();
        grid_to_particle(&mut mp, &grid, &cfg, &material);
        for v in mp.velocities() {
            assert!((*v - vel).length() < 1.0e-4);
        }
    }
}
