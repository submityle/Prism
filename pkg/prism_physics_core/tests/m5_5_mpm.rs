//! Cross-module acceptance tests for the MLS-MPM solver (milestone M5.5).
//!
//! These exercise the full public MPM pipeline through `MpmSolver`: a free-
//! falling elastic blob (mass/momentum bookkeeping), snow-plasticity
//! stability, bit-for-bit determinism, and finiteness.

use glam::Vec3;
use prism_physics_core::math::scalar::Real;
use prism_physics_core::mpm::{
    BoundaryCondition, Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver,
};

fn blob() -> MaterialPoints {
    let mut mp = MaterialPoints::new();
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let x = Vec3::new(
                    1.0 + a as Real * 0.05,
                    1.4 + b as Real * 0.05,
                    1.0 + c as Real * 0.05,
                );
                mp.spawn(x, Vec3::ZERO, 1.0, 1.0e-3);
            }
        }
    }
    mp
}

fn solver_with(cfg: MpmConfig, material: MpmMaterial) -> MpmSolver {
    let grid = Grid::new(24, 24, 24, 0.1, Vec3::ZERO);
    MpmSolver::new(blob(), grid, material, cfg)
}

#[test]
fn falling_blob_conserves_mass_and_gains_gravity_momentum() {
    let mut cfg = MpmConfig::new(5.0e-4, Vec3::new(0.0, -9.81, 0.0));
    cfg.boundary = BoundaryCondition::Slip;
    let material = MpmMaterial::default();
    let mut solver = solver_with(cfg, material);
    let mass0 = solver.particles.total_mass();

    let steps = 24;
    solver.advance(steps);

    // Particle masses never change: total mass is exactly conserved.
    assert!((solver.particles.total_mass() - mass0).abs() < 1.0e-6);

    // Before touching the walls the vertical momentum tracks the gravity
    // impulse m·g·(dt·steps).
    let expected = mass0 * cfg.gravity.y * cfg.dt * steps as Real;
    let py = solver.particles.total_momentum().y;
    assert!(
        (py - expected).abs() < 1.0e-2 * expected.abs().max(1.0),
        "py={py} expected={expected}"
    );

    for v in solver.particles.velocities() {
        assert!(v.is_finite(), "non-finite velocity {v:?}");
    }
    for p in solver.particles.positions() {
        assert!(p.is_finite(), "non-finite position {p:?}");
    }
}

#[test]
fn snow_pile_stays_finite_and_bounded() {
    let material = MpmMaterial::new(1.4e5, 0.2, 4.0e2);
    let mut cfg = MpmConfig::new(2.0e-4, Vec3::new(0.0, -9.81, 0.0));
    cfg.plastic = true;
    cfg.boundary = BoundaryCondition::Sticky;
    let mut solver = solver_with(cfg, material);

    solver.advance(60);

    for p in solver.particles.positions() {
        assert!(p.is_finite(), "non-finite position {p:?}");
        // Particles stay inside the domain (clamped by the boundary layer).
        assert!(p.x > 0.0 && p.x < 2.4);
        assert!(p.y > 0.0 && p.y < 2.4);
        assert!(p.z > 0.0 && p.z < 2.4);
    }
    for jp in solver.particles.plastic_det() {
        assert!(jp.is_finite() && *jp > 0.0, "bad plastic det {jp}");
    }
}

#[test]
fn identical_seeds_are_bit_for_bit_deterministic() {
    let cfg = MpmConfig::default();
    let material = MpmMaterial::default();
    let mut a = solver_with(cfg, material);
    let mut b = solver_with(cfg, material);

    for _ in 0..12 {
        a.step();
        b.step();
    }

    for (pa, pb) in a.particles.positions().iter().zip(b.particles.positions()) {
        assert_eq!(pa.to_array(), pb.to_array());
    }
    for (va, vb) in a
        .particles
        .velocities()
        .iter()
        .zip(b.particles.velocities())
    {
        assert_eq!(va.to_array(), vb.to_array());
    }
}
