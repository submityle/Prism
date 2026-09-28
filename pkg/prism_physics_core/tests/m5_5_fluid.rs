//! Cross-module acceptance tests for the FLIP/APIC fluid solver (M5.5).
//!
//! Exercises the full `FluidSolver` pipeline: a falling water column that
//! conserves marker count and stays finite, a near-divergence-free field after
//! projection, and bit-for-bit determinism across identical seeds.

use glam::Vec3;
use prism_physics_core::fluid::{FluidConfig, FluidSolver, MacGrid, MarkerParticles, TransferMode};
use prism_physics_core::math::scalar::Real;

fn dam_break(mode: TransferMode) -> FluidSolver {
    let mut grid = MacGrid::new(16, 16, 16, 0.1, Vec3::ZERO);
    grid.set_solid_walls(1);
    let mut parts = MarkerParticles::new();
    for i in 2..8 {
        for j in 2..12 {
            for k in 2..8 {
                for s in 0..2 {
                    let jitter = s as Real * 0.05;
                    let x = Vec3::new(
                        (i as Real + 0.25 + jitter) * 0.1,
                        (j as Real + 0.5) * 0.1,
                        (k as Real + 0.5) * 0.1,
                    );
                    parts.spawn(x, Vec3::ZERO);
                }
            }
        }
    }
    let mut cfg = FluidConfig::new(5.0e-3, Vec3::new(0.0, -9.81, 0.0));
    cfg.transfer = mode;
    cfg.pressure_iterations = 80;
    FluidSolver::new(grid, parts, cfg)
}

#[test]
fn water_column_conserves_particles_and_is_finite() {
    let mut solver = dam_break(TransferMode::PicFlip);
    let n0 = solver.particle_count();
    for _ in 0..16 {
        solver.step();
    }
    assert_eq!(
        solver.particle_count(),
        n0,
        "marker count must be conserved"
    );
    for x in solver.particles.positions() {
        assert!(x.is_finite(), "non-finite position {x:?}");
    }
    for v in solver.particles.velocities() {
        assert!(v.is_finite(), "non-finite velocity {v:?}");
    }
}

#[test]
fn projection_drives_divergence_to_near_zero() {
    let mut solver = dam_break(TransferMode::PicFlip);
    solver.step();
    let d = solver.max_divergence();
    assert!(d < 5.0e-2, "post-projection divergence too large: {d}");
}

#[test]
fn apic_transfer_runs_and_stays_finite() {
    let mut solver = dam_break(TransferMode::Apic);
    for _ in 0..10 {
        solver.step();
    }
    for v in solver.particles.velocities() {
        assert!(v.is_finite(), "non-finite APIC velocity {v:?}");
    }
    assert!(solver.max_divergence() < 1.0e-1);
}

#[test]
fn identical_seeds_are_bit_for_bit_deterministic() {
    let mut a = dam_break(TransferMode::PicFlip);
    let mut b = dam_break(TransferMode::PicFlip);
    for _ in 0..8 {
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
