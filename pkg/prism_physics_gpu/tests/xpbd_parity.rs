//! Real-device parity: the `GPU` `XPBD` distance solver must reproduce the
//! `CPU` golden twin's trajectory within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! The scenes are a pinned hanging chain (a stress test for colour ordering,
//! since consecutive edges share a particle and must land in different colours)
//! and a pinned cloth grid (a wider constraint graph). Because the device runs
//! `GPU` floating-point with fused multiply-add and differing division and
//! square-root rounding, positions are compared within a tolerance rather than
//! for exact equality.
//!
//! Provenance: substep `XPBD` (Müller et al.). No Unreal Engine source or
//! derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_solve, DistanceConstraint, GpuContext, GpuXpbdSolver, ParticleState, XpbdConfig,
};

/// Maximum allowed per-particle position divergence between the two engines.
const TOLERANCE: f32 = 1e-3;

/// A pinned chain of `n` particles hanging along `-Y`, unit spacing.
fn hanging_chain(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let y = -(i as f32);
        // Particle 0 is pinned (inverse mass 0); the rest are free.
        let inv_mass = if i == 0 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.0, y, 0.0), inv_mass);
    }
    let constraints = (0..n - 1)
        .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 0.0))
        .collect();
    (state, constraints)
}

/// A `w` by `h` cloth grid pinned along its top row, with structural
/// (horizontal and vertical) distance constraints.
fn cloth_grid(w: u32, h: u32) -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    let index = |x: u32, y: u32| y * w + x;
    for y in 0..h {
        for x in 0..w {
            let inv_mass = if y == 0 { 0.0 } else { 1.0 };
            state.push(Vec3::new(x as f32, -(y as f32), 0.0), inv_mass);
        }
    }
    let mut constraints = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                constraints.push(DistanceConstraint::new(
                    index(x, y),
                    index(x + 1, y),
                    1.0,
                    0.0,
                ));
            }
            if y + 1 < h {
                constraints.push(DistanceConstraint::new(
                    index(x, y),
                    index(x, y + 1),
                    1.0,
                    0.0,
                ));
            }
        }
    }
    (state, constraints)
}

/// Asserts every particle in `gpu` is within [`TOLERANCE`] of `cpu`.
fn assert_parity(cpu: &ParticleState, gpu: &ParticleState, scene: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{scene}: particle counts differ");
    for i in 0..cpu.len() {
        let delta = (cpu.positions[i] - gpu.positions[i]).length();
        assert!(
            delta <= TOLERANCE,
            "{scene}: particle {i} diverged by {delta}: cpu {:?} vs gpu {:?}",
            cpu.positions[i],
            gpu.positions[i]
        );
    }
}

/// Runs `frames` of both engines from the same initial state and checks parity.
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuXpbdSolver,
    initial: &ParticleState,
    constraints: &[DistanceConstraint],
    config: &XpbdConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve(&mut cpu, constraints, config, dt).expect("cpu solve");
        solver
            .solve(ctx, &mut gpu, constraints, config, dt)
            .expect("gpu solve");
    }
    assert_parity(&cpu, &gpu, scene);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_xpbd_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU XPBD parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuXpbdSolver::new(&ctx);
    let config = XpbdConfig::default();

    let (chain, chain_cons) = hanging_chain(16);
    run_parity(
        &ctx,
        &solver,
        &chain,
        &chain_cons,
        &config,
        30,
        "hanging_chain",
    );

    let (cloth, cloth_cons) = cloth_grid(8, 8);
    run_parity(
        &ctx,
        &solver,
        &cloth,
        &cloth_cons,
        &config,
        20,
        "cloth_grid",
    );
}
