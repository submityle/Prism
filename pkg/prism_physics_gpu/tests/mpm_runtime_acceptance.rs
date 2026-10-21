//! Production-facing acceptance coverage for the fixed-step MPM runtime.

use glam::Vec3;
use prism_physics_core::mpm::{Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver};
use prism_physics_gpu::{MpmBackend, MpmBackendPolicy, MpmRuntime, MpmRuntimeConfig};

fn build_solver() -> MpmSolver {
    let mut particles = MaterialPoints::new();
    for index in 0..4 {
        particles.spawn(
            Vec3::new(0.9 + index as f32 * 0.025, 0.9, 0.9),
            Vec3::ZERO,
            1.0,
            1.0e-3,
        );
    }
    MpmSolver::new(
        particles,
        Grid::new(24, 24, 24, 0.1, Vec3::ZERO),
        MpmMaterial::default(),
        MpmConfig::new(5.0e-4, Vec3::new(0.0, -9.81, 0.0)),
    )
}

#[test]
fn fixed_step_runtime_advances_and_switches_without_losing_state() {
    let mut runtime = MpmRuntime::new(
        build_solver(),
        MpmRuntimeConfig {
            backend: MpmBackendPolicy::PreferGpu,
            max_steps_per_advance: 4,
        },
    )
    .expect("runtime construction");

    let report = runtime.advance(2);
    assert_eq!(report.steps, 2);
    let before_switch = runtime.snapshot().expect("snapshot after fixed steps");
    assert_eq!(before_switch.positions.len(), 4);
    assert!(before_switch
        .positions
        .iter()
        .chain(before_switch.velocities.iter())
        .all(|value| value.is_finite()));

    if runtime.backend() == MpmBackend::Gpu {
        runtime.switch_to_cpu().expect("GPU to CPU synchronization");
        assert_eq!(runtime.backend(), MpmBackend::Cpu);
        assert_eq!(runtime.snapshot().expect("CPU snapshot"), before_switch);
    }
}
