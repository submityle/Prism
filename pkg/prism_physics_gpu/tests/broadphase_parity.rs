//! Real-device parity: the `GPU` broad-phase kernel must reproduce the `CPU`
//! golden twin's candidate-pair set exactly.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: Teschner et al. 2003 spatial hash. No Unreal Engine source or
//! derived code.

use glam::Vec3;
use prism_physics_gpu::broadphase::GpuBroadphase;
use prism_physics_gpu::{cpu_broadphase, BroadphaseConfig, GpuContext, Particle};

/// Builds a deterministic jittered lattice of overlapping spheres.
fn scene() -> Vec<Particle> {
    let dim = 8;
    let spacing = 1.0_f32;
    let radius = 0.62_f32;
    let mut particles = Vec::new();
    let mut seed = 0x1234_5678_u32;
    let mut next = || {
        // xorshift32 for reproducible jitter, no float transcendentals.
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let jitter = Vec3::new(next(), next(), next()) * 0.15;
                let base = Vec3::new(x as f32, y as f32, z as f32) * spacing;
                particles.push(Particle::new(base + jitter, radius));
            }
        }
    }
    particles
}

fn sorted(
    result: Result<Vec<prism_physics_gpu::CandidatePair>, prism_physics_gpu::BroadphaseError>,
) -> Vec<prism_physics_gpu::CandidatePair> {
    let mut pairs = result.expect("broad phase should succeed on the test scene");
    pairs.sort_unstable();
    pairs
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_broadphase_matches_cpu_golden() {
    let particles = scene();
    // diameter 1.24 < cell_size 1.5, so the hash is exact for this scene.
    let config = BroadphaseConfig::new(1.5, 8192, 64, 1 << 20);
    let cpu = sorted(cpu_broadphase(&particles, &config));
    assert!(
        !cpu.is_empty(),
        "the jittered lattice must produce contacts"
    );

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU broad-phase parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBroadphase::new(&ctx);
    let device_pairs = sorted(gpu.run(&ctx, &particles, &config));
    assert_eq!(
        device_pairs, cpu,
        "GPU pair set must equal the CPU golden set"
    );
}
