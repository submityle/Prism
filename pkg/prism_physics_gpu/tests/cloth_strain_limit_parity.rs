//! Real-device parity for the `GPU` colour-batched cloth strain-limit sweep
//! against its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The sweep is a sequence of graph-coloured biphasic length clamps. The
//! colouring and the colour-class visit order are integer exact and identical
//! on both sides, so the only divergence between device and twin is the
//! per-projection fused-multiply-add and division/square-root rounding; parity
//! is therefore checked within a tight tolerance rather than bit-for-bit, the
//! same model the rest of the solver uses.
//!
//! Provenance: biphasic strain limiting is a standard, publicly documented
//! cloth technique (Provot 1995; Thomaszewski et al. 2009). No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_strain_limit, ClothStrainLimitConstraint, GpuClothStrainLimit};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible jitter.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// Relative per-component tolerance: each projection is a normalise (division
/// and square root, with `GPU` fused multiply-add in the dot product) so only
/// the low bits can diverge.
const TOL: f32 = 1.0e-4;

fn close(a: Vec3, b: Vec3) -> bool {
    let scale = a.abs().max(b.abs()).max(Vec3::ONE);
    (a.x - b.x).abs() <= TOL * scale.x
        && (a.y - b.y).abs() <= TOL * scale.y
        && (a.z - b.z).abs() <= TOL * scale.z
}

/// Builds a `rows x cols` cloth grid in the `xy` plane with unit rest spacing
/// and a seeded radial blow-out so a good fraction of the structural edges
/// overshoot their stretch cap (and some compress below the floor). The
/// structural edges (horizontal and vertical neighbours) each get a biphasic
/// limiter. Returns `(positions, inverse_masses, constraints)`.
fn cloth_grid(
    rows: usize,
    cols: usize,
    seed: u64,
) -> (Vec<Vec3>, Vec<f32>, Vec<ClothStrainLimitConstraint>) {
    let mut rng = Rng::new(seed);
    let mut positions = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let rest = Vec3::new(c as f32, r as f32, 0.0);
            // Push each particle a seeded random distance off its rest point so
            // some edges stretch past the cap and some compress below the floor.
            let jitter = Vec3::new(
                rng.range(-0.5, 0.5),
                rng.range(-0.5, 0.5),
                rng.range(-0.3, 0.3),
            );
            positions.push(rest + jitter);
        }
    }
    let mut constraints = Vec::new();
    let idx = |r: usize, c: usize| (r * cols + c) as u32;
    // Horizontal structural edges.
    for r in 0..rows {
        for c in 0..cols - 1 {
            constraints.push(ClothStrainLimitConstraint::new(
                idx(r, c),
                idx(r, c + 1),
                1.0,
                1.1,
                0.5,
            ));
        }
    }
    // Vertical structural edges.
    for r in 0..rows - 1 {
        for c in 0..cols {
            constraints.push(ClothStrainLimitConstraint::new(
                idx(r, c),
                idx(r + 1, c),
                1.0,
                1.1,
                0.5,
            ));
        }
    }
    let inverse_masses = vec![1.0_f32; positions.len()];
    (positions, inverse_masses, constraints)
}

fn assert_parity(label: &str, gpu: &[Vec3], cpu: &[Vec3]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            close(*g, *c),
            "{label}: particle {i} diverged: gpu={g:?} cpu={c:?}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_across_iterations() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_strain_limit_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothStrainLimit::new(&ctx);
    let (positions, inverse_masses, constraints) = cloth_grid(6, 7, 0x5eed_abcd);
    for iterations in [1_u32, 4, 12] {
        let gpu = kernel.solve(&ctx, &positions, &inverse_masses, &constraints, iterations);
        let cpu = cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, iterations);
        assert_parity(&format!("iterations={iterations}"), &gpu, &cpu);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_with_pinned_particles() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_strain_limit_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothStrainLimit::new(&ctx);
    let (positions, mut inverse_masses, constraints) = cloth_grid(5, 5, 0xfeed_0002);
    let cols = 5;
    let pins = [
        0_usize,
        cols - 1,
        inverse_masses.len() - cols,
        inverse_masses.len() - 1,
    ];
    for &p in &pins {
        inverse_masses[p] = 0.0;
    }
    let gpu = kernel.solve(&ctx, &positions, &inverse_masses, &constraints, 10);
    let cpu = cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, 10);
    assert_parity("pinned", &gpu, &cpu);
    for &p in &pins {
        assert!(close(gpu[p], positions[p]), "pinned corner {p} moved");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_with_shared_particle_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_strain_limit_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothStrainLimit::new(&ctx);
    // A fan of edges all incident on particle 0, forcing a multi-colour
    // sequential schedule that must be reproduced on the device.
    let positions = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(0.2, 0.2, 0.0),
    ];
    let inverse_masses = vec![1.0_f32; positions.len()];
    let constraints = vec![
        ClothStrainLimitConstraint::new(0, 1, 1.0, 1.1, 0.5),
        ClothStrainLimitConstraint::new(0, 2, 1.0, 1.1, 0.5),
        ClothStrainLimitConstraint::new(0, 3, 1.0, 1.1, 0.5),
        ClothStrainLimitConstraint::new(0, 4, 1.0, 1.1, 0.5),
    ];
    for iterations in [1_u32, 5, 16] {
        let gpu = kernel.solve(&ctx, &positions, &inverse_masses, &constraints, iterations);
        let cpu = cpu_cloth_strain_limit(&positions, &inverse_masses, &constraints, iterations);
        assert_parity(&format!("shared/iterations={iterations}"), &gpu, &cpu);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_empty_constraints_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_strain_limit_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothStrainLimit::new(&ctx);
    let (positions, inverse_masses, _) = cloth_grid(4, 4, 0x3);
    let out = kernel.solve(&ctx, &positions, &inverse_masses, &[], 8);
    assert_eq!(out, positions, "empty constraint set must be a no-op");
    let zero = kernel.solve(&ctx, &positions, &inverse_masses, &[], 0);
    assert_eq!(zero, positions, "zero iterations must be a no-op");
}
