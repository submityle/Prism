//! Real-device parity for the `GPU` cloth plasticity kernel against its `CPU`
//! golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Plasticity is *per-edge independent*: one thread owns one edge, reads a
//! read-only position snapshot, and writes only its own rest length, the same
//! work the golden ([`cpu_cloth_plasticity`]) performs. Both delegate the creep
//! arithmetic to the same `prism_physics_core::plastic_rest_length` kernel, so
//! the only divergence is a few `ULP` in the device division; the rest lengths
//! are checked within a tight relative tolerance and the `modified` count must
//! match exactly.
//!
//! Provenance: rest-length creep past a yield strain is a standard, publicly
//! documented plastic-set model for position-based cloth. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_core::PlasticParams;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_plasticity, ClothPlasticEdge, GpuClothPlasticity};

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

    fn index(&mut self, len: usize) -> u32 {
        (self.next_u64() % (len as u64)) as u32
    }
}

/// Relative tolerance: the device recomputes the strain ratio and the residual
/// clamp with divisions, so only the low bits can diverge from the sequential
/// golden.
const TOL: f32 = 1.0e-4;

fn assert_rest_match(cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "rest-length vector length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!((c - g).abs() <= TOL * scale, "edge {i}: cpu {c} != gpu {g}");
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth plasticity parity: no GPU adapter available");
            None
        }
    }
}

#[test]
fn within_yield_band_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(1.05, 0.0, 0.0)];
    let edges = [ClothPlasticEdge::new(0, 1, 1.0)];
    let params = PlasticParams::new(0.1, 0.5, 1.0);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, 0);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
}

#[test]
fn beyond_yield_creep_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
    let edges = [ClothPlasticEdge::new(0, 1, 1.0)];
    let params = PlasticParams::new(0.1, 0.5, 10.0);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, 1);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
}

#[test]
fn residual_cap_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
    let edges = [ClothPlasticEdge::new(0, 1, 1.0)];
    let params = PlasticParams::new(0.1, 1.0, 0.2);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
    // Residual elastic strain is clamped to max_strain on both paths.
    let residual = (2.0 - gpu_rest[0]) / gpu_rest[0];
    assert!(residual.abs() <= 0.2 + 1e-4, "residual {residual}");
}

#[test]
fn compressed_edge_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    // Current length 0.5 well below rest 1.0 -> compressive strain -0.5.
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let edges = [ClothPlasticEdge::new(0, 1, 1.0)];
    let params = PlasticParams::new(0.1, 0.6, 5.0);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, 1);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
    assert!(gpu_rest[0] < 1.0, "rest should shrink: {}", gpu_rest[0]);
}

#[test]
fn degenerate_and_out_of_range_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
    let edges = [
        ClothPlasticEdge::new(0, 1, 0.0), // degenerate rest length
        ClothPlasticEdge::new(0, 9, 1.0), // out-of-range endpoint
        ClothPlasticEdge::new(0, 1, 1.0), // stretched, should creep
    ];
    let params = PlasticParams::new(0.1, 0.5, 10.0);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, 1);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
}

#[test]
fn empty_positions_leave_edges_untouched() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let edges = [
        ClothPlasticEdge::new(0, 1, 1.0),
        ClothPlasticEdge::new(2, 3, 2.0),
    ];
    let params = PlasticParams::new(0.1, 0.5, 10.0);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&[], &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &[], &edges, params);
    assert_eq!(cpu_mod, 0);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
}

#[test]
fn large_random_batch_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let mut rng = Rng::new(0x5eed_1234_9abc_def0);

    let particle_count = 400usize;
    let positions: Vec<Vec3> = (0..particle_count)
        .map(|_| {
            Vec3::new(
                rng.range(-5.0, 5.0),
                rng.range(-5.0, 5.0),
                rng.range(-5.0, 5.0),
            )
        })
        .collect();

    // ~1500 edges spanning multiple workgroups, with assorted rest lengths so a
    // healthy fraction land beyond the yield band in both directions.
    let edges: Vec<ClothPlasticEdge> = (0..1500)
        .map(|_| {
            let a = rng.index(particle_count);
            let mut b = rng.index(particle_count);
            if b == a {
                b = (b + 1) % particle_count as u32;
            }
            let rest = rng.range(0.2, 6.0);
            ClothPlasticEdge::new(a, b, rest)
        })
        .collect();

    let params = PlasticParams::new(0.07, 0.35, 0.4);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, gpu_mod, "modified counts diverged");
    assert!(cpu_mod > 0, "expected some edges to creep");
    assert_rest_match(&cpu_rest, &gpu_rest);
}

#[test]
fn zero_creep_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothPlasticity::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)];
    let edges = [ClothPlasticEdge::new(0, 1, 1.0)];
    // creep = 0 leaves the rest length at its input value even beyond yield;
    // the edge is still counted as visited, so parity covers the count too.
    let params = PlasticParams::new(0.1, 0.0, 10.0);
    let (cpu_rest, cpu_mod) = cpu_cloth_plasticity(&positions, &edges, params);
    let (gpu_rest, gpu_mod) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_mod, gpu_mod);
    assert_rest_match(&cpu_rest, &gpu_rest);
}
