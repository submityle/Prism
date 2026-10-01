//! Real-device parity for the `GPU` colour-batched cloth bending sweep against
//! its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The sweep is a sequence of graph-coloured compliant `XPBD` bending
//! projections. The colouring and the colour-class visit order are integer
//! exact and identical on both sides, so the only divergence between device and
//! twin is the per-projection fused-multiply-add and division/square-root
//! rounding; parity is therefore checked within a tight tolerance rather than
//! bit-for-bit, the same model the rest of the solver uses.
//!
//! Provenance: the compliant `XPBD` bending projection is the published Müller
//! et al. position-based-dynamics technique. No Unreal Engine source or derived
//! code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_bending, ClothBendingConstraint, GpuClothBending};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible fold jitter.
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

/// Builds a `rows x cols` cloth grid lying in the `xy` plane with a seeded `z`
/// fold jitter, plus the horizontal and vertical bending triples that resist
/// the folds. Returns `(positions, inverse_masses, constraints)`.
fn cloth_grid(rows: usize, cols: usize, seed: u64) -> (Vec<Vec3>, Vec<f32>, Vec<ClothBendingConstraint>) {
    let mut rng = Rng::new(seed);
    let mut positions = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            positions.push(Vec3::new(
                c as f32,
                r as f32,
                // A real out-of-plane fold so bending has something to flatten.
                rng.range(-0.4, 0.4),
            ));
        }
    }
    let inverse_masses = vec![1.0_f32; positions.len()];
    let idx = |r: usize, c: usize| (r * cols + c) as u32;

    let mut constraints = Vec::new();
    // Horizontal hinges: (r,c-1) - (r,c) - (r,c+1).
    for r in 0..rows {
        for c in 1..cols.saturating_sub(1) {
            constraints.push(ClothBendingConstraint {
                a: idx(r, c - 1),
                center: idx(r, c),
                b: idx(r, c + 1),
                rest_offset: 0.0,
                compliance: 0.0,
            });
        }
    }
    // Vertical hinges: (r-1,c) - (r,c) - (r+1,c).
    for r in 1..rows.saturating_sub(1) {
        for c in 0..cols {
            constraints.push(ClothBendingConstraint {
                a: idx(r - 1, c),
                center: idx(r, c),
                b: idx(r + 1, c),
                rest_offset: 0.0,
                compliance: 1.0e-5,
            });
        }
    }
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
        eprintln!("skipping cloth_bending_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothBending::new(&ctx);
    let (positions, inverse_masses, constraints) = cloth_grid(6, 7, 0x5eed_1234);
    let dt = 1.0 / 60.0;
    for iterations in [1_u32, 4, 12] {
        let gpu = kernel.solve(&ctx, &positions, &inverse_masses, &constraints, dt, iterations);
        let cpu = cpu_cloth_bending(&positions, &inverse_masses, &constraints, dt, iterations);
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
        eprintln!("skipping cloth_bending_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothBending::new(&ctx);
    let (positions, mut inverse_masses, constraints) = cloth_grid(5, 5, 0xabcd_0001);
    // Pin the four corners.
    let cols = 5;
    for &p in &[0_usize, cols - 1, inverse_masses.len() - cols, inverse_masses.len() - 1] {
        inverse_masses[p] = 0.0;
    }
    let dt = 1.0 / 60.0;
    let gpu = kernel.solve(&ctx, &positions, &inverse_masses, &constraints, dt, 10);
    let cpu = cpu_cloth_bending(&positions, &inverse_masses, &constraints, dt, 10);
    assert_parity("pinned", &gpu, &cpu);
    // The pinned corners must not have moved on the device.
    for &p in &[0_usize, cols - 1, inverse_masses.len() - cols, inverse_masses.len() - 1] {
        assert!(close(gpu[p], positions[p]), "pinned corner {p} moved");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_empty_constraints_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_bending_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothBending::new(&ctx);
    let (positions, inverse_masses, _) = cloth_grid(4, 4, 0x1);
    let out = kernel.solve(&ctx, &positions, &inverse_masses, &[], 1.0 / 60.0, 8);
    assert_eq!(out, positions, "empty constraint set must be a no-op");
    let zero = kernel.solve(&ctx, &positions, &inverse_masses, &[], 1.0 / 60.0, 0);
    assert_eq!(zero, positions, "zero iterations must be a no-op");
}
