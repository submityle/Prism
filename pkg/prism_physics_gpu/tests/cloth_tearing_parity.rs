//! Real-device parity for the `GPU` cloth tearing (break-flag) kernel against
//! its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Tearing is *per-edge independent*: one thread owns one edge, reads a
//! read-only position snapshot, and writes only its own break flag, the same
//! work the golden ([`cpu_cloth_tearing`]) performs. Both delegate the break
//! decision to the same `prism_physics_core::tear_flag` predicate; the result is
//! an integer flag, so unlike the floating-point projection kernels the flags
//! and the torn count must match *exactly*, with no tolerance.
//!
//! Provenance: removing a constraint whose strain exceeds a threshold is a
//! standard, publicly documented position-based-dynamics technique. No Unreal
//! Engine source or derived code.

use glam::Vec3;
use prism_physics_core::TearingParams;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_tearing, ClothTearEdge, GpuClothTearing};

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

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth tearing parity: no GPU adapter available");
            None
        }
    }
}

#[test]
fn over_strained_edge_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
    let edges = [ClothTearEdge::new(0, 1, 1.0)];
    let params = TearingParams::new(0.5);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&positions, &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_torn, 1);
    assert_eq!(cpu_torn, gpu_torn);
    assert_eq!(cpu_flags, gpu_flags);
}

#[test]
fn within_threshold_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(1.4, 0.0, 0.0)];
    let edges = [ClothTearEdge::new(0, 1, 1.0)];
    let params = TearingParams::new(0.5);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&positions, &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_torn, 0);
    assert_eq!(cpu_torn, gpu_torn);
    assert_eq!(cpu_flags, gpu_flags);
}

#[test]
fn compression_never_tears_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    // Current length 0.1 well below rest 1.0 -> compressive strain, never tears.
    let positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
    let edges = [ClothTearEdge::new(0, 1, 1.0)];
    let params = TearingParams::new(0.5);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&positions, &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_torn, 0);
    assert_eq!(cpu_torn, gpu_torn);
    assert_eq!(cpu_flags, gpu_flags);
}

#[test]
fn degenerate_and_out_of_range_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
    let edges = [
        ClothTearEdge::new(0, 1, 0.0), // degenerate rest length -> never tears
        ClothTearEdge::new(0, 9, 1.0), // out-of-range endpoint -> never tears
        ClothTearEdge::new(0, 1, 1.0), // stretched 5x -> tears
    ];
    let params = TearingParams::new(0.5);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&positions, &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_torn, 1);
    assert_eq!(cpu_torn, gpu_torn);
    assert_eq!(cpu_flags, gpu_flags);
    assert_eq!(gpu_flags, [0, 0, 1]);
}

#[test]
fn nan_threshold_tears_nothing_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    // A huge strain, but a NaN threshold sanitizes to +inf on both paths.
    let positions = [Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
    let edges = [ClothTearEdge::new(0, 1, 1.0)];
    let params = TearingParams::new(f32::NAN);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&positions, &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_torn, 0);
    assert_eq!(cpu_torn, gpu_torn);
    assert_eq!(cpu_flags, gpu_flags);
}

#[test]
fn empty_positions_leave_edges_untouched() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    let edges = [ClothTearEdge::new(0, 1, 1.0), ClothTearEdge::new(2, 3, 2.0)];
    let params = TearingParams::new(0.5);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&[], &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &[], &edges, params);
    assert_eq!(cpu_torn, 0);
    assert_eq!(cpu_torn, gpu_torn);
    assert_eq!(cpu_flags, gpu_flags);
    assert_eq!(gpu_flags, [0, 0]);
}

#[test]
fn boundary_is_strict_on_cpu_and_parity_just_off_it() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    let params = TearingParams::new(0.5);

    // The CPU golden's predicate is strict (`strain > break`), so a strain of
    // exactly 0.5 (len 1.5, rest 1.0) must NOT tear. This is checked on the
    // deterministic CPU path alone: on a real device the sqrt in `distance`
    // can perturb a knife-edge strain by a `ULP`, so an exactly-on-threshold
    // input is inherently ambiguous across hardware and is not a parity target.
    let on_edge = [ClothTearEdge::new(0, 1, 1.0)];
    let on_edge_pos = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&on_edge_pos, &on_edge, params);
    assert_eq!(
        cpu_torn, 0,
        "exact-threshold strain must not tear on the CPU"
    );
    assert_eq!(cpu_flags, [0]);

    // Just inside the threshold (strain 0.49) survives on both paths, and just
    // outside (strain 0.51) tears on both paths; held clearly off the knife
    // edge, the device and the golden must agree exactly.
    let edges = [ClothTearEdge::new(0, 1, 1.0)];
    let inside = [Vec3::ZERO, Vec3::new(1.49, 0.0, 0.0)];
    let (cpu_in, cpu_in_torn) = cpu_cloth_tearing(&inside, &edges, params);
    let (gpu_in, gpu_in_torn) = kernel.solve(&ctx, &inside, &edges, params);
    assert_eq!(cpu_in_torn, 0);
    assert_eq!(cpu_in_torn, gpu_in_torn);
    assert_eq!(cpu_in, gpu_in);

    let outside = [Vec3::ZERO, Vec3::new(1.51, 0.0, 0.0)];
    let (cpu_out, cpu_out_torn) = cpu_cloth_tearing(&outside, &edges, params);
    let (gpu_out, gpu_out_torn) = kernel.solve(&ctx, &outside, &edges, params);
    assert_eq!(cpu_out_torn, 1);
    assert_eq!(cpu_out_torn, gpu_out_torn);
    assert_eq!(cpu_out, gpu_out);
}

#[test]
fn large_random_batch_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothTearing::new(&ctx);
    let mut rng = Rng::new(0x7ea2_4321_0fed_cba9);

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
    // healthy fraction land beyond the break threshold.
    let edges: Vec<ClothTearEdge> = (0..1500)
        .map(|_| {
            let a = rng.index(particle_count);
            let mut b = rng.index(particle_count);
            if b == a {
                b = (b + 1) % particle_count as u32;
            }
            let rest = rng.range(0.2, 6.0);
            ClothTearEdge::new(a, b, rest)
        })
        .collect();

    let params = TearingParams::new(0.3);
    let (cpu_flags, cpu_torn) = cpu_cloth_tearing(&positions, &edges, params);
    let (gpu_flags, gpu_torn) = kernel.solve(&ctx, &positions, &edges, params);
    assert_eq!(cpu_torn, gpu_torn, "torn counts diverged");
    assert!(cpu_torn > 0, "expected some edges to tear");
    assert_eq!(cpu_flags, gpu_flags);
}
