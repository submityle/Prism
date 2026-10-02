//! Real-device parity tests for the incremental resident `BVH` refit.
//!
//! Each test builds a tree resident on device with
//! [`GpuLbvh::build_resident`], refits it in place for a new set of primitive
//! boxes with [`GpuBvhRefit::refit`], reads the updated internal-node bounds
//! back, and asserts they equal the [`cpu_refit_lbvh`] golden twin's bounds
//! over the host-built [`Lbvh`]. Because the refit is only an integer
//! permutation (the unchanged sorted payload) plus exact componentwise min/max
//! reductions over the same parent links, the device and twin bounds are
//! bit-for-bit identical. Every test skips cleanly when no adapter is available
//! (for example inside a sandbox) so the suite never fails for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own refit kernel (bottom-up bounds over the
//! linear `BVH` of Karras, 2012) against its `CPU` twin. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_refit_lbvh, Aabb, GpuBvhRefit, GpuContext, GpuLbvh,
};

/// A small xorshift generator so the tests are deterministic.
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

    fn coord(&mut self, lo: f32, span: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * span
    }
}

/// A box of half-extent `h` centred on a random point in a cube of side
/// `2 * span`.
fn random_box(rng: &mut Rng, span: f32, h: f32) -> Aabb {
    let c = Vec3::new(
        rng.coord(-span, 2.0 * span),
        rng.coord(-span, 2.0 * span),
        rng.coord(-span, 2.0 * span),
    );
    let half = Vec3::splat(h);
    Aabb::new(c - half, c + half)
}

/// Builds `original` resident, refits to `moved` on device, and asserts the
/// internal bounds match the `CPU` twin refit bit-for-bit.
fn run_refit_parity(ctx: &GpuContext, original: &[Aabb], moved: &[Aabb]) {
    let builder = GpuLbvh::new(ctx);
    let refitter = GpuBvhRefit::new(ctx);

    let tree = builder.build_resident(ctx, original);
    refitter.refit(ctx, &tree, moved);
    let gpu_bounds = refitter.read_internal_aabb(ctx, &tree);

    let host = cpu_build_lbvh(original);
    let cpu = cpu_refit_lbvh(&host, moved);

    assert_eq!(
        gpu_bounds.len(),
        cpu.internal_aabb.len(),
        "internal node count must match"
    );
    for (i, (g, c)) in gpu_bounds.iter().zip(cpu.internal_aabb.iter()).enumerate() {
        assert_eq!(g.min, c.min, "node {i} min");
        assert_eq!(g.max, c.max, "node {i} max");
    }

    // A second refit on the same resident tree must reset and recompute cleanly,
    // not accumulate the previous bounds.
    refitter.refit(ctx, &tree, original);
    let gpu_again = refitter.read_internal_aabb(ctx, &tree);
    let cpu_again = cpu_refit_lbvh(&host, original);
    for (i, (g, c)) in gpu_again
        .iter()
        .zip(cpu_again.internal_aabb.iter())
        .enumerate()
    {
        assert_eq!(g.min, c.min, "re-refit node {i} min");
        assert_eq!(g.max, c.max, "re-refit node {i} max");
    }

}

#[test]
#[expect(clippy::print_stderr, reason = "surface GPU-skip reason")]
fn refit_small_translation_matches_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = Rng::new(0x0bu64 ^ 0x9e37_79b9);
    let original: Vec<Aabb> = (0..64).map(|_| random_box(&mut rng, 10.0, 0.5)).collect();
    let moved: Vec<Aabb> = original
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let d = Vec3::new(0.1 * ((i % 9) as f32 - 4.0), -0.05, 0.02 * i as f32);
            Aabb::new(b.min + d, b.max + d)
        })
        .collect();
    run_refit_parity(&ctx, &original, &moved);
}

#[test]
#[expect(clippy::print_stderr, reason = "surface GPU-skip reason")]
fn refit_large_resize_matches_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = Rng::new(0x1234_5678_9abc_def0);
    let original: Vec<Aabb> = (0..129).map(|_| random_box(&mut rng, 25.0, 1.0)).collect();
    // Grow and shift every box substantially; the refit must still produce the
    // exact union bounds over the unchanged topology.
    let moved: Vec<Aabb> = original
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let grow = Vec3::splat(0.5 + 0.1 * (i % 7) as f32);
            let shift = Vec3::new(2.0, -1.5, 0.75);
            Aabb::new(b.min - grow + shift, b.max + grow + shift)
        })
        .collect();
    run_refit_parity(&ctx, &original, &moved);
}

#[test]
#[expect(clippy::print_stderr, reason = "surface GPU-skip reason")]
fn refit_two_leaf_tree_matches_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let original = vec![
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        Aabb::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0)),
    ];
    let moved = vec![
        Aabb::new(Vec3::new(-3.0, -2.0, -1.0), Vec3::new(0.0, 0.0, 0.0)),
        Aabb::new(Vec3::new(8.0, 4.0, 2.0), Vec3::new(9.0, 5.0, 3.0)),
    ];
    run_refit_parity(&ctx, &original, &moved);
}

#[test]
#[expect(clippy::print_stderr, reason = "surface GPU-skip reason")]
fn refit_trivial_tree_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let refitter = GpuBvhRefit::new(&ctx);

    // One-leaf tree has no internal nodes; refit is a no-op and reads back empty.
    let tree = builder.build_resident(&ctx, &[Aabb::new(Vec3::ZERO, Vec3::ONE)]);
    refitter.refit(&ctx, &tree, &[Aabb::new(Vec3::splat(4.0), Vec3::splat(5.0))]);
    assert!(refitter.read_internal_aabb(&ctx, &tree).is_empty());
}
